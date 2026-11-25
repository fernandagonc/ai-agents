"""Step 3: Provider-agnostic tool-calling agent orchestration loop (no frameworks)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from calendar_agent.llm.protocol import LLMClient, Message, ToolCall
from calendar_agent.prompts import build_system_prompt
from calendar_agent.tools import ToolRegistry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ToolExecutionTrace:
    """Record of one tool call and the JSON output returned to the model."""

    call: ToolCall
    output: str


@dataclass(frozen=True)
class AgentStep:
    """One tool-calling iteration of the agent loop (may contain parallel tool calls)."""

    executions: tuple[ToolExecutionTrace, ...]


@dataclass(frozen=True)
class AgentRunResult:
    """Final output and full trace produced by a single `CalendarAgent.run()` call."""

    reply: str
    steps: tuple[AgentStep, ...]
    messages: tuple[Message, ...]
    reached_max_steps: bool = False


class CalendarAgent:
    """Minimal, provider-agnostic ReAct / tool-calling orchestration loop.

    Loop structure:
        1. Send conversation history + tool specs to the LLM.
        2. Stop condition #1: if the LLM returns no tool calls, return its reply.
        3. Otherwise, execute EVERY requested tool call and append a paired
           `Message.tool_result(tool_call_id=...)` for each one.
        4. Repeat until Stop condition #1 or Stop condition #2 (`max_steps` cap).
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str | None = None,
        clock: Callable[[], datetime] | None = None,
        timezone_name: str | None = None,
        max_steps: int = 8,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._llm = llm
        self._tools = tools
        self._static_system_prompt = system_prompt
        self._clock = clock or _utc_now
        self._timezone_name = timezone_name
        self._max_steps = max_steps

    def _resolve_system_prompt(self) -> str:
        if self._static_system_prompt is not None:
            return self._static_system_prompt
        return build_system_prompt(now=self._clock(), timezone_name=self._timezone_name)

    def run(
        self,
        user_input: str,
        history: list[Message] | None = None,
    ) -> AgentRunResult:
        system_message = Message.system(self._resolve_system_prompt())
        if history:
            messages = list(history)
            # Keep current datetime fresh across multi-turn sessions.
            if messages[0].role == "system":
                messages[0] = system_message
            else:
                messages.insert(0, system_message)
        else:
            messages = [system_message]
        messages.append(Message.user(user_input))

        tool_specs = self._tools.tool_specs()
        steps: list[AgentStep] = []

        for _ in range(self._max_steps):
            turn = self._llm.complete(messages=messages, tools=tool_specs)
            messages.append(
                Message.assistant(content=turn.content, tool_calls=turn.tool_calls)
            )

            # Stop condition #1: model chose to answer the user directly.
            if not turn.tool_calls:
                return AgentRunResult(
                    reply=turn.content or "",
                    steps=tuple(steps),
                    messages=tuple(messages),
                    reached_max_steps=False,
                )

            # Execute every requested tool call and pair each result by tool_call_id.
            executions: list[ToolExecutionTrace] = []
            for call in turn.tool_calls:
                output = self._tools.execute(call.name, call.arguments)
                messages.append(
                    Message.tool_result(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=output,
                    )
                )
                executions.append(ToolExecutionTrace(call=call, output=output))

            steps.append(AgentStep(executions=tuple(executions)))

        # Stop condition #2: safety cap reached—exit gracefully instead of looping forever.
        return AgentRunResult(
            reply=(
                "I reached the maximum number of tool steps before finishing. "
                "Please try narrowing or rephrasing your request."
            ),
            steps=tuple(steps),
            messages=tuple(messages),
            reached_max_steps=True,
        )
