"""Step 5: Multi-turn conversation state and turn-boundary context trimming."""

from __future__ import annotations

from calendar_agent.agent import AgentRunResult, CalendarAgent
from calendar_agent.llm.protocol import Message


def trim_history_by_turns(messages: list[Message], max_turns: int) -> list[Message]:
    """Keep the system message plus the last `max_turns` complete user turns.

    Why turn-boundary trimming matters:
        Every user turn starts at a `role="user"` message where zero tool calls are
        pending. Dropping whole turns from the front preserves every
        `assistant(tool_calls)` <-> `tool(tool_call_id)` pair intact, preventing
        HTTP 400 orphaned-tool-result errors from LLM APIs.
    """
    if max_turns <= 0:
        raise ValueError("max_turns must be positive.")
    if not messages:
        return []

    prefix: list[Message] = []
    start_idx = 0
    if messages[0].role == "system":
        prefix.append(messages[0])
        start_idx = 1

    # Group subsequent messages into indivisible blocks starting at each user message.
    turns: list[list[Message]] = []
    for msg in messages[start_idx:]:
        if msg.role == "user" or not turns:
            turns.append([msg])
        else:
            turns[-1].append(msg)

    kept_turns = turns[-max_turns:]
    flattened = [msg for turn in kept_turns for msg in turn]
    return [*prefix, *flattened]


class ConversationSession:
    """Manages multi-turn memory across sequential `send()` calls to a CalendarAgent."""

    def __init__(self, agent: CalendarAgent, max_turns: int = 10) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive.")
        self._agent = agent
        self._max_turns = max_turns
        self._history: list[Message] = []

    @property
    def history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    @property
    def turn_count(self) -> int:
        return sum(1 for msg in self._history if msg.role == "user")

    def send(self, user_input: str) -> AgentRunResult:
        """Send a user message, run the agent loop, and trim retained history safely."""
        result = self._agent.run(
            user_input=user_input,
            history=self._history if self._history else None,
        )
        self._history = trim_history_by_turns(
            list(result.messages),
            max_turns=self._max_turns,
        )
        return result

    def reset(self) -> None:
        """Clear all conversation memory so the next message starts fresh."""
        self._history = []
