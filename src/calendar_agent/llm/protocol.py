"""Provider-agnostic conversation and tool-calling primitives.

These types capture the universal concepts shared across OpenAI, Anthropic Claude,
and Google Gemini without tying our agent loop to any single SDK's wire syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral description of a tool an LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Message:
    """One turn in the stateless conversation history sent to an LLM."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, tool_name: str, content: str) -> Message:
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )


@dataclass(frozen=True)
class ModelTurn:
    """Normalized output from a single LLM completion call."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class LLMClient(Protocol):
    """Minimal contract every LLM provider adapter (OpenAI, Claude, Gemini, Fake) satisfies."""

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> ModelTurn:
        """Send the conversation history and available tools, returning the model's next turn."""
        ...
