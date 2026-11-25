"""OpenAI Chat Completions adapter implementing the provider-agnostic LLMClient protocol."""

from __future__ import annotations

from typing import Any

from calendar_agent.llm.protocol import Message, ModelTurn, ToolCall, ToolSpec


class OpenAIClient:
    """Translates between our provider-neutral types and OpenAI's Chat Completions API.

    Accepts an injected `client` (matching the official `openai.OpenAI` shape) so it can
    be unit-tested with a stub without making network calls. For live runs, construct it
    with `OpenAIClient.from_env()`.
    """

    def __init__(self, client: Any, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_env(
        cls,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> OpenAIClient:
        """Build a live OpenAIClient using OPENAI_API_KEY (or an explicit api_key)."""
        from openai import OpenAI

        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        return cls(client=client, model=model)

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [self.format_message(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = [self.format_tool(t) for t in tools]

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        raw_calls = message.tool_calls or []
        parsed_calls = tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in raw_calls
        )
        return ModelTurn(content=message.content, tool_calls=parsed_calls)

    @staticmethod
    def format_tool(tool: ToolSpec) -> dict[str, Any]:
        """Translate a neutral ToolSpec into OpenAI's function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def format_message(message: Message) -> dict[str, Any]:
        """Translate a neutral Message into an OpenAI Chat Completions message dict."""
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content or "",
            }

        payload: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.role == "assistant" and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return payload
