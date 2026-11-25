"""Step 3 tests: provider-agnostic agent loop, tool_call_id pairing, recovery, and OpenAI adapter."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from calendar_agent.agent import CalendarAgent
from calendar_agent.llm import Message, ModelTurn, OpenAIClient, ToolCall, ToolSpec
from calendar_agent.tools import build_calendar_tools
from tests.fakes import FakeLLMClient, InMemoryCalendar


class TestCalendarAgentLoop:
    def test_direct_reply_stops_after_one_llm_call(
        self, memory_calendar: InMemoryCalendar
    ) -> None:
        llm = FakeLLMClient(turns=[ModelTurn(content="Hello! How can I help with your calendar?")])
        agent = CalendarAgent(llm=llm, tools=build_calendar_tools(memory_calendar))

        result = agent.run("Hi there!")

        assert result.reply == "Hello! How can I help with your calendar?"
        assert result.steps == ()
        assert result.reached_max_steps is False
        assert len(llm.recorded_calls) == 1

    def test_check_availability_then_create_event(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        llm = FakeLLMClient(
            turns=[
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="call_1",
                            name="list_events",
                            arguments=json.dumps(
                                {
                                    "start": "2026-12-01T13:00:00-08:00",
                                    "end": "2026-12-01T18:00:00-08:00",
                                }
                            ),
                        ),
                    )
                ),
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="call_2",
                            name="create_event",
                            arguments=json.dumps(
                                {
                                    "title": "Meeting with John",
                                    "start": "2026-12-01T15:00:00-08:00",
                                    "duration_minutes": 60,
                                }
                            ),
                        ),
                    )
                ),
                ModelTurn(content="Booked 'Meeting with John' on Dec 1 from 3:00 PM to 4:00 PM."),
            ]
        )
        agent = CalendarAgent(llm=llm, tools=build_calendar_tools(memory_calendar))

        result = agent.run("Schedule a meeting with John on Dec 1 at 3 PM for one hour.")

        assert result.reached_max_steps is False
        assert len(result.steps) == 2
        assert result.steps[0].executions[0].call.name == "list_events"
        assert result.steps[1].executions[0].call.name == "create_event"
        assert "Booked 'Meeting with John'" in result.reply

        # Verify the event actually exists in the calendar backend
        booked = memory_calendar.list_events(
            datetime(2026, 12, 1, 15, 0, tzinfo=tz),
            datetime(2026, 12, 1, 16, 0, tzinfo=tz),
        )
        assert [e.title for e in booked] == ["Meeting with John"]

    def test_parallel_tool_calls_pair_every_tool_call_id(
        self, memory_calendar: InMemoryCalendar
    ) -> None:
        llm = FakeLLMClient(
            turns=[
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="call_tue",
                            name="list_events",
                            arguments=json.dumps(
                                {
                                    "start": "2026-12-01T12:00:00-08:00",
                                    "end": "2026-12-01T17:00:00-08:00",
                                }
                            ),
                        ),
                        ToolCall(
                            id="call_wed",
                            name="list_events",
                            arguments=json.dumps(
                                {
                                    "start": "2026-12-02T12:00:00-08:00",
                                    "end": "2026-12-02T17:00:00-08:00",
                                }
                            ),
                        ),
                    )
                ),
                ModelTurn(content="Tuesday has Design Review 2-3 PM; Wednesday afternoon is free."),
            ]
        )
        agent = CalendarAgent(llm=llm, tools=build_calendar_tools(memory_calendar))

        result = agent.run("Am I free Tuesday or Wednesday afternoon?")

        assert len(result.steps) == 1
        assert len(result.steps[0].executions) == 2

        # Inspect the exact message history sent to the LLM on iteration #2
        second_call_messages, _ = llm.recorded_calls[1]
        assert [m.role for m in second_call_messages] == [
            "system",
            "user",
            "assistant",
            "tool",
            "tool",
        ]
        assert second_call_messages[3].tool_call_id == "call_tue"
        assert second_call_messages[4].tool_call_id == "call_wed"
        assert json.loads(second_call_messages[3].content or "")["data"]["count"] == 1
        assert json.loads(second_call_messages[4].content or "")["data"]["count"] == 0

    def test_recovers_from_conflict_error_without_crashing(
        self, memory_calendar: InMemoryCalendar
    ) -> None:
        llm = FakeLLMClient(
            turns=[
                # Model mistakenly tries to book over Design Review (14:00-15:00)
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="call_conflict",
                            name="create_event",
                            arguments=json.dumps(
                                {
                                    "title": "Meeting with John",
                                    "start": "2026-12-01T14:00:00-08:00",
                                    "duration_minutes": 60,
                                }
                            ),
                        ),
                    )
                ),
                ModelTurn(
                    content=(
                        "2:00 PM conflicts with Design Review (2:00-3:00 PM). "
                        "Would you like me to book 3:00 PM instead?"
                    )
                ),
            ]
        )
        agent = CalendarAgent(llm=llm, tools=build_calendar_tools(memory_calendar))

        result = agent.run("Book a meeting with John Dec 1 at 2 PM for 1 hour.")

        assert result.reached_max_steps is False
        tool_output = json.loads(result.steps[0].executions[0].output)
        assert tool_output["ok"] is False
        assert tool_output["error_type"] == "conflict"
        assert "Design Review" in result.reply

    def test_self_corrects_after_validation_error(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        llm = FakeLLMClient(
            turns=[
                # Attempt 1: naive timestamp (missing timezone offset)
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="call_bad_tz",
                            name="create_event",
                            arguments=json.dumps(
                                {
                                    "title": "Coffee with John",
                                    "start": "2026-12-01T16:00:00",
                                    "duration_minutes": 30,
                                }
                            ),
                        ),
                    )
                ),
                # Attempt 2: model reads validation_error and retries with offset
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="call_fixed_tz",
                            name="create_event",
                            arguments=json.dumps(
                                {
                                    "title": "Coffee with John",
                                    "start": "2026-12-01T16:00:00-08:00",
                                    "duration_minutes": 30,
                                }
                            ),
                        ),
                    )
                ),
                ModelTurn(content="Booked 'Coffee with John' at 4:00 PM."),
            ]
        )
        agent = CalendarAgent(llm=llm, tools=build_calendar_tools(memory_calendar))

        result = agent.run("Schedule Coffee with John at 4 PM for 30 minutes.")

        first_output = json.loads(result.steps[0].executions[0].output)
        second_output = json.loads(result.steps[1].executions[0].output)
        assert first_output["error_type"] == "validation_error"
        assert second_output["ok"] is True
        assert len(
            memory_calendar.list_events(
                datetime(2026, 12, 1, 16, 0, tzinfo=tz),
                datetime(2026, 12, 1, 16, 30, tzinfo=tz),
            )
        ) == 1

    def test_stops_gracefully_at_max_steps_cap(
        self, memory_calendar: InMemoryCalendar
    ) -> None:
        infinite_tool_turn = ModelTurn(
            tool_calls=(
                ToolCall(
                    id="call_loop",
                    name="list_events",
                    arguments=json.dumps(
                        {
                            "start": "2026-12-01T12:00:00-08:00",
                            "end": "2026-12-01T13:00:00-08:00",
                        }
                    ),
                ),
            )
        )
        llm = FakeLLMClient(turns=[infinite_tool_turn, infinite_tool_turn, infinite_tool_turn])
        agent = CalendarAgent(
            llm=llm,
            tools=build_calendar_tools(memory_calendar),
            max_steps=2,
        )

        result = agent.run("Keep checking forever.")

        assert result.reached_max_steps is True
        assert len(result.steps) == 2
        assert len(llm.recorded_calls) == 2
        assert "maximum number of tool steps" in result.reply

    def test_rejects_non_positive_max_steps(self, memory_calendar: InMemoryCalendar) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            CalendarAgent(
                llm=FakeLLMClient(turns=[]),
                tools=build_calendar_tools(memory_calendar),
                max_steps=0,
            )


class TestOpenAIClientAdapter:
    def test_translates_messages_tools_and_response(self) -> None:
        captured_kwargs: dict[str, Any] = {}

        class _FakeCompletions:
            def create(self, **kwargs: Any) -> Any:
                captured_kwargs.update(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        id="call_openai_1",
                                        function=SimpleNamespace(
                                            name="list_events",
                                            arguments='{"start": "2026-12-01T12:00:00-08:00", "end": "2026-12-01T17:00:00-08:00"}',
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                )

        fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
        client = OpenAIClient(client=fake_sdk, model="gpt-4o-mini")

        messages = [
            Message.system("You are helpful."),
            Message.user("Check Tuesday."),
            Message.assistant(
                content=None,
                tool_calls=(ToolCall(id="call_prev", name="list_events", arguments="{}"),),
            ),
            Message.tool_result(tool_call_id="call_prev", tool_name="list_events", content='{"ok": true}'),
        ]
        tools = [
            ToolSpec(
                name="list_events",
                description="List events.",
                parameters={"type": "object", "properties": {}, "required": []},
            )
        ]

        turn = client.complete(messages=messages, tools=tools)

        # Verify OpenAI wire format sent outbound
        assert captured_kwargs["model"] == "gpt-4o-mini"
        assert captured_kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "list_events",
                    "description": "List events.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
        assert captured_kwargs["messages"][2]["tool_calls"][0]["id"] == "call_prev"
        assert captured_kwargs["messages"][3] == {
            "role": "tool",
            "tool_call_id": "call_prev",
            "content": '{"ok": true}',
        }

        # Verify normalized ModelTurn returned inbound
        assert turn.content is None
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].id == "call_openai_1"
        assert turn.tool_calls[0].name == "list_events"
