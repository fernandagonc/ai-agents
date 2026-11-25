"""Step 5 tests: turn-boundary memory trimming, multi-turn sessions, html_link, and CLI REPL."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from calendar_agent.agent import CalendarAgent
from calendar_agent.calendar import GoogleCalendarClient
from calendar_agent.cli import format_step_trace, run_repl
from calendar_agent.llm import Message, ModelTurn, ToolCall
from calendar_agent.models import CreateEventRequest
from calendar_agent.session import ConversationSession, trim_history_by_turns
from calendar_agent.tools import build_calendar_tools
from tests.fakes import FakeLLMClient, InMemoryCalendar


def _assert_valid_tool_pairing(messages: list[Message]) -> None:
    """Invariant check: every tool message must follow an assistant call with the same ID."""
    pending_ids: list[str] = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            assert pending_ids == [], "Previous tool calls were not answered before new assistant turn."
            pending_ids = [call.id for call in msg.tool_calls]
        elif msg.role == "tool":
            assert msg.tool_call_id in pending_ids, f"Orphaned tool_call_id: {msg.tool_call_id}"
            pending_ids.remove(msg.tool_call_id)
        else:
            assert pending_ids == [], "Unanswered tool calls before non-tool message."
    assert pending_ids == []


class TestTurnBoundaryTrimming:
    def test_drops_whole_turns_preserving_system_and_tool_pairs(self) -> None:
        history = [
            Message.system("System prompt"),
            # Turn 1 (4 messages: user + assistant tool call + tool result + assistant reply)
            Message.user("Turn 1"),
            Message.assistant(tool_calls=(ToolCall(id="c1", name="list_events", arguments="{}"),)),
            Message.tool_result(tool_call_id="c1", tool_name="list_events", content='{"ok": true}'),
            Message.assistant("Reply 1"),
            # Turn 2 (4 messages)
            Message.user("Turn 2"),
            Message.assistant(tool_calls=(ToolCall(id="c2", name="create_event", arguments="{}"),)),
            Message.tool_result(tool_call_id="c2", tool_name="create_event", content='{"ok": true}'),
            Message.assistant("Reply 2"),
            # Turn 3 (2 messages)
            Message.user("Turn 3"),
            Message.assistant("Reply 3"),
        ]

        trimmed = trim_history_by_turns(history, max_turns=2)

        assert trimmed[0].role == "system"
        assert [m.content for m in trimmed if m.role == "user"] == ["Turn 2", "Turn 3"]
        assert len(trimmed) == 1 + 4 + 2  # system + Turn 2 + Turn 3
        _assert_valid_tool_pairing(trimmed)

    def test_handles_empty_and_systemless_histories(self) -> None:
        assert trim_history_by_turns([], max_turns=3) == []
        no_sys = [
            Message.user("U1"),
            Message.assistant("A1"),
            Message.user("U2"),
            Message.assistant("A2"),
        ]
        trimmed = trim_history_by_turns(no_sys, max_turns=1)
        assert [m.content for m in trimmed] == ["U2", "A2"]

    def test_rejects_non_positive_max_turns(self) -> None:
        with pytest.raises(ValueError, match="max_turns"):
            trim_history_by_turns([], max_turns=0)


class TestConversationSession:
    def test_multi_turn_follow_up_and_trimming(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        llm = FakeLLMClient(
            turns=[
                # Turn 1: user asks for 2 PM -> hits conflict -> agent asks about 3 PM
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="call_1",
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
                ModelTurn(content="2 PM conflicts with Design Review. Book 3 PM instead?"),
                # Turn 2: user says "Yes, book 3 PM" -> agent books 3 PM
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
                ModelTurn(content="Booked 'Meeting with John' from 3:00-4:00 PM!"),
                # Turn 3: simple follow-up triggers trimming with max_turns=2
                ModelTurn(content="You're welcome!"),
            ]
        )
        agent = CalendarAgent(
            llm=llm,
            tools=build_calendar_tools(memory_calendar),
            clock=lambda: datetime(2026, 11, 25, 9, 0, tzinfo=tz),
            timezone_name="America/Los_Angeles",
        )
        session = ConversationSession(agent=agent, max_turns=2)

        r1 = session.send("Schedule a meeting with John Dec 1 at 2 PM for 1 hour.")
        assert "Design Review" in r1.reply
        assert session.turn_count == 1

        r2 = session.send("Yes, book 3 PM instead.")
        assert "3:00-4:00 PM" in r2.reply
        assert session.turn_count == 2

        # On Turn 2, the LLM received Turn 1's full context + Turn 2's user message
        turn2_first_call_msgs, _ = llm.recorded_calls[2]
        assert [m.content for m in turn2_first_call_msgs if m.role == "user"] == [
            "Schedule a meeting with John Dec 1 at 2 PM for 1 hour.",
            "Yes, book 3 PM instead.",
        ]

        # Turn 3 exceeds max_turns=2 -> Turn 1 dropped cleanly as a whole unit
        session.send("Thanks!")
        assert session.turn_count == 2
        assert [m.content for m in session.history if m.role == "user"] == [
            "Yes, book 3 PM instead.",
            "Thanks!",
        ]
        _assert_valid_tool_pairing(list(session.history))

        session.reset()
        assert session.turn_count == 0
        assert session.history == ()


class TestCliAndLinks:
    def test_google_event_preserves_html_link_through_tool_serialization(
        self, tz: ZoneInfo
    ) -> None:
        class _Exec:
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload

            def execute(self) -> dict[str, Any]:
                return self._payload

        class _Events:
            def list(self, **kwargs: Any) -> _Exec:
                return _Exec({"items": []})

            def insert(self, **kwargs: Any) -> _Exec:
                return _Exec(
                    {
                        "id": "g_link_1",
                        "htmlLink": "https://calendar.google.com/event?eid=abc123",
                        **kwargs["body"],
                    }
                )

        class _Service:
            def events(self) -> _Events:
                return _Events()

        client = GoogleCalendarClient(service=_Service(), default_tz=tz)
        created = client.create_event(
            CreateEventRequest(
                title="Meeting with John",
                start=datetime(2026, 12, 1, 15, 0, tzinfo=tz),
                duration_minutes=60,
            )
        )
        assert created.html_link == "https://calendar.google.com/event?eid=abc123"

        registry = build_calendar_tools(client)
        payload = json.loads(
            registry.execute(
                "create_event",
                json.dumps(
                    {
                        "title": "Coffee",
                        "start": "2026-12-01T16:00:00-08:00",
                        "duration_minutes": 30,
                    }
                ),
            )
        )
        assert payload["data"]["event"]["html_link"] == "https://calendar.google.com/event?eid=abc123"

    def test_repl_traces_tools_and_handles_commands(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        llm = FakeLLMClient(
            turns=[
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="c1",
                            name="list_events",
                            arguments=json.dumps(
                                {
                                    "start": "2026-12-01T12:00:00-08:00",
                                    "end": "2026-12-01T17:00:00-08:00",
                                }
                            ),
                        ),
                    )
                ),
                ModelTurn(content="You have Design Review from 2:00-3:00 PM."),
            ]
        )
        agent = CalendarAgent(
            llm=llm,
            tools=build_calendar_tools(memory_calendar),
            clock=lambda: datetime(2026, 11, 25, 9, 0, tzinfo=tz),
            timezone_name="America/Los_Angeles",
        )
        session = ConversationSession(agent=agent)

        scripted_inputs = iter(["", "What's on Dec 1 afternoon?", "/history", "/reset", "/quit"])
        outputs: list[str] = []

        run_repl(
            session=session,
            input_fn=lambda _prompt: next(scripted_inputs),
            output_fn=outputs.append,
            show_trace=True,
        )

        joined = "\n".join(outputs)
        assert "[tool] list_events(" in joined
        assert "-> ok (1 event)" in joined
        assert "Agent > You have Design Review from 2:00-3:00 PM." in joined
        assert "Retained 1 turn(s)" in joined
        assert "Conversation memory cleared." in joined
        assert "Goodbye!" in joined

    def test_format_step_trace_highlights_conflicts(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        llm = FakeLLMClient(
            turns=[
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="c_conf",
                            name="create_event",
                            arguments=json.dumps(
                                {
                                    "title": "Overlap",
                                    "start": "2026-12-01T14:00:00-08:00",
                                    "duration_minutes": 60,
                                }
                            ),
                        ),
                    )
                ),
                ModelTurn(content="Conflict detected."),
            ]
        )
        agent = CalendarAgent(
            llm=llm,
            tools=build_calendar_tools(memory_calendar),
            clock=lambda: datetime(2026, 11, 25, 9, 0, tzinfo=tz),
        )
        result = agent.run("Book 2 PM")
        trace = format_step_trace(result)
        assert len(trace) == 1
        assert "-> conflict ('Design Review')" in trace[0]
