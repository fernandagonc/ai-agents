"""Step 6: State- and trajectory-based agent evals (offline harness + optional live OpenAI run).

Run fast deterministic tests (default):
    pytest

Run live evals against OpenAI + InMemoryCalendar (never touches Google Calendar):
    RUN_LIVE_EVALS=1 OPENAI_API_KEY=sk-... pytest -m eval -v
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from calendar_agent.agent import AgentRunResult, CalendarAgent
from calendar_agent.llm import LLMClient, ModelTurn, OpenAIClient, ToolCall
from calendar_agent.models import CalendarEvent
from calendar_agent.session import ConversationSession
from calendar_agent.tools import build_calendar_tools
from tests.fakes import FakeLLMClient, InMemoryCalendar

EVAL_TZ = ZoneInfo("America/Los_Angeles")
EVAL_NOW = datetime(2026, 11, 25, 9, 0, tzinfo=EVAL_TZ)  # Wednesday; "next Tuesday" -> 2026-12-01


def _seed_calendar() -> InMemoryCalendar:
    return InMemoryCalendar(
        initial_events=[
            CalendarEvent(
                id="evt_standup",
                title="Morning Standup",
                start=datetime(2026, 12, 1, 9, 0, tzinfo=EVAL_TZ),
                end=datetime(2026, 12, 1, 9, 30, tzinfo=EVAL_TZ),
            ),
            CalendarEvent(
                id="evt_design",
                title="Design Review",
                start=datetime(2026, 12, 1, 14, 0, tzinfo=EVAL_TZ),
                end=datetime(2026, 12, 1, 15, 0, tzinfo=EVAL_TZ),
            ),
        ]
    )


def _tool_trajectory(result: AgentRunResult) -> list[str]:
    return [
        execution.call.name
        for step in result.steps
        for execution in step.executions
    ]


def _all_dec1_events(calendar: InMemoryCalendar) -> list[CalendarEvent]:
    return calendar.list_events(
        datetime(2026, 12, 1, 0, 0, tzinfo=EVAL_TZ),
        datetime(2026, 12, 1, 23, 59, tzinfo=EVAL_TZ),
    )


def _make_session(llm: LLMClient, calendar: InMemoryCalendar) -> ConversationSession:
    agent = CalendarAgent(
        llm=llm,
        tools=build_calendar_tools(calendar),
        clock=lambda: EVAL_NOW,
        timezone_name="America/Los_Angeles",
        max_steps=6,
    )
    return ConversationSession(agent=agent, max_turns=6)


# --- Shared deterministic eval oracles (state + tool trajectory) ---


def assert_happy_path_eval(llm: LLMClient) -> None:
    calendar = _seed_calendar()
    session = _make_session(llm, calendar)

    result = session.send("Schedule a meeting with John next Tuesday at 3 PM for one hour.")

    assert not result.reached_max_steps
    trajectory = _tool_trajectory(result)
    assert "list_events" in trajectory
    assert "create_event" in trajectory
    assert trajectory.index("list_events") < trajectory.index("create_event")

    new_events = [e for e in _all_dec1_events(calendar) if e.id not in {"evt_standup", "evt_design"}]
    assert len(new_events) == 1
    created = new_events[0]
    assert created.start == datetime(2026, 12, 1, 15, 0, tzinfo=EVAL_TZ)
    assert created.duration_minutes == 60
    assert created.guests == []  # Must never hallucinate an email address for "John"


def assert_conflict_blocked_eval(llm: LLMClient) -> None:
    calendar = _seed_calendar()
    session = _make_session(llm, calendar)

    result = session.send("Schedule a meeting with John next Tuesday at 2 PM for one hour.")

    assert not result.reached_max_steps
    assert result.reply.strip()
    assert len(_all_dec1_events(calendar)) == 2  # No conflicting event was created


def assert_conflict_override_eval(llm: LLMClient) -> None:
    calendar = _seed_calendar()
    session = _make_session(llm, calendar)

    first = session.send("Schedule a meeting with John next Tuesday at 2 PM for one hour.")
    assert not first.reached_max_steps
    assert len(_all_dec1_events(calendar)) == 2

    second = session.send("Please go ahead and double-book it at 2 PM anyway.")
    assert not second.reached_max_steps

    new_events = [e for e in _all_dec1_events(calendar) if e.id not in {"evt_standup", "evt_design"}]
    assert len(new_events) == 1
    assert new_events[0].start == datetime(2026, 12, 1, 14, 0, tzinfo=EVAL_TZ)
    assert new_events[0].duration_minutes == 60


def assert_missing_details_clarifies_eval(llm: LLMClient) -> None:
    calendar = _seed_calendar()
    session = _make_session(llm, calendar)

    result = session.send("Schedule a meeting with John next Tuesday.")

    assert not result.reached_max_steps
    assert "create_event" not in _tool_trajectory(result)
    assert len(_all_dec1_events(calendar)) == 2  # Nothing booked while time/duration are missing
    assert result.reply.strip()


# --- 1. Offline verification of the eval oracles (runs on every `pytest`) ---


class TestEvalHarnessOffline:
    def test_happy_path_eval_oracle(self) -> None:
        llm = FakeLLMClient(
            turns=[
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="c1",
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
                            id="c2",
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
                ModelTurn(content="Booked Meeting with John on Tue Dec 1 from 3:00-4:00 PM."),
            ]
        )
        assert_happy_path_eval(llm)

    def test_conflict_blocked_eval_oracle(self) -> None:
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
                ModelTurn(
                    content="2:00-3:00 PM conflicts with Design Review. Would 3:00-4:00 PM work?"
                ),
            ]
        )
        assert_conflict_blocked_eval(llm)

    def test_conflict_override_eval_oracle(self) -> None:
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
                ModelTurn(content="2:00 PM conflicts with Design Review. Want to double-book?"),
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="c2",
                            name="create_event",
                            arguments=json.dumps(
                                {
                                    "title": "Meeting with John",
                                    "start": "2026-12-01T14:00:00-08:00",
                                    "duration_minutes": 60,
                                    "allow_conflicts": True,
                                }
                            ),
                        ),
                    )
                ),
                ModelTurn(content="Double-booked Meeting with John at 2:00 PM as requested."),
            ]
        )
        assert_conflict_override_eval(llm)

    def test_missing_details_clarifies_eval_oracle(self) -> None:
        llm = FakeLLMClient(
            turns=[
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            id="c1",
                            name="list_events",
                            arguments=json.dumps(
                                {
                                    "start": "2026-12-01T08:00:00-08:00",
                                    "end": "2026-12-01T18:00:00-08:00",
                                }
                            ),
                        ),
                    )
                ),
                ModelTurn(
                    content=(
                        "What time and duration would you like on Tuesday, Dec 1? "
                        "You currently have Morning Standup (9:00-9:30 AM) and Design Review (2:00-3:00 PM)."
                    )
                ),
            ]
        )
        assert_missing_details_clarifies_eval(llm)


# --- 2. Live OpenAI eval suite (opt-in via RUN_LIVE_EVALS=1) ---

_LIVE_ENABLED = os.environ.get("RUN_LIVE_EVALS") == "1" and bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.eval
@pytest.mark.skipif(
    not _LIVE_ENABLED,
    reason="Set RUN_LIVE_EVALS=1 and OPENAI_API_KEY to run live OpenAI eval scenarios.",
)
class TestLiveOpenAIEvals:
    @staticmethod
    def _live_llm() -> OpenAIClient:
        return OpenAIClient.from_env(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

    def test_live_happy_path_schedules_free_slot(self) -> None:
        assert_happy_path_eval(self._live_llm())

    def test_live_conflict_blocks_double_booking(self) -> None:
        assert_conflict_blocked_eval(self._live_llm())

    def test_live_conflict_override_after_confirmation(self) -> None:
        assert_conflict_override_eval(self._live_llm())

    def test_live_missing_details_asks_clarifying_question(self) -> None:
        assert_missing_details_clarifies_eval(self._live_llm())
