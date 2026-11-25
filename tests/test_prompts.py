"""Step 4 tests: temporal grounding, 14-day lookup table, DST handling, and policy rules."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from calendar_agent.agent import CalendarAgent
from calendar_agent.llm import ModelTurn
from calendar_agent.prompts import (
    build_system_prompt,
    build_upcoming_days_table,
    format_utc_offset,
)
from calendar_agent.tools import build_calendar_tools
from tests.fakes import FakeLLMClient, InMemoryCalendar


class TestTemporalGrounding:
    def test_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            build_system_prompt(now=datetime(2026, 11, 25, 9, 0))

    def test_formats_positive_negative_and_half_hour_offsets(self) -> None:
        la = datetime(2026, 11, 25, 9, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        utc = datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc)
        india = datetime(2026, 11, 25, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

        assert format_utc_offset(la) == "-08:00"
        assert format_utc_offset(utc) == "+00:00"
        assert format_utc_offset(india) == "+05:30"

    def test_upcoming_table_handles_month_rollover(self, tz: ZoneInfo) -> None:
        now = datetime(2026, 11, 25, 4, 10, tzinfo=tz)
        table = build_upcoming_days_table(now, horizon_days=14)
        lines = table.splitlines()

        assert len(lines) == 14
        assert lines[0] == "- Today:    Wed 2026-11-25 (UTC-08:00)"
        assert lines[1] == "- Tomorrow: Thu 2026-11-26 (UTC-08:00)"
        assert "-           Tue 2026-12-01 (UTC-08:00)" in lines
        assert lines[-1] == "-           Tue 2026-12-08 (UTC-08:00)"

    def test_upcoming_table_tracks_dst_transition(self, tz: ZoneInfo) -> None:
        # In America/Los_Angeles, clocks fall back from PDT (-07:00) to PST (-08:00) on 2026-11-01.
        before_dst = datetime(2026, 10, 30, 10, 0, tzinfo=tz)
        table = build_upcoming_days_table(before_dst, horizon_days=4)
        lines = table.splitlines()

        assert lines[0] == "- Today:    Fri 2026-10-30 (UTC-07:00)"
        assert lines[1] == "- Tomorrow: Sat 2026-10-31 (UTC-07:00)"
        assert lines[2] == "-           Sun 2026-11-01 (UTC-08:00)"
        assert lines[3] == "-           Mon 2026-11-02 (UTC-08:00)"

    def test_system_prompt_localizes_utc_clock_to_requested_timezone(self) -> None:
        # 2026-11-25 18:00 UTC == 2026-11-25 10:00 America/Los_Angeles (UTC-08:00)
        utc_now = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)
        prompt = build_system_prompt(now=utc_now, timezone_name="America/Los_Angeles")

        assert "Wednesday, 2026-11-25 10:00 America/Los_Angeles (UTC-08:00)" in prompt
        assert "Tue 2026-12-01 (UTC-08:00)" in prompt
        # Core scheduling policies are explicitly encoded
        assert "list_events" in prompt
        assert "Never guess missing required details" in prompt
        assert "Never guess or fabricate guest email addresses" in prompt
        assert "allow_conflicts=false" in prompt
        assert "in the past" in prompt

    def test_agent_uses_injected_clock_and_refreshes_system_message(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        current_time = [datetime(2026, 11, 25, 9, 0, tzinfo=tz)]
        llm = FakeLLMClient(
            turns=[
                ModelTurn(content="First reply."),
                ModelTurn(content="Second reply."),
            ]
        )
        agent = CalendarAgent(
            llm=llm,
            tools=build_calendar_tools(memory_calendar),
            clock=lambda: current_time[0],
            timezone_name="America/Los_Angeles",
        )

        first = agent.run("Hello")
        first_system = llm.recorded_calls[0][0][0].content or ""
        assert "2026-11-25 09:00 America/Los_Angeles" in first_system

        # Advance the clock to the next day and pass previous history
        current_time[0] = datetime(2026, 11, 26, 15, 30, tzinfo=tz)
        agent.run("How about tomorrow?", history=list(first.messages))
        second_system = llm.recorded_calls[1][0][0].content or ""
        assert "2026-11-26 15:30 America/Los_Angeles" in second_system
        assert "- Today:    Thu 2026-11-26 (UTC-08:00)" in second_system
