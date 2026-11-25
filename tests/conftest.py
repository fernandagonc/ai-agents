"""Shared pytest fixtures for fast, deterministic calendar tests."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from calendar_agent.models import CalendarEvent
from tests.fakes import InMemoryCalendar


@pytest.fixture
def tz() -> ZoneInfo:
    return ZoneInfo("America/Los_Angeles")


@pytest.fixture
def sample_events(tz: ZoneInfo) -> list[CalendarEvent]:
    return [
        CalendarEvent(
            id="evt_standup",
            title="Morning Standup",
            start=datetime(2026, 12, 1, 9, 0, tzinfo=tz),
            end=datetime(2026, 12, 1, 9, 30, tzinfo=tz),
            guests=["team@example.com"],
            location="Zoom",
        ),
        CalendarEvent(
            id="evt_design",
            title="Design Review",
            start=datetime(2026, 12, 1, 14, 0, tzinfo=tz),
            end=datetime(2026, 12, 1, 15, 0, tzinfo=tz),
            guests=["alice@example.com"],
            location="Room 3B",
        ),
    ]


@pytest.fixture
def memory_calendar(sample_events: list[CalendarEvent]) -> InMemoryCalendar:
    return InMemoryCalendar(initial_events=sample_events)
