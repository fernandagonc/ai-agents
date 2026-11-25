"""Step 1 tests: domain models, overlap logic, InMemoryCalendar, and GoogleCalendarClient."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from calendar_agent.calendar import GoogleCalendarClient
from calendar_agent.models import (
    CalendarEvent,
    CreateEventRequest,
    EventConflictError,
    intervals_overlap,
)
from tests.fakes import InMemoryCalendar


class TestModelsAndOverlap:
    def test_rejects_naive_datetime_in_event(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            CalendarEvent(
                id="evt_1",
                title="Lunch",
                start=datetime(2026, 12, 1, 12, 0),
                end=datetime(2026, 12, 1, 13, 0),
            )

    def test_rejects_end_before_or_equal_start(self, tz: ZoneInfo) -> None:
        dt = datetime(2026, 12, 1, 12, 0, tzinfo=tz)
        with pytest.raises(ValidationError, match="strictly after"):
            CalendarEvent(id="evt_1", title="Zero length", start=dt, end=dt)

    def test_create_request_computes_end_from_duration(self, tz: ZoneInfo) -> None:
        req = CreateEventRequest(
            title="Meeting with John",
            start=datetime(2026, 12, 1, 14, 0, tzinfo=tz),
            duration_minutes=60,
            guests=["john@example.com"],
            location="Coffee Shop",
        )
        assert req.end == datetime(2026, 12, 1, 15, 0, tzinfo=tz)

    def test_create_request_rejects_non_positive_duration(self, tz: ZoneInfo) -> None:
        with pytest.raises(ValidationError):
            CreateEventRequest(
                title="Bad duration",
                start=datetime(2026, 12, 1, 14, 0, tzinfo=tz),
                duration_minutes=0,
            )

    @pytest.mark.parametrize(
        ("b_start", "b_end", "expected"),
        [
            # Back-to-back before and after: NOT a conflict
            ((13, 0), (14, 0), False),
            ((15, 0), (16, 0), False),
            # Completely disjoint
            ((10, 0), (11, 0), False),
            ((16, 0), (17, 0), False),
            # Exact match, partial left, partial right, containment
            ((14, 0), (15, 0), True),
            ((13, 30), (14, 30), True),
            ((14, 30), (15, 30), True),
            ((14, 15), (14, 45), True),
            ((13, 0), (16, 0), True),
        ],
    )
    def test_interval_overlap_cases(
        self,
        tz: ZoneInfo,
        b_start: tuple[int, int],
        b_end: tuple[int, int],
        expected: bool,
    ) -> None:
        a_start = datetime(2026, 12, 1, 14, 0, tzinfo=tz)
        a_end = datetime(2026, 12, 1, 15, 0, tzinfo=tz)
        assert (
            intervals_overlap(
                a_start,
                a_end,
                datetime(2026, 12, 1, *b_start, tzinfo=tz),
                datetime(2026, 12, 1, *b_end, tzinfo=tz),
            )
            is expected
        )

    def test_overlap_works_across_different_timezones(self, tz: ZoneInfo) -> None:
        # 14:00-15:00 PST (UTC-8) == 22:00-23:00 UTC
        pst_start = datetime(2026, 12, 1, 14, 0, tzinfo=tz)
        pst_end = datetime(2026, 12, 1, 15, 0, tzinfo=tz)
        utc_overlapping_start = datetime(2026, 12, 1, 22, 30, tzinfo=timezone.utc)
        utc_overlapping_end = datetime(2026, 12, 1, 23, 30, tzinfo=timezone.utc)
        assert intervals_overlap(pst_start, pst_end, utc_overlapping_start, utc_overlapping_end)


class TestInMemoryCalendar:
    def test_list_events_returns_sorted_matches(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        events = memory_calendar.list_events(
            datetime(2026, 12, 1, 0, 0, tzinfo=tz),
            datetime(2026, 12, 1, 23, 59, tzinfo=tz),
        )
        assert [e.title for e in events] == ["Morning Standup", "Design Review"]

    def test_create_event_succeeds_for_back_to_back_slot(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        created = memory_calendar.create_event(
            CreateEventRequest(
                title="1:1 with John",
                start=datetime(2026, 12, 1, 15, 0, tzinfo=tz),
                duration_minutes=60,
                guests=["john@example.com"],
            )
        )
        assert created.title == "1:1 with John"
        assert created.duration_minutes == 60
        assert len(memory_calendar.list_events(created.start, created.end)) == 1

    def test_create_event_raises_on_conflict_by_default(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        with pytest.raises(EventConflictError) as exc_info:
            memory_calendar.create_event(
                CreateEventRequest(
                    title="Double Booked",
                    start=datetime(2026, 12, 1, 14, 30, tzinfo=tz),
                    duration_minutes=60,
                )
            )
        assert [c.title for c in exc_info.value.conflicts] == ["Design Review"]

    def test_create_event_allows_conflict_when_explicitly_overridden(
        self, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        created = memory_calendar.create_event(
            CreateEventRequest(
                title="Urgent Override",
                start=datetime(2026, 12, 1, 14, 30, tzinfo=tz),
                duration_minutes=30,
                allow_conflicts=True,
            )
        )
        assert created.title == "Urgent Override"


class _FakeExecutable:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def execute(self) -> dict[str, Any]:
        return self._payload


class _FakeGoogleEventsResource:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items
        self.list_calls: list[dict[str, Any]] = []
        self.insert_calls: list[dict[str, Any]] = []

    def list(self, **kwargs: Any) -> _FakeExecutable:
        self.list_calls.append(kwargs)
        return _FakeExecutable({"items": self._items})

    def insert(self, **kwargs: Any) -> _FakeExecutable:
        self.insert_calls.append(kwargs)
        body = kwargs["body"]
        return _FakeExecutable({"id": "gcal_new_1", **body})


class _FakeGoogleService:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.events_resource = _FakeGoogleEventsResource(items)

    def events(self) -> _FakeGoogleEventsResource:
        return self.events_resource


class TestGoogleCalendarClient:
    def test_list_events_expands_recurring_and_parses_all_day_events(self, tz: ZoneInfo) -> None:
        fake_service = _FakeGoogleService(
            items=[
                {
                    "id": "g_1",
                    "summary": "Team Sync",
                    "start": {"dateTime": "2026-12-01T14:00:00-08:00"},
                    "end": {"dateTime": "2026-12-01T15:00:00-08:00"},
                    "attendees": [{"email": "john@example.com"}],
                    "location": "HQ",
                },
                {
                    "id": "g_allday",
                    "summary": "Company Offsite",
                    "start": {"date": "2026-12-01"},
                    "end": {"date": "2026-12-02"},
                },
                {
                    "id": "g_cancelled",
                    "status": "cancelled",
                    "summary": "Cancelled Event",
                    "start": {"dateTime": "2026-12-01T14:00:00-08:00"},
                    "end": {"dateTime": "2026-12-01T15:00:00-08:00"},
                },
            ]
        )
        client = GoogleCalendarClient(service=fake_service, default_tz=tz)
        events = client.list_events(
            datetime(2026, 12, 1, 0, 0, tzinfo=tz),
            datetime(2026, 12, 1, 23, 59, tzinfo=tz),
        )

        assert fake_service.events_resource.list_calls[0]["singleEvents"] is True
        assert [e.title for e in events] == ["Company Offsite", "Team Sync"]
        assert events[1].guests == ["john@example.com"]
        assert events[1].location == "HQ"

    def test_create_event_blocks_insert_when_conflict_exists(self, tz: ZoneInfo) -> None:
        fake_service = _FakeGoogleService(
            items=[
                {
                    "id": "g_1",
                    "summary": "Existing Meeting",
                    "start": {"dateTime": "2026-12-01T14:00:00-08:00"},
                    "end": {"dateTime": "2026-12-01T15:00:00-08:00"},
                }
            ]
        )
        client = GoogleCalendarClient(service=fake_service, default_tz=tz)

        with pytest.raises(EventConflictError):
            client.create_event(
                CreateEventRequest(
                    title="Conflicting Call",
                    start=datetime(2026, 12, 1, 14, 0, tzinfo=tz),
                    duration_minutes=60,
                )
            )
        assert fake_service.events_resource.insert_calls == []

    def test_create_event_maps_fields_to_google_payload(self, tz: ZoneInfo) -> None:
        fake_service = _FakeGoogleService(items=[])
        client = GoogleCalendarClient(service=fake_service, default_tz=tz)

        created = client.create_event(
            CreateEventRequest(
                title="Meeting with John",
                start=datetime(2026, 12, 1, 14, 0, tzinfo=tz),
                duration_minutes=60,
                guests=["john@example.com"],
                location="Room 2",
            )
        )

        assert created.id == "gcal_new_1"
        assert len(fake_service.events_resource.insert_calls) == 1
        body = fake_service.events_resource.insert_calls[0]["body"]
        assert body["summary"] == "Meeting with John"
        assert body["attendees"] == [{"email": "john@example.com"}]
        assert body["location"] == "Room 2"
