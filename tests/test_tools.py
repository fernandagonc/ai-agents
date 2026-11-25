"""Step 2 tests: OpenAI tool schemas, validation, and structured error recovery."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from calendar_agent.models import CalendarError, CalendarEvent, CreateEventRequest
from calendar_agent.tools import ToolRegistry, build_calendar_tools
from tests.fakes import InMemoryCalendar


@pytest.fixture
def registry(memory_calendar: InMemoryCalendar) -> ToolRegistry:
    return build_calendar_tools(memory_calendar)


class TestToolSpecs:
    def test_exports_provider_neutral_tool_specs(self, registry: ToolRegistry) -> None:
        specs = registry.tool_specs()
        assert len(specs) == 2

        by_name = {spec.name: spec for spec in specs}
        assert set(by_name.keys()) == {"list_events", "create_event"}

        params = by_name["create_event"].parameters
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert set(params["required"]) == {"title", "start", "duration_minutes"}
        # Descriptions guide the model against common hallucinations
        assert "EMAIL" in params["properties"]["guests"]["description"]
        assert "timezone" in params["properties"]["start"]["description"]


class TestToolExecution:
    def test_list_events_executes_from_json_string(self, registry: ToolRegistry) -> None:
        raw_args = json.dumps(
            {
                "start": "2026-12-01T12:00:00-08:00",
                "end": "2026-12-01T17:00:00-08:00",
            }
        )
        result = json.loads(registry.execute("list_events", raw_args))

        assert result["ok"] is True
        assert result["data"]["count"] == 1
        assert result["data"]["events"][0]["title"] == "Design Review"

    def test_create_event_persists_in_calendar(
        self, registry: ToolRegistry, memory_calendar: InMemoryCalendar, tz: ZoneInfo
    ) -> None:
        raw_args = json.dumps(
            {
                "title": "Meeting with John",
                "start": "2026-12-01T15:00:00-08:00",
                "duration_minutes": 60,
                "location": "Cafe",
            }
        )
        result = json.loads(registry.execute("create_event", raw_args))

        assert result["ok"] is True
        assert result["data"]["event"]["title"] == "Meeting with John"
        assert result["data"]["event"]["duration_minutes"] == 60
        stored = memory_calendar.list_events(
            datetime(2026, 12, 1, 15, 0, tzinfo=tz),
            datetime(2026, 12, 1, 16, 0, tzinfo=tz),
        )
        assert len(stored) == 1
        assert stored[0].title == "Meeting with John"


class TestToolErrorRecovery:
    def test_unknown_tool_returns_structured_error(self, registry: ToolRegistry) -> None:
        result = json.loads(registry.execute("delete_all_events", "{}"))

        assert result["ok"] is False
        assert result["error_type"] == "unknown_tool"
        assert "list_events" in result["hint"]
        assert "create_event" in result["hint"]

    def test_malformed_json_returns_structured_error(self, registry: ToolRegistry) -> None:
        result = json.loads(registry.execute("list_events", "{not valid json"))

        assert result["ok"] is False
        assert result["error_type"] == "invalid_json"

    def test_hallucinated_extra_parameter_rejected(self, registry: ToolRegistry) -> None:
        raw_args = json.dumps(
            {
                "title": "Meeting with John",
                "start": "2026-12-01T15:00:00-08:00",
                "duration_minutes": 60,
                "attendee": "john@example.com",  # Wrong key name!
            }
        )
        result = json.loads(registry.execute("create_event", raw_args))

        assert result["ok"] is False
        assert result["error_type"] == "validation_error"
        assert any("attendee" in detail for detail in result["details"])

    def test_naive_timestamp_rejected_with_validation_error(self, registry: ToolRegistry) -> None:
        raw_args = json.dumps(
            {
                "title": "Meeting with John",
                "start": "2026-12-01T15:00:00",  # Missing timezone offset!
                "duration_minutes": 60,
            }
        )
        result = json.loads(registry.execute("create_event", raw_args))

        assert result["ok"] is False
        assert result["error_type"] == "validation_error"
        assert any("timezone-aware" in detail for detail in result["details"])

    def test_non_email_guest_name_rejected(self, registry: ToolRegistry) -> None:
        raw_args = json.dumps(
            {
                "title": "Meeting",
                "start": "2026-12-01T15:00:00-08:00",
                "duration_minutes": 60,
                "guests": ["John"],  # Not an email!
            }
        )
        result = json.loads(registry.execute("create_event", raw_args))

        assert result["ok"] is False
        assert result["error_type"] == "validation_error"
        assert any("Invalid guest email" in detail for detail in result["details"])

    def test_conflict_returns_overlapping_events_and_hint(self, registry: ToolRegistry) -> None:
        raw_args = json.dumps(
            {
                "title": "Meeting with John",
                "start": "2026-12-01T14:00:00-08:00",
                "duration_minutes": 60,
            }
        )
        result = json.loads(registry.execute("create_event", raw_args))

        assert result["ok"] is False
        assert result["error_type"] == "conflict"
        assert [c["title"] for c in result["conflicts"]] == ["Design Review"]
        assert "allow_conflicts" in result["hint"]

    def test_calendar_backend_error_caught_cleanly(self) -> None:
        class _BrokenCalendar:
            def list_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
                raise CalendarError("Google API 503 Service Unavailable")

            def create_event(self, request: CreateEventRequest) -> CalendarEvent:
                raise CalendarError("Google API 503 Service Unavailable")

        registry = build_calendar_tools(_BrokenCalendar())
        raw_args = json.dumps(
            {
                "start": "2026-12-01T12:00:00-08:00",
                "end": "2026-12-01T17:00:00-08:00",
            }
        )
        result = json.loads(registry.execute("list_events", raw_args))

        assert result["ok"] is False
        assert result["error_type"] == "calendar_error"
        assert "503" in result["message"]
