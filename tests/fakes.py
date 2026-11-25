"""Test doubles for deterministic, network-free tests."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from calendar_agent.llm.protocol import Message, ModelTurn, ToolSpec
from calendar_agent.models import (
    CalendarEvent,
    CreateEventRequest,
    EventConflictError,
    validate_time_range,
)


class InMemoryCalendar:
    """Fake CalendarProvider matching GoogleCalendarClient's behavior without network calls."""

    def __init__(self, initial_events: list[CalendarEvent] | None = None) -> None:
        self._events: list[CalendarEvent] = list(initial_events or [])

    def list_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        validate_time_range(start, end)
        matching = [event for event in self._events if event.overlaps(start, end)]
        return sorted(matching, key=lambda event: event.start)

    def create_event(self, request: CreateEventRequest) -> CalendarEvent:
        if not request.allow_conflicts:
            conflicts = self.list_events(request.start, request.end)
            if conflicts:
                raise EventConflictError(conflicts)

        event = CalendarEvent(
            id=f"evt_{uuid4().hex[:8]}",
            title=request.title,
            start=request.start,
            end=request.end,
            guests=list(request.guests),
            location=request.location,
        )
        self._events.append(event)
        return event


class FakeLLMClient:
    """Scripted LLMClient that replays predefined ModelTurns and records every request."""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)
        self.recorded_calls: list[tuple[list[Message], list[ToolSpec]]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> ModelTurn:
        # Snapshot the message list as it looked on this exact iteration.
        self.recorded_calls.append((list(messages), list(tools)))
        if not self._turns:
            raise AssertionError("FakeLLMClient ran out of scripted turns.")
        return self._turns.pop(0)
