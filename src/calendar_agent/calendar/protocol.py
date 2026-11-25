"""Protocol defining the boundary between the agent and any calendar backend."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from calendar_agent.models import CalendarEvent, CreateEventRequest


class CalendarProvider(Protocol):
    """Minimal contract every calendar backend (Google, in-memory fake, etc.) must satisfy."""

    def list_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """Return all events overlapping [start, end), ordered by start time."""
        ...

    def create_event(self, request: CreateEventRequest) -> CalendarEvent:
        """Create a new event, raising EventConflictError if blocked unless allow_conflicts=True."""
        ...
