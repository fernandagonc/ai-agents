"""Tool-calling AI agent for coordinating calendar events."""

from calendar_agent.models import (
    CalendarError,
    CalendarEvent,
    CreateEventRequest,
    EventConflictError,
    intervals_overlap,
)

__all__ = [
    "CalendarError",
    "CalendarEvent",
    "CreateEventRequest",
    "EventConflictError",
    "intervals_overlap",
]
