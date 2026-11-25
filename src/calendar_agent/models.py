"""Core domain models and time-interval rules for the calendar agent."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, field_validator, model_validator

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_guest_emails(guests: list[str]) -> list[str]:
    """Ensure every guest entry looks like an email address (required by Google Calendar)."""
    cleaned: list[str] = []
    for raw in guests:
        email = raw.strip()
        if not _EMAIL_RE.match(email):
            raise ValueError(
                f"Invalid guest email '{raw}'. Guests must be valid email addresses "
                "(put plain names like 'John' in the title instead)."
            )
        cleaned.append(email)
    return cleaned


def ensure_aware(dt: datetime, field_name: str = "datetime") -> datetime:
    """Require timezone-aware datetimes so naive timestamps never leak into the calendar."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware (include a UTC offset or timezone).")
    return dt


def validate_time_range(start: datetime, end: datetime) -> None:
    """Validate that a query or event interval has timezone-aware bounds with end > start."""
    ensure_aware(start, "start")
    ensure_aware(end, "end")
    if end <= start:
        raise ValueError("end must be strictly after start.")


def intervals_overlap(
    start_a: datetime,
    end_a: datetime,
    start_b: datetime,
    end_b: datetime,
) -> bool:
    """Return True if half-open intervals [start_a, end_a) and [start_b, end_b) overlap.

    Back-to-back events (e.g., 14:00-15:00 and 15:00-16:00) do NOT conflict because
    end_a == start_b makes `start_b < end_a` False.
    """
    validate_time_range(start_a, end_a)
    validate_time_range(start_b, end_b)
    return start_a < end_b and start_b < end_a


class CalendarError(Exception):
    """Base exception for calendar backend errors."""


class EventConflictError(CalendarError):
    """Raised when attempting to create an event that overlaps existing events."""

    def __init__(self, conflicts: list[CalendarEvent]) -> None:
        self.conflicts = conflicts
        titles = ", ".join(f"'{e.title}' ({e.start.isoformat()} - {e.end.isoformat()})" for e in conflicts)
        super().__init__(f"Time slot conflicts with existing event(s): {titles}")


class CalendarEvent(BaseModel):
    """Canonical representation of an existing calendar event."""

    id: str
    title: str
    start: datetime
    end: datetime
    guests: list[str] = Field(default_factory=list)
    location: str | None = None
    html_link: str | None = None

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("title cannot be blank.")
        return cleaned

    @field_validator("start")
    @classmethod
    def _validate_start(cls, value: datetime) -> datetime:
        return ensure_aware(value, "start")

    @field_validator("end")
    @classmethod
    def _validate_end(cls, value: datetime) -> datetime:
        return ensure_aware(value, "end")

    @model_validator(mode="after")
    def _validate_bounds(self) -> CalendarEvent:
        if self.end <= self.start:
            raise ValueError("end must be strictly after start.")
        return self

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return intervals_overlap(self.start, self.end, start, end)


class CreateEventRequest(BaseModel):
    """Validated input for scheduling a new calendar event.

    Uses start + duration_minutes (instead of start + end) so callers and LLMs
    cannot accidentally construct inverted time ranges.
    """

    title: str
    start: datetime
    duration_minutes: int = Field(gt=0, le=24 * 60)
    guests: list[str] = Field(default_factory=list)
    location: str | None = None
    allow_conflicts: bool = False

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("title cannot be blank.")
        return cleaned

    @field_validator("start")
    @classmethod
    def _validate_start(cls, value: datetime) -> datetime:
        return ensure_aware(value, "start")

    @field_validator("guests")
    @classmethod
    def _validate_guests(cls, value: list[str]) -> list[str]:
        return validate_guest_emails(value)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.duration_minutes)
