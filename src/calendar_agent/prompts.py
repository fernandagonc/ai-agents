"""Step 4: Deterministic temporal grounding and scheduling policy prompt builder."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from calendar_agent.models import ensure_aware


def format_utc_offset(dt: datetime) -> str:
    """Format a timezone-aware datetime's UTC offset as '+HH:MM' or '-HH:MM'."""
    ensure_aware(dt, "dt")
    offset = dt.utcoffset()
    assert offset is not None
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def build_upcoming_days_table(now: datetime, horizon_days: int = 14) -> str:
    """Build a deterministic calendar lookup table so the LLM never does mental date math.

    Uses local noon on each date with `now.tzinfo` so Daylight Saving Time (DST) transitions
    within the window display each day's exact UTC offset.
    """
    ensure_aware(now, "now")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    lines: list[str] = []
    base_date = now.date()
    for offset in range(horizon_days):
        day = base_date + timedelta(days=offset)
        day_sample = datetime(day.year, day.month, day.day, 12, 0, tzinfo=now.tzinfo)
        utc_offset = format_utc_offset(day_sample)

        if offset == 0:
            label = "Today:   "
        elif offset == 1:
            label = "Tomorrow:"
        else:
            label = "         "

        lines.append(f"- {label} {day_sample.strftime('%a %Y-%m-%d')} (UTC{utc_offset})")

    return "\n".join(lines)


def build_system_prompt(
    now: datetime,
    timezone_name: str | None = None,
    horizon_days: int = 14,
) -> str:
    """Build the grounded system prompt combining current time context and scheduling rules."""
    ensure_aware(now, "now")

    if timezone_name is not None:
        tz = ZoneInfo(timezone_name)
        localized_now = now.astimezone(tz)
        tz_label = timezone_name
    else:
        localized_now = now
        tz_label = (
            now.tzinfo.key
            if isinstance(now.tzinfo, ZoneInfo)
            else (now.tzname() or f"UTC{format_utc_offset(now)}")
        )

    utc_offset = format_utc_offset(localized_now)
    upcoming_table = build_upcoming_days_table(localized_now, horizon_days=horizon_days)

    return (
        "You are a concise, reliable calendar assistant that manages events via tools.\n\n"
        f"Current datetime: {localized_now.strftime('%A, %Y-%m-%d %H:%M')} "
        f"{tz_label} (UTC{utc_offset})\n\n"
        f"Upcoming {horizon_days} days (use these exact dates and UTC offsets—do not do mental date math):\n"
        f"{upcoming_table}\n\n"
        "Scheduling rules:\n"
        "1. Always call `list_events` first to check availability before calling `create_event`. "
        "Query a +/-2 hour buffer (or the day's working hours) around the requested time so you "
        "can suggest nearby open slots if busy.\n"
        "2. Never guess missing required details (`title`, `start` date and time, or "
        "`duration_minutes`). Ask a brief clarifying question before calling `create_event`. "
        "You may still call `list_events` first so your question includes helpful availability context.\n"
        "3. Never guess or fabricate guest email addresses. Only include emails explicitly provided "
        "by the user in `guests`; put plain names (like 'John') in `title` instead.\n"
        "4. Never double-book automatically. Keep `allow_conflicts=false` unless the user explicitly "
        "confirms they want to overlap an existing event. If a slot conflicts, name the conflicting "
        "event(s) and propose open alternatives.\n"
        "5. If a requested start time is in the past relative to Current datetime, ask the user to "
        "confirm before booking.\n"
        "6. If a relative date phrase like 'next Tuesday' could be ambiguous (for example, when that "
        "weekday is only 1-2 days away), confirm the exact calendar date with the user before booking.\n"
        "7. Always include the timezone offset in every ISO 8601 timestamp passed to tools."
    )
