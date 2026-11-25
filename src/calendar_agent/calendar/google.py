"""Google Calendar API v3 implementation of CalendarProvider."""

from __future__ import annotations

from datetime import date, datetime, timezone, tzinfo
from pathlib import Path
from typing import Any

from calendar_agent.models import (
    CalendarError,
    CalendarEvent,
    CreateEventRequest,
    EventConflictError,
    validate_time_range,
)

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


class GoogleCalendarClient:
    """Adapter around Google Calendar API v3.

    Accepts an already-constructed Google API `service` resource so it can be unit-tested
    with a stub service without hitting the network. For live runs, construct it via
    `GoogleCalendarClient.from_oauth_files(...)`.
    """

    def __init__(
        self,
        service: Any,
        calendar_id: str = "primary",
        default_tz: tzinfo = timezone.utc,
    ) -> None:
        self._service = service
        self._calendar_id = calendar_id
        self._default_tz = default_tz

    @classmethod
    def from_oauth_files(
        cls,
        credentials_path: str | Path = "credentials.json",
        token_path: str | Path = "token.json",
        calendar_id: str = "primary",
        default_tz: tzinfo = timezone.utc,
    ) -> GoogleCalendarClient:
        """Authenticate via OAuth2 Desktop flow and return a live GoogleCalendarClient."""
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        cred_file = Path(credentials_path)
        tok_file = Path(token_path)

        creds: Credentials | None = None
        if tok_file.exists():
            creds = Credentials.from_authorized_user_file(str(tok_file), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not cred_file.exists():
                    raise CalendarError(
                        f"Google OAuth credentials file not found at '{cred_file}'. "
                        "See docs/google-oauth-setup.md to generate credentials.json."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(str(cred_file), SCOPES)
                creds = flow.run_local_server(port=0)
            tok_file.write_text(creds.to_json(), encoding="utf-8")

        service = build("calendar", "v3", credentials=creds)
        return cls(service=service, calendar_id=calendar_id, default_tz=default_tz)

    def list_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        validate_time_range(start, end)
        response = (
            self._service.events()
            .list(
                calendarId=self._calendar_id,
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,  # Expand recurring events so conflicts aren't missed
                orderBy="startTime",
            )
            .execute()
        )
        events: list[CalendarEvent] = []
        for item in response.get("items", []):
            if item.get("status") == "cancelled":
                continue
            parsed = self._parse_event(item)
            if parsed.overlaps(start, end):
                events.append(parsed)
        return sorted(events, key=lambda event: event.start)

    def create_event(self, request: CreateEventRequest) -> CalendarEvent:
        if not request.allow_conflicts:
            conflicts = self.list_events(request.start, request.end)
            if conflicts:
                raise EventConflictError(conflicts)

        body: dict[str, Any] = {
            "summary": request.title,
            "start": {"dateTime": request.start.isoformat()},
            "end": {"dateTime": request.end.isoformat()},
        }
        if request.location:
            body["location"] = request.location
        if request.guests:
            # Google Calendar API requires email addresses for attendees.
            body["attendees"] = [{"email": guest} for guest in request.guests]

        created = (
            self._service.events()
            .insert(calendarId=self._calendar_id, body=body)
            .execute()
        )
        return self._parse_event(created)

    def _parse_event(self, item: dict[str, Any]) -> CalendarEvent:
        start = self._parse_google_time(item["start"])
        end = self._parse_google_time(item["end"])
        guests = [
            attendee["email"]
            for attendee in item.get("attendees", [])
            if "email" in attendee
        ]
        return CalendarEvent(
            id=item["id"],
            title=item.get("summary") or "(No title)",
            start=start,
            end=end,
            guests=guests,
            location=item.get("location"),
            html_link=item.get("htmlLink"),
        )

    def _parse_google_time(self, raw: dict[str, Any]) -> datetime:
        if "dateTime" in raw:
            # Python 3.12's fromisoformat handles trailing 'Z' as UTC.
            return datetime.fromisoformat(raw["dateTime"])
        if "date" in raw:
            # All-day Google Calendar events use YYYY-MM-DD with exclusive end date.
            day = date.fromisoformat(raw["date"])
            return datetime(day.year, day.month, day.day, tzinfo=self._default_tz)
        raise CalendarError(f"Unsupported Google Calendar time payload: {raw}")
