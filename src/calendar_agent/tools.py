"""Step 2: LLM tool schemas, validation, and safe execution registry (no framework)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from calendar_agent.calendar.protocol import CalendarProvider
from calendar_agent.llm.protocol import ToolSpec
from calendar_agent.models import (
    CalendarError,
    CalendarEvent,
    CreateEventRequest,
    EventConflictError,
    ensure_aware,
    validate_guest_emails,
)


def _serialize_event(event: CalendarEvent) -> dict[str, Any]:
    """Serialize a CalendarEvent into a compact JSON-friendly dict for the LLM."""
    payload: dict[str, Any] = {
        "id": event.id,
        "title": event.title,
        "start": event.start.isoformat(),
        "end": event.end.isoformat(),
        "duration_minutes": event.duration_minutes,
    }
    if event.guests:
        payload["guests"] = event.guests
    if event.location:
        payload["location"] = event.location
    if event.html_link:
        payload["html_link"] = event.html_link
    return payload


def _build_parameters_schema(args_model: type[BaseModel]) -> dict[str, Any]:
    """Generate a clean JSON Schema object from a Pydantic input model."""
    schema = args_model.model_json_schema()
    schema.pop("title", None)
    return schema


@dataclass(frozen=True)
class ToolDefinition:
    """Pairs an LLM-visible schema with the Python handler that executes it."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any], dict[str, Any]]

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=_build_parameters_schema(self.args_model),
        )


class ToolRegistry:
    """Holds tools, exports provider-neutral ToolSpecs, and executes tool calls safely.

    Crucial agent invariant: `execute()` catches recoverable errors (bad JSON,
    hallucinated params, validation errors, scheduling conflicts, calendar errors)
    and returns structured JSON so the LLM can inspect the failure and recover.
    """

    def __init__(self, tools: list[ToolDefinition] | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def tool_specs(self) -> list[ToolSpec]:
        return [tool.to_spec() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str | dict[str, Any]) -> str:
        """Validate arguments, invoke the tool, and return a JSON string for the LLM."""
        tool = self._tools.get(name)
        if tool is None:
            return self._error(
                error_type="unknown_tool",
                message=f"Tool '{name}' does not exist.",
                hint=f"Call one of the available tools: {', '.join(self.tool_names)}.",
            )

        parsed_args, parse_error = self._parse_arguments(raw_arguments)
        if parse_error is not None:
            return parse_error

        try:
            validated = tool.args_model.model_validate(parsed_args)
        except ValidationError as exc:
            details = [
                f"{'.'.join(str(part) for part in err['loc']) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            ]
            return self._error(
                error_type="validation_error",
                message="Invalid tool arguments.",
                details=details,
                hint="Fix the highlighted argument(s) to match the tool schema and retry.",
            )

        try:
            data = tool.handler(validated)
        except EventConflictError as exc:
            return self._error(
                error_type="conflict",
                message=str(exc),
                conflicts=[_serialize_event(event) for event in exc.conflicts],
                hint=(
                    "Do NOT double-book automatically. Inform the user of the conflicting "
                    "event(s) and suggest nearby open slots, or retry with allow_conflicts=true "
                    "only if the user explicitly confirms."
                ),
            )
        except CalendarError as exc:
            return self._error(
                error_type="calendar_error",
                message=str(exc),
                hint="Inform the user that the calendar backend returned an error.",
            )

        return json.dumps({"ok": True, "data": data})

    def _parse_arguments(
        self, raw_arguments: str | dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        if isinstance(raw_arguments, dict):
            return raw_arguments, None
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return None, self._error(
                error_type="invalid_json",
                message=f"Tool arguments were not valid JSON: {exc.msg}.",
                hint="Provide a valid JSON object matching the tool schema.",
            )
        if not isinstance(parsed, dict):
            return None, self._error(
                error_type="invalid_json",
                message="Tool arguments must be a JSON object.",
                hint="Wrap parameters in a JSON object like {\"start\": \"...\", \"end\": \"...\"}.",
            )
        return parsed, None

    @staticmethod
    def _error(error_type: str, message: str, hint: str, **extra: Any) -> str:
        payload: dict[str, Any] = {
            "ok": False,
            "error_type": error_type,
            "message": message,
            "hint": hint,
            **extra,
        }
        return json.dumps(payload)


class ListEventsArgs(BaseModel):
    """Schema exposed to the LLM for querying calendar events."""

    model_config = ConfigDict(extra="forbid")

    start: datetime = Field(
        description=(
            "Start of the query window (inclusive) as ISO 8601 datetime WITH timezone offset, "
            "e.g. '2026-12-01T12:00:00-08:00'."
        )
    )
    end: datetime = Field(
        description=(
            "End of the query window (exclusive) as ISO 8601 datetime WITH timezone offset, "
            "e.g. '2026-12-01T17:00:00-08:00'. Tip: query a +/-2 hour buffer (or the full day) "
            "around a requested slot so you can immediately suggest nearby free times if busy."
        )
    )

    @field_validator("start")
    @classmethod
    def _validate_start(cls, value: datetime) -> datetime:
        return ensure_aware(value, "start")

    @field_validator("end")
    @classmethod
    def _validate_end(cls, value: datetime) -> datetime:
        return ensure_aware(value, "end")

    @model_validator(mode="after")
    def _validate_bounds(self) -> ListEventsArgs:
        if self.end <= self.start:
            raise ValueError("end must be strictly after start.")
        return self


class CreateEventArgs(BaseModel):
    """Schema exposed to the LLM for creating a calendar event."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=1,
        description="Short, clear title for the event (e.g. 'Meeting with John').",
    )
    start: datetime = Field(
        description=(
            "Event start time as ISO 8601 datetime WITH timezone offset, "
            "e.g. '2026-12-01T14:00:00-08:00'."
        )
    )
    duration_minutes: int = Field(
        gt=0,
        le=24 * 60,
        description="Duration in minutes (e.g. 60 for one hour, 30 for half an hour).",
    )
    guests: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of guest EMAIL ADDRESSES (e.g. 'john@company.com'). "
            "Only include emails explicitly provided by the user—never guess or invent an email. "
            "If only a person's name is mentioned, include it in `title` instead."
        ),
    )
    location: str | None = Field(
        default=None,
        description="Optional physical location or meeting link.",
    )
    allow_conflicts: bool = Field(
        default=False,
        description=(
            "Keep false by default. Set to true ONLY if the user explicitly confirms they "
            "want to double-book an overlapping slot."
        ),
    )

    @field_validator("start")
    @classmethod
    def _validate_start(cls, value: datetime) -> datetime:
        return ensure_aware(value, "start")

    @field_validator("guests")
    @classmethod
    def _validate_guests(cls, value: list[str]) -> list[str]:
        return validate_guest_emails(value)


def build_calendar_tools(calendar: CalendarProvider) -> ToolRegistry:
    """Create a ToolRegistry wired to any CalendarProvider (Google or InMemory)."""

    def handle_list_events(args: ListEventsArgs) -> dict[str, Any]:
        events = calendar.list_events(args.start, args.end)
        return {
            "count": len(events),
            "events": [_serialize_event(event) for event in events],
        }

    def handle_create_event(args: CreateEventArgs) -> dict[str, Any]:
        created = calendar.create_event(
            CreateEventRequest(
                title=args.title,
                start=args.start,
                duration_minutes=args.duration_minutes,
                guests=args.guests,
                location=args.location,
                allow_conflicts=args.allow_conflicts,
            )
        )
        return {"event": _serialize_event(created)}

    return ToolRegistry(
        tools=[
            ToolDefinition(
                name="list_events",
                description=(
                    "List all calendar events overlapping the given [start, end) window. "
                    "Always call this before scheduling to check availability and spot nearby "
                    "open slots."
                ),
                args_model=ListEventsArgs,
                handler=handle_list_events,
            ),
            ToolDefinition(
                name="create_event",
                description=(
                    "Create a new calendar event. Automatically checks for overlapping events "
                    "and returns a structured conflict error unless allow_conflicts is true."
                ),
                args_model=CreateEventArgs,
                handler=handle_create_event,
            ),
        ]
    )
