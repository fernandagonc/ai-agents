"""Step 5: Interactive terminal CLI with readable tool-call tracing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from calendar_agent.agent import AgentRunResult, CalendarAgent
from calendar_agent.calendar.google import GoogleCalendarClient
from calendar_agent.llm.openai import OpenAIClient
from calendar_agent.models import CalendarError
from calendar_agent.session import ConversationSession
from calendar_agent.tools import build_calendar_tools


def _format_arguments(raw_arguments: str) -> str:
    """Render JSON tool arguments compactly as key=value pairs for trace output."""
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return raw_arguments
    if not isinstance(parsed, dict):
        return raw_arguments
    return ", ".join(f"{key}={value!r}" for key, value in parsed.items())


def _format_outcome(raw_output: str) -> str:
    """Summarize a structured JSON tool result into a concise status label."""
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output
    if not isinstance(parsed, dict):
        return raw_output

    if parsed.get("ok") is True:
        data = parsed.get("data", {})
        if isinstance(data, dict) and "count" in data:
            count = data["count"]
            noun = "event" if count == 1 else "events"
            return f"ok ({count} {noun})"
        if isinstance(data, dict) and "event" in data:
            title = data["event"].get("title", "event")
            return f"ok (created '{title}')"
        return "ok"

    error_type = parsed.get("error_type", "error")
    if error_type == "conflict":
        titles = [c.get("title", "?") for c in parsed.get("conflicts", [])]
        return f"conflict ({', '.join(repr(t) for t in titles)})"
    return str(error_type)


def format_step_trace(result: AgentRunResult) -> list[str]:
    """Format all tool executions from an AgentRunResult as human-readable trace lines."""
    lines: list[str] = []
    for step in result.steps:
        for execution in step.executions:
            args_str = _format_arguments(execution.call.arguments)
            outcome = _format_outcome(execution.output)
            lines.append(f"  [tool] {execution.call.name}({args_str}) -> {outcome}")
    return lines


def run_repl(
    session: ConversationSession,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    show_trace: bool = True,
) -> None:
    """Run an interactive terminal chat loop over a ConversationSession."""
    output_fn("Calendar Agent ready. Commands: /reset, /history, /quit")

    while True:
        try:
            raw = input_fn("You > ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye!")
            return

        user_input = raw.strip()
        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"/quit", "/exit"}:
            output_fn("Goodbye!")
            return
        if lowered == "/reset":
            session.reset()
            output_fn("Conversation memory cleared.")
            continue
        if lowered == "/history":
            output_fn(
                f"Retained {session.turn_count} turn(s) "
                f"({len(session.history)} total messages)."
            )
            continue

        result = session.send(user_input)
        if show_trace:
            for line in format_step_trace(result):
                output_fn(line)
        output_fn(f"Agent > {result.reply}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calendar-agent",
        description="Natural-language calendar assistant powered by OpenAI and Google Calendar.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI chat model name (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--timezone",
        default=os.environ.get("CALENDAR_TIMEZONE", "America/Los_Angeles"),
        help="IANA timezone name for date grounding (default: America/Los_Angeles).",
    )
    parser.add_argument(
        "--credentials",
        default="credentials.json",
        help="Path to Google OAuth Desktop credentials JSON (default: credentials.json).",
    )
    parser.add_argument(
        "--token",
        default="token.json",
        help="Path to cached Google OAuth token JSON (default: token.json).",
    )
    parser.add_argument(
        "--calendar-id",
        default="primary",
        help="Google Calendar ID (default: primary).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of user turns retained in memory (default: 10).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide tool-call trace lines and only display assistant replies.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint wiring GoogleCalendarClient, OpenAIClient, CalendarAgent, and REPL."""
    args = _build_parser().parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        return 1

    try:
        tz = ZoneInfo(args.timezone)
    except Exception:
        print(f"Error: Unknown IANA timezone '{args.timezone}'.", file=sys.stderr)
        return 1

    if not Path(args.credentials).exists() and not Path(args.token).exists():
        print(
            f"Error: Google OAuth credentials not found at '{args.credentials}'. "
            "Follow docs/google-oauth-setup.md to generate credentials.json.",
            file=sys.stderr,
        )
        return 1

    try:
        calendar = GoogleCalendarClient.from_oauth_files(
            credentials_path=args.credentials,
            token_path=args.token,
            calendar_id=args.calendar_id,
            default_tz=tz,
        )
    except CalendarError as exc:
        print(f"Error connecting to Google Calendar: {exc}", file=sys.stderr)
        return 1

    llm = OpenAIClient.from_env(model=args.model)
    tools = build_calendar_tools(calendar)
    agent = CalendarAgent(
        llm=llm,
        tools=tools,
        clock=lambda: datetime.now(tz),
        timezone_name=args.timezone,
    )
    session = ConversationSession(agent=agent, max_turns=args.max_turns)
    run_repl(session=session, show_trace=not args.quiet)
    return 0
