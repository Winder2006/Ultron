"""iCloud Calendar (CalDAV) read-only access.

Connects to https://caldav.icloud.com using an Apple ID + an
*app-specific password* (NOT your iCloud login). Generate one at
appleid.apple.com → Sign-In and Security → App-Specific Passwords.
Drop the credentials into .env:

    ICLOUD_USERNAME=you@icloud.com
    ICLOUD_APP_PASSWORD=abcd-efgh-ijkl-mnop

The principal/calendar discovery is cached for one hour to avoid
re-doing the multi-step PROPFIND dance on every query.

Tool surface (registered in mother/llm/tools.py):
  • get_calendar_today     — today's events
  • get_calendar_range     — events between two dates
  • search_calendar        — free-text match on title/location

All return human-readable strings suitable for direct TTS or for
narration by the LLM. Raw event dicts are also exposed via
`fetch_today()` etc. for the ambient greeting.

Failure mode: any error (no creds, network down, calendar slow)
returns a short string explaining the problem rather than raising —
the LLM can acknowledge the miss in character.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, date, time as dtime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger("mother.tools.icloud_calendar")

ICLOUD_CALDAV_URL = "https://caldav.icloud.com"
_PRINCIPAL_CACHE_TTL_S = 3600.0  # re-do calendar discovery hourly
# Per-request timeout passed through DAVClient to the HTTP layer.
# Without it a dead network hangs the worker thread indefinitely.
_CALDAV_TIMEOUT_S = 15


# ─────────────────────── data shapes ───────────────────────────────


@dataclass
class CalEvent:
    summary: str
    start: datetime
    end: Optional[datetime]
    all_day: bool
    location: str
    calendar_name: str

    def time_phrase(self) -> str:
        """How a human says the time. '10:30 AM' / 'all day' / '2 to 3 PM'."""
        if self.all_day:
            return "all day"
        s = self.start.astimezone()
        # _fmt_clock is the cross-platform formatter. Do NOT "optimize"
        # this to strftime("%-I:%M %p") — %-I is glibc-only and RAISES
        # ValueError on Windows, which crashed every calendar tool the
        # moment the day contained a timed event.
        s_str = _fmt_clock(s)
        if self.end and self.end > self.start:
            e_str = _fmt_clock(self.end.astimezone())
            return f"{s_str} to {e_str}"
        return s_str


def _fmt_clock(dt: datetime) -> str:
    """Cross-platform '10:30 AM' / '3 PM' formatter (no %-I on Windows)."""
    h = dt.hour
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    if dt.minute == 0:
        return f"{h12} {suffix}"
    return f"{h12}:{dt.minute:02d} {suffix}"


# ─────────────────────── connection cache ──────────────────────────


_lock = threading.Lock()
_cache: dict = {
    "principal": None,
    "calendars": [],
    "last_refresh": 0.0,
    "client": None,
}


def _get_creds() -> tuple[Optional[str], Optional[str]]:
    user = os.environ.get("ICLOUD_USERNAME") or os.environ.get("ICLOUD_USER")
    pw = os.environ.get("ICLOUD_APP_PASSWORD") or os.environ.get("ICLOUD_PASSWORD")
    return user, pw


def _ensure_principal():
    """Connect + discover calendars. Caches for an hour. Raises on auth fail."""
    now = time.monotonic()
    with _lock:
        if _cache["principal"] is not None and (
            now - _cache["last_refresh"]
        ) < _PRINCIPAL_CACHE_TTL_S:
            return _cache["principal"], _cache["calendars"]

        user, pw = _get_creds()
        if not user or not pw:
            raise RuntimeError(
                "iCloud credentials not set — add ICLOUD_USERNAME and "
                "ICLOUD_APP_PASSWORD to .env"
            )

        # Lazy import — keeps cold-start cheap when calendar isn't used.
        import caldav

        client = caldav.DAVClient(
            url=ICLOUD_CALDAV_URL,
            username=user,
            password=pw,
            timeout=_CALDAV_TIMEOUT_S,
        )
        principal = client.principal()
        calendars = principal.calendars()
        _cache.update({
            "principal": principal,
            "calendars": calendars,
            "client": client,
            "last_refresh": now,
        })
        logger.info(
            "[icloud_calendar] connected — %d calendars discovered",
            len(calendars),
        )
        return principal, calendars


# ─────────────────────── core fetch ────────────────────────────────


def _fetch_events(start: datetime, end: datetime) -> List[CalEvent]:
    """Pull events from every visible calendar between start and end (inclusive).

    Times are passed as datetime; iCloud accepts naive UTC but is happiest
    with aware datetimes. We convert if needed.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    _, calendars = _ensure_principal()
    results: List[CalEvent] = []

    for cal in calendars:
        try:
            # search() handles RRULEs (recurring events) by expanding them
            # into individual occurrences within the range — without this,
            # weekly events only return their first instance.
            events = cal.search(
                start=start, end=end,
                event=True, expand=True,
            )
        except Exception as e:
            logger.debug("[icloud_calendar] %s search failed: %s", cal.name, e)
            continue

        cal_name = getattr(cal, "name", "Calendar")
        for ev in events:
            try:
                parsed = _parse_event(ev, cal_name)
                if parsed:
                    results.append(parsed)
            except Exception as e:
                logger.debug("[icloud_calendar] event parse failed: %s", e)
                continue

    results.sort(key=lambda e: e.start)
    return results


def _parse_event(ev, cal_name: str) -> Optional[CalEvent]:
    """Convert a CalDAV Event object to a CalEvent."""
    try:
        from icalendar import Calendar as ICal
    except ImportError:
        return None
    try:
        cal_data = ev.icalendar_instance
    except Exception:
        try:
            cal_data = ICal.from_ical(ev.data)
        except Exception:
            return None

    for component in cal_data.walk():
        if component.name != "VEVENT":
            continue
        summary = str(component.get("SUMMARY", "(no title)"))
        location = str(component.get("LOCATION", ""))
        dtstart = component.get("DTSTART")
        dtend = component.get("DTEND")
        if dtstart is None:
            continue

        start_val = dtstart.dt
        end_val = dtend.dt if dtend is not None else None

        # All-day events come through as `date` rather than `datetime`.
        all_day = isinstance(start_val, date) and not isinstance(start_val, datetime)
        if all_day:
            start_dt = datetime.combine(start_val, dtime.min, tzinfo=timezone.utc)
            end_dt = (
                datetime.combine(end_val, dtime.min, tzinfo=timezone.utc)
                if isinstance(end_val, date) else None
            )
        else:
            start_dt = start_val if start_val.tzinfo else start_val.replace(tzinfo=timezone.utc)
            if isinstance(end_val, datetime):
                end_dt = end_val if end_val.tzinfo else end_val.replace(tzinfo=timezone.utc)
            else:
                end_dt = None

        return CalEvent(
            summary=summary,
            start=start_dt,
            end=end_dt,
            all_day=all_day,
            location=location,
            calendar_name=cal_name,
        )
    return None


# ─────────────────────── public API ────────────────────────────────


def fetch_today() -> List[CalEvent]:
    """Today's events in the local timezone."""
    now = datetime.now().astimezone()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return _fetch_events(start, end)


def fetch_range(start: datetime, end: datetime) -> List[CalEvent]:
    return _fetch_events(start, end)


def search(query: str, lookback_days: int = 7, lookahead_days: int = 30) -> List[CalEvent]:
    """Search by free-text match on summary or location across a window
    around today. iCloud's CalDAV doesn't expose full-text search server-
    side, so we filter client-side after fetching the window."""
    now = datetime.now().astimezone()
    start = now - timedelta(days=lookback_days)
    end = now + timedelta(days=lookahead_days)
    events = _fetch_events(start, end)
    q = query.lower().strip()
    if not q:
        return events
    return [
        e for e in events
        if q in e.summary.lower() or q in e.location.lower()
    ]


# ─────────────────────── string formatting for tools ───────────────


def _format_for_tts(events: Sequence[CalEvent], header: str) -> str:
    """Turn a list of events into a single speakable line."""
    if not events:
        return f"{header} Nothing on the calendar."
    n = len(events)
    if n == 1:
        e = events[0]
        loc = f" at {e.location}" if e.location else ""
        return f"{header} One thing — {e.summary} {e.time_phrase()}{loc}."
    bits = []
    for e in events[:5]:
        loc = f" at {e.location}" if e.location else ""
        bits.append(f"{e.summary} {e.time_phrase()}{loc}")
    overflow = "" if n <= 5 else f" Plus {n - 5} more."
    return f"{header} {n} things — {'; '.join(bits)}.{overflow}"


# ─────────────────────── tool entrypoints ──────────────────────────


def get_calendar_today(args: dict) -> str:
    """Tool: today's calendar."""
    try:
        events = fetch_today()
    except Exception as e:
        return f"Calendar unavailable: {e}"
    return _format_for_tts(events, "Today.")


def get_calendar_range(args: dict) -> str:
    """Tool: events between args['start_date'] and args['end_date']
    (ISO YYYY-MM-DD)."""
    try:
        s_str = (args.get("start_date") or "").strip()
        e_str = (args.get("end_date") or "").strip()
        if not s_str or not e_str:
            return "Need both start_date and end_date (YYYY-MM-DD)."
        s = datetime.fromisoformat(s_str)
        e = datetime.fromisoformat(e_str)
        if e < s:
            return "end_date must be on or after start_date."
        # Make end exclusive at midnight if user gave a date (vs datetime)
        if e.time() == dtime.min and len(e_str) <= 10:
            e = e + timedelta(days=1)
        events = fetch_range(s.astimezone(), e.astimezone())
    except Exception as ex:
        return f"Calendar unavailable: {ex}"
    return _format_for_tts(events, f"{s_str} to {e_str}.")


def search_calendar(args: dict) -> str:
    """Tool: free-text calendar search."""
    query = (args.get("query") or "").strip()
    if not query:
        return "No query provided."
    try:
        events = search(query)
    except Exception as e:
        return f"Calendar search failed: {e}"
    if not events:
        return f"Nothing on the calendar matches {query!r}."
    return _format_for_tts(events, f"Matches for {query!r}:")
