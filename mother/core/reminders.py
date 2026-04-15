"""Proactive reminder / alarm system for MOTHER.

Users can say "remind me at 3 PM to call the dentist" and MOTHER will
speak the reminder at the right time — even mid-conversation.

Storage: ``assistant/memory/reminders.json``
Format:
    [
      {
        "id": "uuid4-string",
        "user_id": "charlie",
        "text": "call the dentist",
        "trigger_iso": "2025-09-01T15:00:00",
        "created_iso": "2025-09-01T09:00:00",
        "fired": false
      },
      ...
    ]

The background thread wakes every 15 seconds, checks for due reminders,
calls the registered speak callback, and marks the reminder as fired.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Callable, Optional, List, Dict

_REMINDERS_PATH = Path("assistant/memory/reminders.json")
_CHECK_INTERVAL = 15  # seconds between checks

# Registered speak callable (set by cli.py at startup)
_speak_fn: Optional[Callable[[str], None]] = None
_stop_event = threading.Event()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load() -> List[Dict]:
    if _REMINDERS_PATH.exists():
        try:
            return json.loads(_REMINDERS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save(reminders: List[Dict]) -> None:
    _REMINDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REMINDERS_PATH.write_text(
        json.dumps(reminders, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_reminder(user_id: str, text: str, trigger: datetime) -> str:
    """Schedule a reminder. Returns the new reminder's ID."""
    rid = str(uuid.uuid4())[:8]
    with _lock:
        reminders = _load()
        reminders.append({
            "id": rid,
            "user_id": user_id,
            "text": text,
            "trigger_iso": trigger.isoformat(),
            "created_iso": datetime.now().isoformat(),
            "fired": False,
        })
        _save(reminders)
    return rid


def list_reminders(user_id: str = None) -> List[Dict]:
    """Return pending reminders, optionally filtered to a user."""
    with _lock:
        all_r = _load()
    pending = [r for r in all_r if not r.get("fired")]
    if user_id:
        pending = [r for r in pending if r.get("user_id") == user_id]
    return pending


def delete_reminder(rid: str) -> bool:
    """Delete a reminder by ID."""
    with _lock:
        reminders = _load()
        before = len(reminders)
        reminders = [r for r in reminders if r.get("id") != rid]
        if len(reminders) < before:
            _save(reminders)
            return True
    return False


def register_speak(fn: Callable[[str], None]) -> None:
    """Register the TTS speak callback used by the background thread."""
    global _speak_fn
    _speak_fn = fn


def start_background_thread() -> threading.Thread:
    """Start the reminder polling thread. Call once at startup."""
    th = threading.Thread(target=_reminder_loop, daemon=True)
    th.start()
    return th


def stop() -> None:
    """Signal the background thread to exit cleanly."""
    _stop_event.set()


# ---------------------------------------------------------------------------
# Natural language time parser
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}


def parse_reminder(utterance: str) -> Optional[tuple[str, datetime]]:
    """Parse "remind me [at/in/on TIME] to TEXT" into (text, datetime).

    Supports:
        "remind me at 3 PM to call the dentist"
        "remind me in 10 minutes to take my meds"
        "remind me tomorrow at 9 AM to submit the report"
        "remind me on Friday at 2 PM to prep for the meeting"

    Returns (reminder_text, trigger_datetime) or None if not parseable.
    """
    low = utterance.lower().strip()
    m = re.match(
        r"remind me\s+(.+?)\s+to\s+(.+)$",
        low,
        re.IGNORECASE,
    )
    if not m:
        return None
    time_part = m.group(1).strip()
    text_part = m.group(2).strip()
    if not text_part:
        return None

    trigger = _parse_time_expr(time_part)
    if trigger is None:
        return None
    return (text_part, trigger)


def _parse_time_expr(expr: str) -> Optional[datetime]:
    """Convert a natural-language time expression to a datetime."""
    now = datetime.now().replace(second=0, microsecond=0)
    expr = expr.strip().lower()

    # "in X minutes/hours"
    m = re.match(r"in (\d+)\s*(minute|minutes|min|hour|hours|hr|second|seconds|sec)", expr)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("min"):
            return now + timedelta(minutes=n)
        if unit.startswith("hour") or unit == "hr":
            return now + timedelta(hours=n)
        if unit.startswith("sec"):
            return now + timedelta(seconds=n)

    # "at HH:MM [AM/PM]" or "at H [AM/PM]"
    time_match = re.search(
        r"(?:^|at\s+)(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", expr
    )
    base_date = now.date()

    # Day modifiers
    if "tomorrow" in expr:
        base_date = base_date + timedelta(days=1)
    else:
        for day_name, day_num in _WEEKDAYS.items():
            if day_name in expr:
                days_ahead = (day_num - now.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7  # "Friday" means next Friday if today is Friday
                base_date = now.date() + timedelta(days=days_ahead)
                break

    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        ampm = time_match.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        trigger = datetime(
            base_date.year, base_date.month, base_date.day, hour, minute
        )
        # If trigger is in the past and no day modifier, push to tomorrow
        if trigger <= now and "tomorrow" not in expr and not any(d in expr for d in _WEEKDAYS):
            trigger += timedelta(days=1)
        return trigger

    # Just a day modifier, no time — default to 9 AM
    if base_date != now.date():
        return datetime(base_date.year, base_date.month, base_date.day, 9, 0)

    return None


# ---------------------------------------------------------------------------
# Background polling loop
# ---------------------------------------------------------------------------

def _reminder_loop() -> None:
    while not _stop_event.wait(timeout=_CHECK_INTERVAL):
        _fire_due_reminders()


def _fire_due_reminders() -> None:
    now = datetime.now()
    with _lock:
        reminders = _load()
        changed = False
        for r in reminders:
            if r.get("fired"):
                continue
            try:
                trigger = datetime.fromisoformat(r["trigger_iso"])
            except (KeyError, ValueError):
                continue
            if trigger <= now:
                r["fired"] = True
                changed = True
                text = r.get("text", "reminder")
                _fire(r.get("user_id", ""), text)
        if changed:
            _save(reminders)


def _fire(user_id: str, text: str) -> None:
    msg = f"Reminder: {text}."
    print(f"\n[REMINDER] {msg}")
    if _speak_fn is not None:
        try:
            _speak_fn(msg)
        except Exception:
            pass
