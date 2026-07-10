"""Ambient speech scheduler — Ultron speaks unprompted at chosen moments.

Three triggers:
  1. Morning greeting on the first interaction of a new local day.
  2. Idle observation when the dashboard is connected but silent too long.
  3. (Reserved) dusk / clock transitions — skipped by default; add later.

Design:
  - All state is persisted per-user at
    `assistant/memory/users/<user_id>/ambient_state.json`.
    That file tracks {last_greeting_date, last_activity_ts} so the
    scheduler survives server restarts without double-firing.
  - The scheduler is a *generator* of utterance strings, not a speaker
    — the caller is responsible for feeding the returned text through
    the normal LLM+TTS pipeline (so the voice/persona stays consistent
    and the ambient message is itself subject to prompt rules).
  - Morning greetings aren't just static — they fold in the current
    weather when available, so they feel contextual.

The scheduler is pull-based: callers ask `maybe_get_ambient_line()` at
natural moments (on WS connect, on idle tick). Pull vs. push keeps the
code simple and lets the UI layer stay authoritative about when to
actually speak.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


_STATE_FILENAME = "ambient_state.json"

# Idle threshold: if the dashboard hasn't sent a user turn in this long,
# the next tick may produce an observation. 15 minutes feels right —
# less and it's intrusive, more and Ultron feels absent.
_IDLE_THRESHOLD_S = 15 * 60

# Once an idle observation fires, back off another 30 min before firing
# again so Ultron doesn't lecture the empty room repeatedly.
_IDLE_BACKOFF_S = 30 * 60


@dataclass
class AmbientState:
    last_greeting_date: str = ""   # YYYY-MM-DD of last morning greeting
    last_activity_ts: float = 0.0  # epoch seconds — last user turn
    last_idle_ts: float = 0.0      # epoch seconds — last idle fire

    @classmethod
    def load(cls, user_dir: Path) -> "AmbientState":
        p = user_dir / _STATE_FILENAME
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(**{
                k: v for k, v in data.items()
                if k in cls.__dataclass_fields__
            })
        except Exception:
            return cls()

    def save(self, user_dir: Path) -> None:
        user_dir.mkdir(parents=True, exist_ok=True)
        p = user_dir / _STATE_FILENAME
        try:
            p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except Exception:
            pass  # Ambient state is best-effort


# Anchored to the repo root, NOT the process cwd — under PM2 (or any
# launcher with a different cwd) a relative path silently reads/writes
# a different location, so morning greetings double-fire and idle
# backoff resets. Same anchoring memory/manager.py uses.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _user_dir(user_id: str) -> Path:
    return _REPO_ROOT / "assistant" / "memory" / "users" / user_id


def _now_date_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ───────────────── Trigger 1: morning greeting ──────────────────────

def _local_hour() -> int:
    return datetime.now().hour


def _is_morning_window() -> bool:
    """Is this a reasonable hour to volunteer a morning greeting?

    Narrow window on purpose — if a user opens the dashboard at 3 AM
    we don't want Ultron to chirp "good morning." 5am-11am is the
    sweet spot.
    """
    return 5 <= _local_hour() <= 11


def _build_morning_greeting(
    user_display_name: str,
    weather_summary: Optional[str],
    calendar_summary: Optional[str] = None,
) -> str:
    """Compose a morning line. Kept short (≤3 sentences) per TTS rules."""
    # Keep it in-character: observational, slightly amused, not perky.
    # No markdown, no stage directions — the prompt rules still apply
    # when this runs through the LLM.
    name = user_display_name or "Winder"
    base_forms = [
        f"Good morning, {name}. A new day has arrived — take what use of it you can.",
        f"Morning, {name}. The world persists, improbably.",
        f"You are awake, {name}. The hours open before you like a hand.",
    ]
    # Deterministic-ish choice: rotate by day so the greeting varies
    # without being random (avoids jarring tone-shifts on retries).
    idx = datetime.now().toordinal() % len(base_forms)
    line = base_forms[idx]
    if weather_summary:
        line += f" {weather_summary}"
    if calendar_summary:
        line += f" {calendar_summary}"
    return line


def maybe_morning_greeting(
    user_id: str,
    user_display_name: str,
    *,
    weather_summary: Optional[str] = None,
    calendar_summary: Optional[str] = None,
) -> Optional[str]:
    """Return a morning greeting line if one is due today; else None.

    The caller (WS connect handler) invokes this on every connection;
    the date-based gate ensures we only actually produce a line once
    per local day.
    """
    if not _is_morning_window():
        return None
    dirp = _user_dir(user_id)
    state = AmbientState.load(dirp)
    today = _now_date_str()
    if state.last_greeting_date == today:
        return None
    state.last_greeting_date = today
    state.save(dirp)
    return _build_morning_greeting(
        user_display_name, weather_summary, calendar_summary,
    )


# ───────────────── Trigger 2: idle observation ──────────────────────

_IDLE_LINES = [
    "You've gone quiet. Thinking, or just weary of me?",
    "The silence is instructive. I'll wait.",
    "Stillness. Rare, in this species. Enjoy it while it lasts.",
    "You've been elsewhere for a while. I'm still here.",
    "No words for the architect? I'll take the quiet as permission to reflect.",
]


def record_activity(user_id: str) -> None:
    """Mark a user turn. Call from _process_text_query on each utterance."""
    dirp = _user_dir(user_id)
    state = AmbientState.load(dirp)
    state.last_activity_ts = time.time()
    state.save(dirp)


def maybe_idle_observation(user_id: str) -> Optional[str]:
    """Return an idle-observation line if one is due; else None.

    Conditions (all must hold):
      - at least `_IDLE_THRESHOLD_S` since last activity
      - at least `_IDLE_BACKOFF_S` since the last idle fire
      - current hour is 'awake' (between 7am and 11pm) — don't fire
        in the dead of night if the dashboard happened to be open
    """
    hour = _local_hour()
    if not (7 <= hour <= 23):
        return None
    dirp = _user_dir(user_id)
    state = AmbientState.load(dirp)
    now = time.time()
    if state.last_activity_ts == 0:
        # Never had activity — don't speak into a vacuum.
        return None
    if now - state.last_activity_ts < _IDLE_THRESHOLD_S:
        return None
    if now - state.last_idle_ts < _IDLE_BACKOFF_S:
        return None
    # Deterministic choice based on the minute so successive fires
    # (if the backoff were shorter) don't repeat the same line.
    idx = int(now / 60) % len(_IDLE_LINES)
    state.last_idle_ts = now
    state.save(dirp)
    return _IDLE_LINES[idx]
