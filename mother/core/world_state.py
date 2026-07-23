"""Ambient world-state brief for context injection.

JARVIS-style awareness: a background refresh gathers today's calendar,
recent inbox activity, weather, and headlines into one compact text
block that the voice route injects into every turn's dynamic context.
The model then already knows the state of the day — "anything I should
know?" gets a real answer with zero tool round-trips, and ordinary
answers can reference the 3pm meeting without being asked about it.

Design constraints:
- NEVER blocks a turn. get_brief() returns whatever is cached
  (possibly empty) instantly; refreshes run in a daemon thread kicked
  by maybe_refresh_async() at connection-open and turn-start.
- Each source runs concurrently with its own hard timeout and fails
  silently — a dead IMAP server costs one missing line, not a hung
  brief. Sources that report "unavailable" are dropped rather than
  injected as noise.
- The brief lives in the UNCACHED dynamic context block, so its churn
  never busts the Anthropic prompt cache on the static persona.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger("mother.world_state")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _watch_tickers() -> list:
    """ambient.watch_tickers from configs/app.yaml (best-effort)."""
    try:
        import yaml
        data = yaml.safe_load(
            (_REPO_ROOT / "configs" / "app.yaml").read_text(encoding="utf-8")
        ) or {}
        return [str(t).upper() for t in (data.get("ambient") or {}).get("watch_tickers") or []][:3]
    except Exception:
        return []

BRIEF_TTL_S = 600.0        # refresh cadence while the assistant is in use
_SOURCE_TIMEOUT_S = 20.0   # per-source hard cap (IMAP/CalDAV can crawl)

_state = {"text": "", "fetched_at": 0.0}
_refresh_lock = threading.Lock()
_refreshing = False


def _sources() -> list:
    """One zero-arg callable per brief line. Import inside each so a
    missing optional dependency disables one line, not the module."""

    def _cal() -> str:
        from mother.tools.icloud_calendar import get_calendar_today
        return "Calendar today: " + get_calendar_today({})

    def _mail() -> str:
        from mother.tools.icloud_email import summarize_inbox
        # dedicated=True: own IMAP connection. Sharing the cached one
        # would hold its lock for the whole background fetch and make a
        # user's email question queue behind ambient work.
        return "Inbox (12h): " + summarize_inbox(
            {"hours": 12, "max": 10, "dedicated": True}
        )

    def _wx() -> str:
        from mother.llm.tools import _handle_get_weather
        return "Weather: " + _handle_get_weather({}, None)

    def _news() -> str:
        from mother.tools.utility_tools import get_news_headlines
        return "Headlines: " + get_news_headlines({"count": 4})

    def _cal_tomorrow() -> str:
        from mother.tools.icloud_calendar import get_calendar_range
        d = (date.today() + timedelta(days=1)).isoformat()
        return "Calendar tomorrow: " + get_calendar_range(
            {"start_date": d, "end_date": d}
        )

    def _markets() -> str:
        # Quotes for the tickers the user tracks (ambient.watch_tickers
        # in app.yaml) so market awareness rides along for free.
        tickers = _watch_tickers()
        if not tickers:
            return ""
        from mother.llm.tools import _handle_get_stock_price
        quotes = []
        for t in tickers:
            q = _handle_get_stock_price({"symbol": t}, None)
            if "unreachable" not in q and "No market data" not in q:
                quotes.append(q.rstrip("."))
        return ("Markets: " + " | ".join(quotes)) if quotes else ""

    def _last_session() -> str:
        # One line of continuity across restarts: the final exchange of
        # the PREVIOUS session. Skipped when the history file is fresh
        # (< 15 min old) — that's the live conversation, already in the
        # model's history window.
        try:
            from mother.identity.speaker import get_or_fallback_user
            u = get_or_fallback_user()
            if u is None:
                return ""
            p = _REPO_ROOT / "assistant" / "memory" / "users" / u.user_id / "conv_history.json"
            if not p.exists() or time.time() - p.stat().st_mtime < 900:
                return ""
            import json
            msgs = (json.loads(p.read_text(encoding="utf-8")) or {}).get("messages") or []
            last_u = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
            last_a = next((m["content"] for m in reversed(msgs) if m.get("role") == "assistant"), "")
            if not last_u:
                return ""
            age_h = (time.time() - p.stat().st_mtime) / 3600
            return (
                f"Last session ({age_h:.0f}h ago) ended with — user: "
                f"\"{last_u[:120]}\" / you: \"{last_a[:120]}\""
            )
        except Exception:
            return ""

    def _health() -> str:
        # Empty when everything is nominal — the brief only carries a
        # line when something is actually wrong, so Ultron NOTICES a
        # dead service and can mention it unprompted.
        from mother.tools.system_tools import degraded_summary
        s = degraded_summary()
        return f"System alert: {s}" if s else ""

    return [_cal, _cal_tomorrow, _mail, _wx, _news, _markets, _last_session, _health]


def _gather() -> str:
    lines: list[str] = []
    fns = _sources()
    with ThreadPoolExecutor(max_workers=len(fns)) as pool:
        futures = [pool.submit(fn) for fn in fns]
        for fut in futures:
            try:
                line = (fut.result(timeout=_SOURCE_TIMEOUT_S) or "").strip()
            except Exception as e:
                logger.debug("[world-state] source failed: %s", e)
                continue
            # A source that answers "unavailable" adds nothing the
            # model should carry around — drop it.
            if line and "unavailable" not in line.lower():
                lines.append(line)
    return "\n".join(lines)


def refresh(force: bool = False) -> None:
    """Blocking refresh — call from a worker thread, never the loop."""
    global _refreshing
    if not force and time.time() - _state["fetched_at"] < BRIEF_TTL_S:
        return
    with _refresh_lock:
        if _refreshing:
            return
        _refreshing = True
    try:
        text = _gather()
        _state["text"] = text
        _state["fetched_at"] = time.time()
        logger.info("[world-state] brief refreshed (%d chars)", len(text))
    except Exception as e:
        logger.warning("[world-state] refresh failed: %s", e)
    finally:
        _refreshing = False


def maybe_refresh_async() -> None:
    """Kick a background refresh if the brief is stale. Returns instantly."""
    if time.time() - _state["fetched_at"] < BRIEF_TTL_S:
        return
    threading.Thread(target=refresh, daemon=True).start()


def get_brief() -> str:
    """The cached brief wrapped for context injection ('' if empty)."""
    if not _state["text"]:
        return ""
    age_min = int((time.time() - _state["fetched_at"]) / 60)
    stamp = "just now" if age_min < 1 else f"{age_min} min ago"
    return (
        f"[World state, refreshed {stamp}:\n{_state['text']}\n"
        "This is your ambient awareness — reference it naturally when "
        "relevant. Call tools only when the question needs more detail "
        "than this snapshot holds.]"
    )
