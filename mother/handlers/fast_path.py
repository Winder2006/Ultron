"""Fast-path handlers that bypass the LLM entirely.

Consolidated dispatch for all fast-path intents. Called by both the
async orchestrator and the WebSocket voice endpoint.
"""
from __future__ import annotations

from typing import Optional, Tuple

from mother.core.intent import Intent
from mother.core.logging_config import get_logger

logger = get_logger("mother.handlers.fast_path")


def dispatch(
    intent: Intent,
    text: str,
    *,
    memory_manager=None,
    current_user=None,
) -> Optional[str]:
    """Route a fast-path intent to the appropriate handler.

    Returns the response string, or None if the intent isn't handled here.
    All handlers return (handled: bool, response: str | None) — we unwrap.
    """
    if intent == Intent.WEATHER:
        return _handle_weather(text)

    if intent in (Intent.FINANCE_QUOTE, Intent.FINANCE_NEWS):
        return _handle_finance(intent, text)

    if intent in (Intent.REMINDER_SET, Intent.REMINDER_LIST):
        return _handle_reminder(intent, text, current_user)

    if intent in (Intent.IDENTITY_CLAIM, Intent.IDENTITY_QUERY):
        return _handle_identity(intent, text)

    if intent == Intent.MEMORY_EXPLICIT:
        return _handle_memory(text, memory_manager)

    return None


# ── Individual handlers ──────────────────────────────────────────────────────

def _handle_weather(text: str) -> str:
    try:
        from mother.handlers.weather import handle_weather_command
        handled, response = handle_weather_command(text)
        return response or "Weather data unavailable."
    except Exception as e:
        logger.warning("Weather handler error: %s", e)
        return f"Weather data unavailable: {e}"


def _handle_finance(intent: Intent, text: str) -> str:
    try:
        from mother.handlers.finance import handle_finance_command, handle_finance_news
        if intent == Intent.FINANCE_NEWS:
            handled, response = handle_finance_news(text)
        else:
            handled, response = handle_finance_command(text)
        return response or "Finance data unavailable."
    except Exception as e:
        logger.warning("Finance handler error: %s", e)
        return f"Finance data unavailable: {e}"


def _handle_reminder(intent: Intent, text: str, current_user) -> str:
    try:
        from mother.core.reminders import parse_reminder, add_reminder, list_reminders
        uid = current_user.user_id if current_user else "unknown"

        if intent == Intent.REMINDER_LIST:
            pending = list_reminders(uid)
            if pending:
                lines = [r.get("text", "?") for r in pending[:5]]
                return "Your reminders: " + "; ".join(lines) + "."
            return "You have no pending reminders."

        # REMINDER_SET
        parsed = parse_reminder(text)
        if parsed:
            rtext, rtime = parsed
            add_reminder(uid, rtext, rtime)
            when = rtime.strftime("%I:%M %p").lstrip("0")
            return f"Reminder set for {when}: {rtext}."
        return "I couldn't parse that reminder. Try: remind me at 3 PM to call the dentist."
    except Exception as e:
        logger.warning("Reminder handler error: %s", e)
        return f"Reminder error: {e}"


def _handle_identity(intent: Intent, text: str) -> str:
    try:
        from mother.identity.speaker import (
            get_registry, get_current_user, set_current_user,
        )
        low = text.lower()

        if intent == Intent.IDENTITY_QUERY:
            cu = get_current_user()
            if cu:
                return f"You are {cu.display_name}. I identified you by your voice."
            return "I haven't been able to identify you yet. Say your name or speak so I can match your voice."

        # IDENTITY_CLAIM — "this is Oliver", "I'm Oliver"
        import re
        match = re.search(r"(?:this is|i'?m|i am|my name is)\s+(\w+)", low)
        if not match:
            return None  # fall through

        claimed = match.group(1)
        # Skip false positives
        _FP = {
            "already", "not", "going", "trying", "here", "there", "ready",
            "sorry", "happy", "sad", "fine", "good", "okay", "ok", "sure",
            "just", "also", "still", "now", "back", "done", "enrolled",
            "looking", "asking", "wondering", "thinking", "speaking", "talking",
        }
        if claimed in _FP:
            return None

        registry = get_registry()
        for uid in registry.list_users():
            profile = registry.get_user(uid)
            if profile and (claimed == uid or claimed == profile.display_name.lower()):
                set_current_user(uid, confidence=1.0, method="explicit")
                return f"Hello, {profile.display_name}. I've switched to your profile."

        # Not found — check if it looks like a real name
        orig_match = re.search(r"(?:this is|i'?m|i am|my name is)\s+(\w+)", text)
        orig_name = orig_match.group(1) if orig_match else claimed
        if len(orig_name) >= 2 and orig_name[0].isupper():
            return f"I don't have a profile for {orig_name}. Would you like to enroll?"
        return None
    except Exception as e:
        logger.warning("Identity handler error: %s", e)
        return None


def _handle_memory(text: str, memory_manager) -> str:
    low = text.lower()

    # "Remember" commands
    if "remember" in low:
        try:
            from mother.handlers.memory_commands import handle_remember_command
            handled, response = handle_remember_command(text, memory_manager)
            return response or "I'll try to remember that."
        except Exception as e:
            logger.warning("Remember command error: %s", e)
            return f"Memory error: {e}"

    # "What do you know about me" queries
    if any(p in low for p in ["what do you know", "what do you remember", "what have you learned"]):
        if memory_manager:
            try:
                summary = memory_manager.get_memory_summary(max_facts=8, max_episodic=5)
                if summary and summary != "No memories stored yet.":
                    return f"Here's what I know: {summary[:300]}"
                return "I haven't learned much about you yet. Tell me about yourself!"
            except Exception:
                pass
        return "I need to identify you first to access your memories."

    # Specific memory queries
    try:
        from mother.handlers.memory_commands import handle_memory_query
        handled, response = handle_memory_query(text, memory_manager)
        return response or "I don't have that information."
    except Exception as e:
        logger.warning("Memory query error: %s", e)
        return f"Memory error: {e}"
