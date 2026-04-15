"""Shim: re-exports from mother.core.reminders for backward compatibility."""
from mother.core.reminders import *  # noqa: F401,F403
from mother.core.reminders import (  # explicit for IDE
    add_reminder, list_reminders, parse_reminder, delete_reminder,
    register_speak, start_background_thread, stop,
    _parse_time_expr,  # used by tools_registry
)
