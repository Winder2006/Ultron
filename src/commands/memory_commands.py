"""Shim: re-exports from mother.handlers.memory_commands for backward compatibility."""
from mother.handlers.memory_commands import *  # noqa: F401,F403
from mother.handlers.memory_commands import (  # explicit for IDE
    handle_memory_query, handle_remember_command,
)
