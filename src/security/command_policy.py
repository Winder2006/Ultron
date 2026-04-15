"""Shim: re-exports from mother.security.command_policy for backward compatibility."""
from mother.security.command_policy import *  # noqa: F401,F403
from mother.security.command_policy import register_command, run_command  # explicit for IDE
