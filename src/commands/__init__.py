"""Shim: re-exports from mother.handlers for backward compatibility."""
from mother.handlers import *  # noqa: F401,F403
from mother.handlers import (  # explicit for IDE
    handle_finance_command, handle_finance_news,
    handle_weather_command, handle_info_search,
    handle_memory_query, handle_remember_command,
)
