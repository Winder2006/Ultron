"""Shim: re-exports from mother.handlers.weather for backward compatibility."""
from mother.handlers.weather import *  # noqa: F401,F403
from mother.handlers.weather import handle_weather_command  # explicit for IDE
