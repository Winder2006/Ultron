"""Shim: re-exports from mother.tools for backward compatibility."""
from mother.tools.info_search import get_info
from mother.tools.weather_tool import get_weather

try:
    ALLOWED_COMMANDS
except NameError:
    ALLOWED_COMMANDS = {}

ALLOWED_COMMANDS.update({
    "info.search": get_info,
    "weather.get": get_weather,
})
