"""Weather command handlers for MOTHER."""
from __future__ import annotations

from typing import Tuple, Optional, Dict
import httpx

from mother.core.logging_config import get_logger

logger = get_logger("commands.weather")

# Known locations
LOCATION_MAP = {
    "milwaukee": (43.0389, -87.9065),
    "chicago": (41.8781, -87.6298),
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "san francisco": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321),
    "boston": (42.3601, -71.0589),
    "miami": (25.7617, -80.1918),
    "denver": (39.7392, -104.9903),
    "atlanta": (33.7490, -84.3880),
    "home": (43.0389, -87.9065),  # Default to Milwaukee
}


def resolve_location(user_input: str) -> Tuple[float, float]:
    """Resolve location from user input to lat/lon.
    
    Returns:
        (latitude, longitude) tuple
    """
    low = (user_input or "").lower()
    
    for loc_name, coords in LOCATION_MAP.items():
        if loc_name in low:
            logger.debug(f"Resolved location: {loc_name}")
            return coords
    
    # Default to Milwaukee
    logger.debug("Using default location: Milwaukee")
    return LOCATION_MAP["home"]


def get_weather(lat: float, lon: float) -> Optional[Dict]:
    """Fetch weather from Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Weather data dict or None
    """
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit": "mph",
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("current_weather", {})
    except httpx.TimeoutException:
        logger.warning("Weather API timeout")
    except httpx.HTTPError as e:
        logger.warning(f"Weather API HTTP error: {e}")
    except Exception as e:
        logger.error(f"Weather API unexpected error: {e}")
    
    return None


def handle_weather_command(user_input: str) -> Tuple[bool, Optional[str]]:
    """Handle weather query.
    
    Args:
        user_input: User's text input
        
    Returns:
        (handled, response_text)
    """
    low = (user_input or "").lower()
    
    # Check if this is a weather query
    if not any(kw in low for kw in ["weather", "temperature", "forecast"]):
        return False, None
    
    logger.info("Weather query detected")
    
    lat, lon = resolve_location(user_input)
    data = get_weather(lat, lon)
    
    if data:
        temp = data.get("temperature")
        wind = data.get("windspeed")
        
        if temp is not None:
            response = f"The temperature is {round(temp)} degrees Fahrenheit"
            if wind is not None:
                response += f" with wind {round(wind)} miles per hour."
            else:
                response += "."
        else:
            response = "Weather data incomplete."
    else:
        response = "Sorry, I couldn't get the weather."
    
    return True, response

