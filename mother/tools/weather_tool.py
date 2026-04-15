import requests


def get_weather(lat: float = 43.0389, lon: float = -87.9065, *,
			    fahrenheit: bool = False, mph: bool = False) -> dict:
	"""Fetch current weather using Open-Meteo (no API key).

	Args:
		lat: Latitude
		lon: Longitude
		fahrenheit: If True, return temperature in Fahrenheit
		mph: If True, return windspeed in mph
	"""
	params = [
		f"latitude={lat}",
		f"longitude={lon}",
		"current_weather=true",
	]
	if fahrenheit:
		params.append("temperature_unit=fahrenheit")
	if mph:
		params.append("windspeed_unit=mph")
	url = "https://api.open-meteo.com/v1/forecast?" + "&".join(params)
	try:
		r = requests.get(url, timeout=5)
		r.raise_for_status()
		data = r.json().get("current_weather", {})
		return data or {"error": "No weather data returned."}
	except Exception as e:
		return {"error": str(e)}


def speak_weather(data: dict) -> str:
	if isinstance(data, dict) and "temperature" in data:
		temp = data.get("temperature")
		wind = data.get("windspeed")
		return f"The temperature is {temp}° with wind speed {wind}."
	return "Sorry, I couldn’t get the weather right now."
