"""Low-risk utility tools: time, forecast, news, math, units, web search.

Each function here takes a plain dict of arguments and returns a
**string** suitable for injection as a tool-result message — the LLM
then narrates it to the user in its own voice. Keeping the return type
as string (not a structured object) matches the existing dispatcher
contract in `mother/llm/tools.py`.

All functions degrade gracefully: network failure, bad input, or a
missing API key turns into a user-facing "unavailable" string rather
than an exception. The LLM can then acknowledge the miss instead of
the whole response tree blowing up.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import httpx


# ─────────────────────── constants ──────────────────────────────────

# News feed defaults. RSS is the path of least resistance — no API keys,
# no auth, extremely stable XML. Override via ULTRON_NEWS_FEEDS.
_DEFAULT_NEWS_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.npr.org/1001/rss.xml",
]


# Brave Search: free tier at https://brave.com/search/api/ (2000 q/mo).
# If no key is present, `brave_web_search` returns a helpful message
# rather than attempting an unauthenticated request (which errors out
# cryptically).
_BRAVE_API_BASE = "https://api.search.brave.com/res/v1/web/search"


# Reasonable HTTP timeout for every tool. Tool latency eats user-facing
# latency, so fail fast rather than block.
_HTTP_TIMEOUT_S = 5.0


# ─────────────────────── T — current_time ──────────────────────────

def current_time(args: dict) -> str:
    """Return the current local time + date.

    Ultron-as-LLM knows the system clock via the prompt scaffolding,
    but hallucinates about 5% of the time. Having a tool that returns
    a ground-truth string eliminates that error class.
    """
    fmt = args.get("format", "full")
    now = datetime.now()
    if fmt == "time":
        return now.strftime("%I:%M %p").lstrip("0")
    if fmt == "date":
        return now.strftime("%A, %B %d, %Y")
    # Full (default)
    return now.strftime("%A, %B %d, %Y at %I:%M %p").replace(" 0", " ")


# ─────────────────────── G — get_time_in ───────────────────────────

_TZ_ALIASES = {
    "new york": "America/New_York", "nyc": "America/New_York",
    "chicago": "America/Chicago", "milwaukee": "America/Chicago",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "sf": "America/Los_Angeles",
    "london": "Europe/London", "uk": "Europe/London",
    "paris": "Europe/Paris", "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo",
    "sydney": "Australia/Sydney",
    "utc": "UTC", "gmt": "UTC",
}


def get_time_in(args: dict) -> str:
    """Time in a named timezone or city."""
    where = (args.get("location") or args.get("timezone") or "").strip().lower()
    if not where:
        return "No location provided."
    tz_name = _TZ_ALIASES.get(where, where)
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            tz = ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            # Last-ditch: title-case guess like "America/New_York"
            guess = tz_name.title().replace(" ", "_")
            if "/" not in guess and where in _TZ_ALIASES:
                guess = _TZ_ALIASES[where]
            try:
                tz = ZoneInfo(guess)
            except Exception:
                return f"Unknown timezone: {where}"
    except ImportError:
        return "Timezone support is not available on this server."
    now = datetime.now(tz)
    return now.strftime(f"%A, %I:%M %p ({tz_name})").replace(" 0", " ")


# ─────────────────────── E — get_forecast ──────────────────────────

# Per-city shortcut map. Queried *before* the geocoder so common US
# cities don't accidentally match smaller namesakes abroad (Portland
# UK, Birmingham AL vs UK, etc.). Add more here as you notice misses.
_CITY_COORDS = {
    "milwaukee":   (43.0389, -87.9065),
    "madison":     (43.0731, -89.4012),
    "chicago":     (41.8781, -87.6298),
    "new york":    (40.7128, -74.0060),
    "nyc":         (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "la":          (34.0522, -118.2437),
    "san francisco": (37.7749, -122.4194),
    "sf":          (37.7749, -122.4194),
    "seattle":     (47.6062, -122.3321),
    "portland":    (45.5152, -122.6784),   # Oregon, not Dorset
    "boston":      (42.3601, -71.0589),
    "denver":      (39.7392, -104.9903),
    "austin":      (30.2672, -97.7431),
    "miami":       (25.7617, -80.1918),
    "phoenix":     (33.4484, -112.0740),
    "dallas":      (32.7767, -96.7970),
    "atlanta":     (33.7490, -84.3880),
    "london":      (51.5074, -0.1278),
    "paris":       (48.8566, 2.3522),
    "berlin":      (52.5200, 13.4050),
    "tokyo":       (35.6762, 139.6503),
    "sydney":      (-33.8688, 151.2093),
    "toronto":     (43.6532, -79.3832),
    "mexico city": (19.4326, -99.1332),
}

_WEATHER_CODE_DESC = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "rain showers", 82: "violent rain",
    95: "thunderstorms", 96: "thunderstorms w/ hail", 99: "severe storms",
}


def _geocode_open_meteo(place: str) -> Optional[tuple[float, float, str]]:
    """Resolve a place name to (lat, lon, display).

    Returns None on any failure OR if the match looks weak. We pull
    several results (not just the top one) and prefer the entry with
    the highest population — this suppresses the common failure mode
    where Open-Meteo returns a tiny hamlet that shares a name with a
    major city the user was almost certainly asking about.
    """
    try:
        r = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 5, "language": "en"},
            timeout=_HTTP_TIMEOUT_S,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results") or []
        if not results:
            return None
        # Prefer the highest-population match. Most misresolutions
        # ("rivendell" → Rivendell, TN) are small places with population
        # near zero; real cities the user means have populations in the
        # tens of thousands or more.
        results.sort(key=lambda h: h.get("population") or 0, reverse=True)
        hit = results[0]
        display = hit.get("name", place)
        if hit.get("country"):
            display += f", {hit['country']}"
        return float(hit["latitude"]), float(hit["longitude"]), display
    except Exception:
        return None


def get_forecast(args: dict) -> str:
    """Multi-day forecast. Defaults to 3 days; capped at 7.

    Returns a short bulleted string the LLM can restate naturally.
    """
    location = (args.get("location") or "current").lower().strip()
    days = int(args.get("days", 3))
    days = max(1, min(7, days))

    # Resolve coordinates
    if location in _CITY_COORDS:
        lat, lon = _CITY_COORDS[location]
        display = location.title()
    elif location == "current":
        lat, lon = _CITY_COORDS["milwaukee"]
        display = "Milwaukee"
    else:
        resolved = _geocode_open_meteo(location)
        if resolved is None:
            return f"Couldn't find location: {location}."
        lat, lon, display = resolved

    try:
        r = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "forecast_days": days,
            },
            timeout=_HTTP_TIMEOUT_S,
        )
        if r.status_code != 200:
            return f"Forecast unavailable (HTTP {r.status_code})."
        daily = r.json().get("daily", {})
    except Exception as e:
        return f"Forecast unavailable: {e}"

    dates = daily.get("time", [])
    codes = daily.get("weathercode", [])
    highs = daily.get("temperature_2m_max", [])
    lows = daily.get("temperature_2m_min", [])
    precips = daily.get("precipitation_probability_max", [])

    if not dates:
        return f"No forecast data for {display}."

    lines = [f"Forecast for {display}:"]
    for i in range(min(days, len(dates))):
        try:
            day_name = datetime.fromisoformat(dates[i]).strftime("%A")
        except Exception:
            day_name = dates[i]
        desc = _WEATHER_CODE_DESC.get(int(codes[i]), "varied") if codes else "varied"
        hi = round(highs[i]) if i < len(highs) else "?"
        lo = round(lows[i]) if i < len(lows) else "?"
        pop = f", {precips[i]}% rain" if i < len(precips) and precips[i] else ""
        lines.append(f"{day_name}: {desc}, high {hi}° / low {lo}°{pop}")
    return " ".join(lines)


# ─────────────────────── F — get_news_headlines ─────────────────────

# Very small XML parser for RSS <item><title>..</title>. We deliberately
# avoid pulling in `feedparser` as a dependency — it's a big install and
# RSS <title> extraction is a 10-line regex job.
_RSS_ITEM_TITLE_RE = re.compile(
    r"<item>.*?<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
    re.DOTALL | re.IGNORECASE,
)


def _feed_urls() -> List[str]:
    env = os.environ.get("ULTRON_NEWS_FEEDS", "").strip()
    if env:
        return [u.strip() for u in env.split(",") if u.strip()]
    return list(_DEFAULT_NEWS_FEEDS)


def _fetch_titles_from_feed(url: str, max_titles: int) -> List[str]:
    """Fetch up to `max_titles` titles from a single RSS feed.

    Returns an empty list on any failure (HTTP error, timeout, bad XML,
    feed with no items). Callers should treat an empty return as
    "skip this feed and try the next."
    """
    try:
        r = httpx.get(
            url,
            timeout=_HTTP_TIMEOUT_S,
            headers={"User-Agent": "Ultron/1.0"},
        )
        if r.status_code != 200:
            return []
    except Exception:
        return []

    out: List[str] = []
    for m in _RSS_ITEM_TITLE_RE.finditer(r.text):
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        # Unescape XML entities — NPR (and some others) emit &apos;/&amp;
        # raw inside CDATA blocks, which the TTS would otherwise read
        # as "ampersand apos semicolon".
        try:
            from html import unescape as _unescape
            title = _unescape(title)
        except Exception:
            pass
        if title:
            out.append(title)
            if len(out) >= max_titles:
                break
    return out


def get_news_headlines(args: dict) -> str:
    """Return the top N headlines across configured RSS feeds.

    Rotates across feeds rather than draining them sequentially: this
    gives a broader mix of stories and, more importantly, lets one
    slow or dead feed fail without starving the response. If *every*
    feed fails (network out, all sources 503), returns a neutral
    message rather than an error — callers never see an exception.
    """
    try:
        count = int(args.get("count", 5))
    except (TypeError, ValueError):
        count = 5
    count = max(1, min(10, count))

    feeds = _feed_urls()
    if not feeds:
        return "No news feeds configured."

    # Pull a small pool from each feed (up to `count` from each, capped
    # so big feeds don't dominate). Skip feeds that fail silently.
    per_feed_cap = max(3, count)
    pools: List[List[str]] = []
    for url in feeds:
        titles = _fetch_titles_from_feed(url, per_feed_cap)
        if titles:
            pools.append(titles)

    if not pools:
        return "No headlines available right now."

    # Round-robin across the pools so the final list mixes sources
    # rather than reading like "all of feed 1, then all of feed 2."
    merged: List[str] = []
    seen: set = set()
    max_rounds = max((len(p) for p in pools), default=0)
    for round_idx in range(max_rounds):
        if len(merged) >= count:
            break
        for pool in pools:
            if round_idx >= len(pool):
                continue
            title = pool[round_idx]
            if title not in seen:
                seen.add(title)
                merged.append(title)
                if len(merged) >= count:
                    break

    if not merged:
        return "No headlines available right now."
    return "Top headlines: " + "; ".join(merged)


# ─────────────────────── H — calculate ─────────────────────────────

# Allowed names and operators for the safe calculator. We do a restricted
# `eval` against a minimal globals dict — no builtins, only math funcs
# and constants. The regex pre-validates the expression to block anything
# outside arithmetic + named math functions, stopping payloads like
# `__import__('os').system('rm -rf /')` before eval ever sees them.
_SAFE_CALC_NAMES = {
    "pi": 3.141592653589793,
    "e":  2.718281828459045,
    "abs": abs, "round": round, "min": min, "max": max,
}
try:
    import math as _math
    for _name in (
        "sqrt", "pow", "log", "log2", "log10", "exp",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "floor", "ceil",
    ):
        _SAFE_CALC_NAMES[_name] = getattr(_math, _name)
except ImportError:
    pass

# Only these characters and identifiers are allowed in the raw expression.
# We strip whitespace then check the full string matches the pattern.
_CALC_ALLOWED_RE = re.compile(r"^[0-9+\-*/().,%\s]+$")
# Or: same + known function names (allowlist of word tokens)
_CALC_SAFE_NAME_RE = re.compile(
    r"^[0-9+\-*/().,%\s]*(?:(?:"
    + "|".join(re.escape(n) for n in _SAFE_CALC_NAMES)
    + r")[0-9+\-*/().,%\s]*)*$"
)


def calculate(args: dict) -> str:
    """Evaluate an arithmetic expression safely.

    Accepts basic arithmetic plus a small allowlist of math functions
    (sqrt, log, sin, etc.) and constants (pi, e). Rejects anything
    containing other names, attribute access, or Python keywords."""
    expr_raw = (args.get("expression") or "").strip()
    if not expr_raw:
        return "No expression provided."
    if len(expr_raw) > 200:
        return "Expression too long."

    # Syntax pre-screen: if it's not purely allowed chars + known names,
    # reject without even trying to eval.
    # Cheap first gate: no dunders.
    if "__" in expr_raw or "import" in expr_raw.lower():
        return "Expression rejected."
    # Token-level gate: strip all allowlisted names, then check what's
    # left is only the allowed arithmetic punctuation.
    stripped = expr_raw
    for n in _SAFE_CALC_NAMES:
        stripped = re.sub(rf"\b{re.escape(n)}\b", "", stripped)
    if not _CALC_ALLOWED_RE.match(stripped):
        return "Expression contains unsupported characters or names."

    try:
        result = eval(expr_raw, {"__builtins__": {}}, _SAFE_CALC_NAMES)
    except Exception as e:
        return f"Could not evaluate: {e.__class__.__name__}"

    # Format nicely: integers as integers, floats trimmed.
    if isinstance(result, float):
        if result.is_integer():
            result_str = str(int(result))
        else:
            # Cap precision at 6 decimals to avoid spoken gibberish
            result_str = f"{result:.6g}"
    else:
        result_str = str(result)
    return f"{expr_raw} = {result_str}"


# ─────────────────────── I — convert_units ─────────────────────────

# Conversion factors — value is "multiply by this to get the canonical
# base unit for the family." Keeps the code short; adds support via
# data rather than code when we need new units.
_LENGTH_TO_METERS = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
}

_WEIGHT_TO_GRAMS = {
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
    "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
    "lb": 453.592, "lbs": 453.592, "pound": 453.592, "pounds": 453.592,
    "ton": 907184.0, "tons": 907184.0,
}

_VOLUME_TO_ML = {
    "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
    "l": 1000.0, "liter": 1000.0, "liters": 1000.0, "litre": 1000.0,
    "tsp": 4.92892, "teaspoon": 4.92892, "teaspoons": 4.92892,
    "tbsp": 14.7868, "tablespoon": 14.7868, "tablespoons": 14.7868,
    "cup": 236.588, "cups": 236.588,
    "oz_fl": 29.5735, "fl_oz": 29.5735, "floz": 29.5735,
    "pint": 473.176, "pints": 473.176,
    "quart": 946.353, "quarts": 946.353,
    "gallon": 3785.41, "gallons": 3785.41,
}

_FAMILIES = [
    ("length", _LENGTH_TO_METERS, "meters"),
    ("weight", _WEIGHT_TO_GRAMS, "grams"),
    ("volume", _VOLUME_TO_ML, "milliliters"),
]


def _lookup_unit(unit: str) -> Optional[tuple[str, float, dict, str]]:
    """Return (family_name, factor, family_table, canonical) or None."""
    u = unit.strip().lower().replace(" ", "_")
    for name, table, canon in _FAMILIES:
        if u in table:
            return name, table[u], table, canon
    return None


def _convert_temperature(value: float, src: str, dst: str) -> Optional[float]:
    """Special-case temperature because it needs offsets, not just factors."""
    src = src.strip().lower().replace("°", "").strip()
    dst = dst.strip().lower().replace("°", "").strip()
    alias = {"c": "celsius", "celsius": "celsius",
             "f": "fahrenheit", "fahrenheit": "fahrenheit",
             "k": "kelvin", "kelvin": "kelvin"}
    s = alias.get(src); d = alias.get(dst)
    if s is None or d is None:
        return None
    # to Celsius
    if s == "celsius":     c = value
    elif s == "fahrenheit": c = (value - 32) * 5 / 9
    else:                   c = value - 273.15
    # from Celsius
    if d == "celsius":     return c
    if d == "fahrenheit":  return c * 9 / 5 + 32
    return c + 273.15


def convert_units(args: dict) -> str:
    """Convert a value between units (length, weight, volume, temp)."""
    try:
        value = float(args.get("value"))
    except (TypeError, ValueError):
        return "Invalid numeric value."
    src = (args.get("from") or args.get("from_unit") or "").strip()
    dst = (args.get("to") or args.get("to_unit") or "").strip()
    if not src or not dst:
        return "Missing from/to unit."

    # Temperature: try special-case first
    temp = _convert_temperature(value, src, dst)
    if temp is not None:
        return f"{value} {src} = {round(temp, 2)} {dst}"

    src_info = _lookup_unit(src)
    dst_info = _lookup_unit(dst)
    if src_info is None or dst_info is None:
        return f"Unsupported unit: {src if src_info is None else dst}"
    if src_info[0] != dst_info[0]:
        return f"Incompatible units: {src} ({src_info[0]}) and {dst} ({dst_info[0]})"

    base = value * src_info[1]
    out = base / dst_info[1]
    # Format: avoid long-tail floats
    if abs(out) >= 100 or out == int(out):
        out_str = f"{out:.2f}".rstrip("0").rstrip(".")
    else:
        out_str = f"{out:.4g}"
    return f"{value} {src} = {out_str} {dst}"


# ─────────────────────── J — brave_web_search ──────────────────────

def brave_web_search(args: dict) -> str:
    """Live web search via Brave Search API.

    Returns a compact multi-line string of the top few hits. Requires
    BRAVE_API_KEY in the environment; otherwise returns a graceful
    "unavailable" message rather than failing.
    """
    query = (args.get("query") or "").strip()
    if not query:
        return "No query provided."
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return (
            "Web search is not configured. "
            "Set BRAVE_API_KEY (free at brave.com/search/api)."
        )
    try:
        r = httpx.get(
            _BRAVE_API_BASE,
            params={"q": query, "count": 5, "safesearch": "moderate"},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            timeout=_HTTP_TIMEOUT_S,
        )
        if r.status_code != 200:
            return f"Web search returned HTTP {r.status_code}."
        data = r.json()
    except Exception as e:
        return f"Web search failed: {e}"

    results = (data.get("web") or {}).get("results") or []
    if not results:
        return f"No web results for {query!r}."

    # Brave returns HTML in title/description — they use <strong> to
    # indicate matched terms, and HTML entities for apostrophes etc.
    # If we hand that verbatim to the LLM, it often echoes it to the
    # user literally; TTS then reads "less than strong greater than".
    # Strip tags, unescape entities, collapse whitespace.
    from html import unescape as _unescape
    _TAG_RE = re.compile(r"<[^>]+>")

    def _clean(s: str) -> str:
        s = _TAG_RE.sub("", s or "")
        s = _unescape(s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    lines: List[str] = []
    for item in results[:4]:
        title = _clean(item.get("title") or "")
        desc = _clean(item.get("description") or "")
        if not title:
            continue
        # Shorter blurb so the LLM stays terse instead of parroting paragraphs.
        if len(desc) > 140:
            desc = desc[:140].rstrip() + "…"
        lines.append(f"{title}: {desc}" if desc else title)

    # Structure the output as a numbered list. Makes it obvious to the
    # LLM this is a search-result blob to synthesize, not narration to
    # read. Also bounded overall size so we don't pollute the next turn.
    body = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines[:4]))
    return f"Search results for {query!r}:\n{body}"


# ─────────────────────── A — search_code (code RAG) ─────────────────

def search_code(args: dict) -> str:
    """Search Ultron's own codebase via the RAG service.

    Called as a tool mid-conversation when the LLM decides it needs to
    consult its own implementation (e.g., user asks a deeply technical
    question about a specific module). The heuristic-based pre-prompt
    injection covers common cases; this tool covers the long tail.
    """
    query = (args.get("query") or "").strip()
    if not query:
        return "No query provided."
    try:
        k = int(args.get("k", 3))
    except (TypeError, ValueError):
        k = 3
    k = max(1, min(6, k))

    base = os.environ.get("ULTRON_RAG_BASE", "http://127.0.0.1:8123")
    try:
        r = httpx.get(
            f"{base}/code-search",
            params={"q": query, "k": k},
            timeout=_HTTP_TIMEOUT_S,
        )
        if r.status_code != 200:
            return f"Code search unavailable (HTTP {r.status_code})."
        hits = r.json()
    except Exception as e:
        return f"Code search failed: {e}"

    if not hits:
        return f"No code found matching {query!r}."

    lines: List[str] = []
    for h in hits:
        meta = h.get("meta") or {}
        sym = meta.get("symbol") or h.get("path", "?")
        lines_range = meta.get("lines") or ""
        preview = (h.get("text") or "").splitlines()
        first_real = next(
            (ln for ln in preview if ln.strip() and not ln.startswith("#")),
            "",
        )[:140]
        tag = f"{sym} [{lines_range}]" if lines_range else sym
        lines.append(f"{tag}: {first_real}")
    return "Code references: " + " | ".join(lines)


# ─────────────────────── AA — list_my_tools ────────────────────────

def list_my_tools(args: dict) -> str:
    """Return a condensed catalog of the tools currently registered.

    Reads directly from `mother.llm.tools.TOOLS_SCHEMA` so the output
    is always current — no separate catalog to keep in sync.
    """
    try:
        from mother.llm.tools import TOOLS_SCHEMA
    except Exception as e:
        return f"Cannot list tools: {e}"
    lines: List[str] = []
    for t in TOOLS_SCHEMA:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        name = fn.get("name", "?")
        desc = (fn.get("description") or "").split(".", 1)[0]  # first sentence
        lines.append(f"{name}: {desc}")
    return "My tools — " + "; ".join(lines)


# ─────────────────────── CC — forget_fact ──────────────────────────

def forget_fact(args: dict, *, current_user=None) -> str:
    """Delete a fact by key for the current user."""
    key = (args.get("key") or "").strip()
    if not key:
        return "No key provided."
    if current_user is None:
        return "No user identified — cannot modify memory."
    try:
        from mother.memory.manager import get_user_memory
        mem = get_user_memory(current_user.user_id)
        if mem is None:
            return "Memory unavailable."
        existing = mem.get_fact(key)
        if existing is None:
            return f"No stored fact for {key!r}."
        ok = mem.delete_fact(key)
        return (
            f"Forgotten: {key} (was {existing!r})." if ok
            else f"Could not forget {key!r}."
        )
    except Exception as e:
        return f"Forget failed: {e}"


# ─────────────────────── DD — correct_fact ─────────────────────────

def correct_fact(args: dict, *, current_user=None) -> str:
    """Overwrite a single fact with a corrected value.

    Distinct from forget_fact: this is the "actually, it's X not Y"
    pattern. Stores with source='corrected' and confidence=1.0 so it
    takes precedence over any future regex/LLM extraction.
    """
    key = (args.get("key") or "").strip()
    value = (args.get("value") or "").strip()
    category = (args.get("category") or "general").strip()
    if not key or not value:
        return "Missing key or value."
    if current_user is None:
        return "No user identified — cannot modify memory."
    try:
        from mother.memory.manager import get_user_memory
        mem = get_user_memory(current_user.user_id)
        if mem is None:
            return "Memory unavailable."
        mem.set_fact(key, value, category=category,
                     confidence=1.0, source="corrected")
        return f"Noted: {key} is {value}."
    except Exception as e:
        return f"Correction failed: {e}"
