"""Tool-call routing for MOTHER via Ollama's function-calling API.

Registers first-class tools (weather, finance, info, reminders, memory) and
dispatches LLM tool-call requests to the appropriate handlers.

Usage (in cli.py LLM call):
    from .tools_registry import TOOLS_SCHEMA, dispatch_tool_call

    # Pass schema to Ollama
    for chunk in llm.stream_chat(messages, tools=TOOLS_SCHEMA):
        ...
    # If LLM emitted a tool_call, dispatch it:
    result = dispatch_tool_call(tool_name, tool_args, context)

The tool system is additive — queries that match a tool get a grounded
answer; queries that don't match fall through to plain LLM generation.
Models that don't support tools (older Ollama builds) are handled
gracefully: if the LLM returns plain text instead of a tool call, the
normal streaming path runs unchanged.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Tool schema (Ollama / OpenAI function-call format)
# ---------------------------------------------------------------------------

TOOLS_SCHEMA: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or 'current' for the default location.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock or cryptocurrency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol, e.g. TSLA, BTC, AAPL.",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_info",
            "description": (
                "Look up a factual answer about a PERSON, PLACE, ORGANIZATION, "
                "HISTORICAL EVENT, or DEFINITION. Use for 'who is X', 'what is "
                "X', 'where is X', 'tell me about X'. Wikipedia-first, with "
                "DuckDuckGo as fallback. Prefer this over brave_web_search "
                "for biographical or encyclopedic queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or question to look up.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for the user at a specific time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "What to remind the user about.",
                    },
                    "time_expr": {
                        "type": "string",
                        "description": "Natural language time: 'at 3 PM', 'in 10 minutes', 'tomorrow at 9 AM'.",
                    },
                },
                "required": ["text", "time_expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory",
            "description": "Retrieve stored facts and memories about the current user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional topic to focus the memory retrieval.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": (
                "Read a note from the user's sandboxed notes directory "
                "(~/AI_Workspace). Only use for notes, todo lists, scratch "
                "files, logs — not for system files or code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path inside the notes directory, e.g. "
                            "'todo.md' or 'ideas/project-plan.md'."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": (
                "Write or append to a note inside the user's sandboxed notes "
                "directory (~/AI_Workspace). Use this when the user asks you "
                "to save something, write it down, or add to a list. Never "
                "use this to write code, logs, or system files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside the notes directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text to write.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "When true, append to existing file instead of overwriting. Default false.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    # ─────────── T: current_time ───────────
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": (
                "Get the current local time and/or date as ground truth. "
                "Use this instead of guessing — LLMs sometimes hallucinate "
                "times from prior context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["full", "time", "date"],
                        "description": "'time' for just clock, 'date' for just date, 'full' for both (default).",
                    }
                },
                "required": [],
            },
        },
    },
    # ─────────── G: get_time_in ───────────
    {
        "type": "function",
        "function": {
            "name": "get_time_in",
            "description": "Get the current time in a specified city or timezone (e.g. 'Tokyo', 'UTC', 'America/New_York').",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "A city name or IANA timezone.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    # ─────────── E: get_forecast ───────────
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Multi-day weather forecast for a named city. Use when the user asks about upcoming weather, not just today.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g. 'Seattle', 'Tokyo'). Defaults to Milwaukee if omitted.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "How many days to forecast (1-7). Default 3.",
                    },
                },
                "required": [],
            },
        },
    },
    # ─────────── F: get_news_headlines ───────────
    {
        "type": "function",
        "function": {
            "name": "get_news_headlines",
            "description": (
                "ONLY for current breaking news. Returns general world "
                "headlines from BBC/NPR RSS. Use ONLY when the user "
                "explicitly asks 'what's in the news', 'what happened "
                "today', 'news headlines'. Do NOT use for factual "
                "questions, people, places, or specific topics — those "
                "need search_info or brave_web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "How many headlines to return (1-10). Default 5.",
                    }
                },
                "required": [],
            },
        },
    },
    # ─────────── H: calculate ───────────
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Safely evaluate an arithmetic expression (+, -, *, /, **, %, "
                "parentheses, plus math functions like sqrt, log, sin, cos). "
                "Use for any numeric computation — LLMs are unreliable at arithmetic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "E.g. '1040 / 8.5', 'sqrt(144) + 3', '(100 - 32) * 5/9'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    # ─────────── I: convert_units ───────────
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Convert a value between units. Supports length, weight, volume, temperature.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "The numeric value to convert."},
                    "from": {"type": "string", "description": "Source unit (e.g. 'miles', 'celsius', 'lbs')."},
                    "to": {"type": "string", "description": "Target unit (e.g. 'km', 'fahrenheit', 'kg')."},
                },
                "required": ["value", "from", "to"],
            },
        },
    },
    # ─────────── J: brave_web_search ───────────
    {
        "type": "function",
        "function": {
            "name": "brave_web_search",
            "description": (
                "Live web search via Brave. Use when the answer would "
                "change over time (current officials, recent events, "
                "prices, scores, who currently holds a position, "
                "today's date-specific queries, anything where "
                "Wikipedia might be stale). For stable encyclopedic "
                "facts (definitions, dead people, historical events) "
                "prefer search_info. NEVER claim you 'lack web access' "
                "without trying this tool first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    # ─────────── A: search_code ───────────
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Search Ultron's own codebase for relevant modules or functions. "
                "Use when the user asks a deeply technical question about how "
                "you are implemented that wasn't covered by pre-prompt context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to find in the code."},
                    "k": {"type": "integer", "description": "How many matches (1-6). Default 3."},
                },
                "required": ["query"],
            },
        },
    },
    # ─────────── AA: list_my_tools ───────────
    {
        "type": "function",
        "function": {
            "name": "list_my_tools",
            "description": "List the tools Ultron can currently use. Useful when asked 'what can you do?' so the answer is truthful and current.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ─────────── CC: forget_fact ───────────
    {
        "type": "function",
        "function": {
            "name": "forget_fact",
            "description": (
                "Delete a stored fact about the current user by key. Use when "
                "the user explicitly asks you to forget something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The fact key to delete (e.g. 'employer', 'favorite_color').",
                    }
                },
                "required": ["key"],
            },
        },
    },
    # ─────────── DD: correct_fact ───────────
    {
        "type": "function",
        "function": {
            "name": "correct_fact",
            "description": (
                "Overwrite a stored fact with a corrected value. Use when the "
                "user says 'actually it's X not Y'. Records as high-confidence "
                "correction so it persists over future auto-extraction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Fact key (e.g. 'name', 'employer')."},
                    "value": {"type": "string", "description": "The corrected value."},
                    "category": {
                        "type": "string",
                        "description": "Optional category: personal, work, preference, contact, general.",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Context bag passed to every dispatcher
# ---------------------------------------------------------------------------

class ToolContext:
    """Holds live references needed by tool handlers."""

    def __init__(
        self,
        http_client=None,
        rag_base: str = "http://127.0.0.1:8123",
        rag_timeout: float = 0.5,
        user_memory=None,
        current_user=None,
        add_reminder_fn: Optional[Callable] = None,
    ):
        self.http = http_client
        self.rag_base = rag_base
        self.rag_timeout = rag_timeout
        self.user_memory = user_memory
        self.current_user = current_user
        self.add_reminder_fn = add_reminder_fn


# ---------------------------------------------------------------------------
# Individual tool handlers
# ---------------------------------------------------------------------------

def _handle_get_weather(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.weather_tool import get_weather as _get_weather
    location = (args.get("location") or "current").lower()
    loc_map = {
        "milwaukee": (43.0389, -87.9065),
        "madison":   (43.0731, -89.4012),
        "chicago":   (41.8781, -87.6298),
        "new york":  (40.7128, -74.0060),
        "los angeles": (34.0522, -118.2437),
    }
    lat, lon = loc_map.get(location, (43.0389, -87.9065))
    try:
        data = _get_weather(lat, lon, fahrenheit=True, mph=True)
    except Exception:
        return "Weather data unavailable."
    if data.get("error"):
        return "Weather data unavailable."
    temp = round(data.get("temperature", 0))
    wind = round(data.get("windspeed", 0))
    desc = data.get("description", "")
    loc_name = location.title() if location != "current" else "Milwaukee"
    return f"{loc_name}: {temp}°F, {desc}, wind {wind} mph."


def _handle_get_stock_price(args: Dict, ctx: ToolContext) -> str:
    symbol = (args.get("symbol") or "").upper().strip()
    if not symbol:
        return "No symbol provided."
    data = {}
    if ctx.http:
        try:
            resp = ctx.http.get(
                f"{ctx.rag_base}/finance/quote",
                params={"symbol": symbol},
                timeout=ctx.rag_timeout,
            )
            data = resp.json() if resp.status_code == 200 else {}
        except Exception:
            pass
    price = data.get("regularMarketPrice")
    if price is None:
        return f"Price for {symbol} unavailable."
    curr = data.get("currency") or "USD"
    name = data.get("shortName") or symbol
    return f"{name} ({symbol}): {price:.2f} {curr}."


def _handle_search_info(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.info_search import get_info
    query = (args.get("query") or "").strip()
    if not query:
        return "No query provided."
    import concurrent.futures as _cf
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(get_info, query)
            info = fut.result(timeout=4.0)
    except Exception:
        return "Search timed out or failed."
    if info.get("error"):
        return "No information found."
    summary = info.get("summary", "")
    return summary[:320] if summary else "No summary available."


def _handle_set_reminder(args: Dict, ctx: ToolContext) -> str:
    text = (args.get("text") or "").strip()
    time_expr = (args.get("time_expr") or "").strip()
    if not text or not time_expr:
        return "Missing reminder text or time."
    if ctx.add_reminder_fn is None:
        return "Reminder system unavailable."
    from mother.core.reminders import _parse_time_expr
    trigger = _parse_time_expr(time_expr)
    if trigger is None:
        return f"Couldn't parse time '{time_expr}'. Try 'at 3 PM' or 'in 10 minutes'."
    user_id = ctx.current_user.user_id if ctx.current_user else "unknown"
    ctx.add_reminder_fn(user_id, text, trigger)
    when = trigger.strftime("%I:%M %p").lstrip("0")
    return f"Reminder set for {when}: {text}."


def _handle_get_memory(args: Dict, ctx: ToolContext) -> str:
    if ctx.user_memory is None:
        return "No memory available — user not identified."
    query = (args.get("query") or "").strip()
    return ctx.user_memory.get_context_for_prompt(query, max_items=5) or "No memories stored yet."


def _handle_read_note(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.notes_tool import read_note, SandboxError
    path = (args.get("path") or "").strip()
    if not path:
        return "No path provided."
    try:
        data = read_note(path)
    except SandboxError as e:
        return f"Cannot read: {e}"
    except Exception as e:
        return f"Read failed: {e}"
    # Cap what we return to the LLM (it re-emits this for TTS). 800 chars
    # is plenty for Ultron to summarize from; the user can ask for more.
    if len(data) > 800:
        return data[:800] + " … (note continues; ask to read more)"
    return data or "(note is empty)"


def _handle_write_note(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.notes_tool import write_note, SandboxError
    path = (args.get("path") or "").strip()
    content = args.get("content") or ""
    append = bool(args.get("append", False))
    if not path or not content:
        return "Missing path or content."
    try:
        return write_note(path, content, append=append)
    except SandboxError as e:
        return f"Cannot write: {e}"
    except Exception as e:
        return f"Write failed: {e}"


def _handle_current_time(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import current_time
    return current_time(args)


def _handle_get_time_in(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import get_time_in
    return get_time_in(args)


def _handle_get_forecast(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import get_forecast
    return get_forecast(args)


def _handle_get_news_headlines(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import get_news_headlines
    return get_news_headlines(args)


def _handle_calculate(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import calculate
    return calculate(args)


def _handle_convert_units(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import convert_units
    return convert_units(args)


def _handle_brave_web_search(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import brave_web_search
    return brave_web_search(args)


def _handle_search_code(args: Dict, ctx: ToolContext) -> str:
    # Prefer the configured RAG base on ctx when present; the tool's
    # own env-var fallback handles stand-alone dev/test paths.
    import os
    if ctx.rag_base:
        os.environ.setdefault("ULTRON_RAG_BASE", ctx.rag_base)
    from mother.tools.utility_tools import search_code
    return search_code(args)


def _handle_list_my_tools(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import list_my_tools
    return list_my_tools(args)


def _handle_forget_fact(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import forget_fact
    return forget_fact(args, current_user=ctx.current_user)


def _handle_correct_fact(args: Dict, ctx: ToolContext) -> str:
    from mother.tools.utility_tools import correct_fact
    return correct_fact(args, current_user=ctx.current_user)


_HANDLERS: Dict[str, Callable[[Dict, ToolContext], str]] = {
    # Existing tools
    "get_weather":       _handle_get_weather,
    "get_stock_price":   _handle_get_stock_price,
    "search_info":       _handle_search_info,
    "set_reminder":      _handle_set_reminder,
    "get_memory":        _handle_get_memory,
    "read_note":         _handle_read_note,
    "write_note":        _handle_write_note,
    # New batch
    "current_time":       _handle_current_time,
    "get_time_in":        _handle_get_time_in,
    "get_forecast":       _handle_get_forecast,
    "get_news_headlines": _handle_get_news_headlines,
    "calculate":          _handle_calculate,
    "convert_units":      _handle_convert_units,
    "brave_web_search":   _handle_brave_web_search,
    "search_code":        _handle_search_code,
    "list_my_tools":      _handle_list_my_tools,
    "forget_fact":        _handle_forget_fact,
    "correct_fact":       _handle_correct_fact,
}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_tool_call(
    tool_name: str,
    tool_args: Dict[str, Any],
    ctx: ToolContext,
) -> str:
    """Execute the named tool and return a plain-text result string.

    Returns an error string (never raises) so the caller can always hand
    the result back to the LLM as a tool-result message.
    """
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return f"Unknown tool: {tool_name}"
    try:
        return handler(tool_args, ctx)
    except Exception as exc:
        return f"Tool '{tool_name}' failed: {exc}"


def extract_tool_call(chunk_or_message) -> Optional[tuple[str, Dict]]:
    """Try to extract a tool_call from an Ollama response chunk/message dict.

    Ollama returns tool calls as:
        {"message": {"role": "assistant", "tool_calls": [{"function": {"name": ..., "arguments": ...}}]}}

    Returns (tool_name, args_dict) or None if no tool call present.
    """
    if not isinstance(chunk_or_message, dict):
        return None
    msg = chunk_or_message.get("message", {})
    tool_calls = msg.get("tool_calls", [])
    if not tool_calls:
        return None
    first = tool_calls[0]
    fn = first.get("function", {})
    name = fn.get("name", "")
    raw_args = fn.get("arguments", {})
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except Exception:
            raw_args = {}
    return (name, raw_args) if name else None
