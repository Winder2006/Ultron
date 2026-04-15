"""Intent classification for MOTHER voice commands.

Classifies user utterances into a fixed set of intents using a fast
keyword-based first pass, falling back to an LLM call for ambiguous inputs.

Intent enum values:
    FINANCE_QUOTE   - stock/crypto price query
    FINANCE_NEWS    - market news/headlines
    WEATHER         - weather / temperature / forecast
    INFO_SEARCH     - Wikipedia/DDG lookup ("what is", "who is", "tell me about")
    MEMORY_EXPLICIT - "remember X", "what do you know about me"
    IDENTITY_CLAIM  - "I'm Charlie", "this is Oliver"
    IDENTITY_QUERY  - "who am I", "do you know who I am"
    REMINDER_SET    - "remind me at X to Y"
    REMINDER_LIST   - "what are my reminders"
    GENERAL         - everything else → goes to LLM
"""
from __future__ import annotations

import re
from enum import Enum, auto
from typing import Optional, Callable


class Intent(Enum):
    FINANCE_QUOTE = auto()
    FINANCE_NEWS = auto()
    WEATHER = auto()
    INFO_SEARCH = auto()
    MEMORY_EXPLICIT = auto()
    IDENTITY_CLAIM = auto()
    IDENTITY_QUERY = auto()
    REMINDER_SET = auto()
    REMINDER_LIST = auto()
    GENERAL = auto()


# ---------------------------------------------------------------------------
# Fast keyword rules — evaluated in priority order.
# Each rule is (intent, list-of-required-any-keywords, optional-regex).
# The first matching rule wins.
# ---------------------------------------------------------------------------
_KEYWORD_RULES: list[tuple[Intent, list[str], Optional[str]]] = [
    # Identity queries take priority over identity claims
    (Intent.IDENTITY_QUERY, ["who am i", "do you know who i am",
                              "can you identify me", "who is speaking",
                              "what's my name", "what is my name"], None),

    # Identity claims
    (Intent.IDENTITY_CLAIM, [], r"(?:this is|i'?m|i am|my name is)\s+[A-Z][a-z]+"),

    # Reminders
    (Intent.REMINDER_SET, ["remind me", "set a reminder", "set reminder",
                            "add reminder"], None),
    (Intent.REMINDER_LIST, ["what are my reminders", "show reminders",
                             "list reminders", "do i have any reminders"], None),

    # Memory explicit
    (Intent.MEMORY_EXPLICIT, ["remember ", "what do you know about me",
                               "what do you remember", "what have you learned",
                               "forget that", "forget my"], None),

    # Finance news before finance quote so "finance news" doesn't match quote
    (Intent.FINANCE_NEWS, ["finance news", "market news", "market headlines",
                            "stock news", "stock headlines"], None),

    # Finance quote — "price of", "how much is", trading at, quote for
    (Intent.FINANCE_QUOTE, ["price of", "price for", "how much is",
                             "stock price", "trading at", "quote for",
                             "what is the price", "what's the price"], None),
    # Bare ticker-ish: "what is TSLA" / "TSLA price"
    (Intent.FINANCE_QUOTE, [], r"\b([A-Z]{2,5}|BTC|ETH|crypto)\b.{0,20}(?:price|worth|value|quote)"),

    # Weather
    (Intent.WEATHER, ["weather", "temperature outside", "forecast",
                       "how hot", "how cold", "will it rain",
                       "chance of rain", "humidity"], None),

    # Info search
    (Intent.INFO_SEARCH, ["what is ", "what are ", "who is ", "who was ",
                           "tell me about", "explain ", "describe ",
                           "when was ", "where is ", "how does ",
                           "what happened", "give me info"], None),
]

# Compiled regex for identity claim (uses original case input)
_IDENTITY_CLAIM_RE = re.compile(r"(?:this is|i'?m|i am|my name is)\s+([A-Z][a-z]+)", re.IGNORECASE)

# Ambiguous phrases that warrant an LLM classification call
_AMBIGUOUS_TRIGGERS = {"can you", "could you", "would you", "is it", "are you",
                        "do you", "should i", "what should"}


def classify(text: str, llm_fn: Optional[Callable[[str], str]] = None) -> Intent:
    """Classify *text* into an Intent.

    Uses a fast keyword pass first. If the result is GENERAL and *llm_fn* is
    provided and the input looks genuinely ambiguous, runs a single cheap LLM
    classification call as a second opinion.

    The LLM is only invoked when the keyword pass returns GENERAL AND the
    utterance starts with an ambiguous phrase — so it never adds latency to
    clearly-routed queries.
    """
    text_lower = text.lower().strip()

    for intent, keywords, pattern in _KEYWORD_RULES:
        if keywords and any(kw in text_lower for kw in keywords):
            return intent
        if pattern and re.search(pattern, text, re.IGNORECASE):
            return intent

    # LLM fallback for genuinely ambiguous inputs
    if llm_fn is not None and any(text_lower.startswith(a) for a in _AMBIGUOUS_TRIGGERS):
        return _llm_classify(text, llm_fn)

    return Intent.GENERAL


def _llm_classify(text: str, llm_fn: Callable[[str], str]) -> Intent:
    """Ask the LLM to classify the utterance. Returns GENERAL on any failure."""
    _LABELS = {
        "finance_quote": Intent.FINANCE_QUOTE,
        "finance_news": Intent.FINANCE_NEWS,
        "weather": Intent.WEATHER,
        "info_search": Intent.INFO_SEARCH,
        "memory": Intent.MEMORY_EXPLICIT,
        "identity_claim": Intent.IDENTITY_CLAIM,
        "identity_query": Intent.IDENTITY_QUERY,
        "reminder_set": Intent.REMINDER_SET,
        "general": Intent.GENERAL,
    }
    prompt = (
        "Classify the following voice command into exactly one label.\n"
        "Labels: finance_quote, finance_news, weather, info_search, "
        "memory, identity_claim, identity_query, reminder_set, general\n"
        "Reply with only the label, nothing else.\n\n"
        f"Command: {text}\nLabel:"
    )
    try:
        raw = llm_fn(prompt).strip().lower().split()[0]
        return _LABELS.get(raw, Intent.GENERAL)
    except Exception:
        return Intent.GENERAL
