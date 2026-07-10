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

# Question pivots used by the compound-query detector.
_QUESTION_WORDS = ("who", "what", "what's", "where", "when", "why", "how", "which", "whose")

# Phrases that signal "this query depends on resolving someone or
# something else first" — e.g. "weather in their HQ" needs a CEO
# lookup BEFORE the weather lookup. Single-intent fast-paths can't
# do that resolution; they need the LLM + tool-chaining loop.
_REFERENTIAL_PHRASES = (
    "their hq", "their headquarters", "their office", "their location",
    "his hq", "her hq", "his office", "her office",
    "his location", "her location",
    "where they", "where he", "where she", "where it",
    "where they are", "where he is", "where she is",
    "in his city", "in her city", "in their city",
    "at their", "at his", "at her",
)

# Topic clusters used by the compound-query detector. If a query
# contains " and " and the two halves hit DIFFERENT topics, the
# query needs tool chaining — e.g. "price of TSLA and the weather in
# Austin" splits cleanly into a finance question and a weather
# question, neither of which the other's fast-path can serve.
_TOPIC_HINTS: dict[str, tuple[str, ...]] = {
    "weather":  ("weather", "forecast", "temperature", "rain",
                 "humidity", "hot", "cold", "snow", "sunny"),
    "finance":  ("price", "stock", "ticker", "quote", "shares",
                 "trading", "market cap"),
    "time":     ("time", "clock", "timezone", "what hour"),
    "news":     ("news", "headlines"),
    "person":   ("ceo", "president", "founder", "leader", "who is",
                 "who's", "current head"),
    "location": ("hq", "headquarters", "office", "city", "country",
                 "address"),
}


def _topic_of(fragment: str) -> Optional[str]:
    """Return the first matching topic for a fragment, or None.

    Used to split " X and Y " queries — if X and Y land on different
    topics, the query is compound.
    """
    for topic, hints in _TOPIC_HINTS.items():
        for h in hints:
            if h in fragment:
                return topic
    return None


def _is_compound_query(text_lower: str) -> bool:
    """True when the query asks for two things at once.

    Compound queries can't be served by a single-purpose fast-path
    (WEATHER, FINANCE_QUOTE, etc.) — they need the LLM + tools loop
    so it can resolve one piece, then chain to the next. Falsely
    flagging a query as compound just sends it to tier 2 with tools,
    which is slightly slower but still correct; falsely flagging a
    compound query as single-intent is the bad failure mode (only
    half the question gets answered).
    """
    # Two distinct question pivots = "who is X and what is Y"
    pivot_count = 0
    padded = f" {text_lower} "
    for w in _QUESTION_WORDS:
        if f" {w} " in padded:
            pivot_count += 1
            if pivot_count >= 2:
                return True

    # Referential phrase that requires resolving a subject first
    if any(p in text_lower for p in _REFERENTIAL_PHRASES):
        return True

    # "X and Y" with either:
    #   (a) question word on each side, OR
    #   (b) different topics on each side (catches "price of TSLA and
    #       the weather in Austin" where only one side has "what is")
    if " and " in text_lower:
        left, _, right = text_lower.partition(" and ")
        left_q = any(f" {w} " in f" {left} " for w in _QUESTION_WORDS)
        right_q = any(f" {w} " in f" {right} " for w in _QUESTION_WORDS)
        if left_q and right_q:
            return True
        left_topic = _topic_of(left)
        right_topic = _topic_of(right)
        if left_topic and right_topic and left_topic != right_topic:
            return True

    return False


def classify(text: str, llm_fn: Optional[Callable[[str], str]] = None) -> Intent:
    """Classify *text* into an Intent.

    Uses a fast keyword pass first. If the result is GENERAL and *llm_fn* is
    provided and the input looks genuinely ambiguous, runs a single cheap LLM
    classification call as a second opinion.

    The LLM is only invoked when the keyword pass returns GENERAL AND the
    utterance starts with an ambiguous phrase — so it never adds latency to
    clearly-routed queries.

    Compound queries ("who is X and what's the weather in their HQ")
    bypass single-intent fast-paths even when a fast-path keyword
    matches, because answering them correctly requires tool chaining.
    """
    text_lower = text.lower().strip()
    compound = _is_compound_query(text_lower)

    for intent, keywords, pattern in _KEYWORD_RULES:
        if keywords and any(kw in text_lower for kw in keywords):
            # Drop to GENERAL so the LLM + tool-chaining loop handles it.
            # IDENTITY_CLAIM is exempt — "I'm Charlie and I want the
            # weather" is still a valid identity claim worth recording.
            if compound and intent != Intent.IDENTITY_CLAIM:
                return Intent.GENERAL
            return intent
        if pattern and re.search(pattern, text, re.IGNORECASE):
            if compound and intent != Intent.IDENTITY_CLAIM:
                return Intent.GENERAL
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
