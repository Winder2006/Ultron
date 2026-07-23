"""Lightweight query complexity classifier for LLM tier routing.

Pure heuristics — no LLM call, <5ms execution time.

Tiers (as configured in configs/app.yaml → llm.routing):
    tier1 (Cerebras Llama 3.1 8B, ~190ms TTFT):
        - Very short queries (<=3 words), greetings, acknowledgements
        - Simple arithmetic ("what is 2+2")
        - Fast-path commands that slipped past the intent classifier
        - Anything matching TIER1_PATTERNS
    tier2 (Claude Haiku 4.5, ~530ms TTFT):
        - Default for general conversation
        - Tool-calling queries (most of them)
        - INFO_SEARCH queries
    tier3 (Claude Sonnet 4, ~1200ms TTFT):
        - Long queries (>50 words)
        - Explicit reasoning triggers ("explain", "analyze")
        - Code-related queries

Routing precedence (enforced in classify_complexity, in this order):
    1. TIER1_PATTERNS — exact-pattern match wins immediately. Stops a
       query like "explain yes" from escalating to tier3 just because
       it contains the word "explain".
    2. Tier3 triggers — only checked AFTER tier1 patterns failed.
    3. Default by intent — INFO_SEARCH → tier2, else tier2.
"""
from __future__ import annotations

import re
from typing import Literal

Tier = Literal["tier1", "tier2", "tier3"]

# Words/phrases that signal complex reasoning → tier3.
# Removed from earlier list:
#   "convert" — now tier 2 (Haiku + convert_units tool is the right path).
#   "calculate" — kept, because "calculate the area of a circle" is real
#                 math that benefits from Sonnet. Simple arithmetic is
#                 caught earlier by the arithmetic tier1 pattern.
TIER3_TRIGGERS = [
    "explain", "analyze", "analyse", "write a", "write me",
    "help me with", "how do i", "what is the difference",
    "compare", "summarize", "summarise", "research",
    "code", "script", "debug", "program", "function",
    "step by step", "in detail", "pros and cons",
    "what are the implications", "break down",
    "create a plan", "design", "architect",
    "translate", "calculate",
    # Opinion/reasoning shapes that deserve the strong model when the
    # question is substantial (the >=8 word gate below still applies,
    # so "why not" or a short "what do you think" stays on Haiku).
    "what do you think", "your opinion", "your take",
    "why is", "why does", "why do", "why did", "why would",
    "how should", "what would happen",
]

# Patterns that signal simple/short queries → tier1.
# Checked BEFORE TIER3_TRIGGERS so short arithmetic or greetings can't
# be escalated to Sonnet just because they happen to contain a
# tier3-trigger substring.
TIER1_PATTERNS = [
    re.compile(r"^(what|what's) (time|date|day) ", re.I),
    re.compile(r"^(turn|switch|set|toggle) ", re.I),
    re.compile(r"^(play|pause|stop|skip|next|previous) ", re.I),
    re.compile(r"^(yes|no|ok|okay|sure|thanks|thank you|goodbye|bye|good night)\.?$", re.I),
    re.compile(r"^(hello|hi|hey|good morning|good evening|good afternoon)\.?$", re.I),
    # Simple arithmetic — "what is 2+2" or "3 times 4". Uses trivial
    # mental math, doesn't need Haiku, shouldn't trigger calculate
    # tool. tier 1 Cerebras answers these correctly and in ~200ms.
    re.compile(
        r"^(what(?:'s| is)?\s+)?\d+\s*(?:\+|-|\*|x|times|plus|minus|divided by|/)"
        r"\s*\d+\s*\??$",
        re.I,
    ),
]

# Words that signal the query needs real computation (execute_python /
# calculate). Tier1 (Cerebras Llama 8B) gets NO tools — so a 2-3 word
# query like "factorial of twenty" or "hash hello" that falls into the
# short-query tier1 shortcut would be answered from the model's head
# instead of running code. Any of these keywords forces at least tier2
# regardless of word count.
COMPUTE_HINTS = re.compile(
    r"\b(hash|sha-?\d*|md5|factorial|fibonacci|prime|primes|plot|graph|"
    r"histogram|median|mean|average|std|variance|sort|shuffle|random|"
    r"compute|simulate|csv|dataframe|regex|encode|decode|base64|"
    r"uuid|permutation|combination|digits)\b",
    re.I,
)

# Speculative questions about volatile topics ("what's gonna happen
# with SpaceX stock?", "who will win the election?") need BOTH signals:
# a predictive/opinion verb AND a topic that actually moves. Requiring
# the pair keeps persona questions ("what do you think of humanity?")
# and command phrasing ("will you remind me...") out of this bucket.
# Matches route to tier3 (a committed prediction is real reasoning) and
# the voice route forces a web search first so the take is grounded in
# this week's facts, not training data.
# "will(?!\s+you)" — "will you remind me / will you check" is a
# request TO the assistant, not a prediction. "should i" alone matched
# "should I bring a jacket to the game" (with 'game' as the volatile
# topic), forcing a web research call on a wardrobe question — so it
# now requires a trade-flavored verb.
PREDICTIVE_HINTS = re.compile(
    r"\b(gonna|going to|will(?!\s+you\b)|predict|prediction|forecast|"
    r"expect|should i (?:buy|sell|invest|short|bet)|"
    r"what do you think|your (?:take|opinion|read|call)|"
    r"odds|chances|likely|bet)\b",
    re.I,
)
VOLATILE_TOPIC_HINTS = re.compile(
    r"\b(stocks?|shares?|share price|markets?|trading|invest(?:ing|ment)?|"
    r"crypto|bitcoin|btc|ethereum|price|valuation|ipo|merger|acquisition|"
    r"earnings|fed|interest rates?|inflation|recession|economy|"
    r"election|polls?|game|match|series|season|playoffs?|championship)\b",
    re.I,
)


def is_speculative(query: str) -> bool:
    """True when the query asks for a prediction/opinion on a volatile topic."""
    return bool(
        PREDICTIVE_HINTS.search(query) and VOLATILE_TOPIC_HINTS.search(query)
    )


# User assertions about recent events ("SpaceX just had their IPO",
# "did you hear X died?"). The model's training snapshot predates the
# conversation, so contradicting these from memory produces confident
# nonsense — the observed case: flatly telling the user a completed
# IPO was "fabricated". A match forces a verification web search
# before the model may agree or disagree.
#
# TWO signals required, like is_speculative: a claim marker alone
# ("i saw...", "did you know...", "apparently...") fires on everyday
# small talk ("I saw a great movie last night"), which forced a
# pointless search AND armed the correct_fact instruction to write
# junk into permanent memory. The claim marker must co-occur with a
# newsworthy-event topic (or a volatile topic from the list above).
NEWS_CLAIM_HINTS = re.compile(
    r"\b(just (?:had|happened|announced|launched|released|dropped|"
    r"died|won|lost|ipo'?d|went public|got|became)|"
    r"did you (?:hear|see|know)|i (?:heard|read|saw)|apparently|"
    r"breaking|(?:is|are|went) (?:now|officially) |went public|"
    r"had (?:their|its|his|her) (?:ipo|launch|release|election)|"
    r"(?:is|are) (?:a )?public(?:ly traded)? compan)",
    re.I,
)
NEWS_EVENT_TOPICS = re.compile(
    r"\b(ipo|public(?:ly traded)?|launch(?:ed)?|release[ds]?|"
    r"announc(?:ed|ement)|died|death|dead|passed away|won|lost|"
    r"elect(?:ed|ion)|resign(?:ed)?|acquir(?:ed|es|ing)|merg(?:ed|er)|"
    r"bankrupt(?:cy)?|crash(?:ed)?|record (?:high|low)|all.time high|"
    r"war|ceasefire|indicted?|arrested|hacked|breach(?:ed)?|"
    r"ceo|president|prime minister)\b",
    re.I,
)


def is_news_claim(query: str) -> bool:
    """True when the user asserts/reports a possibly-recent public event."""
    if not NEWS_CLAIM_HINTS.search(query):
        return False
    return bool(
        NEWS_EVENT_TOPICS.search(query) or VOLATILE_TOPIC_HINTS.search(query)
    )


def classify_complexity(query: str, intent: str = "GENERAL") -> Tier:
    """Classify query complexity into a routing tier.

    Args:
        query: The user's text query.
        intent: The intent class from intent.py (e.g., "GENERAL", "INFO_SEARCH").

    Returns:
        "tier1", "tier2", or "tier3"
    """
    q = query.strip()
    q_lower = q.lower()
    word_count = len(q.split())

    # --- Tier 1 FIRST ---
    # Short greetings and explicit tier1 patterns short-circuit
    # before any tier3 trigger can grab them. Fixes the case where
    # a short or trivial query accidentally contains a tier3 keyword.
    # Compute-flavored queries are exempt from the shortcut: tier1
    # has no tools, so routing "factorial of twenty" there means the
    # answer gets recited instead of executed.
    #
    # NOTE: tier1 is PATTERN-matched only. There used to be a generic
    # "any query of <=3 words" rule here, which sent real questions
    # ("who's Kant?", "define entropy") to the 8B model for mediocre
    # answers. Greetings, acks, media commands, and trivial arithmetic
    # are what tier1 is for; everything else earns Haiku's ~300ms.
    needs_compute = bool(COMPUTE_HINTS.search(q_lower))
    if word_count <= 6 and not needs_compute:
        for pat in TIER1_PATTERNS:
            if pat.search(q):
                return "tier1"

    # --- Tier 3: genuine reasoning work ---

    # Predictions/opinions on volatile topics: a committed, grounded
    # take is exactly what the strongest model is for.
    if is_speculative(q_lower):
        return "tier3"

    # Long queries almost always need real reasoning
    if word_count > 50:
        return "tier3"

    # Explicit reasoning/code triggers — match on word boundaries so
    # ambiguous substrings ("function" appearing inside "functions"
    # in a plot prompt) don't escalate. We also require the query to
    # be at least 8 words for trigger matching: short queries like
    # "translate hello" don't need Sonnet, Haiku handles them fine.
    if word_count >= 8:
        # \b matches anchor on word boundaries. We compile a single
        # alternation pattern for speed instead of looping.
        pattern_parts = []
        for trig in TIER3_TRIGGERS:
            # Multi-word triggers ("write a", "step by step") still match
            # via plain substring — the boundary issue only bites single
            # words that are also substrings of common words.
            if " " in trig:
                pattern_parts.append(re.escape(trig))
            else:
                pattern_parts.append(r"\b" + re.escape(trig) + r"\b")
        if re.search("|".join(pattern_parts), q_lower):
            return "tier3"

    # Multiple sentence-questions stacked → probably compound reasoning
    sentences = [s.strip() for s in re.split(r"[.!?]+", q) if s.strip()]
    question_count = sum(
        1
        for s in sentences
        if "?" in s or s.lower().startswith(("what", "why", "how", "when", "where", "who"))
    )
    if question_count >= 2:
        return "tier3"

    # Code indicators — these specifically include syntax punctuation
    # so they only match when the user is literally pasting code.
    if any(kw in q_lower for kw in ("```", "def ", "function(", "class ", "import ", "var ", "const ")):
        return "tier3"

    # --- Tier 2: default ---
    # INFO_SEARCH and everything else falls through to Haiku.
    return "tier2"
