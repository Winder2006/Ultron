"""Tiered request routing for MOTHER.

Single decision point for every user query.
Determines: fast-path handler OR which LLM tier to use.

Decision tree (in order):
1. Is intent a fast-path handler (WEATHER, FINANCE, REMINDER, IDENTITY, MEMORY)?
   → Return "fast_path" — no LLM call
2. Is intent INFO_SEARCH or GENERAL?
   → Classify query complexity → route to tier1/tier2/tier3
"""
from __future__ import annotations

from typing import Optional

from mother.core.intent import Intent
from mother.llm.classifier import classify_complexity, Tier

# Intents that are handled entirely by fast-path handlers — no LLM needed
FAST_PATH_INTENTS = {
    Intent.WEATHER,
    Intent.FINANCE_QUOTE,
    Intent.FINANCE_NEWS,
    Intent.REMINDER_SET,
    Intent.REMINDER_LIST,
    Intent.IDENTITY_CLAIM,
    Intent.IDENTITY_QUERY,
    Intent.MEMORY_EXPLICIT,
}


class RequestRouter:
    """Routes queries to fast-path handlers or LLM tiers."""

    def route(
        self,
        query: str,
        intent: Intent,
        has_vision_context: bool = False,
    ) -> tuple[str, Optional[Tier]]:
        """Determine routing for a query.

        Returns:
            ("fast_path", None) — handled by fast-path, no LLM
            ("llm", "tier1"|"tier2"|"tier3") — route to LLM tier
        """
        # Fast-path intents bypass LLM entirely
        if intent in FAST_PATH_INTENTS:
            return ("fast_path", None)

        # Everything else goes to LLM — classify complexity
        tier = classify_complexity(query, intent.name)

        # Vision context bumps tier1 → tier2 (needs more reasoning to incorporate)
        if has_vision_context and tier == "tier1":
            tier = "tier2"

        return ("llm", tier)
