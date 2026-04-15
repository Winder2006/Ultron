"""Info search command handlers for MOTHER."""
from __future__ import annotations

import re
from typing import Tuple, Optional, Dict, List
from difflib import SequenceMatcher
import httpx

from mother.core.logging_config import get_logger

logger = get_logger("commands.info_search")

# Info query patterns
INFO_PATTERNS = [
    re.compile(r"^\s*(who is|who are)\s+(.+)$", re.I),
    re.compile(r"^\s*(what is|what are)\s+(.+)$", re.I),
    re.compile(r"^\s*(tell me about)\s+(.+)$", re.I),
    re.compile(r"^\s*(define|explain)\s+(.+)$", re.I),
    re.compile(r"^\s*(where is|where are)\s+(.+)$", re.I),
]

# Heuristic: bare proper-noun phrases like "Christopher Nolan" → info search
_PROPER_NOUN_PHRASE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$")

# Common STT mishearings of lore terms
_COMMON_CORRECTIONS: dict[str, str] = {
    "xenoporph": "xenomorph",
    "xenoporf": "xenomorph",
    "xenomorpf": "xenomorph",
    "lspace": "space",
    "aliens": "alien",
    "emission": "mission",
    "emissions": "mission",
}


def normalize_mishearings(text: str) -> str:
    """Correct common STT mishearings before query processing."""
    t = text or ""
    for wrong, right in _COMMON_CORRECTIONS.items():
        t = re.sub(rf"\b{re.escape(wrong)}\b", right, t, flags=re.I)
    return t


def _fuzzy_has(term: str, text: str, threshold: float = 0.78) -> bool:
    """Return True if *term* is in *text* exactly or with high fuzzy similarity."""
    a = term.lower()
    b = (text or "").lower()
    return SequenceMatcher(None, a, b).ratio() >= threshold or a in b


def is_info_query(text: str) -> bool:
    """Check if text is an info search query."""
    for pattern in INFO_PATTERNS:
        if pattern.match(text):
            return True
    return False


def clean_topic(s: str) -> str:
    """Strip quotes and trailing punctuation from a query topic."""
    t = (s or "").strip().strip('"\'')
    t = re.sub(r"[\.!?\s]+$", "", t)
    return t


def extract_info_query(text: str) -> Optional[str]:
    """Extract the info topic from user input.

    Handles pattern-matched queries ("what is X"), fallback keywords
    ("tell me about", "look up"), and bare proper nouns ("Christopher Nolan").
    """
    text = (text or "").strip()
    for pat in INFO_PATTERNS:
        m = pat.match(text)
        if m:
            return clean_topic((m.group(2) or "").strip())
    low = text.lower()
    for kw in ("wikipedia", "lookup", "look up", "info on", "tell me about"):
        if kw in low:
            return clean_topic(text.replace(kw, "").strip())
    if _PROPER_NOUN_PHRASE.match(text):
        return clean_topic(text)
    return None


# Keep old name as alias for any code still using it
extract_query_topic = extract_info_query


def is_lore_query(text: str) -> bool:
    """Check if query is about MOTHER/Alien universe lore (with fuzzy matching)."""
    t = (text or "").lower()
    if any(k in t for k in [
        "lv-426", "lv 426", "weyland", "yutani", "directive 937", "special order",
        "sevastopol", "fiorina", "prometheus", "winder", "winder corp", "mother",
        "mission", "protocol", "protocols",
    ]):
        return True
    if any(k in t for k in ["xenomorph", "engineers", "pathogen", "black goo", "alien",
                              "nostromo", "ripley", "ash", "bishop", "david"]):
        return True
    # Fuzzy catch for common typos / mishearings
    if _fuzzy_has("xenomorph", t) or _fuzzy_has("weyland", t) or _fuzzy_has("yutani", t):
        return True
    if _fuzzy_has("protocol", t) or _fuzzy_has("mission", t):
        return True
    return False


def shorten_summary(s: str, max_chars: int = 320) -> str:
    """Truncate *s* to *max_chars*, preferring a sentence boundary."""
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    last = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:last + 1] if last > 40 else cut + "..."


def search_wikipedia(query: str) -> Optional[Dict]:
    """Search Wikipedia for info."""
    try:
        with httpx.Client(timeout=2.0) as client:
            # Search
            resp = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 1,
                }
            )
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            if not results:
                return None
            
            title = results[0].get("title")
            
            # Get summary
            resp = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "titles": title,
                    "format": "json",
                }
            )
            if resp.status_code != 200:
                return None
            
            pages = resp.json().get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id != "-1":
                    return {
                        "title": page.get("title"),
                        "summary": page.get("extract", ""),
                        "source": "wikipedia",
                    }
    except httpx.TimeoutException:
        logger.warning("Wikipedia API timeout")
    except Exception as e:
        logger.warning(f"Wikipedia search error: {e}")
    
    return None


def search_rag(
    query: str,
    rag_api_base: str = "http://127.0.0.1:8123",
    k: int = 3,
    timeout_ms: int = 400
) -> List[Dict]:
    """Search RAG for relevant notes."""
    try:
        with httpx.Client(timeout=timeout_ms / 1000.0) as client:
            resp = client.get(
                f"{rag_api_base}/search",
                params={"q": query, "k": k}
            )
            if resp.status_code == 200:
                return resp.json()
    except httpx.TimeoutException:
        logger.debug("RAG search timeout")
    except Exception as e:
        logger.debug(f"RAG search error: {e}")
    
    return []


def handle_info_search(
    user_input: str,
    rag_api_base: str = "http://127.0.0.1:8123",
    rag_k: int = 3,
    rag_timeout_ms: int = 400
) -> Tuple[bool, Optional[str]]:
    """Handle info/knowledge search query.
    
    Args:
        user_input: User's text input
        rag_api_base: RAG API base URL
        rag_k: Number of RAG results
        rag_timeout_ms: RAG timeout in ms
        
    Returns:
        (handled, response_text)
    """
    if not is_info_query(user_input):
        return False, None
    
    topic = extract_query_topic(user_input) or user_input
    logger.info(f"Info search for: {topic}")
    
    is_lore = is_lore_query(user_input)
    max_chars = 160 if is_lore else 320
    
    # Try RAG first for lore queries
    if is_lore:
        hits = search_rag(topic, rag_api_base, rag_k, rag_timeout_ms)
        if hits:
            best = hits[0]
            summary = best.get("text", "")
            if summary:
                logger.debug("Found answer in RAG")
                return True, shorten_summary(summary, max_chars)
    
    # Try Wikipedia
    wiki_result = search_wikipedia(topic)
    if wiki_result and wiki_result.get("summary"):
        logger.debug("Found answer on Wikipedia")
        return True, shorten_summary(wiki_result["summary"], max_chars)
    
    # Fallback to RAG for non-lore
    if not is_lore:
        hits = search_rag(topic, rag_api_base, rag_k, rag_timeout_ms)
        if hits:
            best = hits[0]
            summary = best.get("text", "")
            if summary:
                return True, shorten_summary(summary, max_chars)
    
    return True, f"I couldn't find information about {topic}."

