from __future__ import annotations

from typing import Dict, List

try:
    from yahooquery import Ticker  # type: ignore
except Exception:  # pragma: no cover
    Ticker = None  # type: ignore


def get_quote(symbol: str) -> Dict:
    if not Ticker:
        return {"error": "yahooquery not installed"}
    t = Ticker(symbol)
    q = t.price.get(symbol) or {}
    # extract common fields
    fields = (
        "shortName",
        "longName",
        "symbol",
        "regularMarketPrice",
        "regularMarketChange",
        "regularMarketChangePercent",
        "regularMarketDayHigh",
        "regularMarketDayLow",
        "regularMarketPreviousClose",
        "currency",
        "marketState",
    )
    out = {k: q.get(k) for k in fields}
    return out


def get_news(symbol: str, count: int = 5) -> List[Dict]:
    if not Ticker:
        return []
    t = Ticker(symbol)
    try:
        raw = t.news() if callable(getattr(t, "news", None)) else t.news
    except Exception:
        raw = []
    # Normalize to list of dicts
    news_list: list = []
    if isinstance(raw, dict):
        # sometimes under key 'news' or similar
        cand = raw.get("news") or raw.get("items") or raw.get("result") or []
        if isinstance(cand, list):
            news_list = cand
    elif isinstance(raw, list):
        news_list = raw
    # Filter only dict items
    news_list = [n for n in news_list if isinstance(n, dict)]
    items = []
    for n in news_list[:count]:
        items.append({
            "title": n.get("title"),
            "publisher": n.get("publisher"),
            "link": n.get("link"),
            "providerPublishTime": n.get("providerPublishTime"),
        })
    return items


