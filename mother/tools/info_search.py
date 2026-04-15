from __future__ import annotations

import requests
from urllib.parse import quote


_UA = {"User-Agent": "MotherAssistant/1.0 (https://localhost)"}


def _from_wikipedia(query: str) -> dict | None:
    # REST summary with redirect
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query.replace(' ', '_'))}?redirect=true"
    try:
        r = requests.get(url, timeout=6, headers=_UA)
    except Exception:
        return None
    if r.status_code == 200:
        try:
            j = r.json()
        except Exception:
            return None
        if j.get("extract"):
            return {
                "source": "wikipedia",
                "title": j.get("title"),
                "summary": j.get("extract"),
                "image": (j.get("thumbnail") or {}).get("source"),
                "url": ((j.get("content_urls") or {}).get("desktop") or {}).get("page"),
            }
    # Fallback to action API extracts
    q = quote(query)
    url2 = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exsentences=3&"
        f"explaintext=1&redirects=1&format=json&titles={q}"
    )
    try:
        r2 = requests.get(url2, timeout=6, headers=_UA)
        if r2.status_code == 200:
            j2 = r2.json()
            pages = (j2.get("query") or {}).get("pages") or {}
            for _, p in pages.items():
                extract = p.get("extract")
                title = p.get("title")
                if extract:
                    return {
                        "source": "wikipedia",
                        "title": title,
                        "summary": extract,
                        "image": None,
                        "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                    }
    except Exception:
        pass
    return None


def _from_duckduckgo(query: str) -> dict | None:
    url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
    try:
        r = requests.get(url, timeout=6, headers=_UA)
    except Exception:
        return None
    if r.status_code == 200:
        try:
            j = r.json()
        except Exception:
            return None
        if j.get("AbstractText"):
            return {
                "source": "duckduckgo",
                "title": j.get("Heading"),
                "summary": j.get("AbstractText"),
                "image": j.get("Image"),
                "url": j.get("AbstractURL"),
            }
        # Fallback: use first RelatedTopics item with Text
        def _first_topic(items):
            if not isinstance(items, list):
                return None
            for it in items:
                if isinstance(it, dict):
                    if it.get("Text"):
                        return it
                    # Sometimes nested under 'Topics'
                    sub = it.get("Topics")
                    if sub:
                        t = _first_topic(sub)
                        if t:
                            return t
            return None
        t = _first_topic(j.get("RelatedTopics") or [])
        if t and t.get("Text"):
            title = j.get("Heading") or (t.get("Text", "").split(" - ")[0])
            return {
                "source": "duckduckgo",
                "title": title,
                "summary": t.get("Text"),
                "image": (t.get("Icon") or {}).get("URL"),
                "url": t.get("FirstURL"),
            }
    return None


def get_info(query: str) -> dict:
    return _from_wikipedia(query) or _from_duckduckgo(query) or {"error": "No info found"}


