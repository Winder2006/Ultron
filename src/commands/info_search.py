"""Shim: re-exports from mother.handlers.info_search for backward compatibility."""
from mother.handlers.info_search import *  # noqa: F401,F403
from mother.handlers.info_search import (  # explicit for IDE
    handle_info_search, is_lore_query, shorten_summary,
    extract_info_query, normalize_mishearings, clean_topic, _fuzzy_has,
)
