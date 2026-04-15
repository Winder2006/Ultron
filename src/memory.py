"""Shim: re-exports from mother.memory.manager for backward compatibility."""
from mother.memory.manager import *  # noqa: F401,F403
from mother.memory.manager import (  # explicit for IDE
    get_user_memory, get_current_user_memory, maybe_learn_from_statement,
    extract_fact_from_statement, extract_all_facts_from_statement,
    UserMemory,
)
