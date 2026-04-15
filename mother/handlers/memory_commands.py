"""Memory command handlers for MOTHER."""
from __future__ import annotations

import re
from typing import Tuple, Optional, Dict, List

from mother.core.logging_config import get_logger
from mother.memory.manager import extract_fact_from_statement

logger = get_logger("commands.memory")

# Remember patterns
REMEMBER_PATTERNS = [
    re.compile(r"^remember\s+(?:that\s+)?(.+)$", re.I),
    re.compile(r"^don'?t forget\s+(?:that\s+)?(.+)$", re.I),
    re.compile(r"^note\s+(?:that\s+)?(.+)$", re.I),
    re.compile(r"^save\s+(?:that\s+)?(.+)$", re.I),
]

# Query patterns
QUERY_PATTERNS = [
    re.compile(r"^(?:what is|what's) my (.+?)(?:\?|$)", re.I),
    re.compile(r"^what do you (?:know|remember) about my (.+?)(?:\?|$)", re.I),
    re.compile(r"^do you (?:know|remember) my (.+?)(?:\?|$)", re.I),
    re.compile(r"^tell me (?:about )?my (.+?)(?:\?|$)", re.I),
]


def extract_remember_content(text: str) -> Optional[str]:
    """Extract content from a remember command."""
    for pattern in REMEMBER_PATTERNS:
        match = pattern.match(text.strip())
        if match:
            return match.group(1).strip()
    return None


def extract_memory_query(text: str) -> Optional[str]:
    """Extract query topic from a memory query."""
    for pattern in QUERY_PATTERNS:
        match = pattern.match(text.strip())
        if match:
            return match.group(1).strip()
    return None


def is_remember_command(text: str) -> bool:
    """Check if text is a remember command."""
    return extract_remember_content(text) is not None


def is_memory_query(text: str) -> bool:
    """Check if text is a memory query."""
    return extract_memory_query(text) is not None


def handle_memory_query(
    user_input: str,
    memory_manager=None,
) -> Tuple[bool, Optional[str]]:
    """Handle memory query (e.g., 'What is my birthday?').
    
    Args:
        user_input: User's text input
        memory_manager: MemoryManager instance for current user
        
    Returns:
        (handled, response_text)
    """
    topic = extract_memory_query(user_input)
    if topic is None:
        return False, None
    
    logger.info(f"Memory query for: {topic}")
    
    if memory_manager is None:
        logger.warning("No memory manager available")
        return True, "I don't have access to your memories right now."
    
    # Try exact fact lookup
    fact = memory_manager.get_fact(topic)
    if fact:
        value = fact.get("value", "")
        logger.debug(f"Found fact: {topic} = {value}")
        return True, f"Your {topic} is {value}."
    
    # Try semantic search in episodic memory
    try:
        memories = memory_manager.search_episodic(topic, n=3)
        if memories:
            best = memories[0]
            text = best.get("text", "")
            if text:
                logger.debug("Found relevant episodic memory")
                return True, f"I remember: {text}"
    except Exception as e:
        logger.warning(f"Episodic memory search failed: {e}")
    
    return True, f"I don't have any information about your {topic}."


def handle_remember_command(
    user_input: str,
    memory_manager=None,
) -> Tuple[bool, Optional[str]]:
    """Handle remember command (e.g., 'Remember my birthday is March 15').
    
    Args:
        user_input: User's text input
        memory_manager: MemoryManager instance for current user
        
    Returns:
        (handled, response_text)
    """
    content = extract_remember_content(user_input)
    if content is None:
        return False, None
    
    logger.info(f"Remember command: {content}")
    
    if memory_manager is None:
        logger.warning("No memory manager available")
        return True, "I can't save memories right now. User not identified."
    
    # Process the statement through memory manager
    try:
        extracted = extract_fact_from_statement(f"my {content}")
        if extracted:
            key, value, category = extracted
            memory_manager.set_fact(key, value, category=category, source="explicit")
        else:
            memory_manager.add_episodic(content, tags=["explicit"], confidence=1.0, source="explicit_request")
        logger.debug("Memory saved successfully")
        return True, "I'll remember that."
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        return True, "I had trouble saving that. Please try again."


def format_all_facts(memory_manager) -> str:
    """Format all facts for display."""
    if memory_manager is None:
        return "No memory manager available."
    
    try:
        facts = memory_manager.get_all_facts()
        if not facts:
            return "I don't have any facts stored for you yet."
        
        lines = []
        for key, fact in facts.items():
            value = fact.get("value", "")
            category = fact.get("category", "general")
            lines.append(f"- {key}: {value} ({category})")
        
        return "Here's what I know:\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to retrieve facts: {e}")
        return "I had trouble accessing your memories."

