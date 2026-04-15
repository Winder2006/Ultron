"""Shim: re-exports from mother.identity.speaker for backward compatibility."""
from mother.identity.speaker import *  # noqa: F401,F403
from mother.identity.speaker import (  # explicit for IDE
    get_registry, get_session, get_current_user, set_current_user,
    identify_from_audio, format_user_greeting, get_user_context_for_prompt,
    reset_session, UserProfile, UserRegistry,
)
