"""Shim: re-exports from mother.core.context_awareness for backward compatibility."""
from mother.core.context_awareness import *  # noqa: F401,F403
from mother.core.context_awareness import (  # explicit for IDE
    TimeContext, ContextState, UrgencyDetector,
    get_context, get_urgency_detector,
    build_context_aware_prompt, get_contextual_acknowledgment, reset_context,
)
