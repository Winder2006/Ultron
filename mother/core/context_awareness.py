"""Context awareness for MOTHER - time, urgency, and situational responses."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import random


@dataclass
class TimeContext:
    """Time-based context awareness."""
    
    @staticmethod
    def get_period() -> str:
        """Get current time period."""
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    @staticmethod
    def is_late_night() -> bool:
        """Check if it's late night (after 11 PM or before 5 AM)."""
        hour = datetime.datetime.now().hour
        return hour >= 23 or hour < 5
    
    @staticmethod
    def is_early_morning() -> bool:
        """Check if it's very early (5-7 AM)."""
        hour = datetime.datetime.now().hour
        return 5 <= hour < 7
    
    @staticmethod
    def is_weekend() -> bool:
        """Check if it's a weekend."""
        return datetime.datetime.now().weekday() >= 5
    
    @staticmethod
    def get_greeting() -> str:
        """Get appropriate greeting based on time."""
        period = TimeContext.get_period()
        hour = datetime.datetime.now().hour
        
        greetings = {
            "morning": [
                "Good morning.",
                "Morning. Systems online.",
                "Good morning. All systems operational.",
            ],
            "afternoon": [
                "Good afternoon.",
                "Afternoon. How can I assist?",
                "Good afternoon. Standing by.",
            ],
            "evening": [
                "Good evening.",
                "Evening. Systems ready.",
                "Good evening. What do you need?",
            ],
            "night": [
                "Working late, I see.",
                "Late night session. I'm here.",
                "Burning the midnight oil. How can I help?",
            ],
        }
        
        # Special cases
        if TimeContext.is_late_night():
            late_options = [
                "It's quite late. Everything alright?",
                "Late night session detected. I'm standing by.",
                "The hour is late, but I remain vigilant.",
            ]
            return random.choice(late_options)
        
        if TimeContext.is_early_morning():
            early_options = [
                "You're up early. Coffee recommended.",
                "Early start detected. Systems ready.",
                "Dawn operations. All systems online.",
            ]
            return random.choice(early_options)
        
        return random.choice(greetings.get(period, ["Hello."]))
    
    @staticmethod
    def get_farewell() -> str:
        """Get appropriate farewell based on time."""
        period = TimeContext.get_period()
        
        if TimeContext.is_late_night():
            return random.choice([
                "Get some rest.",
                "Don't stay up too late.",
                "Entering standby. You should do the same.",
            ])
        
        farewells = {
            "morning": "Have a productive day.",
            "afternoon": "Proceeding to standby.",
            "evening": "Have a good evening.",
            "night": "Goodnight.",
        }
        return farewells.get(period, "Standby mode engaged.")
    
    @staticmethod
    def get_time_aware_prompt_addition() -> str:
        """Get additional system prompt context based on time."""
        hour = datetime.datetime.now().hour
        day = datetime.datetime.now().strftime("%A")
        
        context_parts = []
        
        if TimeContext.is_late_night():
            context_parts.append("It's late at night - be concise and consider the user may be tired.")
        elif TimeContext.is_early_morning():
            context_parts.append("It's early morning - the user is starting their day.")
        
        if TimeContext.is_weekend():
            context_parts.append(f"It's {day}, a weekend day.")
        
        return " ".join(context_parts)


@dataclass
class UrgencyDetector:
    """Detect urgency from audio/text features."""
    
    # Thresholds
    fast_speech_threshold: float = 0.15  # seconds per word (faster = more urgent)
    loud_threshold: float = 0.7  # RMS energy normalized
    
    # Urgent keywords
    urgent_keywords: List[str] = field(default_factory=lambda: [
        "urgent", "emergency", "hurry", "quick", "fast", "now", "immediately",
        "asap", "right now", "help", "critical", "important", "quickly",
        "stop", "wait", "hold on", "danger", "alert", "warning"
    ])
    
    def analyze_text_urgency(self, text: str) -> Tuple[bool, float]:
        """Analyze text for urgency indicators.
        
        Returns (is_urgent, urgency_score 0-1).
        """
        if not text:
            return False, 0.0
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check for urgent keywords
        keyword_matches = sum(1 for kw in self.urgent_keywords if kw in text_lower)
        keyword_score = min(keyword_matches * 0.2, 0.6)
        
        # Check for exclamation marks
        exclaim_score = min(text.count("!") * 0.15, 0.3)
        
        # Check for ALL CAPS words (shouting)
        caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
        caps_score = min(caps_words * 0.1, 0.3)
        
        # Short, terse commands tend to be more urgent
        brevity_score = 0.2 if len(words) <= 3 else 0.0
        
        total_score = min(keyword_score + exclaim_score + caps_score + brevity_score, 1.0)
        is_urgent = total_score >= 0.3
        
        return is_urgent, total_score
    
    def analyze_audio_urgency(
        self,
        audio_samples: "np.ndarray",
        sample_rate: int,
        duration_seconds: float,
        word_count: int
    ) -> Tuple[bool, float, dict]:
        """Analyze audio features for urgency.
        
        Returns (is_urgent, urgency_score 0-1, features_dict).
        """
        import numpy as np
        
        features = {}
        scores = []
        
        # 1. Speech rate (words per second)
        if duration_seconds > 0 and word_count > 0:
            speech_rate = word_count / duration_seconds
            features["speech_rate_wps"] = speech_rate
            # Normal speech is ~2-3 words/sec, urgent is >4
            if speech_rate > 4.0:
                scores.append(0.4)
            elif speech_rate > 3.5:
                scores.append(0.2)
        
        # 2. Volume/energy analysis
        if len(audio_samples) > 0:
            rms = np.sqrt(np.mean(np.square(audio_samples)))
            features["rms_energy"] = float(rms)
            # High volume suggests urgency
            if rms > 0.3:
                scores.append(0.3)
            elif rms > 0.2:
                scores.append(0.15)
        
        # 3. Pitch variance (stress indicator)
        # Higher pitch variance often indicates emotional urgency
        # This is a simplified version - real implementation would use pitch detection
        if len(audio_samples) > sample_rate // 10:  # At least 100ms
            # Use zero-crossing rate as a rough pitch proxy
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_samples))))
            zcr = zero_crossings / len(audio_samples)
            features["zero_crossing_rate"] = float(zcr)
            if zcr > 0.15:  # High frequency content
                scores.append(0.2)
        
        # 4. Energy variance (dynamic speech = more urgent)
        if len(audio_samples) > sample_rate // 4:
            # Split into frames and check energy variance
            frame_size = sample_rate // 20  # 50ms frames
            frames = len(audio_samples) // frame_size
            if frames > 2:
                energies = []
                for i in range(frames):
                    frame = audio_samples[i * frame_size:(i + 1) * frame_size]
                    energies.append(np.sqrt(np.mean(np.square(frame))))
                energy_var = np.var(energies)
                features["energy_variance"] = float(energy_var)
                if energy_var > 0.01:
                    scores.append(0.2)
        
        total_score = min(sum(scores), 1.0)
        is_urgent = total_score >= 0.35
        
        return is_urgent, total_score, features


@dataclass  
class ContextState:
    """Maintains current context state."""

    interaction_count: int = 0
    last_urgency_score: float = 0.0
    last_topic: Optional[str] = None
    user_mood_estimate: str = "neutral"  # neutral, rushed, relaxed, frustrated

    def __init__(self):
        from datetime import datetime as _dt
        self._session_start = _dt.now()

    def get_session_duration_minutes(self) -> float:
        from datetime import datetime as _dt
        return (_dt.now() - self._session_start).total_seconds() / 60.0

    def update_interaction(self, urgency_score: float = 0.0):
        """Update after each interaction."""
        self.interaction_count += 1
        self.last_urgency_score = urgency_score
        
        # Estimate mood from recent urgency
        if urgency_score > 0.6:
            self.user_mood_estimate = "rushed"
        elif urgency_score > 0.4:
            self.user_mood_estimate = "focused"
        else:
            self.user_mood_estimate = "relaxed"
    
    def get_adaptive_response_style(self) -> str:
        """Get response style hints based on context."""
        hints = []
        
        # Time-based adjustments
        if TimeContext.is_late_night():
            hints.append("Keep responses brief - it's late.")
        
        # Urgency-based adjustments
        if self.last_urgency_score > 0.5:
            hints.append("User seems rushed - be concise and action-oriented.")
        
        # Session length adjustments
        duration = self.get_session_duration_minutes()
        if duration > 30:
            hints.append("Long session - user may appreciate efficiency.")
        
        # Interaction pattern
        if self.interaction_count > 10:
            hints.append("Multiple interactions - skip pleasantries.")
        
        return " ".join(hints) if hints else ""


# Global context instance
_context: Optional[ContextState] = None
_urgency_detector: Optional[UrgencyDetector] = None


def get_context() -> ContextState:
    """Get or create global context state."""
    global _context
    if _context is None:
        _context = ContextState()
    return _context


def get_urgency_detector() -> UrgencyDetector:
    """Get or create urgency detector."""
    global _urgency_detector
    if _urgency_detector is None:
        _urgency_detector = UrgencyDetector()
    return _urgency_detector


def reset_context():
    """Reset context for new session."""
    global _context
    _context = ContextState()


def build_context_aware_prompt(base_prompt: str, user_text: str = "", urgency_score: float = 0.0) -> str:
    """Build a context-aware system prompt."""
    context = get_context()
    additions = []
    
    # Time context
    time_context = TimeContext.get_time_aware_prompt_addition()
    if time_context:
        additions.append(time_context)
    
    # Urgency context
    if urgency_score > 0.5:
        additions.append("The user seems urgent - prioritize speed and directness.")
    elif urgency_score > 0.3:
        additions.append("User may be in a hurry - be efficient.")
    
    # Session context
    style_hint = context.get_adaptive_response_style()
    if style_hint:
        additions.append(style_hint)
    
    # Combine
    if additions:
        return base_prompt + "\n\nCurrent context: " + " ".join(additions)
    return base_prompt


def get_contextual_acknowledgment(urgency_score: float = 0.0) -> Optional[str]:
    """Get a quick contextual acknowledgment before processing.
    
    Returns None if no acknowledgment needed.
    """
    if urgency_score > 0.6:
        return random.choice([
            "On it.",
            "Right away.",
            "Understood.",
            "Processing.",
        ])
    elif urgency_score > 0.4:
        return random.choice([
            "One moment.",
            "Working on it.",
            "Checking now.",
        ])
    return None

