"""Multi-user identity management with voice recognition.

Handles:
- Voice enrollment (learning a user's voice)
- Speaker identification (who is speaking?)
- User profile management
- Context switching between users
"""
from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

from mother.core.logging_config import get_logger

logger = get_logger("user_identity")

# Voice encoder (lazy loaded)
_encoder = None


def _get_encoder():
    """Lazy load the voice encoder (heavy import)."""
    global _encoder
    if _encoder is None:
        try:
            from resemblyzer import VoiceEncoder
            _encoder = VoiceEncoder()
            logger.debug("Voice encoder loaded successfully")
        except ImportError:
            logger.warning("resemblyzer not installed - voice ID disabled")
            return None
    return _encoder


def _preprocess_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Preprocess audio for voice encoding."""
    from resemblyzer import preprocess_wav
    # Resemblyzer expects float32 in [-1, 1] range
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.max() > 1.0 or audio.min() < -1.0:
        peak = max(abs(audio.max()), abs(audio.min()))
        if peak > 0:
            audio = audio / peak
    return preprocess_wav(audio, source_sr=sample_rate)


# Storage paths
MEMORY_BASE = Path("assistant/memory")
USERS_DIR = MEMORY_BASE / "users"
SHARED_DIR = MEMORY_BASE / "shared"
SYSTEM_DIR = MEMORY_BASE / "system"
REGISTRY_FILE = SYSTEM_DIR / "user_registry.json"


@dataclass
class UserProfile:
    """A user's profile and preferences."""
    user_id: str  # Unique identifier (lowercase, no spaces)
    display_name: str  # How MOTHER addresses them
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Voice enrollment
    voice_enrolled: bool = False
    voice_samples_count: int = 0
    
    # Preferences
    preferences: Dict = field(default_factory=lambda: {
        "response_style": "concise",  # concise, detailed, balanced
        "greeting_enabled": True,
        "remember_conversations": True,
    })
    
    # Quick facts (frequently accessed)
    facts: Dict = field(default_factory=dict)
    
    def get_user_dir(self) -> Path:
        """Get this user's storage directory."""
        return USERS_DIR / self.user_id
    
    def get_voice_embedding_path(self) -> Path:
        """Path to stored voice embedding."""
        return self.get_user_dir() / "voice_embedding.npy"
    
    def get_facts_db_path(self) -> Path:
        """Path to user's facts database."""
        return self.get_user_dir() / "facts.json"
    
    def get_learned_dir(self) -> Path:
        """Directory for learned knowledge about this user."""
        return self.get_user_dir() / "learned"
    
    def save(self):
        """Save profile to disk."""
        user_dir = self.get_user_dir()
        user_dir.mkdir(parents=True, exist_ok=True)
        
        profile_path = user_dir / "profile.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, user_id: str) -> Optional["UserProfile"]:
        """Load profile from disk."""
        profile_path = USERS_DIR / user_id / "profile.json"
        if not profile_path.exists():
            return None
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            return cls(**data)
        except TypeError:
            # Profile file has unknown or missing fields (e.g. written by older version).
            # Build from known fields only, leaving the rest at defaults.
            import dataclasses
            known = {f.name for f in dataclasses.fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in known})


class UserRegistry:
    """Manages all enrolled users."""
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        self._voice_embeddings: Dict[str, np.ndarray] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load all enrolled users."""
        USERS_DIR.mkdir(parents=True, exist_ok=True)
        SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load each user profile
        if USERS_DIR.exists():
            for user_dir in USERS_DIR.iterdir():
                if user_dir.is_dir() and not user_dir.name.startswith("_"):
                    profile = UserProfile.load(user_dir.name)
                    if profile:
                        self.users[profile.user_id] = profile
                        # Load voice embedding if available
                        emb_path = profile.get_voice_embedding_path()
                        if emb_path.exists():
                            self._voice_embeddings[profile.user_id] = np.load(emb_path)
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get a user by ID."""
        return self.users.get(user_id)
    
    def list_users(self) -> List[str]:
        """List all enrolled user IDs."""
        return list(self.users.keys())
    
    def create_user(self, display_name: str, user_id: Optional[str] = None) -> UserProfile:
        """Create a new user profile."""
        if user_id is None:
            # Generate ID from name
            user_id = display_name.lower().replace(" ", "_")
            # Ensure unique
            base_id = user_id
            counter = 1
            while user_id in self.users:
                user_id = f"{base_id}_{counter}"
                counter += 1
        
        profile = UserProfile(user_id=user_id, display_name=display_name)
        profile.save()
        
        # Create user directories
        profile.get_learned_dir().mkdir(parents=True, exist_ok=True)
        
        self.users[user_id] = profile
        return profile
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user and their data."""
        if user_id not in self.users:
            return False
        
        import shutil
        user_dir = USERS_DIR / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)
        
        del self.users[user_id]
        if user_id in self._voice_embeddings:
            del self._voice_embeddings[user_id]
        
        return True
    
    def enroll_voice(self, user_id: str, audio_samples: List[np.ndarray], sample_rate: int = 16000) -> bool:
        """Enroll a user's voice from audio samples.
        
        Args:
            user_id: The user to enroll
            audio_samples: List of audio arrays (should be 10-30 seconds total)
            sample_rate: Sample rate of audio
            
        Returns:
            True if enrollment succeeded
        """
        encoder = _get_encoder()
        if encoder is None:
            return False
        
        profile = self.users.get(user_id)
        if profile is None:
            return False
        
        try:
            # Combine and preprocess audio
            combined = np.concatenate(audio_samples)
            processed = _preprocess_audio(combined, sample_rate)
            
            # Generate embedding
            embedding = encoder.embed_utterance(processed)
            
            # Save embedding
            emb_path = profile.get_voice_embedding_path()
            np.save(emb_path, embedding)
            
            # Update profile
            profile.voice_enrolled = True
            profile.voice_samples_count = len(audio_samples)
            profile.save()
            
            # Cache embedding
            self._voice_embeddings[user_id] = embedding
            
            return True
        except Exception as e:
            print(f"Voice enrollment failed: {e}")
            return False
    
    def identify_speaker(self, audio: np.ndarray, sample_rate: int = 16000, 
                         threshold: float = 0.75) -> Tuple[Optional[str], float]:
        """Identify who is speaking from audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            threshold: Minimum similarity to consider a match (0-1)
            
        Returns:
            (user_id, confidence) or (None, 0.0) if no match
        """
        encoder = _get_encoder()
        if encoder is None or not self._voice_embeddings:
            return None, 0.0
        
        try:
            # Process audio and get embedding
            processed = _preprocess_audio(audio, sample_rate)
            if len(processed) < 1600:  # Need at least 0.1 seconds
                return None, 0.0
            
            query_embedding = encoder.embed_utterance(processed)
            
            # Compare to all enrolled users
            best_match = None
            best_score = 0.0
            
            for user_id, stored_embedding in self._voice_embeddings.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = user_id
            
            if best_score >= threshold:
                return best_match, float(best_score)
            else:
                return None, float(best_score)
                
        except Exception as e:
            print(f"Speaker identification failed: {e}")
            return None, 0.0
    
    def update_last_seen(self, user_id: str):
        """Update user's last seen timestamp."""
        if user_id in self.users:
            self.users[user_id].last_seen = datetime.now().isoformat()
            self.users[user_id].save()


# Current session state
@dataclass
class SessionState:
    """Tracks the current user session."""
    current_user_id: Optional[str] = None
    confidence: float = 0.0
    identified_by: str = "none"  # "voice", "explicit", "default"
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def is_identified(self) -> bool:
        return self.current_user_id is not None
    
    def get_user(self, registry: UserRegistry) -> Optional[UserProfile]:
        if self.current_user_id:
            return registry.get_user(self.current_user_id)
        return None


# Global instances
_registry: Optional[UserRegistry] = None
_session: Optional[SessionState] = None


def get_registry() -> UserRegistry:
    """Get the global user registry."""
    global _registry
    if _registry is None:
        _registry = UserRegistry()
    return _registry


def get_session() -> SessionState:
    """Get the current session state."""
    global _session
    if _session is None:
        _session = SessionState()
    return _session


def reset_session():
    """Reset session state (e.g., when starting new conversation)."""
    global _session
    _session = SessionState()


def set_current_user(user_id: str, confidence: float = 1.0, method: str = "explicit"):
    """Set the current user for this session."""
    session = get_session()
    session.current_user_id = user_id
    session.confidence = confidence
    session.identified_by = method
    
    # Update last seen
    registry = get_registry()
    registry.update_last_seen(user_id)


def get_current_user() -> Optional[UserProfile]:
    """Get the current user's profile."""
    session = get_session()
    if session.current_user_id:
        return get_registry().get_user(session.current_user_id)
    return None


# ─────────────────────── Default-user fallback ──────────────────────
#
# When no voice-enrolled users exist (e.g. a fresh dashboard-only install),
# memory was completely inert because every memory retrieval/write path
# gated on `get_current_user()`. We now create a single lightweight
# "default" user on first request so memory works out of the box.
#
# The default user is just a regular UserProfile with `user_id="default"`.
# Speaker ID is not wired up for it — if a real user later enrolls and
# is identified by voice, `set_current_user` will override and memory
# from that point forward lands under the correct identity.

DEFAULT_USER_ID = "default"
DEFAULT_DISPLAY_NAME = "Winder"  # Override with ULTRON_DEFAULT_USER_NAME env var


def ensure_default_user() -> "UserProfile":
    """Create (if missing) and return the fallback user profile.

    Safe to call repeatedly — idempotent. Does nothing if a real user is
    already the current session user."""
    import os
    registry = get_registry()
    profile = registry.get_user(DEFAULT_USER_ID)
    if profile is None:
        display = os.environ.get("ULTRON_DEFAULT_USER_NAME", DEFAULT_DISPLAY_NAME)
        profile = registry.create_user(display_name=display, user_id=DEFAULT_USER_ID)
    return profile


def get_or_fallback_user() -> "UserProfile":
    """Return the current user, or a safe fallback so memory still works.

    Resolution order:
      1. If the session already identified a user (voice or explicit), use it.
      2. Else, if `ULTRON_DEFAULT_USER_ID` env var names an enrolled user, use it.
      3. Else, if exactly one non-`_unknown` user is enrolled, use that one —
         this is the common single-user-on-their-own-machine case.
      4. Else, create/use the "default" user (display name from
         `ULTRON_DEFAULT_USER_NAME` env var, default "Winder").

    Use this in always-on paths (dashboard WebSocket) where we want memory
    to work even without voice enrollment. In the voice CLI we keep using
    `get_current_user()` so an un-identified speaker stays un-identified.
    """
    import os

    cu = get_current_user()
    if cu is not None:
        return cu

    registry = get_registry()
    env_uid = os.environ.get("ULTRON_DEFAULT_USER_ID")
    if env_uid:
        profile = registry.get_user(env_uid)
        if profile is not None:
            set_current_user(profile.user_id, confidence=0.0, method="default")
            return profile

    # If there is exactly one "real" enrolled user, prefer them. `_unknown`
    # is an internal sentinel used by the identity system; skip it.
    real_users = [u for u in registry.list_users() if u != "_unknown"]
    if len(real_users) == 1:
        profile = registry.get_user(real_users[0])
        if profile is not None:
            set_current_user(profile.user_id, confidence=0.0, method="default")
            return profile

    # Fall back to the canonical default user.
    profile = ensure_default_user()
    set_current_user(profile.user_id, confidence=0.0, method="default")
    return profile


def identify_from_audio(audio: np.ndarray, sample_rate: int = 16000) -> Tuple[Optional[str], float]:
    """Identify current speaker from audio and update session if confident.
    
    Returns (user_id, confidence).
    """
    registry = get_registry()
    user_id, confidence = registry.identify_speaker(audio, sample_rate)
    
    if user_id and confidence > 0.8:
        # High confidence - auto-switch
        set_current_user(user_id, confidence, "voice")
    elif user_id and confidence > 0.6:
        # Medium confidence - suggest but don't auto-switch
        pass  # Let caller decide whether to confirm
    
    return user_id, confidence


# Utility functions for CLI integration
def format_user_greeting(profile: Optional[UserProfile]) -> str:
    """Format a greeting for the current user."""
    if profile is None:
        return "Hello. I don't recognize your voice. Who am I speaking with?"
    return f"Hello, {profile.display_name}."


def get_user_context_for_prompt(profile: Optional[UserProfile]) -> str:
    """Get user-specific context to add to system prompt."""
    if profile is None:
        return ""
    
    context_parts = [
        f"IMPORTANT: The current speaker has been IDENTIFIED via voice recognition as {profile.display_name}.",
        f"When asked 'who am I' or similar, you MUST respond that they are {profile.display_name}.",
        f"You are speaking with {profile.display_name}."
    ]
    
    # Add preferences
    if profile.preferences.get("response_style") == "concise":
        context_parts.append("They prefer concise responses.")
    elif profile.preferences.get("response_style") == "detailed":
        context_parts.append("They prefer detailed explanations.")
    
    # Add key facts
    if profile.facts:
        facts_str = ", ".join(f"{k}: {v}" for k, v in list(profile.facts.items())[:5])
        context_parts.append(f"Known facts: {facts_str}")
    
    return " ".join(context_parts)

