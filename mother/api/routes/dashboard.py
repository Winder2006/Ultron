"""REST endpoints for the MOTHER dashboard.

Provides system status, user memories, conversation history, and vision state
for the Phase 7 React dashboard.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from mother.core.logging_config import get_logger

logger = get_logger("mother.api.dashboard")
router = APIRouter(tags=["dashboard"])


@router.get("/status")
async def system_status():
    """Current system status: active providers, connected clients, vision state."""
    from mother.api.server import get_state
    state = get_state()
    cfg = state.get("config")

    llm_provider = cfg.llm.provider if cfg else "unknown"
    tts_provider = cfg.tts.provider if cfg else "unknown"
    stt_engine = "none"
    if state.get("streaming_stt"):
        stt_engine = "deepgram" if state["streaming_stt"]._deepgram_available else "faster_whisper"
    elif state.get("stt"):
        stt_engine = "faster_whisper"

    mqtt_connected = state.get("mqtt_client") is not None
    vision = state.get("vision", {})
    ws_count = len(state.get("ws_clients", set()))

    return {
        "llm_provider": llm_provider,
        "tts_provider": tts_provider,
        "stt_engine": stt_engine,
        "mqtt_connected": mqtt_connected,
        "vision": {
            "room_occupied": vision.get("room_occupied", False),
            "faces_count": len(vision.get("faces", [])),
            "objects_count": len(vision.get("objects", [])),
            "last_event_ts": vision.get("last_event_ts"),
        },
        "ws_clients": ws_count,
    }


@router.get("/memories/{user_id}")
async def user_memories(user_id: str, max_facts: int = 20, max_episodic: int = 10):
    """Retrieve stored facts and episodic memories for a user."""
    try:
        from mother.memory.manager import get_user_memory
        memory = get_user_memory(user_id)
        if memory is None:
            raise HTTPException(status_code=404, detail=f"No memory found for user: {user_id}")

        facts = memory.get_all_facts() if hasattr(memory, "get_all_facts") else {}
        episodic = []
        if hasattr(memory, "search_episodic"):
            episodic = memory.search_episodic("", top_k=max_episodic)
        elif hasattr(memory, "_episodic_path"):
            # Direct read from JSON
            import json
            try:
                raw = memory._episodic_path.read_text(encoding="utf-8")
                all_ep = json.loads(raw) if raw else []
                episodic = all_ep[:max_episodic]
            except Exception:
                pass

        return {
            "user_id": user_id,
            "facts": facts,
            "episodic": episodic,
            "fact_count": len(facts) if isinstance(facts, dict) else 0,
            "episodic_count": len(episodic),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Memory retrieval failed for %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{user_id}")
async def user_conversation(user_id: str):
    """Retrieve recent conversation history for a user."""
    try:
        from mother.memory.conversation import ConversationMemory
        conv = ConversationMemory()
        conv.load(user_id)
        messages = [
            {"role": m.role, "content": m.content}
            for m in conv.get_context_messages()
        ]
        return {
            "user_id": user_id,
            "messages": messages,
            "turn_count": len(messages) // 2,
        }
    except Exception as e:
        logger.warning("Conversation retrieval failed for %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users")
async def list_users():
    """List all enrolled users."""
    try:
        from mother.identity.speaker import get_registry
        registry = get_registry()
        users = []
        for uid in registry.list_users():
            profile = registry.get_user(uid)
            if profile:
                users.append({
                    "user_id": uid,
                    "display_name": profile.display_name,
                    "voice_enrolled": profile.voice_enrolled,
                    "last_seen": profile.last_seen,
                })
        return {"users": users}
    except Exception as e:
        logger.warning("User listing failed: %s", e)
        return {"users": []}


@router.get("/vision")
async def vision_state():
    """Current vision state from MQTT camera events."""
    from mother.api.server import get_state
    state = get_state()
    return state.get("vision", {
        "faces": [],
        "objects": [],
        "room_occupied": False,
        "last_event_ts": None,
    })
