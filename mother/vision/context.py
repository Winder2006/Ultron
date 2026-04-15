"""Vision context translator for LLM prompts.

Converts raw vision data (faces, objects, room occupancy) from the MQTT
pipeline into natural language fragments injected into the MOTHER system prompt.

This gives the LLM situational awareness — it knows who is in the room,
what objects are visible, and whether the room is occupied.
"""
from __future__ import annotations

import time
from typing import Optional


# Faces older than this are considered stale and excluded
FACE_STALE_SECONDS = 120.0
# Objects older than this are considered stale
OBJECT_STALE_SECONDS = 300.0


def build_vision_context(vision_state: dict) -> Optional[str]:
    """Convert vision state dict into natural language for LLM injection.

    Returns None if there's nothing meaningful to report.

    Example output:
        "You can see Oliver (high confidence) in the room. A cat is also visible.
         The room is occupied."
    """
    if not vision_state:
        return None

    parts = []
    now = time.time()

    # Faces
    faces = vision_state.get("faces", [])
    active_faces = [
        f for f in faces
        if (now - f.get("last_seen", f.get("since", 0))) < FACE_STALE_SECONDS
    ]
    if active_faces:
        names = []
        for f in active_faces:
            name = f.get("name", "someone")
            conf = f.get("confidence", 0.0)
            if conf > 0.85:
                names.append(name)
            elif conf > 0.6:
                names.append(f"{name} (uncertain)")
            else:
                names.append("an unidentified person")
        if len(names) == 1:
            parts.append(f"You can see {names[0]} in the room.")
        else:
            listed = ", ".join(names[:-1]) + f" and {names[-1]}"
            parts.append(f"You can see {listed} in the room.")

    # Objects
    objects = vision_state.get("objects", [])
    active_objects = [
        o for o in objects
        if (now - o.get("last_seen", o.get("since", 0))) < OBJECT_STALE_SECONDS
    ]
    if active_objects:
        labels = [o.get("label", "something") for o in active_objects if o.get("confidence", 0) > 0.7]
        if labels:
            if len(labels) == 1:
                parts.append(f"A {labels[0]} is visible nearby.")
            else:
                listed = ", ".join(labels[:-1]) + f" and a {labels[-1]}"
                parts.append(f"Visible nearby: {listed}.")

    # Room occupancy
    room_occupied = vision_state.get("room_occupied", False)
    if room_occupied and not active_faces:
        parts.append("The room appears occupied.")
    elif not room_occupied and not active_faces:
        parts.append("The room appears empty.")

    return " ".join(parts) if parts else None
