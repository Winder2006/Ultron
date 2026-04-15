"""MQTT client for Jetson vision events.

Subscribes to face/object detection events from the Jetson camera pipeline.
Maintains in-memory room state for LLM prompt injection.

Expected MQTT topics (published by Jetson camera service):
    mother/vision/face_detected    {"name": "Oliver", "confidence": 0.92, "ts": 1713000000}
    mother/vision/face_lost        {"name": "Oliver", "ts": 1713000030}
    mother/vision/object_detected  {"label": "cat", "confidence": 0.87, "ts": 1713000000}
    mother/vision/room_occupancy   {"occupied": true, "person_count": 2, "ts": 1713000000}
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from mother.core.logging_config import get_logger

logger = get_logger("mother.vision.mqtt")

# MQTT topic prefix
TOPIC_PREFIX = "mother/vision/"
TOPICS = [
    f"{TOPIC_PREFIX}face_detected",
    f"{TOPIC_PREFIX}face_lost",
    f"{TOPIC_PREFIX}object_detected",
    f"{TOPIC_PREFIX}room_occupancy",
]

# Default broker config — overridable via app.yaml in future
DEFAULT_BROKER = "localhost"
DEFAULT_PORT = 1883


class VisionMQTTClient:
    """Async MQTT subscriber that updates shared vision state.

    Vision state dict (shared with server.py):
        {
            "faces": [{"name": "Oliver", "confidence": 0.92, "since": 1713...}],
            "objects": [{"label": "cat", "confidence": 0.87, "since": 1713...}],
            "room_occupied": True,
            "last_event_ts": 1713000000.0,
        }
    """

    def __init__(
        self,
        vision_state: dict,
        event_bus: Optional[asyncio.Queue] = None,
        broker: str = DEFAULT_BROKER,
        port: int = DEFAULT_PORT,
    ):
        self._state = vision_state
        self._event_bus = event_bus
        self._broker = broker
        self._port = port
        self._client = None
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Connect to the MQTT broker and subscribe to vision topics."""
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="mother-vision",
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        # Non-blocking connect — will raise if broker unreachable
        self._client.connect(self._broker, self._port, keepalive=60)

        # Run the MQTT network loop in a background thread
        self._client.loop_start()
        logger.info("MQTT client started (broker=%s:%d)", self._broker, self._port)

    async def stop(self):
        """Disconnect from the MQTT broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            logger.info("MQTT client disconnected")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Subscribe to all vision topics on connect."""
        if rc == 0:
            for topic in TOPICS:
                client.subscribe(topic)
            logger.info("Subscribed to %d vision topics", len(TOPICS))
        else:
            logger.warning("MQTT connect failed with code %d", rc)

    def _on_message(self, client, userdata, msg):
        """Handle incoming vision events and update shared state."""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Bad MQTT payload on %s: %s", msg.topic, e)
            return

        topic = msg.topic
        now = time.time()
        self._state["last_event_ts"] = now

        if topic == f"{TOPIC_PREFIX}face_detected":
            name = payload.get("name", "unknown")
            conf = payload.get("confidence", 0.0)
            # Update or add face
            faces = self._state.get("faces", [])
            existing = next((f for f in faces if f["name"] == name), None)
            if existing:
                existing["confidence"] = conf
                existing["last_seen"] = now
            else:
                faces.append({
                    "name": name,
                    "confidence": conf,
                    "since": now,
                    "last_seen": now,
                })
            self._state["faces"] = faces
            self._push_event("face_detected", {"name": name, "confidence": conf})

        elif topic == f"{TOPIC_PREFIX}face_lost":
            name = payload.get("name", "unknown")
            faces = self._state.get("faces", [])
            self._state["faces"] = [f for f in faces if f["name"] != name]
            self._push_event("face_lost", {"name": name})

        elif topic == f"{TOPIC_PREFIX}object_detected":
            label = payload.get("label", "unknown")
            conf = payload.get("confidence", 0.0)
            objects = self._state.get("objects", [])
            existing = next((o for o in objects if o["label"] == label), None)
            if existing:
                existing["confidence"] = conf
                existing["last_seen"] = now
            else:
                objects.append({
                    "label": label,
                    "confidence": conf,
                    "since": now,
                    "last_seen": now,
                })
            self._state["objects"] = objects
            self._push_event("object_detected", {"label": label, "confidence": conf})

        elif topic == f"{TOPIC_PREFIX}room_occupancy":
            occupied = payload.get("occupied", False)
            count = payload.get("person_count", 0)
            self._state["room_occupied"] = occupied
            self._push_event("room_occupancy", {"occupied": occupied, "person_count": count})

    def _push_event(self, event_type: str, data: dict):
        """Push a vision event to the SSE event bus (non-blocking)."""
        if self._event_bus is None:
            return
        event = {"type": f"vision_{event_type}", **data}
        try:
            self._event_bus.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop if bus is full — SSE is best-effort
