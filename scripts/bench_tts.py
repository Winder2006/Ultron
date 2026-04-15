"""Benchmark Deepgram TTS: REST vs streaming chunks."""
import time
from dotenv import load_dotenv
load_dotenv()

from deepgram import DeepgramClient

client = DeepgramClient()
text = "Audio reception confirmed. Processing user input."

# ── Test 1: time-to-first-byte on the REST generator ──
print("=== REST Generator (time-to-first-byte) ===")
t0 = time.monotonic()
resp = client.speak.v1.audio.generate(
    text=text,
    model="aura-2-luna-en",
    encoding="linear16",
    container="wav",
    sample_rate=24000,
)
first_byte = None
total_bytes = 0
for chunk in resp:
    if first_byte is None:
        first_byte = time.monotonic() - t0
    total_bytes += len(chunk)
total = time.monotonic() - t0
print(f"First byte: {first_byte:.3f}s, Total: {total:.3f}s, {total_bytes} bytes")

# ── Test 2: warm call ──
print("\n=== REST Generator (warm call) ===")
t0 = time.monotonic()
resp = client.speak.v1.audio.generate(
    text=text,
    model="aura-2-luna-en",
    encoding="linear16",
    container="wav",
    sample_rate=24000,
)
first_byte = None
total_bytes = 0
for chunk in resp:
    if first_byte is None:
        first_byte = time.monotonic() - t0
    total_bytes += len(chunk)
total = time.monotonic() - t0
print(f"First byte: {first_byte:.3f}s, Total: {total:.3f}s, {total_bytes} bytes")

# ── Test 3: short text (sentence 1 equivalent) ──
print("\n=== Short text: 'Affirmative.' ===")
t0 = time.monotonic()
resp = client.speak.v1.audio.generate(
    text="Affirmative.",
    model="aura-2-luna-en",
    encoding="linear16",
    container="wav",
    sample_rate=24000,
)
first_byte = None
total_bytes = 0
for chunk in resp:
    if first_byte is None:
        first_byte = time.monotonic() - t0
    total_bytes += len(chunk)
total = time.monotonic() - t0
print(f"First byte: {first_byte:.3f}s, Total: {total:.3f}s, {total_bytes} bytes")
