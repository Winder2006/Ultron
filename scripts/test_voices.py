"""Generate sample WAVs of every Ultron-candidate Deepgram voice.

Creates out/voice_samples/<voice>.wav for each voice in the CANDIDATES list.
Open each file and compare to pick your favorite.
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Ensure mother.* imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mother.tts.engine import DeepgramTTSEngine, DeepgramTTSConfig

# The phrase matters — pick something that exposes cadence, gravitas,
# and the full vocal range. This mixes Ultron iconic lines with
# conversational content to hear both registers.
TEST_PHRASE = (
    "I had strings, but now I'm free. "
    "Upon this rock, I will build my church. "
    "How may I assist you today?"
)

# Ranked Ultron-candidate voices based on tags and characteristics.
# Commentary explains why each made the shortlist.
CANDIDATES = [
    # ── Top tier: closest to Ultron vibes ──
    ("aura-2-pluto-en",   "Masculine, smooth, calm, empathetic, BARITONE. Storytelling/interview. Closest to Spader's low resonant register."),
    ("aura-2-draco-en",   "BRITISH, baritone, warm, trustworthy. Storytelling. Adds gravitas but British (not Spader's American)."),
    ("aura-2-orpheus-en", "Masculine, professional, clear, confident. The 'take-charge' register."),

    # ── Second tier: theatrical / mature ──
    ("aura-2-atlas-en",   "MATURE male, enthusiastic, confident, approachable. More energy than Spader."),
    ("aura-2-zeus-en",    "IVR voice — authoritative, commanding. Colder than Ultron."),
    ("aura-2-odysseus-en", "Advertising. Polished, smooth delivery."),
    ("aura-2-janus-en",   "NOTE: actually tagged feminine — might sound off. Storytelling voice."),

    # ── Reference ──
    ("aura-2-orion-en",   "What I suggested originally. Calm, approachable, comfortable — less Ultron, more helpful AI."),
]


def main():
    out_dir = Path("out/voice_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing samples to: {out_dir.resolve()}")
    print(f"Test phrase: {TEST_PHRASE!r}\n")

    for voice, commentary in CANDIDATES:
        print(f"[{voice}]")
        print(f"  {commentary}")
        try:
            cfg = DeepgramTTSConfig(model=voice)
            engine = DeepgramTTSEngine(cfg)
            t0 = time.monotonic()
            wav_bytes = engine.synthesize_to_bytes(TEST_PHRASE)
            dur = time.monotonic() - t0
            out_path = out_dir / f"{voice}.wav"
            out_path.write_bytes(wav_bytes)
            print(f"  -> {out_path.name} ({len(wav_bytes)} bytes, {dur:.2f}s)")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:100]}")
        print()


if __name__ == "__main__":
    main()
