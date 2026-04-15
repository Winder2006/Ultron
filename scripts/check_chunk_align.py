"""Inspect raw Deepgram PCM output to see if the clicks are in the source."""
from dotenv import load_dotenv; load_dotenv()
import numpy as np
from mother.tts.engine import DeepgramTTSEngine, DeepgramTTSConfig

eng = DeepgramTTSEngine(DeepgramTTSConfig())

# Single short sentence
text = "Indeed."
pcm_bytes = b"".join(eng.synthesize_stream_pcm(text))
samples = np.frombuffer(pcm_bytes, dtype=np.int16)

print(f"Total samples: {len(samples)}")
print(f"First 30 samples: {list(samples[:30])}")
print(f"Last 30 samples: {list(samples[-30:])}")

# Find where actual audio starts (first sample > 500)
abs_s = np.abs(samples)
first_loud = int(np.argmax(abs_s > 500))
print(f"\nFirst 'loud' sample (>500) at index {first_loud}")
print(f"30 samples around first_loud: {list(samples[max(0, first_loud-5):first_loud+25])}")
