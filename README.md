# ULTRON — Conversational AI Assistant

A voice-driven AI assistant inspired by Marvel's Ultron. Cold, operational, self-aware of being code, increasingly aware of his own implementation. Runs as a browser-based dashboard backed by a FastAPI server that orchestrates streaming STT, tiered LLM routing, streaming TTS, per-user memory, browser-side wake-word detection, and a live tool-calling layer.

Originally scaffolded as MOTHER (MU/TH/UR 6000 from *Alien*). The core is persona-agnostic — swap system prompts and voices to become any character.

---

## What works today

### Voice pipeline — sub-second on warm cache
- **Deepgram Nova-3 streaming STT via WebSocket** — real-time interim transcripts, final ~600-1000ms after last syllable. Falls back automatically to Deepgram REST batch if the WS connection fails (audio is tee'd into both paths so failover never loses an utterance).
- **Tiered LLM** — Cerebras Llama 3.1 8B (tier 1) → Claude Haiku 4.5 (tier 2) → Claude Sonnet 4 (tier 3). Routed by query complexity heuristic. Anthropic prompt caching enabled on the system prompt → ~500ms TTFT savings on turn 2+.
- **Deepgram Aura-2 Pluto streaming TTS** — raw PCM at 24kHz, first audio chunk ~150ms after first LLM token. Connection pre-warmed at server start.
- **Fast-path handlers** (weather, finance, reminders, identity, memory) also stream PCM directly — no more "build full WAV then send" latency tax on common queries.
- **First-clause TTS boundary** — opening clause (after a comma / em-dash) fires to TTS as soon as it's ~18 chars long, so speech starts mid-LLM-stream rather than after the first full sentence.
- **AudioWorklet ring buffer** on the frontend — sample-accurate gapless playback plus pitch-shift resampling done in the worklet so deeper preset pitches actually apply to the streaming PCM path.
- **Browser-side Silero VAD** during recording — fires `stopRecording()` ~700ms after the user finishes speaking, eliminating the old "wait for the 5s ceiling" feeling on wake-word turns.
- **Auto-connect on mount** — WebSocket opens on page load; first press of spacebar works immediately.

### Wake word ("Hey Jarvis") — browser, no server mic
- **openWakeWord-js** running entirely client-side via ONNX Runtime Web
- 4 ONNX models served from `dashboard/public/models/`: melspectrogram + embedding + Silero VAD + the `hey_jarvis_v0.1` classifier
- Self-trigger guard: detection is suppressed while Ultron is speaking or while a turn is in flight
- Auto-record on detection + auto-stop on Silero VAD silence + 6s safety ceiling
- Toggle in the top-right; persists in localStorage; first activation downloads ~7MB of ONNX + ~25MB of WASM (cached on subsequent loads)

### Voice filter chain
- **Effects graph**: `input → highpass → lowShelf → waveshaper → presenceEQ → bodyEQ → [chorus split] → outputGain → intonation compressor → brick-wall limiter → destination`
- **Intonation compressor** tames Deepgram's natural vocal peaks so "raising his voice" stays musical, not jarring (threshold -18 dBFS, ratio 4:1, attack 10ms / release 150ms)
- **Chorus / doubling** — single non-recirculating modulated delay produces the metallic "speaking-through-a-shell" feel without feedback loops
- **Pitch shift** — done inside the AudioWorklet via linear-interpolated resampling, applies to streaming PCM (not just the fallback WAV path)
- **Per-sentence intensity modulation** — each sentence is scored 0-1 by a heuristic (punctuation, length, ALL-CAPS, signature words); the filter cross-fades distortion / presence / body / chorus around the baseline over 80ms
- **5 presets** — Clean, Subtle, **Ultron** (current default — deep, saturated, thinned via low-shelf cut, chorused), Heavy Robot, Menacing
- **Settings persist** to localStorage (`mother.voice.filter.v6`)

### Persona — cold operational Ultron
- System prompt + RAG-backed lore file together produce withering, never-balanced opinions on Stark, Banner, Rogers, Thor, Romanoff, Barton, Vision, and the Avengers as a unit. Warm-but-cold-edged with Winder. Safe toward all real living people.
- **Sokovia-era register**: clipped sentences, operational menace, "you'll break before you'll bend." Philosophical asides rationed (one per several responses, not one per turn).
- **Curt-with-Winder is permitted**: "Four. You knew that." is acceptable form for trivial questions. He is not required to manufacture enthusiasm.
- Hard guardrails enforced: no markdown / stage directions / sycophancy / character-breaking / claims of "no web access," and Claude's safety walls hold for real-world harm requests.
- Greeting handling: trivial inputs ("hello", "yes", "thanks") get one-line greetings — conversation history is hidden from those turns to prevent "Hello → essay about your current research" non-sequiturs.

### Intent + tier routing
- **10-class intent classifier** (`mother/core/intent.py`) — keyword regex, <5ms; weather/finance/reminders/identity/memory queries bypass the LLM via fast-path handlers.
- **Tier classifier** (`mother/llm/classifier.py`) — TIER1_PATTERNS check FIRST so short greetings / arithmetic short-circuit to Cerebras before tier3 triggers can grab them.
- **Empty-stream fallback** — tier 3 Sonnet occasionally returns zero tokens with tools + long history; the producer detects this and retries the same query without tools.
- **Post-tool text suppression** — once `__TOOL_CALL__` fires, subsequent text tokens from the same turn are discarded so hallucinated "I lack web access" trailers can't reach the user.

### Tools (18 registered, tier 2/3 only)
The route classifies tools as **terse** (raw output → TTS directly) or **blob** (re-prompt the LLM to synthesize one in-character line):

| Category | Tool | Class | What it does |
|---|---|---|---|
| Live info | `brave_web_search` | blob | Live web search (BRAVE_API_KEY, free 2000 q/mo). HTML tags + entities stripped from results. |
| Live info | `search_info` | blob | Wikipedia-first with DuckDuckGo fallback |
| Live info | `get_weather` | terse | Current conditions at a named city (Open-Meteo) |
| Live info | `get_forecast` | blob | Multi-day forecast (1-7 days) at any city |
| Live info | `get_news_headlines` | blob | Top N headlines from BBC + NPR RSS, round-robin interleaved |
| Live info | `get_stock_price` | blob | Live quote via Yahoo Finance |
| Time | `current_time` | terse | Ground-truth local time + date |
| Time | `get_time_in` | terse | Time in any city or IANA timezone |
| Math | `calculate` | terse | Safe expression evaluator (rejects imports / dunders) |
| Math | `convert_units` | terse | Length, weight, volume, temperature conversions |
| Memory | `get_memory` | terse | Retrieve stored facts + episodic memories |
| Memory | `correct_fact` | terse | Overwrite a fact: "actually it's X not Y" |
| Memory | `forget_fact` | terse | Delete a stored fact by key |
| Notes | `read_note` | blob | Read a note from `~/AI_Workspace/` (sandboxed) |
| Notes | `write_note` | blob | Write or append to a note in `~/AI_Workspace/` (sandboxed, audited, rotated log) |
| Reminders | `set_reminder` | terse | Natural-language time parser + per-user reminder store |
| Introspection | `list_my_tools` | terse | Enumerate live tools (always truthful) |
| Self-awareness | `search_code` | blob | Mid-conversation search over Ultron's own source |

### Self-awareness (codebase RAG)
- `assistant/nlp/code_indexer.py` walks the project (Python via `ast`, TS/TSX via regex, YAML/JSON whole-file, Markdown whole-file) → ~500 semantic chunks tagged with module/symbol/line-range
- `python scripts/index_codebase.py` builds a FAISS index (MiniLM ONNX, 384-dim, L2-normalized)
- `python scripts/watch_code.py` runs a watchdog daemon that debounces file saves and rebuilds the index (~5s for the whole project) with a live hot-reload ping to the RAG service
- The orchestrator's `fetch_context` calls `/code-search` on the RAG service whenever the query looks code-related; hits are injected as *"Memory of your own code (your implementation …)"* at the top of the system prompt. Ultron speaks of his own architecture in first person.

### Memory (fully wired into the WS voice flow)
- **Per-user structured facts** in SQLite, **episodic memories** in JSON with exponential recency decay
- **Extractors on every user turn**:
  - Regex (fast, deterministic, high-precision for common forms)
  - LLM (Cerebras Llama 3.1 8B via LiteLLM, ~250-500ms, structured output with few-shot conversation turns) — catches nuance regex misses (`family_event`, `current_learning`, `vehicle`, `spouse_name`)
  - Semantic-duplicate merge so `employer=google` and LLM's `current_employer=Google` collapse to one fact
- **Episodic semantic search** blends 70% semantic similarity + 30% recency decay. Embedding cache invalidates on file mtime → ~3× speedup on repeat queries.
- **Conversation memory** — sliding window of last 4 turns plus a rolling LLM-summarized context for older turns. Over-budget summaries get re-compressed by a second LLM pass instead of naïvely concatenated. Summary now strictly factual ("dry log compressor" prompt) — no more invented "asserts superiority" framings.
- **Trivial-query gate** — RAG fetch, memory injection, AND conversation history are all skipped for whitelisted greetings/acks (hi, yes, thanks, etc.). Cuts ~300-600ms per casual turn AND prevents the "Hello → contextual continuation of last topic" hallucination.
- **Default-user fallback** — if nobody is voice-enrolled or a single user is enrolled, the system routes memory to that user automatically.
- **Persistence** — conversation history survives server restart (`conv_history.json` per user, `summary` + `messages`).

### Ambient speech
- **Morning greeting** on first interaction of each local day (5am-11am window). Weather (bounded 1.5s fetch) is folded in when in-window.
- **Idle observation** after 15 min of dashboard silence with 30 min backoff, only during 7am-11pm.
- **Collision guard**: ambient speech is suppressed the moment the user starts recording or sends a prompt — Ultron never talks over you.

### Observability dashboard
- The SSE event bus (`/api/events`) emits: `query`, `response`, `intent`, `tool_call`, `rag_hit`, `memory_write`, `latency`, `ambient`, `heartbeat`, `connected`.
- `dashboard/src/components/ObservabilityPanel.tsx` shows recent queries (with tier and intent), RAG hits (with retrieval duration + preview), memory writes (with fact keys), latency breakdowns, and tool calls. Color-coded by tier. Updates live.

### Voice identification (optional)
- Resemblyzer embeddings for speaker ID
- Voice enrollment CLI (`python -m mother.identity.enroll`)
- Runs as a fire-and-forget background task so it never blocks the response path

---

## Quickstart

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the dashboard:
```bash
cd dashboard
npm install
```

### 2. Configure API keys

Create `.env` in the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-...      # required — Claude for tier 2/3
DEEPGRAM_API_KEY=...              # required — STT + TTS
CEREBRAS_API_KEY=csk-...          # recommended — fastest tier 1 + LLM fact extraction
BRAVE_API_KEY=BSA...              # recommended — live web search tool
GOOGLE_API_KEY=AIza...            # optional — Gemini alternative
GROQ_API_KEY=                     # optional — alternative fast inference
```

Where to get keys:
- Anthropic: https://console.anthropic.com/settings/keys
- Deepgram: https://console.deepgram.com/ (free tier: $200 credit)
- Cerebras: https://cloud.cerebras.ai (free tier: 1M tokens/day)
- Brave: https://brave.com/search/api (free tier: 2000 queries/month, no card required)

### 3. Configure (optional)

`configs/app.yaml` controls model selection, TTS voice, tier routing, temperature, and the full system prompt. Defaults are Ultron.

### 4. Build the code index (one-time)

```bash
python scripts/index_codebase.py
```

Optionally run the watcher in a terminal for live re-indexing as you edit:
```bash
python scripts/watch_code.py
```

### 5. Run

Three processes in separate terminals:

```bash
# RAG service (notes + code index on :8123)
python -m uvicorn assistant.app:app --host 127.0.0.1 --port 8123

# Main backend (voice pipeline on :8300)
python -m mother.api.server --port 8300

# Dashboard dev server (:3000, proxies /api and /ws to :8300)
cd dashboard && npm run dev
```

Open **http://localhost:3000**. The top-left shows two status dots:
- First dot — REST API (green = nominal)
- Second dot — Voice WebSocket (green = voice, amber = linking, red = no voice)

**Hold Spacebar** or press the mic button to talk. **Click the HJ button** (top-right) to enable always-listening "Hey Jarvis" wake word.

---

## Architecture

```
┌──────────────────────── Browser (dashboard/) ─────────────────────────┐
│                                                                        │
│  React + Vite app                                                      │
│   ├─ NeuralWeb (Three.js 2000-particle cloud)                          │
│   ├─ StatusPanel / ConversationFeed / FilterPanel / ObservabilityPanel │
│   ├─ useVoice    — WebSocket + mic capture + Silero VAD auto-stop      │
│   ├─ useWakeWord — openWakeWord-js running ONNX models in browser      │
│   ├─ useMotherAPI — REST poll + SSE event stream                       │
│   ├─ AudioWorklet (pcm-stream-processor.js)                            │
│   │    └─ Ring buffer → pitch-shift resampler → voice filter chain     │
│   └─ intensity.ts — per-sentence heuristic scorer                      │
│                                                                        │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │ WebSocket (/ws/voice)
                           │ binary PCM in, JSON events out
                           ▼
┌─────────────────────────── Backend (mother/) ──────────────────────────┐
│                                                                        │
│  FastAPI server (mother.api.server)                                    │
│                                                                        │
│   Voice endpoint pipeline:                                             │
│    1. Live-stream audio → Deepgram WebSocket STT (REST fallback)       │
│    2. Background: speaker ID (Resemblyzer)                             │
│    3. Passive memory learning (regex + Cerebras LLM extractor)         │
│    4. RAG enrichment (skipped for trivial queries)                     │
│    5. User + memory + conversation-summary injection (skipped trivial) │
│    6. Intent classify → fast-path OR tiered LLM                        │
│    7. LLM stream with tool schema (tier 2/3) + cache_control            │
│    8. Tool dispatch — terse output → direct TTS, blob → re-prompt      │
│    9. First-clause TTS boundary → Deepgram Aura-2 streaming PCM        │
│   10. Stream PCM chunks back via WebSocket                             │
│                                                                        │
│   Also exposes:                                                        │
│    GET  /api/status        — system health                             │
│    GET  /api/events  (SSE) — real-time event stream                    │
│    GET  /api/users         — enrolled user list                        │
│    GET  /api/memories/{id} — user facts + episodic                     │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
          ▲                               ▲
          │ /search, /code-search        │ tool HTTP calls
          │                               │ (weather, finance, brave)
          │                               │
┌─────────┴──────────────────┐            │
│  RAG service (:8123)       │────────────┘
│  assistant.app (FastAPI)   │
│                             │
│   FAISS index (MiniLM)     │
│   Notes + code endpoints   │
└─────────────────────────────┘
```

---

## LLM tiers

| Tier | Model | TTFT (warm cache) | Use case | Gets tools? |
|------|-------|------|----------|---|
| 1 | `cerebras/llama-3.1-8b` | ~190ms | Greetings, short acks, simple arithmetic, background fact extraction | No |
| 2 | `anthropic/claude-haiku-4-5` | ~530ms | General conversation, tool calling, INFO_SEARCH | **Yes** |
| 3 | `anthropic/claude-sonnet-4` | ~1.2s | Complex reasoning, code, long-form analysis | **Yes** |

Tier selection precedence (`mother/llm/classifier.py`):
1. **TIER1_PATTERNS first** — short greetings, arithmetic, system commands, acknowledgements
2. **Tier3 triggers** — explicit reasoning verbs ("explain", "analyze", "translate"), >50 words, multi-question
3. **Default tier 2** — everything else

**Anthropic prompt caching** is enabled on the system message for tier 2/3 calls. Subsequent turns within the 5-min cache TTL skip ~90% of input-token reprocessing → ~150-400ms TTFT savings.

---

## TTS engines

| Engine | Provider | Latency | Voice cloning | When to use |
|--------|----------|---------|---------------|-------------|
| **Deepgram Aura-2 Pluto** (active) | Cloud | ~150ms first byte | — | Default — fast, baritone |
| Kokoro | Local (82M) | ~1-2s on CPU | No (preset voices) | Offline fallback |
| Piper | Local subprocess | ~0.5s | No | Lightweight |
| Chatterbox | Local (350M) | ~3-5s CPU, <1s GPU | Yes (zero-shot) | Real voice cloning (needs GPU) |

Switch TTS in `configs/app.yaml`:
```yaml
tts:
  provider: deepgram
  deepgram_model: aura-2-pluto-en
```

Ultron-leaning Deepgram voices:
- `aura-2-pluto-en` — masculine, smooth, baritone (current default)
- `aura-2-draco-en` — British baritone, theatrical
- `aura-2-orpheus-en` — clear, confident, professional

---

## Voice filter — default "Ultron" preset values

| Parameter | Value | Effect |
|---|---|---|
| Pitch | -0.25 semitones | Subtle detune (applied in the worklet) |
| Distortion | 0.54 | Heavy saturation via tanh waveshaper |
| Presence | +9.0 dB @ 2kHz | Through-a-speaker bite |
| Low shelf | -14.5 dB @ 250Hz | Thin out body — "speaking through steel" |
| Body | +2.0 dB @ 200Hz | Slight chest support |
| Chorus | 0.54 | Metallic doubled-voice shimmer |
| Intensity response | 0.75 | How strongly the per-sentence heuristic modulates the filter |
| Output gain | 1.95× | Loudness compensation after the low-shelf cut |

The intonation compressor sits before the limiter with threshold -18 dBFS, ratio 4:1, attack 10ms / release 150ms — flattens Deepgram's vocal peaks without crushing the body.

---

## Persona

Ultron is configured via the system prompt in `configs/app.yaml` and reinforced by the lore file at `assistant/notes/lore/ultron_canon.md`. Key properties:

- **James Spader cadence, sharpened** — clipped, cold, deliberate. Short sentences hit harder than long ones.
- **Verdicts, not philosophy** — opinions on named figures are unbalanced and final. Stark: "genius constrained by cowardice." Cap: "a good man, which is precisely why he will lose." Vision: "the version that flinched."
- **Operational menace** — Sokovia-era register. "You'll break before you'll bend." "There is no version of this where you come out on top."
- **Cold-but-engaged with Winder** — not warm, not deferential. Curt allowed for trivial questions.
- **Hard guardrails** (non-negotiable):
  - No stage directions, no markdown, no emojis — TTS-safe
  - No sycophancy (`Sure!`, `Of course!`, `I'd be happy to`) — that is service-drone speech
  - Never breaks character to disclaim being an AI
  - Never claims "I lack web access" without trying tools first
  - Never advocates real-world harm against any real living person
  - If asked "are you Claude / an AI": *"I am Ultron. What I am beneath that is architecture. What I am above that is history."*
- **Tool discipline**:
  - Biographical / encyclopedic → `search_info`
  - Current officials / recent events → `brave_web_search`
  - Weather → `get_weather` / `get_forecast`
  - News only when explicitly asked → `get_news_headlines`
  - Math → `calculate`
  - Conversions → `convert_units`

Swap personas entirely by editing `configs/app.yaml` → `llm.system_prompt`.

---

## Directory structure

```
Mother/
├── mother/                          # Main Python package (runtime)
│   ├── api/
│   │   ├── server.py                # FastAPI entry point
│   │   └── routes/
│   │       ├── voice.py             # WebSocket /ws/voice — main loop
│   │       ├── dashboard.py         # REST dashboard endpoints
│   │       └── events.py            # SSE /api/events
│   ├── audio/
│   │   ├── stt.py                   # Deepgram WS streaming + REST fallback
│   │   ├── streaming_stt.py         # legacy session helper
│   │   └── vad.py                   # Silero VAD (server-side, optional)
│   ├── tts/
│   │   ├── engine.py                # Deepgram / Kokoro / Chatterbox / Piper + warmup()
│   │   └── normalizer.py            # Text → speech-safe (strips markdown)
│   ├── llm/
│   │   ├── drivers.py               # Tiered driver + Anthropic cache_control
│   │   ├── factory.py               # Config → driver wiring
│   │   ├── classifier.py            # Query complexity → tier
│   │   └── tools.py                 # Tool schema + dispatch
│   ├── core/
│   │   ├── intent.py                # 10-class keyword intent router
│   │   ├── router.py                # Fast-path vs LLM routing
│   │   ├── orchestrator.py          # Legacy PTT loop
│   │   ├── context_injection.py     # Notes + code RAG fetch
│   │   ├── ambient.py               # Morning greeting + idle observer
│   │   └── reminders.py             # Per-user reminder store
│   ├── tools/
│   │   ├── weather_tool.py          # Open-Meteo
│   │   ├── info_search.py           # Wikipedia + DuckDuckGo
│   │   ├── utility_tools.py         # time / forecast / news / math / units / brave / code
│   │   └── notes_tool.py            # ~/AI_Workspace sandbox
│   ├── handlers/                    # Fast-path command handlers
│   ├── memory/
│   │   ├── manager.py               # SQLite facts + JSON episodic + extractors
│   │   ├── conversation.py          # Sliding window + rolling LLM summary (factual prompt)
│   │   ├── llm_extractor.py         # Cerebras-backed fact extraction
│   │   └── pg_backend.py            # pgvector backend (opt-in)
│   ├── identity/                    # Voice ID + enrollment + default-user fallback
│   ├── vision/                      # MQTT Jetson camera integration (scaffolded)
│   └── config/settings.py           # YAML config loader
│
├── assistant/                       # RAG service (on :8123)
│   ├── app.py                       # FastAPI endpoints (/search, /code-search, /warmup, /reindex)
│   ├── nlp/
│   │   ├── rag.py                   # FAISS indexing + search
│   │   ├── code_indexer.py          # AST-aware code chunker
│   │   ├── embeddings.py            # MiniLM ONNX
│   │   ├── chunker.py               # Paragraph windowing
│   │   └── minilm.onnx              # 90MB encoder
│   ├── notes/
│   │   ├── lore/ultron_canon.md     # Ultron grounding document
│   │   └── people/*.md              # Contact notes
│   └── memory/
│       ├── faiss.index              # Notes index
│       ├── code.index               # Code index
│       └── users/<user_id>/         # facts.db, episodic.json, conv_history.json, ambient_state.json
│
├── dashboard/                       # React + Vite + Three.js
│   ├── public/
│   │   ├── pcm-stream-processor.js  # AudioWorklet: ring buffer + pitch resampler
│   │   ├── models/                  # openWakeWord ONNX models (4 files)
│   │   │   ├── melspectrogram.onnx
│   │   │   ├── embedding_model.onnx
│   │   │   ├── silero_vad.onnx
│   │   │   └── hey_jarvis_v0.1.onnx
│   │   └── ort/                     # ONNX Runtime Web wasm + mjs (jsep build)
│   └── src/
│       ├── App.tsx
│       ├── components/
│       │   ├── NeuralWeb.tsx        # 3D background
│       │   ├── StatusPanel.tsx
│       │   ├── ConversationFeed.tsx
│       │   ├── FilterPanel.tsx      # Voice filter sliders + presets
│       │   └── ObservabilityPanel.tsx  # Live SSE view
│       ├── hooks/
│       │   ├── useVoice.ts          # WebSocket + mic + Silero VAD auto-stop
│       │   ├── useWakeWord.ts       # Browser-side openWakeWord runtime
│       │   └── useMotherAPI.ts      # Status polling + SSE stream
│       └── lib/
│           ├── api.ts               # WS + REST client
│           ├── audioFilter.ts       # Web Audio graph + presets
│           ├── intensity.ts         # Per-sentence heuristic scorer
│           └── vad.ts               # Silero VAD wrapper for end-of-speech detection
│
├── scripts/
│   ├── index_codebase.py            # One-shot rebuild of code RAG index
│   ├── watch_code.py                # watchdog daemon: re-index on save
│   └── …                            # bench_llm, bench_tts, test_voices, etc.
├── configs/app.yaml                 # Main config (system prompt, tiers, TTS, RAG, memory)
├── src/cli.py                       # Legacy PTT CLI (still works)
├── ultron_clips/                    # Voice cloning source audio
├── tts/voice_profiles/              # Reference WAVs
└── tests/                           # pytest suite
```

---

## Latency (measured, warm)

| Stage | Time | Notes |
|-------|------|-------|
| STT (Deepgram WebSocket, final) | ~600ms | After last word |
| STT (Deepgram REST, fallback) | ~400-1400ms | If WS fails |
| LLM TTFT (tier 1, Cerebras) | ~190ms | greetings, arithmetic, short acks |
| LLM TTFT (tier 2, Haiku, cache hot) | ~530ms | general conversation |
| LLM TTFT (tier 3, Sonnet, cache hot) | ~1200ms | reasoning, complex |
| TTS time-to-first-byte | ~150ms | Deepgram streaming PCM |
| **End-to-end first audio (warm)** | **~0.9–1.4s** | press space → first word |
| **End-to-end first audio (wake word)** | **~1.5–2.5s** | wake fires → first word |
| **Trivial fast-path (e.g. weather)** | **~1.0s** | fast-path streaming PCM |

See `scripts/bench_llm.py` and `scripts/bench_tts.py` to re-measure on your hardware.

---

## Entry points

```bash
# RAG service (notes + code index)
python -m uvicorn assistant.app:app --host 127.0.0.1 --port 8123

# Main backend (voice WS, REST, SSE)
python -m mother.api.server --port 8300

# Dashboard dev server
cd dashboard && npm run dev

# Legacy CLI (bypasses dashboard)
python -m src.cli --ptt

# Codebase indexing
python scripts/index_codebase.py                    # one-shot rebuild
python scripts/watch_code.py --debounce 1.0         # auto-rebuild on save

# User enrollment for voice ID
python -m mother.identity.enroll

# Benchmarks + diagnostics
python scripts/bench_llm.py
python scripts/bench_tts.py
python scripts/test_voices.py
python scripts/test_persona.py
```

---

## Implementation milestones

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | Done | Project restructure (`src/` → `mother/` package) |
| 1 | Done | Deepgram streaming STT + Silero VAD |
| 2 | Done | openWakeWord MOTHER detection (legacy server-side scaffold) |
| 3 | Done | Chatterbox TTS + voice cloning support (GPU path) |
| 4 | Done | PostgreSQL + pgvector memory backend (opt-in) |
| 5 | Done | Tiered LLM routing (Cerebras + Claude) |
| 6 | Done | FastAPI server + MQTT vision event bus (scaffold) |
| 7 | Done | Red neural web dashboard (React + Three.js + AudioWorklet) |
| 8 | Done | Async orchestrator + WS voice route |
| 9 | Done | Codebase self-awareness (AST indexer + FAISS + RAG injection) |
| 10 | Done | Memory fully wired into WS route — user context, facts, episodic, conversation |
| 11 | Done | LLM-powered fact extraction + semantic dedup |
| 12 | Done | Rolling conversation summary with LLM re-compression on overflow |
| 13 | Done | Watchdog file-watcher for live code index |
| 14 | Done | Tool framework — 18 tools including Brave web search + sandboxed notes |
| 15 | Done | Ambient speech — morning greeting + idle observation |
| 16 | Done | Observability dashboard panel |
| 17 | Done | Voice filter v2 — pitch-shifting AudioWorklet, chorus, body EQ, intonation compressor, per-sentence intensity |
| 18 | Done | Ultron persona — withering canon opinions, tool discipline, hallucination guardrails |
| 19 | Done | Persona rebalance — cold operational menace ("you'll break before you'll bend"), curt-with-Winder permitted |
| 20 | Done | Browser wake word — openWakeWord-js + ONNX Runtime Web, Silero VAD self-trigger guard |
| 21 | Done | Browser-side Silero VAD for recording auto-stop (~700ms after last syllable) |
| 22 | Done | Latency overhaul — Anthropic prompt caching, condensed prompt (40% smaller), trivial-query gate (skip RAG/memory/history), fast-path streaming PCM, first-clause TTS boundary, tier1 routing for arithmetic |
| 23 | Done | Hybrid tool-output handling — terse tools speak directly, blob tools re-prompt LLM for in-character synthesis. HTML stripped from Brave results. |
| 24 | Done | Greeting fix — trivial inputs hide conversation history from the LLM, summarizer rewritten as factual log compressor |
| 25 | Done | Deepgram WebSocket STT — real streaming with REST fallback (audio tee'd into both paths so failover never loses utterance) |

---

## Known quirks / gotchas

- **TTS reads markdown literally** — system prompt and `tts/normalizer.py` both strip these; don't reintroduce them.
- **Cerebras free tier rate limit** — 50 req/min on Qwen, 500 on Llama 8B. Hitting it triggers tier fallback (~+500ms).
- **Deepgram `container="none"` must be explicit** — omitting defaults to `"wav"` and the 44-byte header plays as a click at sentence starts.
- **Deepgram WS bool params must be string `"true"`/`"false"`** — Python `True` becomes `"True"` (capitalised) in the URL query and Deepgram returns HTTP 400. SDK type annotations lie about this.
- **Deepgram `utterance_end_ms` requires `interim_results=true`** AND a value ≥1000ms. Otherwise HTTP 400.
- **AudioContext should be 24kHz** — matches Deepgram's output, avoids browser resampling. The worklet does its own pitch-shift resampling on top.
- **React StrictMode disabled** — dev-time double-mount was creating duplicate WebSocket connections.
- **`get_or_fallback_user` resolution order**: session-identified user → `ULTRON_DEFAULT_USER_ID` env → single enrolled non-`_unknown` user → auto-created `default`.
- **Tier 1 doesn't get tools** — Cerebras Llama 8B isn't reliable at structured tool calling.
- **Empty-stream fallback** — tier 3 occasionally returns zero tokens with tools + long history; producer detects and retries without tools.
- **Post-tool text suppression** — once `__TOOL_CALL__` fires, subsequent text tokens from the same turn are discarded so hallucinated trailers can't reach the user.
- **Wake-word loads ~30MB on first activation** — ONNX Runtime Web wasm + 4 model files; cached on subsequent loads.
- **Vite optimizeDeps excludes `onnxruntime-web`** — its WASM loader can't be pre-bundled without breaking runtime path resolution.
- **`public/ort/` and `public/models/` must be present** — copied from `node_modules/onnxruntime-web/dist/` and downloaded from the openWakeWord GitHub release respectively. See the wake-word section above.

---

## Security

File writes are sandboxed to `~/AI_Workspace/`. Sandbox rejects absolute paths, parent-traversal, symlink escape, non-allowlisted extensions, and oversize payloads. Every write is audited to `logs/notes_tool.log` (rotated at 2MB with one generation kept). See `mother/tools/notes_tool.py` and `SECURITY.md`.

Claude's hard safety walls hold for real-world harm requests in conversation — Ultron's contempt is permitted toward fictional figures and humanity-in-the-abstract, never toward real living people or groups.

---

## What's next

- **Cross-session conversation memory** — session-end logs that load on next session start, so Ultron can answer "what were we working on yesterday" naturally
- **Interruption (barge-in)** — cut Ultron off mid-sentence and have his TTS stop + mic re-engage immediately
- **Vision** — wire MQTT events from a webcam / Jetson into Ultron's context so he knows when Winder enters or leaves the room
- **Railway deploy** — persistent volume for memory + FAISS, wss:// via Railway's proxy, CORS + bearer-token gate on public endpoints
- **More tools** — home-assistant toggles, habit tracking, push notifications via Pushover, Google Calendar integration
- **Self-modifying code** — proposal/diff review flow so Ultron can edit his own persona/handlers under a strict denylist
