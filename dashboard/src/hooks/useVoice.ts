/** Hook for WebSocket voice interaction. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { VoiceSocket, type WSEvent } from '../lib/api';
import {
  VoiceFilterChain,
  PRESETS,
  type FilterParams,
  type PresetName,
} from '../lib/audioFilter';
import { scoreIntensity } from '../lib/intensity';
import { RecordingVAD } from '../lib/vad';

export interface VoiceState {
  connected: boolean;
  connecting: boolean;  // true between connect() call and socket.onopen firing
  recording: boolean;
  processing: boolean;
  transcript: string;
  response: string;
  lastEvent: WSEvent | null;
  micError: string | null;  // populated when getUserMedia fails or is blocked
  audioWarning: string | null;  // worklet ring buffer overflow, sped-up playback, etc.
}

// ── Audio playback queue ──
// Keeps TTS segments ordered even when they arrive faster than they play.
let playbackAudioContext: AudioContext | null = null;
let voiceFilter: VoiceFilterChain | null = null;
// Persist user's filter selection across page reloads / navigation
// Bumped version key to invalidate cached filter params from older builds.
// v3: removed comb filter, added presence/lowCut.
// v4: added bodyBoost, chorus, intensityResponsiveness + worklet pitch shift.
// v5: retuned Ultron preset as the default (was close to Subtle).
// v6: further-tuned Ultron preset (stronger presence + chorus, lighter distortion/body).
const STORED_FILTER_KEY = 'mother.voice.filter.v6';
const STORED_FILTER_LEGACY_KEYS = [
  'mother.voice.filter',
  'mother.voice.filter.v1',
  'mother.voice.filter.v2',
  'mother.voice.filter.v3',
  'mother.voice.filter.v4',
  'mother.voice.filter.v5',
];

// Evict old versions of the filter key so they don't sit in
// localStorage forever. Called once at module init.
(function cleanupLegacyFilterKeys() {
  try {
    STORED_FILTER_LEGACY_KEYS.forEach((k) => localStorage.removeItem(k));
  } catch { /* localStorage disabled, private mode, etc. */ }
})();

function loadStoredFilter(): FilterParams {
  try {
    const raw = localStorage.getItem(STORED_FILTER_KEY);
    if (!raw) return PRESETS.Clean;
    const parsed = JSON.parse(raw);
    // Guard against corrupted / third-party-injected values — if the
    // shape is wrong we fall back to defaults rather than feeding
    // garbage numbers into the filter chain (NaN = silent audio).
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return PRESETS.Clean;
    }
    // Validate numeric fields are actually finite numbers; ignore
    // non-numeric entries by falling back to preset defaults for each.
    const merged: any = { ...PRESETS.Clean };
    for (const [k, v] of Object.entries(parsed)) {
      if (typeof v === 'number' && Number.isFinite(v)) merged[k] = v;
      else if (typeof v === 'boolean') merged[k] = v;
      // drop strings / objects / nulls silently
    }
    return merged as FilterParams;
  } catch { /* corrupted JSON */ }
  return PRESETS.Clean;
}

function saveStoredFilter(p: FilterParams) {
  try { localStorage.setItem(STORED_FILTER_KEY, JSON.stringify(p)); } catch { /* ignore */ }
}

let pendingAudio: Array<() => Promise<void>> = [];
let currentSource: AudioBufferSourceNode | null = null;
let audioPlaying = false;
// Dedup window: if the same base64 chunk arrives within 500ms, drop it.
// Protects against accidental duplicate WebSocket connections or retries.
let recentHashes: Map<string, number> = new Map();

function hashB64(b64: string): string {
  // Cheap: first 40 + last 40 chars + length
  return `${b64.slice(0, 40)}:${b64.slice(-40)}:${b64.length}`;
}

async function processAudioQueue() {
  if (audioPlaying || pendingAudio.length === 0) return;
  audioPlaying = true;
  while (pendingAudio.length > 0) {
    const task = pendingAudio.shift()!;
    try { await task(); } catch (e) { console.warn('[TTS] playback error:', e); }
  }
  audioPlaying = false;
}

// ── Streaming PCM playback via AudioWorklet ──
//
// This uses an AudioWorkletNode with an internal ring buffer. The main thread
// pushes PCM samples via messages; the worklet drains the buffer at exact
// sample-accurate timing on the audio thread. No per-chunk AudioBufferSource
// nodes, no scheduling drift, no resampler artifacts — just a single
// continuous audio stream that plays as samples arrive.
//
// This is the canonical solution per Chrome Audio Team's design docs:
// https://developer.chrome.com/blog/audio-worklet-design-pattern

let pcmStreamNode: AudioWorkletNode | null = null;
let workletReady: Promise<void> | null = null;

// Hook instances subscribe here to receive audio-subsystem warnings
// (worklet overflow, context creation errors, etc.) so the UI can
// actually surface them instead of swallowing them to console.
const audioWarningListeners = new Set<(msg: string) => void>();
function notifyAudioWarning(msg: string) {
  audioWarningListeners.forEach((fn) => {
    try { fn(msg); } catch { /* ignore */ }
  });
}

async function ensureAudioContext(): Promise<AudioContext> {
  if (!playbackAudioContext) {
    // Lock context to 24kHz to match Deepgram's TTS output. Otherwise the
    // browser resamples the output, which can introduce artifacts.
    try {
      playbackAudioContext = new AudioContext({ sampleRate: 24000 });
    } catch {
      playbackAudioContext = new AudioContext();
    }
  }
  if (playbackAudioContext.state === 'suspended') {
    playbackAudioContext.resume().catch(() => { /* ignore */ });
  }
  if (!voiceFilter) {
    voiceFilter = new VoiceFilterChain(playbackAudioContext, loadStoredFilter());
  }
  if (!workletReady) {
    // Load the worklet module once — it's served from /pcm-stream-processor.js
    workletReady = playbackAudioContext.audioWorklet.addModule('/pcm-stream-processor.js')
      .then(() => {
        if (!playbackAudioContext) return;
        pcmStreamNode = new AudioWorkletNode(playbackAudioContext, 'pcm-stream-processor', {
          numberOfInputs: 0,
          numberOfOutputs: 1,
          outputChannelCount: [1],
        });
        // Log overflow events from the worklet AND surface to UI so
        // the user knows why their playback sounds wrong instead of
        // guessing.
        pcmStreamNode.port.onmessage = (ev) => {
          if (ev.data?.type === 'overflow') {
            const msg = `Audio buffer overflow — dropped ${ev.data.dropped} samples. Playback may sound sped up.`;
            console.warn('[TTS]', msg);
            notifyAudioWarning(msg);
          }
        };
        pcmStreamNode.connect(voiceFilter!.getInputNode());
        // Initial pitch sync — the worklet was just created, push the
        // current preset's pitchSemitones so the first spoken utterance
        // already plays at the intended depth.
        if (voiceFilter) {
          const p = voiceFilter.getParams();
          const rate = p.enabled ? Math.pow(2, p.pitchSemitones / 12) : 1.0;
          try {
            pcmStreamNode.port.postMessage({ type: 'pitch', rate });
          } catch { /* ignore */ }
        }
      })
      .catch((err) => {
        console.error('[TTS] Failed to load PCM worklet:', err);
        notifyAudioWarning('Audio playback worklet failed to load. TTS will use fallback WAV path.');
        // Reset workletReady so a subsequent call retries instead of
        // forever waiting on the rejected promise. Also clear the node
        // ref so the fallback WAV path doesn't try to use a half-built
        // graph.
        workletReady = null;
        pcmStreamNode = null;
        throw err;
      });
  }
  try {
    await workletReady;
  } catch {
    // Failed to load — return the context anyway so the WAV-blob
    // fallback path can still play audio. The WAV path doesn't use
    // pcmStreamNode so it's unaffected.
  }
  return playbackAudioContext;
}

// ── Public filter API (consumed by the UI) ──

/**
 * Push the current pitchSemitones down to the worklet so speech
 * actually plays deeper/slower. playbackRate on AudioBufferSourceNode
 * only applies to the fallback WAV path — the primary streaming PCM
 * path goes through pcm-stream-processor.js, which has its own
 * interpolated-resample pitch handling. Call this whenever params
 * change or the worklet (re)loads.
 */
function syncPitchToWorklet(params: FilterParams) {
  if (!pcmStreamNode) return;
  const rate = params.enabled
    ? Math.pow(2, params.pitchSemitones / 12)
    : 1.0;
  try {
    pcmStreamNode.port.postMessage({ type: 'pitch', rate });
  } catch {
    // ignore — worklet may not be fully ready on first call
  }
}

export function getFilterParams(): FilterParams {
  if (voiceFilter) return voiceFilter.getParams();
  return loadStoredFilter();
}

export function setFilterParams(params: Partial<FilterParams>) {
  ensureAudioContext();  // lazily create filter + context
  if (!voiceFilter) return;
  voiceFilter.setParams(params);
  saveStoredFilter(voiceFilter.getParams());
  // Keep the PCM worklet in sync with the pitch setting.
  syncPitchToWorklet(voiceFilter.getParams());
}

export function applyFilterPreset(name: PresetName) {
  setFilterParams(PRESETS[name]);
}

export { PRESETS };
export type { FilterParams, PresetName };

/**
 * Push a PCM chunk into the worklet's ring buffer.
 *
 * The worklet maintains a continuous audio stream, draining samples at
 * sample-accurate timing on the audio thread. No per-chunk scheduling,
 * no AudioBufferSourceNode churn, no clicks.
 *
 * If the worklet isn't ready yet (first chunk), the samples are queued
 * and flushed as soon as it loads.
 */
let pendingSamples: Float32Array[] = [];

function playPCMChunk(b64: string, _sampleRate: number) {
  try {
    // Decode base64 → Int16 → Float32
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const pcm16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const samples = pcm16.length;
    if (samples === 0) return;

    const floatBuf = new Float32Array(samples);
    for (let i = 0; i < samples; i++) floatBuf[i] = pcm16[i] / 32768;

    // If worklet is ready, push immediately; otherwise queue up for when it is.
    // Don't use transferable — structured clone is fine and avoids detach errors.
    if (pcmStreamNode) {
      pcmStreamNode.port.postMessage({ type: 'push', samples: floatBuf });
    } else {
      pendingSamples.push(floatBuf);
      // Kick off worklet initialization if not already started
      ensureAudioContext().then(() => {
        if (pcmStreamNode) {
          for (const q of pendingSamples) {
            pcmStreamNode.port.postMessage({ type: 'push', samples: q });
          }
          pendingSamples = [];
        }
      }).catch((err) => {
        console.error('[TTS] Audio context init failed:', err);
      });
    }
  } catch (err) {
    console.error('[TTS] playPCMChunk error:', err);
  }
}

export function stopAllAudio() {
  pendingAudio = [];
  if (currentSource) {
    try { currentSource.stop(); } catch { /* already stopped */ }
    currentSource = null;
  }
  // Tell the worklet to drop its buffered samples so the next utterance
  // doesn't get mixed in with leftover audio.
  if (pcmStreamNode) {
    pcmStreamNode.port.postMessage({ type: 'clear' });
  }
  pendingSamples = [];
  recentHashes.clear();
}

function playAudioFromBase64(b64: string) {
  // Drop duplicates that arrive within 3 seconds
  const hash = hashB64(b64);
  const now = Date.now();
  const last = recentHashes.get(hash);
  if (last && now - last < 3000) {
    console.warn('[TTS] Dropping duplicate audio chunk');
    return;
  }
  recentHashes.set(hash, now);
  // Clean old entries. Evict on EVERY insert beyond the 10s window so
  // long-idle sessions don't carry ghost entries forever; also cap
  // absolute size at 100 as a belt-and-braces guard.
  const cutoff = now - 10000;
  for (const [k, t] of recentHashes) {
    if (t < cutoff) recentHashes.delete(k);
  }
  if (recentHashes.size > 100) {
    // Map preserves insertion order — delete oldest first.
    const excess = recentHashes.size - 100;
    let i = 0;
    for (const k of recentHashes.keys()) {
      if (i++ >= excess) break;
      recentHashes.delete(k);
    }
  }

  pendingAudio.push(async () => {
    const ctx = await ensureAudioContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    // Decode base64 → ArrayBuffer
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const buffer = await ctx.decodeAudioData(bytes.buffer);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    // Route through filter (same as streaming path)
    const rate = voiceFilter ? voiceFilter.getPitchRate() : 1;
    source.playbackRate.value = rate;
    if (voiceFilter) {
      source.connect(voiceFilter.getInputNode());
    } else {
      source.connect(ctx.destination);
    }
    currentSource = source;
    source.start();
    // Wait for this clip to finish before playing the next
    await new Promise<void>((resolve) => {
      source.onended = () => {
        if (currentSource === source) currentSource = null;
        resolve();
      };
    });
  });
  processAudioQueue();
}

// ── Adaptive endpointing ──
// The VAD's silence window is the single largest fixed delay at the end
// of every turn. A flat 700ms is tuned for the worst case (speaker
// pausing mid-sentence); when the live transcript already reads as a
// complete utterance we can stop sooner, and when it clearly trails
// off mid-clause we should wait longer.
const VAD_SILENCE_DEFAULT_MS = 700;

// Words that signal the speaker is mid-clause — if the interim
// transcript ends on one of these, they're almost certainly not done.
const TRAILING_INCOMPLETE = new Set([
  'and', 'or', 'but', 'so', 'the', 'a', 'an', 'to', 'of', 'in', 'on',
  'at', 'for', 'with', 'my', 'your', 'his', 'her', 'their', 'its',
  'is', 'are', 'was', 'were', 'if', 'then', 'that', 'what', 'whats',
  'how', 'who', 'when', 'where', 'why', 'uh', 'um', 'plus', 'minus',
  'times', 'versus', 'about', 'from', 'by', 'as', 'than',
]);

function silenceMsForTranscript(text: string): number {
  const t = text.trim();
  if (!t) return VAD_SILENCE_DEFAULT_MS;
  const words = t.toLowerCase().replace(/[.,!?]+$/, '').split(/\s+/);
  if (words.length < 3) return VAD_SILENCE_DEFAULT_MS;
  const last = words[words.length - 1];
  if (TRAILING_INCOMPLETE.has(last)) return 850;  // clearly mid-clause
  // Floor at 550/650 (not 450/550): the backend dispatches only 250ms
  // after our end-of-speech signal, and Deepgram punctuates interims
  // at natural breath pauses. 450ms + 250ms stacked was cutting off
  // multi-sentence utterances mid-thought ("Search for Tesla news."
  // <breath> "...from this week") — the two margins must not both be
  // aggressive at once.
  if (/[.!?]$/.test(t)) return 550;               // Deepgram punctuated it — done
  return 650;                                     // plausible complete clause
}

export function useVoice() {
  const [state, setState] = useState<VoiceState>({
    connected: false,
    connecting: false,
    recording: false,
    processing: false,
    transcript: '',
    response: '',
    lastEvent: null,
    micError: null,
    audioWarning: null,
  });

  const socketRef = useRef<VoiceSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  // Muted sink for the mic processor — keeps the graph driving without
  // routing audio back to the speakers (would cause feedback).
  const sinkRef = useRef<GainNode | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  // Browser-side Silero VAD. One instance per hook, reused across
  // recordings — reset() is called at the start of each turn. Loaded
  // lazily on first recording so initial page-load isn't slowed by
  // another ORT model fetch.
  const vadRef = useRef<RecordingVAD | null>(null);

  const handleEvent = useCallback((event: WSEvent) => {
    setState((s) => {
      const updates: Partial<VoiceState> = { lastEvent: event };

      switch (event.event) {
        case 'stt': {
          const text = (event.text as string) || '';
          updates.transcript = text;
          if (!event.final) {
            // Adaptive endpointing: tune the VAD's silence window to
            // how complete the live transcript looks. No-op when no
            // VAD is active (push-to-talk).
            vadRef.current?.setSilenceDurationMs(silenceMsForTranscript(text));
          }
          if (event.final) {
            if (text.trim()) {
              // Got a transcript — backend is now generating a response
              updates.processing = true;
            } else {
              // Empty transcript — backend won't send llm_done; clear processing
              updates.processing = false;
            }
          }
          break;
        }
        case 'llm_token':
          updates.response = s.response + (event.token as string);
          break;
        case 'llm_done':
          updates.response = event.full_text as string;
          updates.processing = false;
          break;
        case 'tts_ready': {
          // Legacy: full WAV blob (Kokoro/Piper fallback path)
          const b64 = event.audio_b64 as string;
          if (b64) playAudioFromBase64(b64);
          break;
        }
        case 'tts_start': {
          // New sentence starting. Don't reset the schedule here —
          // chunks naturally queue onto the tail of previous audio.
          // Ensure the AudioContext is resumed (for autoplay policy compliance).
          ensureAudioContext();
          // Per-sentence intensity modulation: score the text the
          // backend is about to speak, feed it to the filter chain,
          // and let the filter parameters cross-fade toward the
          // sentence's dramatic temperature. Subtle on Clean/Subtle,
          // dramatic on Menacing (which has intensityResponsiveness=0.8).
          const sentText = (event.text as string) || '';
          if (sentText && voiceFilter) {
            const i = scoreIntensity(sentText);
            voiceFilter.setIntensity(i);
          }
          break;
        }
        case 'tts_chunk': {
          // Raw PCM chunk — append to gap-free stream playback
          const b64 = event.pcm_b64 as string;
          const sr = (event.sample_rate as number) || 24000;
          if (b64) playPCMChunk(b64, sr);
          break;
        }
        case 'tts_end': {
          // Sentence complete — worklet continues playing out its ring buffer.
          // Nothing to do here; the audio tail streams out naturally.
          break;
        }
        case 'cancelled': {
          // Server confirmed barge-in. Drop whatever's still buffered
          // in the worklet so we don't keep speaking after the user
          // cut us off.
          stopAllAudio();
          updates.processing = false;
          updates.response = '';
          break;
        }
        case 'recording_started':
          updates.recording = true;
          updates.micError = null;  // clear stale error now that it works
          break;
        case 'recording_stopped':
          updates.recording = false;
          // Don't flip processing here. The `stt` event with final=true
          // already sets processing=true the moment a transcript is
          // ready. By the time recording_stopped arrives, the LLM may
          // already be streaming — re-flipping processing would make
          // the UI flash "thinking" after an answer that's already
          // underway. If the user aborts (no speech), the empty-
          // transcript branch of `stt` clears processing correctly.
          break;
        case 'connected':
          // Real socket-open event — see VoiceSocket.onopen. Previously
          // we flipped `connected=true` optimistically on connect(),
          // which raced with startRecording() before the WS handshake
          // finished.
          updates.connected = true;
          updates.connecting = false;
          break;
        case 'disconnected':
          updates.connected = false;
          updates.connecting = false;
          break;
        case 'error':
          updates.processing = false;
          updates.connecting = false;
          break;
      }

      return { ...s, ...updates };
    });
  }, []);

  const connect = useCallback(() => {
    if (socketRef.current) return;
    const sock = new VoiceSocket(handleEvent);
    sock.connect();
    socketRef.current = sock;
    // Mark `connecting`, not `connected` — connected flips true only
    // when the real WebSocket `onopen` event fires (see handleEvent).
    setState((s) => ({ ...s, connecting: true }));
  }, [handleEvent]);

  const disconnect = useCallback(() => {
    socketRef.current?.disconnect();
    socketRef.current = null;
    setState((s) => ({ ...s, connected: false, connecting: false }));
  }, []);

  /**
   * Start recording. Optional `onSilenceStop` callback fires when the
   * browser's Silero VAD detects end-of-speech — used by the wake-word
   * flow to auto-stop the mic without waiting for the 5s ceiling or
   * the server's STT. For push-to-talk, omit the callback: the user's
   * key release handles stopping.
   */
  const startRecording = useCallback(async (
    onSilenceStop?: () => void,
  ) => {
    const sock = socketRef.current;
    if (!sock) return;
    // Wait for the WS handshake if it hasn't completed yet. This is the
    // critical fix — previously we bailed out silently when the socket
    // wasn't open yet, so the first press of the mic button did nothing
    // on cold start.
    try {
      await sock.ready;
    } catch {
      setState((s) => ({ ...s, micError: 'Cannot connect to backend.' }));
      return;
    }
    if (!sock.connected) {
      setState((s) => ({ ...s, micError: 'Backend socket not open.' }));
      return;
    }
    // Stop any in-progress TTS playback when user starts speaking
    stopAllAudio();

    // Pre-warm the playback pipeline while the user is still talking.
    // We're inside a user gesture here, so the AudioContext can be
    // created/resumed and the PCM worklet module fetched NOW — instead
    // of paying that 50-200ms setup when the first TTS chunk arrives.
    ensureAudioContext().catch(() => { /* fallback path handles it */ });

    // Prep VAD in parallel with mic permission. Only instantiate when
    // a silence callback is requested (push-to-talk doesn't need it).
    // Reuse the instance across recordings — setCallback() swaps the
    // closure without rebuilding the ONNX session (saves 50-150ms
    // per wake-word activation).
    let vad: RecordingVAD | null = null;
    if (onSilenceStop) {
      const wrappedCallback = () => {
        // eslint-disable-next-line no-console
        console.log('[vad] end of speech → stopRecording');
        onSilenceStop();
      };
      if (!vadRef.current) {
        vadRef.current = new RecordingVAD({
          speechThreshold: 0.5,
          silenceDurationMs: 700,  // stop 700ms after last speech
          minSpeechMs: 250,        // require at least 250ms of speech
          // 45s ceiling: the VAD's silence detection ends normal turns;
          // this only caps runaway noise. The old 10s cap cut off long
          // questions while the user was mid-sentence.
          maxDurationMs: 45000,
          onEndOfSpeech: wrappedCallback,
        });
      } else {
        vadRef.current.setCallback(wrappedCallback);
      }
      vad = vadRef.current;
      // Fire-and-forget load — first call pays ~100ms, subsequent
      // recordings are instant.
      vad.load().catch((err) => {
        // eslint-disable-next-line no-console
        console.warn('[vad] load failed, auto-stop on silence disabled:', err);
      });
      vad.reset();
      // Fresh turn — restore the default silence window; the previous
      // turn's adaptive adjustment must not leak into this one.
      vad.setSilenceDurationMs(VAD_SILENCE_DEFAULT_MS);
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
      });
      mediaStreamRef.current = stream;

      const ctx = new AudioContext({ sampleRate: 16000 });
      contextRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      // ScriptProcessor for raw PCM — will migrate to AudioWorklet later.
      // 1024 samples = 64ms at 16kHz. The buffer size sets the floor on
      // stopRecording's tail wait (we must let one full buffer flush
      // before teardown): at 4096 that was ~256ms of dead time appended
      // to EVERY turn. 64ms buffers still hold 2 full Silero VAD frames
      // and the ~2ms inference fits the callback budget comfortably.
      const processor = ctx.createScriptProcessor(1024, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const data = e.inputBuffer.getChannelData(0);
        // Send to server first so the network pipeline isn't delayed
        // by VAD inference.
        socketRef.current?.sendAudio(data.buffer.slice(0));
        // Feed VAD (only if enabled for this recording). Runs on the
        // main thread but Silero inference is ~2ms per 32ms frame so
        // it comfortably fits inside the ScriptProcessor budget.
        if (vad) {
          // Copy because the underlying AudioBuffer gets reused each callback.
          const copy = new Float32Array(data.length);
          copy.set(data);
          vad.process(copy).catch(() => { /* swallow per-frame errors */ });
        }
      };

      source.connect(processor);
      // ScriptProcessorNode requires SOMETHING downstream to drive its
      // onaudioprocess callback. Routing the mic to ctx.destination
      // would echo it out the speakers (acoustic feedback on any
      // machine without a headset). Route through a muted gain node
      // instead — keeps the graph alive without producing output.
      const sink = ctx.createGain();
      sink.gain.value = 0;
      processor.connect(sink);
      sink.connect(ctx.destination);
      sinkRef.current = sink;

      socketRef.current?.send('start');
      setState((s) => ({ ...s, transcript: '', response: '', micError: null }));
    } catch (err) {
      // Surface permission / hardware errors so the user knows why
      // nothing happened. getUserMedia throws on denied permission,
      // no mic device, or secure-context violations (HTTP on LAN).
      const message = err instanceof Error ? err.message : String(err);
      console.error('Mic access failed:', err);
      setState((s) => ({
        ...s,
        micError: message || 'Microphone access failed.',
      }));
    }
  }, []);

  const stopRecording = useCallback(() => {
    // Idempotent — safe to call from both VAD and server-STT final
    // paths without double-sending the stop message or double-closing
    // the AudioContext (AudioContext.close() on a closed context
    // throws in some browsers).
    const proc = processorRef.current;
    const ctx = contextRef.current;
    const stream = mediaStreamRef.current;
    const sink = sinkRef.current;
    if (!proc && !stream) return;

    // ── Tail capture before cutoff ──
    // The ScriptProcessorNode buffers 1024 samples (~64 ms at
    // 16 kHz) before each onaudioprocess fires. If we disconnect
    // immediately, that pending buffer is dropped — which is why
    // releasing PTT mid-word loses the last syllable. Hold the
    // audio path open for one full buffer period plus a small
    // margin, let the next callback flush, THEN tear down and
    // signal stop. The server won't close Deepgram's input stream
    // until it sees `stop`, so the trailing chunk gets transcribed
    // properly. This value must stay > the processor buffer period
    // (64ms) — it was 300ms when the buffer was 4096 samples, which
    // added ~a quarter second of dead air to every single turn.
    const TAIL_MS = 120;
    setTimeout(() => {
      try { proc?.disconnect(); } catch { /* already disconnected */ }
      try { sink?.disconnect(); } catch { /* already disconnected */ }
      try { ctx?.close(); } catch { /* already closed */ }
      stream?.getTracks().forEach((t) => t.stop());
      // Null out refs only AFTER the teardown so a re-press during
      // the tail window is rejected by the startRecording guard
      // (we don't want two overlapping audio contexts).
      processorRef.current = null;
      contextRef.current = null;
      mediaStreamRef.current = null;
      sinkRef.current = null;
      // Send stop AFTER the trailing chunk has had time to leave
      // the browser and reach Deepgram. Otherwise the server-side
      // close-stream beats the last audio packet.
      socketRef.current?.send('stop');
    }, TAIL_MS);
  }, []);

  const sendPrompt = useCallback(async (text: string) => {
    const sock = socketRef.current;
    if (!sock) return;
    try {
      await sock.ready;
    } catch {
      return;
    }
    // Stop any in-progress TTS when sending a new prompt
    stopAllAudio();
    // Pre-warm playback (context + worklet) while the LLM thinks.
    ensureAudioContext().catch(() => { /* fallback path handles it */ });
    setState((s) => ({ ...s, transcript: text, response: '', processing: true }));
    sock.sendPrompt(text);
  }, []);

  // Auto-connect on mount. Without this, the first press of the mic
  // button triggers connect() AND startRecording() in the same tick,
  // but the WebSocket handshake is async (~50-300ms) so the recording
  // attempt silently no-ops. Opening the socket at mount time makes
  // the first press feel instant.
  useEffect(() => {
    if (!socketRef.current) {
      const sock = new VoiceSocket(handleEvent);
      sock.connect();
      socketRef.current = sock;
      setState((s) => ({ ...s, connecting: true }));
    }
    // Subscribe to audio-subsystem warnings (worklet overflow etc).
    const onWarn = (msg: string) => {
      setState((s) => ({ ...s, audioWarning: msg }));
    };
    audioWarningListeners.add(onWarn);
    return () => {
      audioWarningListeners.delete(onWarn);
      socketRef.current?.disconnect();
      socketRef.current = null;
    };
    // handleEvent is stable (useCallback([])) so we only run once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /**
   * Barge-in: stop whatever Ultron is saying right now and cancel the
   * in-flight LLM/TTS pipeline. Safe to call any time — it's a no-op
   * when nothing is playing. Stops local playback immediately so we
   * don't have to wait for the server's `cancelled` confirmation.
   */
  const interrupt = useCallback(() => {
    stopAllAudio();
    socketRef.current?.send('cancel');
    setState((s) => ({ ...s, processing: false, response: '' }));
  }, []);

  return {
    ...state,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    sendPrompt,
    interrupt,
  };
}
