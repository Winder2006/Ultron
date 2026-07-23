import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import NeuralWeb from './components/NeuralWeb';
import StatusPanel from './components/StatusPanel';
import ConversationFeed from './components/ConversationFeed';
import FilterPanel from './components/FilterPanel';
import ObservabilityPanel from './components/ObservabilityPanel';
import ExecPanel from './components/ExecPanel';
import { useMotherAPI } from './hooks/useMotherAPI';
import { useVoice } from './hooks/useVoice';
import { useWakeWord } from './hooks/useWakeWord';
import type { OrbState } from './components/NeuralWeb';

// Persist the wake-word toggle so it survives refreshes. Wake word
// defaults to OFF — it needs mic permission, which we don't want to
// prompt for on first load without user intent.
const WAKE_TOGGLE_KEY = 'mother.wakeword.enabled.v1';

export default function App() {
  const mother = useMotherAPI();
  const voice = useVoice();
  const [showPanels, setShowPanels] = useState(false);
  const [showExec, setShowExec] = useState(false);
  const [time, setTime] = useState(new Date());

  // Auto-open the code window the moment a code_exec event lands —
  // the whole point of running code visibly is lost if it renders on
  // a route the user never has open. Track the latest exec timestamp
  // so re-renders don't re-open a panel the user just closed.
  const lastExecTsRef = useRef<number | null>(null);
  useEffect(() => {
    const latest = mother.events.find((e) => e.type === 'code_exec');
    if (!latest) return;
    const ts = (latest.ts as number) ?? 0;
    if (lastExecTsRef.current === ts) return;
    lastExecTsRef.current = ts;
    setShowExec(true);
  }, [mother.events]);

  // Esc closes the topmost layer: the code window if it's open,
  // otherwise the slide-out panels.
  useEffect(() => {
    if (!showExec && !showPanels) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (showExec) setShowExec(false);
      else setShowPanels(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showExec, showPanels]);

  const [wakeEnabled, setWakeEnabled] = useState<boolean>(() => {
    try { return localStorage.getItem(WAKE_TOGGLE_KEY) === '1'; }
    catch { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem(WAKE_TOGGLE_KEY, wakeEnabled ? '1' : '0'); }
    catch { /* ignore quota */ }
  }, [wakeEnabled]);

  // Auto-stop timer for wake-word triggered recording. Push-to-talk
  // has a key-release event; wake-word doesn't, so we bound the
  // recording length and let the backend's empty-transcript guard
  // handle true silence gracefully.
  const autoStopTimerRef = useRef<number | null>(null);
  // True while the CURRENT recording was started by the wake word.
  // The stop-on-final-transcript effect below must only apply to
  // wake-word turns: during push-to-talk the user is still holding
  // the button, and Deepgram emits a "final" at any natural pause —
  // auto-stopping there cut off multi-sentence input mid-thought.
  const wakeTurnRef = useRef(false);
  const clearAutoStop = useCallback(() => {
    if (autoStopTimerRef.current !== null) {
      window.clearTimeout(autoStopTimerRef.current);
      autoStopTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  // useVoice now auto-connects on mount and its sendPrompt/startRecording
  // awaits the socket's ready promise internally, so no more setTimeout
  // gymnastics here.
  const handleSendPrompt = useCallback((text: string) => {
    voice.sendPrompt(text);
  }, [voice]);

  const handleMicPress = useCallback(() => {
    clearAutoStop();  // user took over from wake-word — cancel timer
    wakeTurnRef.current = false;  // push-to-talk: user controls the stop
    // Barge-in: if Ultron is currently speaking or thinking, cut him
    // off before opening the mic. The interrupt is local-instant so
    // playback stops the moment the button is pressed.
    if (voice.processing) {
      voice.interrupt();
    }
    voice.startRecording();
  }, [voice, clearAutoStop]);

  const handleMicRelease = useCallback(() => {
    clearAutoStop();
    voice.stopRecording();
  }, [voice, clearAutoStop]);

  // ── Wake word ──
  // Opt-in. While enabled, a dedicated mic stream runs openWakeWord
  // in the browser (no server involvement). On detection, auto-start
  // recording and arm a 7s auto-stop — the backend's utterance-end
  // detection tends to finalize earlier, so this is a hard ceiling.
  const wake = useWakeWord({
    // Lower threshold = more sensitive. The official openwakeword-js
    // demo uses 0.09 for their demo models; the hey_jarvis model is
    // less forgiving so 0.4 is a reasonable starting point.
    threshold: 0.4,
    // Short debounce so double-utterances don't both fire, but fast
    // enough to feel responsive if user tries again after a miss.
    debounceSeconds: 1.5,
    // Suppress only while we're already recording or the WS isn't
    // open. We DELIBERATELY allow wake-word to fire while Ultron is
    // mid-response so the user can interrupt by name — barge-in.
    // Echo cancellation on the mic stream + the voice filter chain's
    // distinctive timbre keep self-triggering rare in practice; if
    // false-fires become a problem on speaker setups, gate this with
    // a settings flag.
    isSuppressed: () =>
      voice.recording || !voice.connected,
    onDetected: () => {
      if (voice.recording) return;
      // If Ultron is currently speaking/thinking, this wake-word fire
      // is a barge-in. Cut him off first so the new turn lands clean.
      if (voice.processing) {
        voice.interrupt();
      }
      clearAutoStop();
      wakeTurnRef.current = true;  // hands-free turn — auto-stop applies
      // Pass a silence callback so the browser's Silero VAD auto-stops
      // as soon as the user finishes talking (typically ~700ms after
      // last syllable) — no more waiting for the 5s ceiling or for
      // the server-side STT to emit a final. This is the fix that
      // makes wake-word turns feel instant instead of sluggish.
      voice.startRecording(() => {
        voice.stopRecording();
        clearAutoStop();
      });
      // 30s hard ceiling — a true safety net, not a turn length. The
      // old 6s ceiling chopped any long question mid-sentence; with
      // the VAD handling normal end-of-speech, this only trips on
      // constant loud background noise defeating the VAD.
      autoStopTimerRef.current = window.setTimeout(() => {
        voice.stopRecording();
        autoStopTimerRef.current = null;
      }, 30000);
    },
  });

  // Auto-stop the moment Deepgram reports a final transcript — that's
  // the authoritative signal that the user finished speaking. Cuts
  // typical wake-word turnaround from 5-7s down to ~1s after the last
  // syllable, which is what every competent voice assistant does.
  // WAKE-WORD TURNS ONLY: during push-to-talk the user is still
  // holding the button, and Deepgram finalizes at every natural pause
  // — auto-stopping there truncated multi-sentence input.
  useEffect(() => {
    const ev = voice.lastEvent;
    if (!ev || ev.event !== 'stt' || !ev.final) return;
    if (!voice.recording) return;
    if (!wakeTurnRef.current) return;
    // FALLBACK stopper only. Deepgram emits a "final" at every ~300ms
    // breath pause, so stopping 150ms after one was cutting off
    // multi-sentence questions mid-thought. The Silero VAD (adaptive
    // 550-850ms silence) is the primary end-of-turn signal; this
    // timer only matters when the VAD failed to load. 1.2s of silence
    // after a final means the user genuinely stopped — and if they
    // kept talking, the next audio makes Deepgram emit new events and
    // this effect re-arms with a fresh timer.
    const t = window.setTimeout(() => {
      voice.stopRecording();
      clearAutoStop();
    }, 1200);
    return () => window.clearTimeout(t);
  }, [voice.lastEvent, voice.recording, voice, clearAutoStop]);

  useEffect(() => {
    // Only auto-start when the user just flipped the toggle ON. Don't
    // re-start whenever wake.listening flips false (e.g. from an
    // error) — that would create an infinite retry loop on a model
    // that's failing to load.
    if (wakeEnabled && !wake.listening && !wake.initializing && !wake.error) {
      wake.start();
    } else if (!wakeEnabled && wake.listening) {
      wake.stop();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wakeEnabled]);

  // If the user grabs spacebar mid-wake-session, kill the auto-stop.
  useEffect(() => { if (!voice.recording) clearAutoStop(); }, [voice.recording, clearAutoStop]);

  // Spacebar push-to-talk
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat && !voice.recording) {
        // Don't trigger if user is typing in the text input
        const active = document.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) return;
        e.preventDefault();
        handleMicPress();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space' && voice.recording) {
        e.preventDefault();
        handleMicRelease();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [handleMicPress, handleMicRelease, voice.recording]);

  const orbState: OrbState = useMemo(() => {
    if (voice.recording) return 'listening';
    if (voice.processing) return 'thinking';
    if (mother.activity > 0.5) return 'speaking';
    return 'idle';
  }, [voice.recording, voice.processing, mother.activity]);

  const lastQuery = voice.transcript || mother.lastQuery;
  const lastResponse = voice.response || mother.lastResponse;

  const statusLabel = useMemo(() => {
    switch (orbState) {
      case 'listening': return 'listening...';
      case 'thinking': return 'thinking...';
      case 'speaking': return '';
      default: return '';
    }
  }, [orbState]);

  return (
    <>
      {/* Fullscreen particle canvas */}
      <NeuralWeb state={orbState} activity={mother.activity} />

      {/* Scan-line overlay */}
      <div style={{
        position: 'fixed',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 5,
        background: `repeating-linear-gradient(
          0deg,
          transparent,
          transparent 2px,
          rgba(0, 0, 0, 0.03) 2px,
          rgba(0, 0, 0, 0.03) 4px
        )`,
      }} />

      {/* Top-right controls. zIndex must stay ABOVE the slide-out
          panels (30) — at 20 the right panel covered these buttons,
          so once the panels were open there was no visible way to
          close them again. Kept below the exec overlay (40), which
          has its own close controls. */}
      <div style={{
        position: 'fixed',
        top: 16,
        right: 16,
        display: 'flex',
        gap: 8,
        zIndex: 35,
      }}>
        {/* Mic button — hold to talk */}
        <button
          onMouseDown={handleMicPress}
          onMouseUp={handleMicRelease}
          onMouseLeave={handleMicRelease}
          onTouchStart={handleMicPress}
          onTouchEnd={handleMicRelease}
          title="Hold to speak (or hold Spacebar)"
          style={{
            width: 36,
            height: 36,
            border: voice.recording
              ? '1px solid rgba(255, 51, 68, 0.6)'
              : '1px solid rgba(255, 51, 68, 0.15)',
            borderRadius: 8,
            background: voice.recording
              ? 'rgba(255, 51, 68, 0.25)'
              : 'rgba(255, 51, 68, 0.05)',
            color: voice.recording
              ? '#ff3344'
              : 'rgba(255, 51, 68, 0.5)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: 'var(--font-mono)',
            transition: 'all 0.15s',
            boxShadow: voice.recording ? '0 0 16px rgba(255,51,68,0.4)' : 'none',
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="23"/>
          </svg>
        </button>
        {/* Wake-word toggle. Opt-in; click to activate continuous
            listening for "Hey Jarvis". Shows amber while initializing,
            green when listening, red if an error stopped it. */}
        <button
          onClick={() => setWakeEnabled((v) => !v)}
          title={
            wake.error
              ? `Wake word error: ${wake.error}`
              : wakeEnabled
              ? 'Wake word ON — say "Hey Jarvis" to summon Ultron'
              : 'Click to enable "Hey Jarvis" wake word'
          }
          style={{
            width: 36,
            height: 36,
            borderRadius: 8,
            border: `1px solid ${
              wake.error
                ? 'rgba(255, 51, 68, 0.6)'
                : wake.listening
                ? 'rgba(0, 255, 136, 0.5)'
                : wake.initializing
                ? 'rgba(255, 204, 68, 0.5)'
                : 'rgba(255, 51, 68, 0.15)'
            }`,
            background: wake.listening
              ? 'rgba(0, 255, 136, 0.12)'
              : wake.initializing
              ? 'rgba(255, 204, 68, 0.12)'
              : wake.error
              ? 'rgba(255, 51, 68, 0.12)'
              : 'rgba(255, 51, 68, 0.05)',
            color: wake.listening
              ? 'rgba(0, 255, 136, 0.9)'
              : wake.initializing
              ? 'rgba(255, 204, 68, 0.9)'
              : wake.error
              ? 'rgba(255, 51, 68, 0.9)'
              : 'rgba(255, 51, 68, 0.5)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 10,
            letterSpacing: 1.5,
            fontFamily: 'var(--font-mono)',
            transition: 'all 0.15s',
            fontWeight: 600,
            boxShadow: wake.listening
              ? '0 0 12px rgba(0, 255, 136, 0.35)'
              : 'none',
            animation: wake.initializing
              ? 'ultron-pulse 1s ease-in-out infinite'
              : undefined,
          }}
        >
          HJ
        </button>
        {/* Code window toggle — same cells as /exec, on this screen */}
        <button
          onClick={() => setShowExec((v) => !v)}
          title="Toggle code window (auto-opens when Ultron runs Python)"
          style={{
            width: 36,
            height: 36,
            border: showExec
              ? '1px solid rgba(92, 255, 142, 0.5)'
              : '1px solid rgba(255, 51, 68, 0.15)',
            borderRadius: 8,
            background: showExec ? 'rgba(92, 255, 142, 0.10)' : 'rgba(255, 51, 68, 0.05)',
            color: showExec ? 'rgba(92, 255, 142, 0.9)' : 'rgba(255, 51, 68, 0.5)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 11,
            fontFamily: 'var(--font-mono)',
            fontWeight: 600,
            letterSpacing: 0.5,
            transition: 'all 0.15s',
          }}
        >
          {'>_'}
        </button>
        <button
          onClick={() => setShowPanels(!showPanels)}
          title="Toggle panels"
          style={{
            width: 36,
            height: 36,
            border: '1px solid rgba(255, 51, 68, 0.15)',
            borderRadius: 8,
            background: showPanels ? 'rgba(255, 51, 68, 0.12)' : 'rgba(255, 51, 68, 0.05)',
            color: 'rgba(255, 51, 68, 0.5)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 14,
            fontFamily: 'var(--font-mono)',
            transition: 'all 0.2s',
          }}
        >
          {showPanels ? '×' : '☰'}
        </button>
      </div>

      {/* Status text — centered bottom */}
      {statusLabel && (
        <div style={{
          position: 'fixed',
          bottom: 40,
          left: '50%',
          transform: 'translateX(-50%)',
          color: 'rgba(255, 68, 85, 0.5)',
          fontSize: 13,
          letterSpacing: 2,
          textTransform: 'lowercase',
          fontWeight: 300,
          pointerEvents: 'none',
          zIndex: 10,
          transition: 'opacity 0.5s ease',
        }}>
          {statusLabel}
        </div>
      )}

      {/* ULTRON label — centered bottom */}
      <div style={{
        position: 'fixed',
        bottom: 16,
        left: '50%',
        transform: 'translateX(-50%)',
        color: 'rgba(255, 51, 68, 0.2)',
        fontSize: 10,
        letterSpacing: 4,
        textTransform: 'uppercase',
        fontWeight: 300,
        fontFamily: 'var(--font-display)',
        pointerEvents: 'none',
        zIndex: 10,
      }}>
        ULTRON
      </div>

      {/* Connection status — top left.
          Two dots: REST API status (green/red) and Voice WebSocket
          status (green/amber/red). Without the second dot, the page
          looked "online" even when the WS handshake had failed — the
          user would press space and nothing would happen, with no
          visible reason. */}
      <div style={{
        position: 'fixed',
        top: 16,
        left: 20,
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        zIndex: 20,
        pointerEvents: 'none',
      }}>
        <div style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          background: mother.connected ? '#00ff88' : '#ff3344',
          boxShadow: mother.connected
            ? '0 0 8px rgba(0,255,136,0.5)'
            : '0 0 8px rgba(255,51,68,0.5)',
        }} title={mother.connected ? 'REST API reachable' : 'REST API unreachable'} />
        <span style={{
          fontSize: 9,
          color: mother.connected ? 'rgba(0,255,136,0.5)' : 'rgba(255,51,68,0.4)',
          letterSpacing: 2,
          textTransform: 'uppercase',
          fontWeight: 300,
        }}>
          {mother.connected ? 'nominal' : 'offline'}
        </span>
        {/* Voice WebSocket indicator. Distinct from REST because one
            can be up while the other is down (e.g. if the WS endpoint
            crashes but the health endpoint still answers). */}
        <div style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          marginLeft: 4,
          background: voice.connected
            ? '#00ff88'
            : voice.connecting
            ? '#ffcc44'
            : '#ff3344',
          boxShadow: voice.connected
            ? '0 0 8px rgba(0,255,136,0.5)'
            : voice.connecting
            ? '0 0 8px rgba(255,204,68,0.6)'
            : '0 0 8px rgba(255,51,68,0.5)',
          animation: voice.connecting ? 'ultron-pulse 1s ease-in-out infinite' : undefined,
        }} title={voice.connected ? 'Voice WS open' : voice.connecting ? 'Voice WS connecting' : 'Voice WS offline'} />
        <span style={{
          fontSize: 9,
          color: voice.connected
            ? 'rgba(0,255,136,0.5)'
            : voice.connecting
            ? 'rgba(255,204,68,0.7)'
            : 'rgba(255,51,68,0.4)',
          letterSpacing: 2,
          textTransform: 'uppercase',
          fontWeight: 300,
        }}>
          {voice.connected ? 'voice' : voice.connecting ? 'linking' : 'no voice'}
        </span>
        <span style={{
          fontSize: 9,
          color: 'rgba(255,255,255,0.15)',
          letterSpacing: 1,
          marginLeft: 8,
          fontFamily: 'var(--font-mono)',
        }}>
          {time.toLocaleTimeString([], { hour12: false })}
        </span>
      </div>

      {/* Mic / backend error banner — shown only when the user tried to
          do something and couldn't. Non-intrusive, auto-dismisses when
          the underlying condition clears. */}
      {voice.micError && (
        <div style={{
          position: 'fixed',
          bottom: 70,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(255, 51, 68, 0.12)',
          border: '1px solid rgba(255, 51, 68, 0.35)',
          borderRadius: 6,
          padding: '8px 14px',
          fontSize: 11,
          color: 'rgba(255, 180, 185, 0.9)',
          letterSpacing: 1,
          fontFamily: 'var(--font-mono)',
          zIndex: 25,
          maxWidth: 420,
          textAlign: 'center',
        }}>
          {voice.micError}
        </div>
      )}

      {voice.audioWarning && (
        <div style={{
          position: 'fixed',
          bottom: voice.micError ? 108 : 70,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(255, 170, 51, 0.10)',
          border: '1px solid rgba(255, 170, 51, 0.30)',
          borderRadius: 6,
          padding: '8px 14px',
          fontSize: 11,
          color: 'rgba(255, 220, 170, 0.85)',
          letterSpacing: 1,
          fontFamily: 'var(--font-mono)',
          zIndex: 25,
          maxWidth: 420,
          textAlign: 'center',
        }}>
          {voice.audioWarning}
        </div>
      )}

      {/* No centered text overlay — conversation panel shows transcript/response */}

      {/* Slide-out panels */}
      {showPanels && (
        <>
          {/* Left: Status */}
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: 280,
            height: '100%',
            background: 'rgba(8, 10, 18, 0.92)',
            borderRight: '1px solid rgba(255, 51, 68, 0.1)',
            backdropFilter: 'blur(24px)',
            zIndex: 30,
            padding: '60px 12px 12px',
            overflowY: 'auto',
            transition: 'transform 0.3s ease',
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
          }}>
            <StatusPanel
              status={mother.status}
              users={mother.users}
              connected={mother.connected}
            />
            <FilterPanel />
            <ObservabilityPanel events={mother.events} />
          </div>

          {/* Right: Conversation */}
          <div style={{
            position: 'fixed',
            top: 0,
            right: 0,
            width: 320,
            height: '100%',
            background: 'rgba(8, 10, 18, 0.92)',
            borderLeft: '1px solid rgba(255, 51, 68, 0.1)',
            backdropFilter: 'blur(24px)',
            zIndex: 30,
            display: 'flex',
            flexDirection: 'column',
            paddingTop: 56,
          }}>
            <ConversationFeed
              events={mother.events}
              lastQuery={lastQuery}
              lastResponse={lastResponse}
              lastIntent={mother.lastIntent}
              onSendPrompt={handleSendPrompt}
              processing={voice.processing}
            />
          </div>
        </>
      )}

      {/* Code execution window — overlay version of /exec */}
      {showExec && (
        <ExecPanel events={mother.events} onClose={() => setShowExec(false)} />
      )}
    </>
  );
}
