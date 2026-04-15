import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import NeuralWeb from './components/NeuralWeb';
import StatusPanel from './components/StatusPanel';
import ConversationFeed from './components/ConversationFeed';
import FilterPanel from './components/FilterPanel';
import ObservabilityPanel from './components/ObservabilityPanel';
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
  const [time, setTime] = useState(new Date());

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
    // Suppress while: already recording, still waiting for response,
    // or the voice WS isn't actually open. Without this, Ultron saying
    // phonetically-adjacent words in his response would re-trigger
    // himself endlessly.
    isSuppressed: () =>
      voice.recording || voice.processing || !voice.connected,
    onDetected: () => {
      if (voice.recording || voice.processing) return;
      clearAutoStop();
      // Pass a silence callback so the browser's Silero VAD auto-stops
      // as soon as the user finishes talking (typically ~700ms after
      // last syllable) — no more waiting for the 5s ceiling or for
      // the server-side STT to emit a final. This is the fix that
      // makes wake-word turns feel instant instead of sluggish.
      voice.startRecording(() => {
        voice.stopRecording();
        clearAutoStop();
      });
      // 6s hard ceiling. With VAD the typical stop is ~1-2s into
      // recording; this only trips if VAD somehow fails to detect
      // end of speech (e.g. constant loud background noise).
      autoStopTimerRef.current = window.setTimeout(() => {
        voice.stopRecording();
        autoStopTimerRef.current = null;
      }, 6000);
    },
  });

  // Auto-stop the moment Deepgram reports a final transcript — that's
  // the authoritative signal that the user finished speaking. Cuts
  // typical wake-word turnaround from 5-7s down to ~1s after the last
  // syllable, which is what every competent voice assistant does.
  useEffect(() => {
    const ev = voice.lastEvent;
    if (!ev || ev.event !== 'stt' || !ev.final) return;
    if (!voice.recording) return;
    // Let the last audio flush through before stopping — a tiny pad
    // so we don't clip the trailing phoneme.
    const t = window.setTimeout(() => {
      voice.stopRecording();
      clearAutoStop();
    }, 150);
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

      {/* Top-right controls */}
      <div style={{
        position: 'fixed',
        top: 16,
        right: 16,
        display: 'flex',
        gap: 8,
        zIndex: 20,
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
    </>
  );
}
