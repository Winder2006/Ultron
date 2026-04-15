/**
 * ConversationFeed — Real-time scrolling conversation display.
 *
 * Shows the latest query and streaming LLM response, plus a history
 * of recent events from SSE. Has a text input for sending prompts.
 */
import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { SSEEvent } from '../lib/api';

interface ConversationFeedProps {
  events: SSEEvent[];
  lastQuery: string;
  lastResponse: string;
  lastIntent: string;
  onSendPrompt: (text: string) => void;
  processing: boolean;
}

function EventBadge({ type }: { type: string }) {
  const colorMap: Record<string, string> = {
    query: 'var(--cyan)',
    response: 'var(--red-bright)',
    vision_face_detected: 'var(--green)',
    vision_face_lost: 'var(--text-dim)',
    vision_object_detected: 'var(--amber)',
    vision_room_occupancy: 'var(--amber)',
    heartbeat: 'var(--text-dim)',
  };
  const color = colorMap[type] ?? 'var(--text-secondary)';
  return (
    <span style={{
      fontSize: 9,
      textTransform: 'uppercase',
      letterSpacing: 1,
      color,
      border: `1px solid ${color}44`,
      borderRadius: 2,
      padding: '1px 4px',
      flexShrink: 0,
    }}>
      {type.replace('vision_', '')}
    </span>
  );
}

export default function ConversationFeed({
  events,
  lastQuery,
  lastResponse,
  lastIntent,
  onSendPrompt,
  processing,
}: ConversationFeedProps) {
  const [input, setInput] = useState('');
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [events.length]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !processing) {
      onSendPrompt(input.trim());
      setInput('');
    }
  };

  return (
    <div style={{
      background: 'rgba(12, 12, 24, 0.5)',
      border: '1px solid var(--border-mid)',
      borderRadius: 'var(--radius)',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      overflow: 'hidden',
      boxShadow: '0 0 20px rgba(0,0,0,0.3), inset 0 0 20px rgba(255,34,68,0.02)',
    }}>
      {/* Header */}
      <div style={{
        padding: '8px 16px',
        borderBottom: '1px solid var(--border-mid)',
        fontSize: 9,
        textTransform: 'uppercase',
        letterSpacing: 3,
        color: 'var(--text-secondary)',
        display: 'flex',
        justifyContent: 'space-between',
        fontFamily: 'var(--font-display)',
        fontWeight: 500,
      }}>
        <span>Datastream</span>
        {lastIntent && (
          <span style={{ color: 'var(--amber)', letterSpacing: 2, textShadow: '0 0 8px rgba(255,170,68,0.3)' }}>
            {lastIntent}
          </span>
        )}
      </div>

      {/* Active exchange */}
      {(lastQuery || lastResponse) && (
        <div style={{
          padding: '12px 16px',
          borderBottom: '1px solid var(--border-dim)',
          background: 'var(--bg-secondary)',
        }}>
          {lastQuery && (
            <div style={{ marginBottom: 8 }}>
              <span style={{ color: 'var(--cyan)', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
                USER
              </span>
              <div style={{ color: 'var(--text-primary)', marginTop: 2, fontSize: 13 }}>
                {lastQuery}
              </div>
            </div>
          )}
          {lastResponse && (
            <div>
              <span style={{ color: 'var(--red-bright)', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
                ULTRON
              </span>
              <motion.div
                style={{ color: 'var(--text-primary)', marginTop: 2, fontSize: 13 }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                {lastResponse}
              </motion.div>
            </div>
          )}
          {processing && (
            <motion.div
              style={{
                marginTop: 8,
                display: 'flex',
                gap: 4,
                alignItems: 'center',
              }}
              animate={{ opacity: [0.4, 1, 0.4] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              <div style={{ width: 4, height: 4, borderRadius: '50%', background: 'var(--red-bright)' }} />
              <div style={{ width: 4, height: 4, borderRadius: '50%', background: 'var(--red-bright)' }} />
              <div style={{ width: 4, height: 4, borderRadius: '50%', background: 'var(--red-bright)' }} />
            </motion.div>
          )}
        </div>
      )}

      {/* Event history */}
      <div
        ref={feedRef}
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '8px 16px',
        }}
      >
        <AnimatePresence>
          {events
            .filter((e) =>
              e.type !== 'heartbeat' &&
              e.type !== 'connected' &&
              e.type !== 'query' &&      // already shown in active exchange
              e.type !== 'response'      // already shown in active exchange
            )
            .slice(0, 30)
            .map((ev, i) => (
              <motion.div
                key={`${ev.ts ?? i}-${ev.type}`}
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 8,
                  padding: '4px 0',
                  borderBottom: '1px solid var(--border-dim)',
                  fontSize: 11,
                }}
              >
                <EventBadge type={ev.type} />
                <span style={{ color: 'var(--text-secondary)', flex: 1, wordBreak: 'break-word' }}>
                  {ev.text as string ??
                    ev.name as string ??
                    (ev.occupied !== undefined ? `occupied: ${ev.occupied}` : '') ??
                    JSON.stringify(ev).slice(0, 80)}
                </span>
                {ev.ts && (
                  <span style={{ color: 'var(--text-dim)', fontSize: 9, flexShrink: 0 }}>
                    {new Date((ev.ts as number) * 1000).toLocaleTimeString()}
                  </span>
                )}
              </motion.div>
            ))}
        </AnimatePresence>
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        style={{
          display: 'flex',
          borderTop: '1px solid var(--border-mid)',
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={processing ? 'Processing...' : 'Type a command...'}
          disabled={processing}
          style={{
            flex: 1,
            background: 'var(--bg-secondary)',
            border: 'none',
            outline: 'none',
            padding: '10px 16px',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-mono)',
            fontSize: 12,
          }}
        />
        <button
          type="submit"
          disabled={processing || !input.trim()}
          style={{
            background: processing ? 'var(--red-dim)' : 'var(--red-mid)',
            color: 'var(--text-bright)',
            border: 'none',
            padding: '10px 16px',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            textTransform: 'uppercase',
            letterSpacing: 1,
            cursor: processing ? 'not-allowed' : 'pointer',
          }}
        >
          Send
        </button>
      </form>
    </div>
  );
}
