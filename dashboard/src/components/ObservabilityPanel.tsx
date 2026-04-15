/**
 * ObservabilityPanel — live view of backend events.
 *
 * Consumes the SSE event buffer maintained by `useMotherAPI` and
 * renders four sections: recent queries (with tier + intent),
 * RAG hits, memory writes, and latency breakdown.
 *
 * This is a dev-side surface for debugging what Ultron is thinking,
 * not a user-facing UI — layout favours density over polish.
 */
import { useMemo } from 'react';
import type { SSEEvent } from '../lib/api';

interface Props {
  events: SSEEvent[];
}

// How many of each event category to show at once. Kept small —
// a wall of logs is a worse UX than a small pulse that updates.
const RECENT_LIMIT = 6;


// Extract the most recent N events matching a type predicate.
function recent<T = SSEEvent>(
  events: SSEEvent[],
  predicate: (e: SSEEvent) => boolean,
  limit: number = RECENT_LIMIT,
): T[] {
  const out: T[] = [];
  for (const e of events) {
    if (predicate(e)) {
      out.push(e as unknown as T);
      if (out.length >= limit) break;
    }
  }
  return out;
}


function formatRelativeTime(ts: number | undefined): string {
  if (!ts) return '';
  const delta = Math.floor((Date.now() / 1000) - ts);
  if (delta < 1) return 'now';
  if (delta < 60) return `${delta}s`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m`;
  return `${Math.floor(delta / 3600)}h`;
}


const tierColor = (tier: string | undefined): string => {
  if (tier === 'tier1') return '#00ff88';  // fast green
  if (tier === 'tier2') return '#ffcc44';  // amber
  if (tier === 'tier3') return '#ff3344';  // red (expensive)
  return 'rgba(255,255,255,0.3)';
};


export default function ObservabilityPanel({ events }: Props) {
  const queries = useMemo(
    () => recent(events, (e) => e.type === 'query'),
    [events],
  );
  const ragHits = useMemo(
    () => recent(events, (e) => e.type === 'rag_hit'),
    [events],
  );
  const memoryWrites = useMemo(
    () => recent(events, (e) => e.type === 'memory_write'),
    [events],
  );
  const latencies = useMemo(
    () => recent(events, (e) => e.type === 'latency'),
    [events],
  );
  const toolCalls = useMemo(
    () => recent(events, (e) => e.type === 'tool_call'),
    [events],
  );

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>OBSERVABILITY</div>

      <Section title={`Queries (${queries.length})`}>
        {queries.length === 0 && <Empty>no queries yet</Empty>}
        {queries.map((e, i) => (
          <Row key={i}>
            <div style={{ ...pillStyle, color: tierColor(e.tier as string), borderColor: tierColor(e.tier as string) }}>
              {(e.tier as string) || '-'}
            </div>
            <div style={{ ...pillStyle, color: 'rgba(255,204,68,0.7)', borderColor: 'rgba(255,204,68,0.2)' }}>
              {(e.intent as string)?.toLowerCase() || '-'}
            </div>
            <div style={textStyle} title={String(e.text || '')}>
              {String(e.text || '').slice(0, 42)}
            </div>
            <div style={tsStyle}>{formatRelativeTime(e.ts as number)}</div>
          </Row>
        ))}
      </Section>

      <Section title={`Latency (${latencies.length})`}>
        {latencies.length === 0 && <Empty>—</Empty>}
        {latencies.map((e, i) => (
          <Row key={i}>
            <div style={{ ...pillStyle, color: tierColor(e.tier as string), borderColor: tierColor(e.tier as string) }}>
              {(e.tier as string) || '-'}
            </div>
            <div style={textStyle}>
              ttft {String(e.ttft_ms)}ms · tts {String((e.tts_first_chunk_ms as number) ?? '—')}ms · total {String(e.total_ms)}ms
            </div>
            <div style={tsStyle}>{formatRelativeTime(e.ts as number)}</div>
          </Row>
        ))}
      </Section>

      <Section title={`RAG hits (${ragHits.length})`}>
        {ragHits.length === 0 && <Empty>no context retrieved yet</Empty>}
        {ragHits.map((e, i) => (
          <Row key={i} column>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <div style={{ ...pillStyle, color: '#4488ff', borderColor: 'rgba(68,136,255,0.3)' }}>
                {String(e.blocks || 0)} blocks · {String(e.duration_ms)}ms
              </div>
              <div style={tsStyle}>{formatRelativeTime(e.ts as number)}</div>
            </div>
            <div style={{ ...textStyle, fontSize: 9, opacity: 0.55, marginTop: 2 }} title={String(e.preview || '')}>
              {String(e.preview || '').slice(0, 90)}
            </div>
          </Row>
        ))}
      </Section>

      <Section title={`Memory writes (${memoryWrites.length})`}>
        {memoryWrites.length === 0 && <Empty>—</Empty>}
        {memoryWrites.map((e, i) => (
          <Row key={i} column>
            <div style={{ display: 'flex', gap: 6 }}>
              <div style={{ ...pillStyle, color: '#ff88cc', borderColor: 'rgba(255,136,204,0.3)' }}>
                {String(e.count)} fact{(e.count as number) === 1 ? '' : 's'}
              </div>
              <div style={tsStyle}>{formatRelativeTime(e.ts as number)}</div>
            </div>
            {Array.isArray(e.items) && (e.items as string[]).map((item, j) => (
              <div key={j} style={{ ...textStyle, fontSize: 9, opacity: 0.6 }}>{item}</div>
            ))}
          </Row>
        ))}
      </Section>

      {toolCalls.length > 0 && (
        <Section title={`Tool calls (${toolCalls.length})`}>
          {toolCalls.map((e, i) => (
            <Row key={i} column>
              <div style={{ display: 'flex', gap: 6 }}>
                <div style={{ ...pillStyle, color: '#88ffcc', borderColor: 'rgba(136,255,204,0.3)' }}>
                  {String(e.name || '?')}
                </div>
                <div style={tsStyle}>{formatRelativeTime(e.ts as number)}</div>
              </div>
              <div style={{ ...textStyle, fontSize: 9, opacity: 0.55 }}>
                {String(e.result || '').slice(0, 90)}
              </div>
            </Row>
          ))}
        </Section>
      )}
    </div>
  );
}


// ─────────────────────── tiny inline subcomponents ──────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={sectionTitleStyle}>{title}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {children}
      </div>
    </div>
  );
}

function Row({ children, column }: { children: React.ReactNode; column?: boolean }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: column ? 'column' : 'row',
      gap: 6,
      alignItems: column ? 'flex-start' : 'center',
      padding: '2px 0',
    }}>
      {children}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ ...textStyle, opacity: 0.3, fontStyle: 'italic' }}>
      {children}
    </div>
  );
}


// ─────────────────────── styles ─────────────────────────────────────

const containerStyle: React.CSSProperties = {
  border: '1px solid rgba(255, 51, 68, 0.1)',
  borderRadius: 6,
  background: 'rgba(15, 18, 28, 0.6)',
  padding: 10,
  fontFamily: 'var(--font-mono)',
  color: 'rgba(255,255,255,0.7)',
};

const headerStyle: React.CSSProperties = {
  fontSize: 9,
  letterSpacing: 3,
  color: 'rgba(255, 51, 68, 0.6)',
  marginBottom: 8,
  fontWeight: 500,
};

const sectionTitleStyle: React.CSSProperties = {
  fontSize: 8,
  letterSpacing: 2,
  color: 'rgba(255,255,255,0.35)',
  textTransform: 'uppercase',
  marginBottom: 4,
};

const pillStyle: React.CSSProperties = {
  fontSize: 8.5,
  padding: '1px 6px',
  borderRadius: 4,
  border: '1px solid',
  textTransform: 'uppercase',
  letterSpacing: 1,
  flexShrink: 0,
};

const textStyle: React.CSSProperties = {
  fontSize: 10,
  color: 'rgba(255,255,255,0.7)',
  flex: 1,
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
};

const tsStyle: React.CSSProperties = {
  fontSize: 9,
  color: 'rgba(255,255,255,0.25)',
  flexShrink: 0,
};
