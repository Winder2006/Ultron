/**
 * ExecPanel — the /exec code window as a closable overlay on the MAIN
 * dashboard.
 *
 * The standalone /exec route assumes a second monitor; on a single
 * screen the user could never see what Ultron actually ran. This
 * panel renders the same cells (code, stdout/stderr, value, figures)
 * from the SSE events the dashboard already receives, auto-opens when
 * a code_exec event arrives, and closes via the × button, Esc, or a
 * click on the backdrop.
 *
 * Reuses the .exec-* styles from index.css so the two views stay
 * visually identical.
 */
import type { SSEEvent } from '../lib/api';

interface ExecImage {
  png_b64: string;
  fig_num: number;
}

export interface ExecPanelProps {
  events: SSEEvent[];
  onClose: () => void;
}

export default function ExecPanel({ events, onClose }: ExecPanelProps) {
  // events arrive newest-first; render oldest-first like a terminal.
  const cells = events
    .filter((e) => e.type === 'code_exec')
    .slice()
    .reverse();

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 40,
        background: 'rgba(0, 0, 0, 0.55)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(860px, 92vw)',
          height: 'min(640px, 86vh)',
          background: 'rgba(8, 10, 16, 0.97)',
          border: '1px solid rgba(255, 51, 68, 0.25)',
          borderRadius: 8,
          boxShadow: '0 0 40px rgba(0,0,0,0.6), 0 0 24px rgba(255,51,68,0.08)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <header className="exec-header" style={{ flexShrink: 0 }}>
          <span className="exec-title">ULTRON :: REPL</span>
          <span className="exec-status">{cells.length} cells</span>
          <button
            onClick={onClose}
            title="Close (Esc)"
            style={{
              background: 'rgba(255, 51, 68, 0.08)',
              border: '1px solid rgba(255, 51, 68, 0.3)',
              borderRadius: 4,
              color: '#ff5566',
              cursor: 'pointer',
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              lineHeight: 1,
              padding: '4px 10px',
            }}
          >
            × close
          </button>
        </header>
        <div className="exec-scroll" style={{ flex: 1 }}>
          {cells.length === 0 ? (
            <div className="exec-empty">
              <pre>{`no code has run this session yet.

ask ultron to compute something —
"what's the sha-256 of the word potato"
— and the code + output land here.`}</pre>
            </div>
          ) : (
            cells.map((c, i) => <ExecCell key={`${c.ts ?? i}-${i}`} ev={c} idx={i} />)
          )}
        </div>
      </div>
    </div>
  );
}

function ExecCell({ ev, idx }: { ev: SSEEvent; idx: number }) {
  const rawImages = Array.isArray(ev.images) ? (ev.images as unknown[]) : [];
  const images: ExecImage[] = rawImages
    .filter((x) => x && typeof x === 'object' && 'png_b64' in (x as object))
    .map((x) => ({
      png_b64: String((x as ExecImage).png_b64 ?? ''),
      fig_num: Number((x as ExecImage).fig_num ?? 0),
    }));
  const stdout = String(ev.stdout ?? '').replace(/\s+$/, '');
  const stderr = String(ev.stderr ?? '').replace(/\s+$/, '');
  const value = String(ev.value ?? '');
  const stamp = ev.ts
    ? new Date((ev.ts as number) * 1000).toLocaleTimeString([], { hour12: false })
    : '';

  return (
    <div className={`exec-cell ${ev.timed_out ? 'cell-timeout' : ''}`}>
      <div className="exec-cell-header">
        <span className="exec-prompt">In [{idx}]</span>
        <span className="exec-stamp">
          {stamp}
          {ev.duration_s !== undefined ? ` · ${Number(ev.duration_s).toFixed(2)}s` : ''}
          {ev.timed_out ? ' · TIMED OUT' : ''}
        </span>
      </div>
      <pre className="exec-code">{String(ev.code ?? '') || '(empty)'}</pre>
      {value ? (
        <pre className="exec-value">
          <span className="exec-out-tag">Out[{idx}]:</span> {value}
        </pre>
      ) : null}
      {stdout ? <pre className="exec-stdout">{stdout}</pre> : null}
      {stderr ? <pre className="exec-stderr">{stderr}</pre> : null}
      {images.length > 0 ? (
        <div className="exec-images">
          {images.map((img) => (
            <figure key={img.fig_num} className="exec-figure">
              <img src={`data:image/png;base64,${img.png_b64}`} alt={`Figure ${img.fig_num}`} />
              <figcaption>Figure {img.fig_num}</figcaption>
            </figure>
          ))}
        </div>
      ) : null}
    </div>
  );
}
