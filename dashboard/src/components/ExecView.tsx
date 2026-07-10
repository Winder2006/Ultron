/**
 * Terminal-styled view for code_exec events.
 *
 * Open this page on a second monitor (http://localhost:3000/exec) so
 * you can see what Ultron is actually running while he answers on the
 * main dashboard. Each exec is shown as a "cell": the code that ran,
 * the stdout/stderr/value output, and the duration.
 *
 * The view subscribes to the same SSE bus the main dashboard uses but
 * filters to just `code_exec` events. State is local to this tab —
 * refreshing clears history (intentional; the persistent REPL keeps
 * the variables, the visual log is just a window into it).
 */
import { useEffect, useRef, useState } from 'react';
import { connectSSE, type SSEEvent } from '../lib/api';

interface ExecImage {
  png_b64: string;
  fig_num: number;
}

interface ExecCell {
  id: number;
  ts: Date;
  code: string;
  stdout: string;
  stderr: string;
  value: string;
  durationS: number;
  timedOut: boolean;
  images: ExecImage[];
}

const MAX_CELLS = 200;  // ring-buffer cap to stop a long session bloating memory

export default function ExecView() {
  const [cells, setCells] = useState<ExecCell[]>([]);
  const [connected, setConnected] = useState(false);
  const cellIdRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Subscribe to SSE on mount, filter to code_exec.
  useEffect(() => {
    const es = connectSSE((event: SSEEvent) => {
      if (event.type !== 'code_exec') return;
      // Defensive: backend sends `images` as an array of {png_b64, fig_num}.
      // Old payloads omit it; treat missing as empty.
      const rawImages = Array.isArray(event.images) ? event.images : [];
      const images: ExecImage[] = rawImages
        .filter((x) => x && typeof x === 'object' && 'png_b64' in x)
        .map((x: any) => ({
          png_b64: String(x.png_b64 ?? ''),
          fig_num: Number(x.fig_num ?? 0),
        }));
      const cell: ExecCell = {
        id: cellIdRef.current++,
        ts: new Date(),
        code: String(event.code ?? ''),
        stdout: String(event.stdout ?? ''),
        stderr: String(event.stderr ?? ''),
        value: String(event.value ?? ''),
        durationS: Number(event.duration_s ?? 0),
        timedOut: Boolean(event.timed_out),
        images,
      };
      setCells((prev) => {
        const next = [...prev, cell];
        // Keep the tail — newer cells matter more than older ones.
        return next.length > MAX_CELLS ? next.slice(-MAX_CELLS) : next;
      });
    });
    setConnected(es.readyState !== EventSource.CLOSED);
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    return () => es.close();
  }, []);

  // Auto-scroll to the bottom when a new cell arrives. Skip if the
  // user has scrolled up — they're probably reading something.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const isAtBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 200;
    if (isAtBottom) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    }
  }, [cells]);

  const clear = () => setCells([]);

  return (
    <div className="exec-view">
      <header className="exec-header">
        <span className="exec-title">ULTRON :: REPL</span>
        <span className="exec-status">
          <span
            className={`exec-dot ${connected ? 'on' : 'off'}`}
            title={connected ? 'connected' : 'disconnected'}
          />
          {cells.length} cells
        </span>
        <button onClick={clear} className="exec-clear">
          clear log
        </button>
      </header>
      <div ref={scrollRef} className="exec-scroll">
        {cells.length === 0 ? (
          <div className="exec-empty">
            <pre>{`waiting for code_exec events.

ask ultron to compute something —
"compute the sha-256 of this string: hello"
"load this CSV and tell me the median revenue"
"plot the distribution of these numbers"

each call's code, output, and duration
shows up here in real time. variables
persist across calls — load data once,
query it across multiple questions.`}</pre>
          </div>
        ) : (
          cells.map((c) => <ExecCellView key={c.id} cell={c} />)
        )}
      </div>
    </div>
  );
}

function ExecCellView({ cell }: { cell: ExecCell }) {
  const t = cell.ts;
  const hh = String(t.getHours()).padStart(2, '0');
  const mm = String(t.getMinutes()).padStart(2, '0');
  const ss = String(t.getSeconds()).padStart(2, '0');
  const stamp = `${hh}:${mm}:${ss}`;

  return (
    <div className={`exec-cell ${cell.timedOut ? 'cell-timeout' : ''}`}>
      <div className="exec-cell-header">
        <span className="exec-prompt">In [{cell.id}]</span>
        <span className="exec-stamp">
          {stamp} · {cell.durationS.toFixed(2)}s
          {cell.timedOut ? ' · TIMED OUT' : ''}
        </span>
      </div>
      <pre className="exec-code">{cell.code || '(empty)'}</pre>
      {cell.value ? (
        <pre className="exec-value">
          <span className="exec-out-tag">Out[{cell.id}]:</span> {cell.value}
        </pre>
      ) : null}
      {cell.stdout ? (
        <pre className="exec-stdout">{cell.stdout.replace(/\s+$/, '')}</pre>
      ) : null}
      {cell.stderr ? (
        <pre className="exec-stderr">{cell.stderr.replace(/\s+$/, '')}</pre>
      ) : null}
      {cell.images.length > 0 ? (
        <div className="exec-images">
          {cell.images.map((img) => (
            <figure key={img.fig_num} className="exec-figure">
              <img
                src={`data:image/png;base64,${img.png_b64}`}
                alt={`Figure ${img.fig_num}`}
              />
              <figcaption>Figure {img.fig_num}</figcaption>
            </figure>
          ))}
        </div>
      ) : null}
    </div>
  );
}
