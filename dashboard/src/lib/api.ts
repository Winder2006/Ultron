/** MOTHER API client — REST, SSE, and WebSocket connections. */

const API_BASE = '/api';
const WS_BASE = `ws://${window.location.host}`;

// ── REST ──

export interface SystemStatus {
  llm_provider: string;
  tts_provider: string;
  stt_engine: string;
  mqtt_connected: boolean;
  vision: {
    room_occupied: boolean;
    faces_count: number;
    objects_count: number;
    last_event_ts: number | null;
  };
  ws_clients: number;
}

export interface UserInfo {
  user_id: string;
  display_name: string;
  voice_enrolled: boolean;
  last_seen: string;
}

export interface MemoryData {
  user_id: string;
  facts: Record<string, unknown>;
  episodic: unknown[];
  fact_count: number;
  episodic_count: number;
}

export interface ConversationMessage {
  role: string;
  content: string;
}

export async function fetchStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
}

export async function fetchUsers(): Promise<UserInfo[]> {
  const res = await fetch(`${API_BASE}/users`);
  const data = await res.json();
  return data.users ?? [];
}

export async function fetchMemories(userId: string): Promise<MemoryData> {
  const res = await fetch(`${API_BASE}/memories/${userId}`);
  return res.json();
}

export async function fetchConversation(userId: string): Promise<ConversationMessage[]> {
  const res = await fetch(`${API_BASE}/conversation/${userId}`);
  const data = await res.json();
  return data.messages ?? [];
}

// ── SSE ──

export type SSEEvent = {
  type: string;
  ts?: number;
  [key: string]: unknown;
};

export function connectSSE(onEvent: (event: SSEEvent) => void): EventSource {
  const es = new EventSource(`${API_BASE}/events`);
  es.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data) as SSEEvent;
      onEvent(data);
    } catch {
      // ignore malformed
    }
  };
  return es;
}

// ── WebSocket Voice ──

export type WSEvent = {
  event: string;
  [key: string]: unknown;
};

export class VoiceSocket {
  private ws: WebSocket | null = null;
  private onEvent: (event: WSEvent) => void;
  // Promise that resolves when the WS hits OPEN state. Lets callers
  // `await socket.ready` before sending data, avoiding the race where
  // `connect()` returns synchronously and a caller tries to send
  // before the handshake completes.
  ready: Promise<void> = Promise.resolve();
  private _readyResolve: (() => void) | null = null;
  private _readyReject: ((err: Error) => void) | null = null;

  constructor(onEvent: (event: WSEvent) => void) {
    this.onEvent = onEvent;
  }

  connect() {
    this.ws = new WebSocket(`${WS_BASE}/ws/voice`);
    this.ws.binaryType = 'arraybuffer';

    // Fresh ready-promise for this connection.
    this.ready = new Promise<void>((resolve, reject) => {
      this._readyResolve = resolve;
      this._readyReject = reject;
    });

    this.ws.onopen = () => {
      // Emit a synthetic 'connected' event so the hook can flip its
      // connected flag based on actual readyState, not optimism.
      this.onEvent({ event: 'connected' });
      this._readyResolve?.();
    };

    this.ws.onmessage = (msg) => {
      if (typeof msg.data === 'string') {
        try {
          const data = JSON.parse(msg.data) as WSEvent;
          this.onEvent(data);
        } catch {
          // ignore
        }
      }
    };

    this.ws.onclose = () => {
      this.onEvent({ event: 'disconnected' });
      this._readyReject?.(new Error('socket closed before open'));
    };

    this.ws.onerror = () => {
      this.onEvent({ event: 'error', message: 'WebSocket error' });
      this._readyReject?.(new Error('websocket error'));
    };
  }

  send(action: string, data?: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action, ...data }));
    }
  }

  sendAudio(pcmData: ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(pcmData);
    }
  }

  sendPrompt(text: string) {
    this.send('prompt', { text });
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
  }

  get connected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
