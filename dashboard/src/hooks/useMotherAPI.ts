/** Hook for MOTHER REST API polling + SSE event stream. */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  connectSSE,
  fetchStatus,
  fetchUsers,
  type SSEEvent,
  type SystemStatus,
  type UserInfo,
} from '../lib/api';

export interface MotherState {
  status: SystemStatus | null;
  users: UserInfo[];
  events: SSEEvent[];
  connected: boolean;
  lastQuery: string;
  lastResponse: string;
  lastIntent: string;
  activity: number; // 0..1 — recent activity level for visualizations
}

const MAX_EVENTS = 100;

export function useMotherAPI() {
  const [state, setState] = useState<MotherState>({
    status: null,
    users: [],
    events: [],
    connected: false,
    lastQuery: '',
    lastResponse: '',
    lastIntent: '',
    activity: 0,
  });

  const activityDecay = useRef<number>(0);

  // Poll status every 5s
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const [status, users] = await Promise.all([fetchStatus(), fetchUsers()]);
        if (active) {
          setState((s) => ({ ...s, status, users, connected: true }));
        }
      } catch {
        if (active) setState((s) => ({ ...s, connected: false }));
      }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { active = false; clearInterval(id); };
  }, []);

  // SSE event stream
  useEffect(() => {
    const es = connectSSE((event) => {
      setState((s) => {
        const updates: Partial<MotherState> = {
          events: [event, ...s.events].slice(0, MAX_EVENTS),
        };

        if (event.type === 'query') {
          updates.lastQuery = event.text as string;
          updates.lastIntent = event.intent as string;
          updates.activity = 1;
        } else if (event.type === 'response') {
          updates.lastResponse = event.text as string;
          updates.activity = Math.min(1, s.activity + 0.3);
        } else if (event.type?.toString().startsWith('vision_')) {
          updates.activity = Math.min(1, s.activity + 0.2);
        }

        return { ...s, ...updates };
      });
    });

    return () => es.close();
  }, []);

  // Decay activity over time
  useEffect(() => {
    const id = setInterval(() => {
      setState((s) => ({
        ...s,
        activity: Math.max(0, s.activity - 0.02),
      }));
    }, 100);
    return () => clearInterval(id);
  }, []);

  return state;
}
