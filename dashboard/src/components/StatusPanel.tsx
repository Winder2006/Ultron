/**
 * StatusPanel — Live system status with holographic styling.
 */
import { motion } from 'framer-motion';
import type { SystemStatus, UserInfo } from '../lib/api';

interface StatusPanelProps {
  status: SystemStatus | null;
  users: UserInfo[];
  connected: boolean;
}

function StatusRow({ label, value, active = true }: { label: string; value: string; active?: boolean }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '5px 0',
      borderBottom: '1px solid var(--border-dim)',
    }}>
      <span style={{
        color: 'var(--text-dim)',
        fontSize: 9,
        textTransform: 'uppercase',
        letterSpacing: 2,
        fontWeight: 300,
      }}>
        {label}
      </span>
      <span style={{
        color: active ? 'var(--red-bright)' : 'var(--text-dim)',
        fontSize: 11,
        fontWeight: 500,
        textShadow: active ? '0 0 8px rgba(255,34,68,0.3)' : 'none',
      }}>
        {value}
      </span>
    </div>
  );
}

function Indicator({ active, label }: { active: boolean; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10 }}>
      <motion.div
        style={{
          width: 5,
          height: 5,
          borderRadius: '50%',
          background: active ? 'var(--green)' : 'var(--text-dim)',
          boxShadow: active ? '0 0 6px var(--green)' : 'none',
        }}
        animate={active ? { opacity: [1, 0.4, 1] } : {}}
        transition={{ duration: 2, repeat: Infinity }}
      />
      <span style={{
        color: active ? 'var(--text-primary)' : 'var(--text-dim)',
        fontWeight: 300,
      }}>
        {label}
      </span>
    </div>
  );
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div style={{
      fontSize: 9,
      textTransform: 'uppercase',
      letterSpacing: 3,
      color: 'var(--text-dim)',
      fontWeight: 500,
      marginTop: 8,
      marginBottom: 4,
      fontFamily: 'var(--font-display)',
    }}>
      {children}
    </div>
  );
}

export default function StatusPanel({ status, users, connected }: StatusPanelProps) {
  return (
    <div style={{
      background: 'rgba(12, 12, 24, 0.5)',
      border: '1px solid var(--border-mid)',
      borderRadius: 'var(--radius)',
      padding: 14,
      display: 'flex',
      flexDirection: 'column',
      gap: 6,
      boxShadow: '0 0 20px rgba(0,0,0,0.3), inset 0 0 20px rgba(255,34,68,0.02)',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingBottom: 6,
        borderBottom: '1px solid var(--border-mid)',
      }}>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: 3,
          color: 'var(--text-secondary)',
          fontWeight: 500,
        }}>
          Systems
        </span>
        <Indicator active={connected} label={connected ? 'ONLINE' : 'OFFLINE'} />
      </div>

      {status ? (
        <>
          <StatusRow label="LLM" value={status.llm_provider} />
          <StatusRow label="TTS" value={status.tts_provider} />
          <StatusRow label="STT" value={status.stt_engine} active={status.stt_engine !== 'none'} />
          <StatusRow
            label="MQTT"
            value={status.mqtt_connected ? 'connected' : 'offline'}
            active={status.mqtt_connected}
          />
          <StatusRow label="Clients" value={String(status.ws_clients)} />

          <SectionLabel>Vision</SectionLabel>
          <div style={{ display: 'flex', gap: 12 }}>
            <Indicator active={status.vision.room_occupied} label="Occupied" />
            <Indicator active={status.vision.faces_count > 0} label={`${status.vision.faces_count} faces`} />
          </div>
        </>
      ) : (
        <div style={{
          color: 'var(--text-dim)',
          fontSize: 11,
          textAlign: 'center',
          padding: 24,
          fontWeight: 300,
        }}>
          Initializing...
        </div>
      )}

      {/* Users */}
      {users.length > 0 && (
        <>
          <SectionLabel>Crew</SectionLabel>
          {users.map((u) => (
            <div key={u.user_id} style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontSize: 10,
              padding: '3px 0',
            }}>
              <span style={{ color: 'var(--text-primary)', fontWeight: 400 }}>{u.display_name}</span>
              <span style={{
                color: u.voice_enrolled ? 'var(--green)' : 'var(--text-dim)',
                fontSize: 8,
                letterSpacing: 1,
                textTransform: 'uppercase',
                textShadow: u.voice_enrolled ? '0 0 6px rgba(0,255,136,0.3)' : 'none',
              }}>
                {u.voice_enrolled ? 'VOICE ID' : 'TEXT'}
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
