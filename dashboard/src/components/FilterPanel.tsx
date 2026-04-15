/**
 * FilterPanel — voice filter controls.
 *
 * Sits in the collapsible right panel. Provides:
 *   - Enable/disable toggle
 *   - Preset buttons (Clean, Subtle, Ultron, Heavy Robot)
 *   - Live tuning sliders for each effect parameter
 *
 * Changes apply instantly — next audio chunk uses the new params.
 */
import { useEffect, useState, type ReactNode } from 'react';
import {
  getFilterParams,
  setFilterParams,
  applyFilterPreset,
  PRESETS,
  type FilterParams,
  type PresetName,
} from '../hooks/useVoice';

const PRESET_NAMES: PresetName[] = ['Clean', 'Subtle', 'Ultron', 'Heavy Robot', 'Menacing'];

export default function FilterPanel() {
  const [params, setParams] = useState<FilterParams>(() => getFilterParams());
  const [expanded, setExpanded] = useState(false);

  // Sync local state from the source of truth on mount
  useEffect(() => {
    setParams(getFilterParams());
  }, []);

  const update = (patch: Partial<FilterParams>) => {
    const next = { ...params, ...patch };
    setParams(next);
    setFilterParams(patch);
  };

  const usePreset = (name: PresetName) => {
    applyFilterPreset(name);
    setParams({ ...PRESETS[name] });
  };

  // Detect which (if any) preset matches current params. Compare all
  // tunable fields so new presets (Menacing) distinguish from older
  // ones that differ only in body/chorus/intensity.
  const activePreset = PRESET_NAMES.find((name) => {
    const p = PRESETS[name];
    return (
      p.enabled === params.enabled &&
      Math.abs(p.pitchSemitones - params.pitchSemitones) < 0.1 &&
      Math.abs(p.distortion - params.distortion) < 0.01 &&
      Math.abs(p.presence - params.presence) < 0.5 &&
      Math.abs(p.lowCut - params.lowCut) < 0.5 &&
      Math.abs(p.bodyBoost - params.bodyBoost) < 0.5 &&
      Math.abs(p.chorus - params.chorus) < 0.05 &&
      Math.abs(p.intensityResponsiveness - params.intensityResponsiveness) < 0.05
    );
  });

  return (
    <div style={{
      background: 'rgba(12, 12, 24, 0.5)',
      border: '1px solid var(--border-mid)',
      borderRadius: 'var(--radius)',
      padding: 12,
      fontFamily: 'var(--font-mono)',
      boxShadow: '0 0 20px rgba(0,0,0,0.3), inset 0 0 20px rgba(255,34,68,0.02)',
    }}>
      {/* Header */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          paddingBottom: expanded ? 8 : 0,
          borderBottom: expanded ? '1px solid var(--border-mid)' : 'none',
          userSelect: 'none',
        }}
      >
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <div style={{
            width: 5,
            height: 5,
            borderRadius: '50%',
            background: params.enabled ? 'var(--red-bright)' : 'var(--text-dim)',
            boxShadow: params.enabled ? '0 0 6px rgba(255,51,68,0.6)' : 'none',
          }} />
          <span style={{
            fontFamily: 'var(--font-display)',
            fontSize: 10,
            textTransform: 'uppercase',
            letterSpacing: 3,
            color: 'var(--text-secondary)',
            fontWeight: 500,
          }}>
            Voice Filter
          </span>
          {activePreset && (
            <span style={{
              fontSize: 9,
              color: 'var(--amber)',
              letterSpacing: 2,
              textTransform: 'uppercase',
            }}>
              {activePreset}
            </span>
          )}
        </div>
        <span style={{
          color: 'var(--text-dim)',
          fontSize: 10,
        }}>
          {expanded ? '▼' : '▶'}
        </span>
      </div>

      {expanded && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 8 }}>
          {/* Master toggle */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
              Enabled
            </label>
            <button
              onClick={() => update({ enabled: !params.enabled })}
              style={{
                background: params.enabled ? 'var(--red-bright)' : 'var(--text-dim)',
                color: '#fff',
                border: 'none',
                padding: '2px 8px',
                borderRadius: 3,
                fontSize: 9,
                letterSpacing: 1,
                textTransform: 'uppercase',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)',
              }}
            >
              {params.enabled ? 'ON' : 'OFF'}
            </button>
          </div>

          {/* Presets */}
          <div>
            <div style={{
              fontSize: 9,
              textTransform: 'uppercase',
              letterSpacing: 2,
              color: 'var(--text-dim)',
              marginBottom: 4,
              fontFamily: 'var(--font-display)',
            }}>
              Preset
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              {PRESET_NAMES.map((name) => (
                <button
                  key={name}
                  onClick={() => usePreset(name)}
                  style={{
                    background: activePreset === name
                      ? 'rgba(255, 51, 68, 0.2)'
                      : 'rgba(255, 255, 255, 0.04)',
                    color: activePreset === name ? 'var(--red-bright)' : 'var(--text-primary)',
                    border: `1px solid ${activePreset === name ? 'var(--red-bright)' : 'var(--border-mid)'}`,
                    padding: '5px 8px',
                    borderRadius: 3,
                    fontSize: 10,
                    cursor: 'pointer',
                    fontFamily: 'var(--font-mono)',
                    letterSpacing: 0.5,
                    transition: 'all 0.15s',
                  }}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>

          {/* Sliders — only shown when enabled */}
          {params.enabled && (
            <>
              <SectionLabel>Core Voice</SectionLabel>
              <Slider
                label="Pitch (deeper ↓)"
                hint="Lowers or raises the voice. Negative numbers go deeper — −3 is noticeably menacing, −6 gets into monster territory."
                value={params.pitchSemitones}
                min={-12} max={6} step={0.25}
                onChange={(v) => update({ pitchSemitones: v })}
                format={(v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
              />
              <Slider
                label="Grit / saturation"
                hint="Warm digital distortion. Low values add edge; high values make every consonant bite."
                value={params.distortion}
                min={0} max={1} step={0.01}
                onChange={(v) => update({ distortion: v })}
                format={(v) => v.toFixed(2)}
              />
              <Slider
                label="Bite (2 kHz boost)"
                hint="Mid-high presence. More = through-a-speaker clarity and menace."
                value={params.presence}
                min={0} max={12} step={0.5}
                onChange={(v) => update({ presence: v })}
                format={(v) => `+${v.toFixed(1)}dB`}
              />
              <Slider
                label="Thin ↔ Weight (low shelf)"
                hint="Negative thins the voice (speaking-through-steel). Positive adds bass body."
                value={params.lowCut}
                min={-18} max={12} step={0.5}
                onChange={(v) => update({ lowCut: v })}
                format={(v) => (v > 0 ? `+${v.toFixed(1)}dB` : `${v.toFixed(1)}dB`)}
              />
              <Slider
                label="Chest resonance (200 Hz)"
                hint="Adds weight around the chest register without muddying. Stacks on top of the low-shelf."
                value={params.bodyBoost}
                min={0} max={12} step={0.5}
                onChange={(v) => update({ bodyBoost: v })}
                format={(v) => `+${v.toFixed(1)}dB`}
              />
              <Slider
                label="Doubled voice (chorus)"
                hint="Short detuned delay mixed back in. Low = subtle shimmer, high = synthetic-chord feel."
                value={params.chorus}
                min={0} max={1} step={0.02}
                onChange={(v) => update({ chorus: v })}
                format={(v) => v.toFixed(2)}
              />

              <SectionLabel>Character FX</SectionLabel>
              <Slider
                label="Ring mod — amount"
                hint="Multiplies the voice with a sine carrier: classic robot / Dalek ring. At 0 it's invisible."
                value={params.ringModAmount}
                min={0} max={1} step={0.01}
                onChange={(v) => update({ ringModAmount: v })}
                format={(v) => v.toFixed(2)}
              />
              <Slider
                label="Ring mod — frequency"
                hint="Carrier pitch. Low (30–80 Hz) = menacing tremolo. Mid (100–300 Hz) = metallic. High = alien."
                value={params.ringModFreq}
                min={30} max={800} step={1}
                onChange={(v) => update({ ringModFreq: v })}
                format={(v) => `${Math.round(v)} Hz`}
              />
              <Slider
                label="Bitcrusher — amount"
                hint="Blends in a bit-reduced copy. Adds digital grit and lo-fi edge."
                value={params.bitcrushAmount}
                min={0} max={1} step={0.01}
                onChange={(v) => update({ bitcrushAmount: v })}
                format={(v) => v.toFixed(2)}
              />
              <Slider
                label="Bitcrusher — bit depth"
                hint="Lower bits = more audible steps. 12 bits is subtle; 6 bits is aggressively crunchy."
                value={params.bitcrushBits}
                min={4} max={16} step={1}
                onChange={(v) => update({ bitcrushBits: v })}
                format={(v) => `${Math.round(v)} bits`}
              />
              <Slider
                label="Reverb — amount"
                hint="Adds a synthetic room tail. Small = presence; large = speaking from inside a steel shell."
                value={params.reverbAmount}
                min={0} max={1} step={0.01}
                onChange={(v) => update({ reverbAmount: v })}
                format={(v) => v.toFixed(2)}
              />
              <Slider
                label="Reverb — decay"
                hint="How long the tail lasts. Short is a small chamber; long is cathedral."
                value={params.reverbDecay}
                min={0.1} max={4.0} step={0.05}
                onChange={(v) => update({ reverbDecay: v })}
                format={(v) => `${v.toFixed(2)}s`}
              />
              <Slider
                label="Wobble — depth"
                hint="Modulates the bite frequency up and down. Creates drift/warble — subtle unease or overt tremor."
                value={params.wobbleDepth}
                min={0} max={1} step={0.01}
                onChange={(v) => update({ wobbleDepth: v })}
                format={(v) => v.toFixed(2)}
              />
              <Slider
                label="Wobble — rate"
                hint="How fast the wobble sweeps. 0.5 Hz is a slow breathing drift; 5 Hz is a nervous flicker."
                value={params.wobbleRate}
                min={0.1} max={8} step={0.05}
                onChange={(v) => update({ wobbleRate: v })}
                format={(v) => `${v.toFixed(2)} Hz`}
              />

              <SectionLabel>Dynamics & Output</SectionLabel>
              <Slider
                label="React to sentence mood"
                hint="How much louder/tenser sentences push grit, bite and doubling up. 0 = static preset; 1 = fully dynamic."
                value={params.intensityResponsiveness}
                min={0} max={1} step={0.05}
                onChange={(v) => update({ intensityResponsiveness: v })}
                format={(v) => v.toFixed(2)}
              />
              <Slider
                label="Volume (output)"
                hint="Final makeup gain. Above 1.0 = louder; useful to compensate for the low-shelf cut."
                value={params.outputGain}
                min={0.5} max={2} step={0.05}
                onChange={(v) => update({ outputGain: v })}
                format={(v) => v.toFixed(2)}
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}

interface SliderProps {
  label: string;
  /** Tooltip text explaining what the slider actually does. Shown on
   *  hover via native `title` (zero-CSS, zero-dependency). */
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}

function Slider({ label, hint, value, min, max, step, onChange, format }: SliderProps) {
  return (
    <div title={hint}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: 10,
        marginBottom: 2,
      }}>
        <span style={{
          color: 'var(--text-secondary)',
          cursor: hint ? 'help' : 'default',
        }}>
          {label}
        </span>
        <span style={{ color: 'var(--red-bright)', fontVariantNumeric: 'tabular-nums' }}>
          {format ? format(value) : value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{
          width: '100%',
          accentColor: 'var(--red-bright)',
          height: 4,
        }}
      />
    </div>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div style={{
      fontSize: 9,
      textTransform: 'uppercase',
      letterSpacing: 2,
      color: 'var(--text-dim)',
      marginTop: 4,
      marginBottom: -2,
      fontFamily: 'var(--font-display)',
      borderTop: '1px solid var(--border-mid)',
      paddingTop: 8,
    }}>
      {children}
    </div>
  );
}
