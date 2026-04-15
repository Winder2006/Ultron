/**
 * VoiceOrb — Dramatic holographic orb representing MOTHER's state.
 *
 * Features concentric rotating rings, intense glow, and particle spray effects.
 * States: idle (breathing), listening (rings expand), thinking (spin), speaking (pulse out).
 */
import { motion, AnimatePresence } from 'framer-motion';
import { useCallback, useState } from 'react';

export type OrbState = 'idle' | 'listening' | 'thinking' | 'speaking';

interface VoiceOrbProps {
  state: OrbState;
  activity: number;
  onPress?: () => void;
  onRelease?: () => void;
}

const stateColors: Record<OrbState, string> = {
  idle: '#ff2244',
  listening: '#ff4466',
  thinking: '#ffaa44',
  speaking: '#ff3355',
};

export default function VoiceOrb({ state, activity, onPress, onRelease }: VoiceOrbProps) {
  const [pressed, setPressed] = useState(false);
  const size = 140;
  const color = stateColors[state];

  const handleDown = useCallback(() => {
    setPressed(true);
    onPress?.();
  }, [onPress]);

  const handleUp = useCallback(() => {
    setPressed(false);
    onRelease?.();
  }, [onRelease]);

  return (
    <div
      style={{
        position: 'relative',
        width: size + 80,
        height: size + 80,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        userSelect: 'none',
      }}
      onMouseDown={handleDown}
      onMouseUp={handleUp}
      onTouchStart={handleDown}
      onTouchEnd={handleUp}
    >
      {/* Outermost halo glow */}
      <motion.div
        style={{
          position: 'absolute',
          width: size + 60,
          height: size + 60,
          borderRadius: '50%',
          background: `radial-gradient(circle, ${color}15 0%, transparent 70%)`,
          filter: 'blur(20px)',
        }}
        animate={{
          scale: state === 'idle' ? [1, 1.15, 1] : [1, 1.3, 1],
          opacity: state === 'idle' ? [0.3, 0.5, 0.3] : [0.5, 0.8, 0.5],
        }}
        transition={{ duration: state === 'thinking' ? 1 : 3, repeat: Infinity, ease: 'easeInOut' }}
      />

      {/* Rotating holographic ring 1 */}
      <motion.div
        style={{
          position: 'absolute',
          width: size + 30,
          height: size + 30,
          borderRadius: '50%',
          border: `1px solid ${color}33`,
          boxShadow: `0 0 15px ${color}22, inset 0 0 15px ${color}11`,
        }}
        animate={{
          rotate: 360,
          scale: state === 'listening' ? [1, 1.15, 1] : 1,
        }}
        transition={{
          rotate: { duration: 20, repeat: Infinity, ease: 'linear' },
          scale: { duration: 2, repeat: Infinity, ease: 'easeInOut' },
        }}
      />

      {/* Rotating holographic ring 2 (counter-rotate) */}
      <motion.div
        style={{
          position: 'absolute',
          width: size + 16,
          height: size + 16,
          borderRadius: '50%',
          border: `1px solid ${color}44`,
          boxShadow: `0 0 10px ${color}22`,
        }}
        animate={{
          rotate: -360,
          scale: state === 'listening' ? [1, 1.1, 1] : 1,
        }}
        transition={{
          rotate: { duration: 15, repeat: Infinity, ease: 'linear' },
          scale: { duration: 1.5, repeat: Infinity, ease: 'easeInOut' },
        }}
      />

      {/* Pulse rings on active states */}
      <AnimatePresence>
        {(state === 'listening' || state === 'speaking') && (
          <>
            {[0, 0.7, 1.4].map((delay) => (
              <motion.div
                key={delay}
                style={{
                  position: 'absolute',
                  width: size,
                  height: size,
                  borderRadius: '50%',
                  border: `1px solid ${color}`,
                }}
                initial={{ scale: 0.8, opacity: 0.6 }}
                animate={{ scale: 2.5, opacity: 0 }}
                exit={{ opacity: 0 }}
                transition={{
                  duration: 2.5,
                  repeat: Infinity,
                  ease: 'easeOut',
                  delay,
                }}
              />
            ))}
          </>
        )}
      </AnimatePresence>

      {/* Main orb body */}
      <motion.div
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          background: `radial-gradient(circle at 38% 32%, ${color}dd, ${color}66 50%, ${color}22 80%, transparent)`,
          boxShadow: `
            0 0 30px ${color}66,
            0 0 60px ${color}33,
            0 0 100px ${color}11,
            inset 0 0 30px ${color}44
          `,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          overflow: 'hidden',
        }}
        animate={{
          scale: pressed
            ? 0.88
            : state === 'idle'
            ? [1, 1.04, 1]
            : state === 'thinking'
            ? [1, 1.02, 0.98, 1]
            : [1, 1.06, 1],
        }}
        transition={{
          duration: state === 'thinking' ? 0.8 : 2.5,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      >
        {/* Specular highlight */}
        <div
          style={{
            position: 'absolute',
            top: '15%',
            left: '20%',
            width: '35%',
            height: '25%',
            borderRadius: '50%',
            background: 'radial-gradient(ellipse, rgba(255,255,255,0.25) 0%, transparent 70%)',
            transform: 'rotate(-20deg)',
          }}
        />

        {/* Inner core */}
        <motion.div
          style={{
            width: size * 0.3,
            height: size * 0.3,
            borderRadius: '50%',
            background: `radial-gradient(circle, #fff 0%, ${color} 60%, transparent 100%)`,
            boxShadow: `0 0 20px ${color}aa`,
          }}
          animate={{
            opacity: [0.7, 1, 0.7],
            scale: state === 'thinking' ? [1, 1.4, 1] : [1, 1.15, 1],
          }}
          transition={{
            duration: state === 'thinking' ? 0.5 : 1.5,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </motion.div>

      {/* State label */}
      <div
        style={{
          position: 'absolute',
          bottom: 2,
          fontSize: 10,
          textTransform: 'uppercase',
          letterSpacing: 3,
          color,
          opacity: 0.7,
          textShadow: `0 0 10px ${color}66`,
        }}
      >
        {state}
      </div>
    </div>
  );
}
