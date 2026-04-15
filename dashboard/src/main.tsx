import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// NOTE: StrictMode is disabled because it double-mounts components in dev,
// which causes duplicate WebSocket connections and duplicate TTS audio playback.
createRoot(document.getElementById('root')!).render(<App />)
