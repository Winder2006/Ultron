import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import ExecView from './components/ExecView.tsx'

// NOTE: StrictMode is disabled because it double-mounts components in dev,
// which causes duplicate WebSocket connections and duplicate TTS audio playback.
//
// Tiny manual router: /exec → terminal view (intended for the second
// monitor); anything else → main dashboard. We avoid react-router
// because the app has only two routes and the dependency isn't worth it.
const path = window.location.pathname.replace(/\/+$/, '');
const root = createRoot(document.getElementById('root')!);
if (path === '/exec') {
  root.render(<ExecView />);
} else {
  root.render(<App />);
}
