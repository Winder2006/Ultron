/**
 * PM2 ecosystem config for ULTRON.
 *
 * What this does:
 *   - Starts the FastAPI backend under the repo's `.venv` Python
 *   - Starts the Vite dashboard dev server
 *   - Both auto-restart on crash and (once `pm2 startup` + `pm2 save`
 *     are run) auto-start at boot
 *
 * Usage:
 *   pm2 start ecosystem.config.js     # boot both services
 *   pm2 status                         # live list + state
 *   pm2 logs                           # tail all logs
 *   pm2 logs ultron-backend            # just backend
 *   pm2 restart ultron-backend         # restart one
 *   pm2 stop all                       # halt everything
 *   pm2 monit                          # live CPU/RAM dashboard
 *
 * After changing this file:
 *   pm2 reload ecosystem.config.js     # zero-downtime restart
 *   pm2 save                           # persist new state so boot works
 *
 * Platform note: this file is Windows-first (uses .venv\\Scripts\\python).
 * On Linux/macOS, swap `script` to `.venv/bin/python` — or use
 * `python3 -m mother` if the venv isn't the default.
 */

const path = require('path');
const repoRoot = __dirname;

// Absolute path to the venv's Python so PM2 doesn't care what's on
// PATH when it spawns the process (especially important for
// boot-time start, where shell PATH isn't set yet).
const pythonExe = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');

module.exports = {
  apps: [
    {
      // FastAPI backend — serves /ws/voice, /health, REST endpoints
      name: 'ultron-backend',
      script: pythonExe,
      args: ['-m', 'mother', '--server'],
      cwd: repoRoot,

      // Auto-restart rules
      autorestart: true,
      max_restarts: 10,       // if it dies more than 10x rapidly, give up
      min_uptime: '30s',      // "rapidly" = died within 30s of starting
      restart_delay: 2000,    // 2s pause between restarts — don't thrash

      // Memory ceiling — restart if RSS exceeds this. Catches slow
      // leaks before they eat the host. 2GB is generous for the LLM
      // clients + embedding models.
      max_memory_restart: '2G',

      // Logs: keep them in the repo so they're discoverable, but
      // outside the source tree's main folders so they don't clutter.
      out_file: path.join(repoRoot, 'logs', 'pm2-backend-out.log'),
      error_file: path.join(repoRoot, 'logs', 'pm2-backend-err.log'),
      merge_logs: true,
      time: true,             // prefix each log line with timestamp

      env: {
        PYTHONUNBUFFERED: '1', // flush prints immediately → live `pm2 logs`
      },
    },

    {
      // RAG service — FAISS over notes + code index, embeddings via
      // MiniLM ONNX. Backend's voice route hits it for self-awareness
      // (search_code tool, code-aware context injection) and notes RAG.
      // Optional: if this process is down, the backend falls back to
      // empty RAG silently — Ultron still works, just loses memory of
      // his own architecture and the user's notes.
      //
      // Bound to 127.0.0.1 only — never expose this directly. The
      // backend is the only legitimate caller; the dashboard never
      // talks to it.
      name: 'ultron-rag',
      script: pythonExe,
      args: [
        '-m', 'uvicorn',
        'assistant.app:app',
        '--host', '127.0.0.1',
        '--port', '8123',
      ],
      cwd: repoRoot,

      autorestart: true,
      max_restarts: 10,
      min_uptime: '30s',
      restart_delay: 2000,
      // FAISS index + MiniLM session sit in RAM. 1.5GB is generous
      // headroom — typical RSS is ~400-600MB.
      max_memory_restart: '1500M',

      out_file: path.join(repoRoot, 'logs', 'pm2-rag-out.log'),
      error_file: path.join(repoRoot, 'logs', 'pm2-rag-err.log'),
      merge_logs: true,
      time: true,

      env: {
        PYTHONUNBUFFERED: '1',
      },
    },

    {
      // Vite dev server for the React dashboard.
      //
      // PM2 on Windows doesn't play well with `npm.cmd` — it tries to
      // parse the .cmd file as JavaScript and crashes. The reliable
      // pattern is to point `script` at vite's actual JS entry and
      // let PM2 invoke node on it directly. `--host` makes Vite bind
      // to 0.0.0.0 so you can reach it from other devices on the LAN
      // (phone, desktop once it's set up).
      name: 'ultron-dashboard',
      script: path.join(repoRoot, 'dashboard', 'node_modules', 'vite', 'bin', 'vite.js'),
      args: ['--host'],
      cwd: path.join(repoRoot, 'dashboard'),
      // Tell PM2 to use node as the interpreter (the default is
      // auto-detect-from-extension, which is .js → node anyway, but
      // being explicit avoids surprises on re-install).
      interpreter: 'node',

      autorestart: true,
      max_restarts: 10,
      min_uptime: '30s',
      restart_delay: 2000,
      max_memory_restart: '1G',

      out_file: path.join(repoRoot, 'logs', 'pm2-dashboard-out.log'),
      error_file: path.join(repoRoot, 'logs', 'pm2-dashboard-err.log'),
      merge_logs: true,
      time: true,
    },
  ],
};
