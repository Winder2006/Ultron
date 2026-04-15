import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Keep onnxruntime-web out of the pre-bundling step — Vite's dep
  // optimizer tries to rewrite its internal WASM loader otherwise,
  // which breaks the `wasmPaths` override we set at runtime.
  optimizeDeps: {
    exclude: ['onnxruntime-web', 'openwakeword-js'],
  },
  server: {
    port: 3000,
    // Large .onnx / .wasm files in public/ need to be served as-is
    // with long-ish cache so the dashboard doesn't re-download them
    // on every reload during dev.
    fs: { strict: false },
    proxy: {
      '/api': {
        target: 'http://localhost:8300',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8300',
        ws: true,
      },
      '/health': {
        target: 'http://localhost:8300',
        changeOrigin: true,
      },
    },
  },
})
