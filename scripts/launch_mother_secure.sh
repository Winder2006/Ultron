#!/bin/bash
set -euo pipefail

if [ "${EUID}" -eq 0 ]; then
  echo "Refusing to run as root." >&2
  exit 1
fi

export NO_REMOTE=1
export MOTHER_SECURE=1

# Show listeners
if command -v lsof >/dev/null 2>&1; then
  lsof -i -P -n | grep LISTEN || true
fi

# Start eDEX-UI (local)
( cd edex-ui && npm start ) &

sleep 4

python -m src.main
