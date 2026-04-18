#!/usr/bin/env bash
# Dev server launcher.
# Usage:
#   ./scripts/dev.sh [port]         # single process, stable (default)
#   RELOAD=1 ./scripts/dev.sh [port] # auto-reload on file change (doubles memory)
#
# --reload is off by default because it spawns a watcher subprocess that
# imports the full app a second time; on memory-tight machines the OOM killer
# targets the child and uvicorn dies with `zsh: killed`.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-8000}"

cd "$REPO"

# Load .env if present
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

# Editable-install fallback: editable .pth isn't honored by anaconda-based venv,
# so we prepend src/ to PYTHONPATH explicitly.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Start postgres if using docker-compose and it isn't already up
if command -v docker-compose >/dev/null 2>&1; then
  docker-compose up -d postgres >/dev/null 2>&1 || true
fi

# Evict any zombie listener on the port (uvicorn --reload leaves TIME_WAIT sockets)
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "⚠  port ${PORT} is in use — killing squatter"
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 2>/dev/null || true
  sleep 1
fi

echo "→ http://127.0.0.1:${PORT}"
UVICORN_ARGS=(
  evidenceengine.api.app:app
  --host 127.0.0.1
  --port "$PORT"
)
if [[ "${RELOAD:-0}" == "1" ]]; then
  # One reload-dir, not three overlapping ones — watchfiles on macOS doesn't
  # like nested watchers.
  UVICORN_ARGS+=(--reload --reload-dir "$REPO/src")
fi

exec "$REPO/.venv/bin/uvicorn" "${UVICORN_ARGS[@]}"
