#!/usr/bin/env bash
# Dev server launcher — works around anaconda-venv not processing .pth files
# Usage: ./scripts/dev.sh [port]

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

echo "→ http://127.0.0.1:${PORT}"
exec "$REPO/.venv/bin/uvicorn" \
  evidenceengine.api.app:app \
  --host 127.0.0.1 \
  --port "$PORT" \
  --reload \
  --reload-dir "$REPO/src" \
  --reload-dir "$REPO/src/evidenceengine/templates" \
  --reload-dir "$REPO/src/evidenceengine/static"
