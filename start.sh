#!/usr/bin/env bash
# One-command launcher for the agent memory demo.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — fill in your credentials, then re-run."
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

PORT="${BACKEND_PORT:-8060}"
echo "Starting demo at http://localhost:${PORT}"
exec uvicorn backend.main:app --port "${PORT}" --reload
