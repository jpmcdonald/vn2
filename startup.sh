#!/usr/bin/env bash
# Portable environment entrypoint (no hardcoded machine paths).
# Repairs a broken .venv (missing libpython dylib) by re-syncing with uv.
#
#   source ./startup.sh
#
# Optional: UV_SYNC_ARGS="--frozen" source ./startup.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

_activate() {
  # shellcheck disable=SC1091
  source "$1/bin/activate"
}

_fail() {
  echo "[startup] $*" >&2
  return 1 2>/dev/null || exit 1
}

if command -v uv >/dev/null 2>&1 && [[ -f "$ROOT/uv.lock" ]]; then
  echo "[startup] uv sync → $ROOT/.venv"
  uv sync ${UV_SYNC_ARGS:-}
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    _fail ".venv/bin/python missing after uv sync"
  fi
  _activate "$ROOT/.venv"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  echo "[startup] activating existing .venv (no uv sync; run 'uv sync' if dylib errors)"
  _activate "$ROOT/.venv"
elif [[ -f "$ROOT/V2env/bin/activate" ]]; then
  echo "[startup] activating V2env (legacy)"
  _activate "$ROOT/V2env"
else
  _fail "No usable venv. Install https://github.com/astral-sh/uv then: cd \"$ROOT\" && uv sync"
fi

echo "[startup] $(python -V) | VIRTUAL_ENV=${VIRTUAL_ENV:-}"
