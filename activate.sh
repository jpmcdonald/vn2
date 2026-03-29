#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Prefer uv-managed .venv when uv.lock exists (fixes stale pyvenv.cfg / dyld paths after moves or uv upgrades).
VENV=""
if [[ -f "$ROOT/uv.lock" ]] && command -v uv >/dev/null 2>&1; then
  echo "Using uv (.venv). For a clean repair: rm -rf .venv && uv sync"
  (cd "$ROOT" && uv sync ${UV_SYNC_ARGS:-})
  VENV="$ROOT/.venv"
  source "$VENV/bin/activate"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  VENV="$ROOT/.venv"
  source "$VENV/bin/activate"
else
  VENV="$ROOT/V2env"
  if [[ ! -d "$VENV" ]]; then
    echo "Creating venv at $VENV"
    python3 -m venv "$VENV"
  fi
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip wheel --quiet

  REQ="$ROOT/requirements.txt"
  STAMP="$VENV/.reqs.sha256"

  if [[ -f "$REQ" ]]; then
    CURR_SHA="$(shasum -a 256 "$REQ" | awk '{print $1}')"
    PREV_SHA="$( [[ -f "$STAMP" ]] && cat "$STAMP" || echo "" )"
    if [[ "$CURR_SHA" != "$PREV_SHA" ]]; then
      echo "Installing/upgrading requirements..."
      pip install -r "$REQ" --quiet
      echo "$CURR_SHA" > "$STAMP"
    fi
  fi
fi

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
echo "✓ Activated $(python -V) in $VENV"
echo "  PYTHONPATH=$PYTHONPATH"

