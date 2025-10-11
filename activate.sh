#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
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

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
echo "✓ Activated $(python -V) in $VENV"
echo "  PYTHONPATH=$PYTHONPATH"

