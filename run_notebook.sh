#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate environment
source "$ROOT/activate.sh"

# Launch Jupyter from project root
cd "$ROOT"
echo "🚀 Launching Jupyter Notebook..."
echo "📂 Working directory: $ROOT"
jupyter notebook "$@"

