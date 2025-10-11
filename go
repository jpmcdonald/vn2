#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate environment
source "$ROOT/activate.sh" > /dev/null 2>&1

# Run CLI
python -m vn2.cli "$@"

