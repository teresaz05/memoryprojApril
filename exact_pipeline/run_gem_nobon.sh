#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$DIR/run_pipeline.py" \
  --backend gemini \
  --model gemini-2.5-flash-lite \
  --single_layer1_attempt \
  --max_merge_attempts 1 \
  "$@"
