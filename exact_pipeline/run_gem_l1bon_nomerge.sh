#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$DIR/run_pipeline.py" \
  --backend gemini \
  --model gemini-2.5-flash-lite \
  --max_layer1_attempts 4 \
  --max_merge_attempts 4 \
  --skip_merge \
  "$@"
