#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$DIR/run_pipeline.py" \
  --backend openrouter \
  --model qwen/qwen3.5-35b-a3b \
  --single_layer1_attempt \
  --max_merge_attempts 4 \
  "$@"
