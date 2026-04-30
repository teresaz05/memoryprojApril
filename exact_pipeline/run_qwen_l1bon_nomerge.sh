#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$DIR/run_pipeline.py" \
  --backend openrouter \
  --model qwen/qwen3.5-35b-a3b \
  --max_layer1_attempts 4 \
  --max_merge_attempts 4 \
  --skip_merge \
  "$@"
