#!/usr/bin/env bash
# Train on the prepared q50 filesystem dataset.
# Extra configuration is forwarded as chz key=value arguments, for example:
#   ./train_q50.sh limit=5 eval_size=1
set -euo pipefail
ROOT='/Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/tinker_fs_qa'
python3 "$ROOT/train_filesystem_qa_rl.py" \
  "index_jsonl=$ROOT/train_q50_fs/index.jsonl" \
  "$@"
