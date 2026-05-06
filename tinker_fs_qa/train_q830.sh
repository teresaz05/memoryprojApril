#!/usr/bin/env bash
# Train on the prepared q830 filesystem dataset.
# Because q830 is much larger, you may want to start with:
#   ./train_q830.sh limit=20 eval_size=2
set -euo pipefail
ROOT='/Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/tinker_fs_qa'
python3 "$ROOT/train_filesystem_qa_rl.py" \
  "index_jsonl=$ROOT/train_q830_fs/index.jsonl" \
  "$@"
