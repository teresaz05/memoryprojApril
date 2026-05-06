#!/usr/bin/env bash
# Train on the prepared q50 filesystem dataset with hybrid reward mode:
# - exact match first
# - LLM judge as a fallback
set -euo pipefail
ROOT='/Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/tinker_fs_qa'
python3 "$ROOT/train_filesystem_qa_rl.py" \
  "index_jsonl=$ROOT/train_q50_fs/index.jsonl" \
  "reward_mode=hybrid" \
  "$@"
