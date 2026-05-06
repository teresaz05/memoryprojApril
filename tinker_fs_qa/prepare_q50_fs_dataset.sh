#!/usr/bin/env bash
# Build the per-question filesystem layout for the q50 support-only split.
# You can append extra args, e.g.:
#   ./prepare_q50_fs_dataset.sh --limit 5
set -euo pipefail
ROOT='/Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline'
python3 "$ROOT/tinker_fs_qa/prepare_support_doc_fs_dataset.py" \
  --source_jsonl "$ROOT/data/browsecomp_plus_support_only_q50_main.jsonl" \
  --out_dir "$ROOT/tinker_fs_qa/train_q50_fs" \
  --overwrite \
  "$@"
