#!/usr/bin/env bash
# Build the per-question filesystem layout for the full q830 support-only split.
# You can append extra args, e.g.:
#   ./prepare_q830_fs_dataset.sh --limit 20
set -euo pipefail
ROOT='/Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline'
python3 "$ROOT/tinker_fs_qa/prepare_support_doc_fs_dataset.py" \
  --source_jsonl "$ROOT/data/browsecomp_plus_support_only_all_q830.jsonl" \
  --out_dir "$ROOT/tinker_fs_qa/train_q830_fs" \
  --overwrite \
  "$@"
