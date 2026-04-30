# Usage

This folder is a self-contained version of the current best-of-n inference pipeline.

Main entrypoint:
- `run_pipeline.py`

Wrapper scripts:
- `run_qwen_s1.sh`
- `run_qwen_nobon.sh`
- `run_qwen_full.sh`
- `run_qwen_l1bon_nomerge.sh`
- `run_gem_s1.sh`
- `run_gem_nobon.sh`
- `run_gem_full.sh`
- `run_gem_l1bon_nomerge.sh`

## Variant meanings

### `qwen_s1` / `gem_s1`
- One layer1 attempt
- Multiple merge attempts
- This isolates best-of-n over merge search only

### `qwen_nobon` / `gem_nobon`
- One layer1 attempt
- One merge attempt
- This is the no-best-of-n htree baseline

### `qwen_full` / `gem_full`
- Multiple layer1 attempts
- Multiple merge attempts
- This is the full best-of-n pipeline

### `qwen_l1bon_nomerge` / `gem_l1bon_nomerge`
- Multiple layer1 attempts
- No merge stage
- This isolates best-of-n at layer1 only

## Example commands

Run help:

```bash
python /Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/run_pipeline.py --help
```

Run one Qwen full dry-run question:

```bash
/Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/run_qwen_full.sh \
  --dataset_jsonl /Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/data/browsecomp_plus_support_only_q50_main.jsonl \
  --out_jsonl /tmp/qwen_full_test.jsonl \
  --trace_jsonl /tmp/qwen_full_test.trace.jsonl \
  --manifest_json /tmp/qwen_full_test.manifest.json \
  --limit 1 \
  --dry_run \
  --skip_answer
```

Run one Gemini s1 dry-run question:

```bash
/Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/run_gem_s1.sh \
  --dataset_jsonl /Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/exact_pipeline/data/browsecomp_plus_support_only_q50_main.jsonl \
  --out_jsonl /tmp/gem_s1_test.jsonl \
  --trace_jsonl /tmp/gem_s1_test.trace.jsonl \
  --manifest_json /tmp/gem_s1_test.manifest.json \
  --limit 1 \
  --dry_run \
  --skip_answer
```

## Notes

- All wrapper scripts forward any extra CLI arguments to `run_pipeline.py`.
- Override dataset paths, output paths, temperatures, or other knobs by passing them after the wrapper name.
- Qwen wrappers use `openrouter` + `qwen/qwen3.5-35b-a3b` by default.
- Gemini wrappers use `gemini` + `gemini-2.5-flash-lite` by default.
