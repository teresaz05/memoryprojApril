
## Experiments included

1. `prose_merge2`
   - per-document prose cluster banks
   - layer-1 concatenation output
   - merge2 answer output
2. `structured_merge2`
   - per-document structured cluster banks
   - layer-1 concatenation output
   - merge2 answer output
3. `docsummaryaux_merge2`
   - per-document query-aware summaries plus cluster banks
   - layer-1 docsummaryaux output
   - merge2 answer output
4. `rlm_promptdocs_from_docsummaryaux`
   - official RLM
   - raw support documents kept directly in prompt context
   - one extra prompt document per source document containing the docsummaryaux summary and cluster-bank text

## Package layout

```text
april_version_code/
  data/
  scripts/
  src/april_version_code/
```

The copied heavy runners live in `src/april_version_code/methods/`.
The user-facing entrypoints live in `scripts/`.

## Data included

The package contains the source and derived data needed for the current experiment matrix:

- raw BrowseComp+ JSONL and qrels
- the controlled support-only q799 source used to build q100
- the derived q100 support-only dataset
- the derived q50 support-only dataset used by the current runs

The data files were created as hard links when possible so the package is self-contained without
wasting disk space in the original workspace.

## Environment setup

Use Python 3.11 or newer.

```bash
cd /Users/teresaz/Downloads/cs191ResearchWinter/BrowseCompV2/april_version_code
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.lock.txt
cp .env.example .env
```

Then fill in `.env` with the backend credentials you plan to use.

## Running the current experiments

All wrappers default to the packaged q50 support-only dataset.

### Prose merge2

```bash
python scripts/run_prose_merge2.py --model qwen/qwen3.5-35b-a3b
```

### Structured merge2

```bash
python scripts/run_structured_merge2.py --model qwen/qwen3.5-35b-a3b
```

### Docsummaryaux merge2

```bash
python scripts/run_docsummaryaux_merge2.py --model qwen/qwen3.5-35b-a3b
```

### Prompt-doc RLM from docsummaryaux

This experiment needs the layer-1 docsummaryaux result JSONL from a completed docsummaryaux run.

```bash
python scripts/run_rlm_promptdocs_from_docsummaryaux.py \
  --model qwen/qwen3.5-35b-a3b \
  --docsummaryaux-run-dir /path/to/docsummaryaux/run
```

If you already know the exact result JSONL path, use `--docsummaryaux-results-jsonl` instead.

### Running the second model

Every wrapper accepts a different generation model, so the qwen3-coder runs are the same commands
with `--model qwen/qwen3-coder:exacto`.

## Output layout

Wrapper scripts create runs under:

```text
runs/support_only_q50/<model-slug>/<experiment-name>/<timestamp>/
```

Each run directory keeps the familiar subdirectories:

- `results/`
- `graded/`
- `traces/` for cluster-bank runs
- `logs/` for the prompt-doc RLM run

## Grading an existing result file

```bash
python scripts/grade_run.py \
  --in-jsonl /path/to/result.jsonl \
  --out-jsonl /path/to/graded.jsonl
```

## Rebuilding the packaged datasets

The package includes builders for the current q100 and q50 support-only datasets:

```bash
python -m april_version_code.data.build_support_only_q100
python -m april_version_code.data.build_support_only_q50
```
