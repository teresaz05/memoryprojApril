##files
- `run_pipeline.py`: top-level experiment entrypoint
- `best_of_many.py`: layer1 / merge best-of-many pipeline logic
- `layer1_merge.py`: layer1 bank construction and merge helper logic
- `cluster_bank.py`: cluster-bank parsing, normalization, scoring, and rendering
- `support.py`: embedding helper, summary-update helper, lightweight IO utilities
- `llm_backends.py`: LLM clients, token counting, prompt formatting base helpers
- `sample_metadata.py`: sample metadata passthrough
- `metrics.py`: exact-match helpe