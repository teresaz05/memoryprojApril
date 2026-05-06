# Tinker 1 --> training a model to answer a question by reading files from a per-question folder


Files:
- `prepare_support_doc_fs_dataset.py`: materializes one directory of support-doc files per question and writes an `index.jsonl`.
- `filesystem_qa_env.py`: simplified full-document-exposure environment, reward function, env-group builder, and RL dataset builder.
- `train_filesystem_qa_rl.py`: Tinker RL entrypoint.


- `prepare_q50_fs_dataset.sh`: materialize the q50 dataset into filesystem form.
- `prepare_q830_fs_dataset.sh`: materialize the full q830 support-only dataset into filesystem form.
- `train_q50.sh`: wrapper to train against `train_q50_fs/index.jsonl`.
- `train_q50_hybrid.sh`: wrapper to train against `train_q50_fs/index.jsonl` with `reward_mode=hybrid`.
- `train_q830.sh`: wrapper to train against `train_q830_fs/index.jsonl`.

Intended flow:
1. Prepare folders from a support-only dataset
2. Point the RL trainer at the resulting `index.jsonl`.
3. Train a model that answers questions after seeing all support files in order and then returning `Answer: ...`.

Prepared dataset layout:
- `agent_data/<qid>/*.txt`
  - the only files visible to the model tools
- `privileged_data/<qid>/query.txt`
- `privileged_data/<qid>/answer.txt`
- `privileged_data/<qid>/manifest.json`
  - privileged supervision/evaluation metadata kept separate from the agent-visible files

Reward:
- `1.0` for a correct normalized final answer.
- `0.0` for an incorrect answer.

Reward modes:
- `reward_mode=exact`
  - default
  - uses normalized exact match only
- `reward_mode=llm`
  - uses an LLM judge to decide whether the extracted answer should count as correct
- `reward_mode=hybrid`
  - uses exact match first, then falls back to the LLM judge if exact match fails

The reward metrics also log:
- `format`
- `correct`
- `exact_match`
- `judge_used`
- `judge_score`
- `read_calls`
- `list_calls`