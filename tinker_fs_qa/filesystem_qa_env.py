from __future__ import annotations
"""Tinker RL environment for question answering after full filesystem document exposure.

Current design:
1. The question is given to the model at the beginning.
2. The environment automatically reads every support-document file from disk.
3. Those files are inserted into the conversation in a fixed order.
4. After the last support document, the model gives one final answer.

So this file doesn't implement interactive tool use. It implements a simpler,
fully-controlled setting where the model is guaranteed to see every support file
before answering.
"""

import json
import random
import string
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, TypedDict

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder

FILESYSTEM_QA_INSTRUCTIONS = """You will be given a question followed by all support documents for that question.

Read the support documents carefully. Use only the information supported by those documents.

When you are ready to answer, your entire final assistant message must be exactly one non-empty line:
Answer: <final answer>

Do not include explanation, reasoning, citations, bullet points, JSON, or extra lines.
Do not invent facts not supported by the support documents.
"""


class FilesystemQADatum(TypedDict):
    # In-memory shape of one training example after combining the index row
    # with privileged files on disk.
    question_id: str
    question: str
    gold_answer: str
    agent_query_dir: str
    privileged_query_dir: str
    num_docs: int
    dataset_type: str
    files: list[dict[str, Any]]


@dataclass
class FilesystemAnswerOnlyMessageEnv(MessageEnv):
    """Very small message environment for the new simplified design.

    The environment gives the model the question plus every support document in
    order. The model then produces one final answer. There are no tools and no
    multi-step interaction policy anymore.
    """

    initial_messages: list[Message]
    reward_fn: Any
    history: list[Message] = field(default_factory=list)

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    async def step(self, message: Message) -> MessageStepResult:
        self.history.append(message)
        reward, reward_metrics = await self.reward_fn(self.history)
        logs: dict[str, Any] = {}
        assistant_text = get_text_content(message)
        if assistant_text:
            logs["assistant_content"] = assistant_text
        return MessageStepResult(
            reward=reward,
            episode_done=True,
            next_messages=self.history,
            metrics=reward_metrics,
            logs=logs,
        )


@dataclass
class FilesystemAnswerReward:
    """Reward function for one completed episode.

    reward_mode controls how we decide if an answer is correct:
    - exact: normalized exact match only
    - llm: use an LLM judge only
    - hybrid: exact match first, LLM judge as fallback
    """

    gold_answer: str
    question: str
    reward_mode: str = "exact"
    judge_model: str = "qwen/qwen3.5-35b-a3b"
    judge_base_url: str = "https://openrouter.ai/api/v1"
    judge_api_key_env: str = "OPENROUTER_API_KEY"

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        final_message = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                final_message = msg
                break

        if final_message is None:
            return 0.0, {
                "format": 0.0,
                "correct": 0.0,
                "exact_match": 0.0,
                "judge_used": 0.0,
                "judge_score": 0.0,
            }

        content = get_text_content(final_message)
        extracted = self._extract_answer(content)
        correct_format = float(extracted is not None)
        exact_match = float(self._check_answer(extracted or ""))
        judge_used = 0.0
        judge_score = 0.0

        if extracted is None:
            correct_answer = 0.0
        elif self.reward_mode == "exact":
            correct_answer = exact_match
        elif self.reward_mode == "llm":
            judge_used = 1.0
            judge_score = await self._llm_judge_score(extracted)
            correct_answer = judge_score
        elif self.reward_mode == "hybrid":
            if exact_match >= 1.0:
                correct_answer = 1.0
            else:
                judge_used = 1.0
                judge_score = await self._llm_judge_score(extracted)
                correct_answer = judge_score
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        return correct_answer, {
            "format": correct_format,
            "correct": correct_answer,
            "exact_match": exact_match,
            "judge_used": judge_used,
            "judge_score": judge_score,
        }

    def _extract_answer(self, text: str) -> str | None:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(lines) != 1:
            return None
        line = lines[0]
        prefix = "Answer:"
        if not line.startswith(prefix):
            return None
        answer = line[len(prefix) :].strip()
        return answer or None

    def _check_answer(self, model_answer: str) -> bool:
        return normalize_answer(model_answer) == normalize_answer(self.gold_answer)

    async def _llm_judge_score(self, model_answer: str) -> float:
        prompt = (
            "You are judging whether a predicted answer should count as correct for a question. "
            "Return only JSON in the form {\"correct\": 0 or 1}.\n\n"
            f"Question: {self.question}\n"
            f"Gold answer: {self.gold_answer}\n"
            f"Predicted answer: {model_answer}\n\n"
            "Count semantically equivalent answers as correct even if formatting differs. "
            "Be strict about factual mismatch."
        )
        response_text = await asyncio.to_thread(self._call_judge_api, prompt)
        try:
            parsed = json.loads(response_text)
            val = float(parsed.get("correct", 0))
            return 1.0 if val >= 1.0 else 0.0
        except Exception:
            return 0.0

    def _call_judge_api(self, prompt: str) -> str:
        import os

        api_key = os.getenv(self.judge_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing judge API key in env var {self.judge_api_key_env}")

        url = self.judge_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.judge_model,
            "messages": [
                {"role": "system", "content": "Return only compact JSON. No explanation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Judge HTTP error {e.code}: {body[:500]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Judge URL error: {e}") from e

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"Judge response missing choices: {body}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(parts).strip()
        return str(content).strip()


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        import re

        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    transformations = [lower, remove_punc, remove_articles, white_space_fix]
    return reduce(lambda text, fn: fn(text), transformations, s)


def load_support_document_messages(datum: FilesystemQADatum) -> list[Message]:
    """Read every support-document file from disk and convert them into ordered messages.

    The order is taken from the per-example file manifest when available. Each
    support document becomes its own user message so the model sees the files in
    a clear one-by-one sequence before answering.
    """
    agent_dir = Path(datum["agent_query_dir"])
    file_entries = datum.get("files", [])

    rel_paths = [
        str(file_info.get("relative_path", ""))
        for file_info in file_entries
        if str(file_info.get("relative_path", ""))
    ]
    if not rel_paths:
        rel_paths = [path.name for path in sorted(agent_dir.glob("*.txt")) if path.is_file()]

    messages: list[Message] = []
    total = len(rel_paths)
    for i, rel_path in enumerate(rel_paths, start=1):
        doc_path = agent_dir / rel_path
        if not doc_path.exists():
            raise ValueError(f"Missing support-document file for {datum['question_id']}: {doc_path}")
        doc_text = doc_path.read_text(encoding="utf-8").strip()
        messages.append(
            {
                "role": "user",
                "content": f"Support document {i} of {total}:\nFile: {rel_path}\n\n{doc_text}",
            }
        )
    return messages


def initial_messages(datum: FilesystemQADatum) -> list[Message]:
    """Build the full initial conversation for one example.

    New simplified design:
    - give the question first
    - then automatically provide every support document in order
    - then ask for one final answer
    """
    messages: list[Message] = [
        {"role": "system", "content": FILESYSTEM_QA_INSTRUCTIONS},
        {
            "role": "user",
            "content": (
                f"Question ID: {datum['question_id']}\n"
                f"Question: {datum['question']}\n\n"
                "You will now be shown every support document for this question, one by one. "
                "Read them carefully. After the last support document, return your final answer "
                "exactly as: Answer: <final answer>"
            ),
        },
    ]
    messages.extend(load_support_document_messages(datum))
    messages.append(
        {
            "role": "user",
            "content": (
                "You have now seen all support documents. Using only the support documents above, "
                "return your final answer as exactly one line in this format:\n"
                "Answer: <final answer>"
            ),
        }
    )
    return messages


@dataclass(frozen=True)
class FilesystemQAEnvGroupBuilder(EnvGroupBuilder):
    """Create the group of environments used for one RL problem."""

    datum: FilesystemQADatum
    model_name: str
    renderer_name: str | None
    group_size: int
    reward_mode: str = "exact"
    judge_model: str = "qwen/qwen3.5-35b-a3b"
    judge_base_url: str = "https://openrouter.ai/api/v1"
    judge_api_key_env: str = "OPENROUTER_API_KEY"
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)

        reward_fn = FilesystemAnswerReward(
            gold_answer=self.datum["gold_answer"],
            question=self.datum["question"],
            reward_mode=self.reward_mode,
            judge_model=self.judge_model,
            judge_base_url=self.judge_base_url,
            judge_api_key_env=self.judge_api_key_env,
        )

        envs: list[Env] = []
        base_messages = initial_messages(self.datum)
        for _ in range(self.group_size):
            msg_env = FilesystemAnswerOnlyMessageEnv(
                initial_messages=list(base_messages),
                reward_fn=reward_fn,
            )
            envs.append(
                EnvFromMessageEnv(
                    renderer=renderer,
                    message_env=msg_env,
                    failed_parse_reward=-0.1,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                    context_overflow_reward=self.context_overflow_reward,
                )
            )
        return envs

    def logging_tags(self) -> list[str]:
        return [self.datum.get("dataset_type", "support_only"), "filesystem_qa"]


class FilesystemQARLDataset(RLDataset):
    """A minimal RL dataset wrapper over a list of EnvGroupBuilders."""

    def __init__(self, env_group_builders: list[FilesystemQAEnvGroupBuilder], batch_size: int):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class FilesystemQADatasetBuilder(RLDatasetBuilder):
    """Tinker dataset builder for the simplified full-document-exposure design."""

    index_jsonl: str
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    reward_mode: str = "exact"
    judge_model: str = "qwen/qwen3.5-35b-a3b"
    judge_base_url: str = "https://openrouter.ai/api/v1"
    judge_api_key_env: str = "OPENROUTER_API_KEY"
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    seed: int = 0
    eval_size: int = 0
    limit: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        data = load_index(Path(self.index_jsonl))
        rng = random.Random(self.seed)
        rng.shuffle(data)
        if self.limit > 0:
            data = data[: self.limit]

        eval_rows: list[FilesystemQADatum] = []
        train_rows = data
        if self.eval_size > 0:
            eval_rows = data[: self.eval_size]
            train_rows = data[self.eval_size :]

        train_builders = [
            FilesystemQAEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                reward_mode=self.reward_mode,
                judge_model=self.judge_model,
                judge_base_url=self.judge_base_url,
                judge_api_key_env=self.judge_api_key_env,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
            )
            for datum in train_rows
        ]
        train_dataset = FilesystemQARLDataset(train_builders, batch_size=self.batch_size)

        if not eval_rows:
            return train_dataset, None

        eval_builders = [
            FilesystemQAEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                reward_mode=self.reward_mode,
                judge_model=self.judge_model,
                judge_base_url=self.judge_base_url,
                judge_api_key_env=self.judge_api_key_env,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
            )
            for datum in eval_rows
        ]
        eval_dataset = FilesystemQARLDataset(eval_builders, batch_size=self.batch_size)
        return train_dataset, eval_dataset


def load_index(path: Path) -> list[FilesystemQADatum]:
    """Load the index produced by prepare_support_doc_fs_dataset.py."""
    rows: list[FilesystemQADatum] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ["question_id", "agent_query_dir", "privileged_query_dir", "num_docs"]
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_num} missing required keys: {missing}")
            row = hydrate_privileged_record(row)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in index: {path}")
    return rows


def hydrate_privileged_record(row: dict[str, Any]) -> FilesystemQADatum:
    """Combine one index row with privileged files on disk."""
    qid = str(row["question_id"])
    privileged_query_dir = Path(row["privileged_query_dir"])
    query_path = privileged_query_dir / "query.txt"
    answer_path = privileged_query_dir / "answer.txt"

    if not query_path.exists():
        raise ValueError(f"Missing privileged query file for {qid}: {query_path}")
    if not answer_path.exists():
        raise ValueError(f"Missing privileged answer file for {qid}: {answer_path}")

    return {
        "question_id": qid,
        "question": query_path.read_text(encoding="utf-8").strip(),
        "gold_answer": answer_path.read_text(encoding="utf-8").strip(),
        "agent_query_dir": str(row["agent_query_dir"]),
        "privileged_query_dir": str(privileged_query_dir),
        "num_docs": int(row["num_docs"]),
        "dataset_type": str(row.get("dataset_type", "support_only")),
        "files": list(row.get("files", [])),
    }
