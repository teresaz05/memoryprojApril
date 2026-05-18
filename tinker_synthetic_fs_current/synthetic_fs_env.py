from __future__ import annotations

import asyncio
import difflib
import gc
import json
import logging
import os
import random
import re
import string
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import reduce
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, TypedDict

import chz
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import simple_tool_result, tool
from tinker_cookbook.tool_use.tools import handle_tool_call
from tinker_cookbook.tool_use.types import Tool, ToolResult

NodeKind = Literal["cluster", "merge_summary"]
JudgeBackend = Literal["openrouter", "gemini"]
AnswererWorkspaceMode = Literal["synthetic_only"]
READ_FILE_MAX_LINES = 100
READ_FILE_MAX_CHARS = 16000
READ_MANY_MAX_FILES = 8
ANSWER_PREFIX = "Answer:"
STOP_MESSAGE = "STOP"
MODEL_HTTP_TIMEOUT_SECONDS = 300
MODEL_HTTP_MAX_ATTEMPTS = 8
MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS = 60
BUILDER_COMPACTION_SUMMARY_MARKER = "[COMPACTED BUILDER CONTEXT SUMMARY]"
APPROX_CHARS_PER_TOKEN = 4
DEFAULT_BUILDER_COMPACTION_TRIGGER_TOKENS = 8000
DEFAULT_BUILDER_COMPACTION_KEEP_RECENT_TURNS = 2
DEFAULT_BUILDER_COMPACTION_MAX_OUTPUT_TOKENS = 1200
DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS = 60000
BAD_SYNTHETIC_TEXT_VALUES = {"", "none", "null", "n/a", "na"}
LOGGER = logging.getLogger(__name__)

BUILDER_SYSTEM_PROMPT = f"""You are building a synthetic filesystem for a later answering agent.

Use the provided filesystem tools to inspect files under:
- raw_docs/
- synthetic_fs/
- history/

Your job is to leave behind a final active synthetic filesystem that helps later agents answer as many plausible future questions about this corpus as possible.
Do not optimize for one guessed target question; build broad, reusable knowledge structure over the corpus.
The later answering agent will start with root-level synthetic entries, browse deeper synthetic files if needed, and use raw_docs/ only as fallback.

In this pipeline, a cluster is one synthetic memory bank for one coherent family of plausible future information needs.
Choose cluster boundaries by downstream retrieval need:
- group facts together when a later answerer would likely want to retrieve them together
- split facts apart when they would support different kinds of future questions
- do not assume a fixed template; let the evidence determine the right cluster shape and scope

Use the tools as follows:
- list_files and read_file inspect the available files; use list_files on paths like raw_docs/ or synthetic_fs/ to browse incrementally
- create_cluster creates exactly one new synthetic file from selected inputs
- merge_clusters groups similar or related active synthetic files under a parent summary; do not merge arbitrary unrelated files
- delete removes active synthetic_fs/ files from the final active view

Important rules:
- only call read_file on exact paths returned by list_files or prior successful tool results; never guess raw_docs/ paths from memory or URL names
- if read_file says a raw document is not visible, do not retry that path; call list_files("raw_docs") and continue with only listed files
- you may include brief reasoning when it helps choose the next action, but keep it short and action-oriented
- avoid long planning monologues, numbered lists of possible next steps, or repeated status summaries
- after reasoning, take a concrete tool action unless you are returning STOP
- use only information supported by the files
- do not produce final answers directly; your job is to build useful synthetic files
- make each cluster a coherent memory bank rather than a loose dump of unrelated facts
- if two facts would likely be retrieved for different purposes, put them in different clusters
- if several clusters belong under a broader theme, create the local clusters first and merge them later
- do not wait to read everything before synthesizing; once you have enough evidence for one coherent memory bank, create it
- work in short synthesis cycles: after a few read/list calls, create or merge a durable synthetic file
- avoid long stretches of read/list-only turns; extra reading is useful only when it leads to a new or improved synthetic file
- returning STOP without creating any synthetic file is usually a failure
- a good early success is to create one focused cluster from a small set of closely related files, then refine or merge later if needed
- when files are long, read only the chunk you need instead of trying to ingest the whole document at once
- preserve salient facts, names, dates, places, relationships, aliases, titles, and other high-signal details that would help with many plausible questions about these documents
- prefer many focused leaf clusters organized under a smaller number of parent merge summaries
- root-level entries should be useful navigation summaries, not a flat dump of every leaf cluster
- use delete only on active synthetic_fs/ paths; never pass raw_docs/ or history/ to delete
- if a raw document is irrelevant, ignore it instead of trying to delete it
- if the final synthetic filesystem has broad coverage, many focused leaf clusters, and useful merge-summary structure, stop immediately
- do not spend the full turn budget just because more turns are available; stop proactively when the filesystem is mature

When you are ready to finish, your entire final assistant message must be exactly one non-empty line:
{STOP_MESSAGE}

Do not emit <think> tags or long hidden-reasoning dumps.
Outside tool calls, keep any free text short and directly tied to the next action or STOP decision.
"""


def build_builder_system_prompt(*, executor_enabled: bool, batch_tools_enabled: bool) -> str:
    prompt = BUILDER_SYSTEM_PROMPT
    if executor_enabled:
        prompt += """

Executor mode is active:
- you are the planner/controller, not the long-form cluster writer
- create_cluster and merge_clusters call a frozen executor model to write the synthetic file content
- provide concise titles, exact source paths/handles, and short generation_instructions describing the intended scope
- do not spend tokens writing full cluster_text or merge_summary_text; the executor writes those from the selected source files
- the executor may only use the files you pass as sources, so source selection and grouping remain your responsibility
"""
    if batch_tools_enabled:
        prompt += """

Batched tools are available:
- read_many reads several exact paths in one tool call
- create_clusters creates several independent leaf clusters in one tool call
- merge_many creates several parent summaries in one tool call
Use batched tools when the operations are independent and already well-scoped. Prefer batching to avoid long read/read/create loops, but keep each operation coherent.
"""
    return prompt


def build_answerer_system_prompt(mode: AnswererWorkspaceMode) -> str:
    workspace_lines = [
        "You are the answering agent for a read-only workspace.",
        "",
        "You have filesystem tools over a final workspace that contains:",
        "- README.md explaining the workspace conventions and navigation only",
        "- root-level synthetic entries that are already loaded into context as the primary entry points",
        "- nested synthetic folders and summary files for more detail",
    ]

    order_lines = [
        "",
        "Use this order:",
        "1. start from the already-loaded README.md and root-level synthetic entries",
        "2. browse deeper synthetic files if needed",
    ]
    guidance = (
        "Do not guess. If the synthetic files you have read so far do not contain enough evidence to "
        "uniquely identify the answer, continue exploring the synthetic filesystem before answering. "
        "README.md only explains the workspace and is not sufficient evidence by itself. "
        "Answer only when the synthetic workspace contains enough evidence."
    )

    footer = [
        "",
        guidance,
        "",
        "At each step, either:",
        "- return exactly one compact JSON action:",
        '  {"action":"list_files","path":"."}',
        f'  {{"action":"read_file","path":"some/path.txt","start_line":1,"num_lines":{READ_FILE_MAX_LINES}}}',
        "- or return exactly one non-empty line:",
        f"  {ANSWER_PREFIX} <final answer>",
        "",
        "Do not include explanation outside the JSON action or final answer line.",
    ]
    return "\n".join(workspace_lines + order_lines + footer)


def build_answerer_user_prompt(question: str, mode: AnswererWorkspaceMode) -> str:
    return (
        f"Question: {question}\n\n"
        "Use the workspace tools to inspect the final workspace. "
        "README.md and the current root-level synthetic entries are already loaded into context. "
        "Answer using only the synthetic workspace."
    )


def call_chat_model_messages(
    *,
    backend: JudgeBackend,
    model: str,
    base_url: str,
    api_key_env: str,
    messages: Sequence[dict[str, str]],
    response_json: bool,
    max_output_tokens: int | None = None,
) -> str:
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {api_key_env}")

    if backend == "openrouter":
        url = base_url.rstrip("/") + "/chat/completions"
        system_messages = [m for m in messages if m.get("role") == "system"]
        non_system_messages = [m for m in messages if m.get("role") != "system"]
        payload: dict[str, Any] = {
            "model": model,
            "messages": [],
            "temperature": 0,
        }
        if max_output_tokens is not None and max_output_tokens > 0:
            payload["max_tokens"] = max_output_tokens
        if response_json and not system_messages:
            payload["messages"].append({"role": "system", "content": "Return only compact JSON."})
        payload["messages"].extend(system_messages)
        payload["messages"].extend(non_system_messages)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        body = urlopen_json_with_retry(req)

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"Response missing choices: {body}")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(parts).strip()
        return str(content).strip()

    if backend == "gemini":
        model_path = model if model.startswith("models/") else f"models/{model}"
        url = f"{base_url.rstrip('/')}/{model_path}:generateContent"
        system_texts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
        contents = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": message.get("content", "")}]})
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0,
            },
        }
        if max_output_tokens is not None and max_output_tokens > 0:
            payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
        if response_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        body = urlopen_json_with_retry(req)

        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Response missing candidates: {body}")
        content = ((candidates[0].get("content") or {}).get("parts") or [])
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("text")]
        return "".join(text_parts).strip()

    raise ValueError(f"Unknown backend: {backend}")


def urlopen_json_with_retry(req: urllib.request.Request) -> dict[str, Any]:
    last_error: Exception | None = None
    retryable_http_codes = {429, 500, 502, 503, 504}

    for attempt in range(1, MODEL_HTTP_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=MODEL_HTTP_TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code in retryable_http_codes and attempt < MODEL_HTTP_MAX_ATTEMPTS:
                time.sleep(min(2 ** (attempt - 1), MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS))
                last_error = RuntimeError(f"HTTP error {e.code}: {body[:500]}")
                continue
            raise RuntimeError(f"HTTP error {e.code}: {body[:500]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < MODEL_HTTP_MAX_ATTEMPTS:
                time.sleep(min(2 ** (attempt - 1), MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS))
                last_error = RuntimeError(f"URL/timeout error: {e}")
                continue
            raise RuntimeError(f"URL/timeout error: {e}") from e

    if last_error is not None:
        raise last_error
    raise RuntimeError("Request failed without a captured error")


class SyntheticFilesystemDatum(TypedDict):
    question_id: str
    question: str
    gold_answer: str
    agent_query_dir: str
    privileged_query_dir: str
    num_docs: int
    dataset_type: str
    files: list[dict[str, Any]]


@dataclass
class SyntheticFileRecord:
    node_id: str
    kind: NodeKind
    title: str
    slug: str
    content: str
    created_round: int
    parent_id: str | None = None
    direct_raw_doc_paths: list[str] = field(default_factory=list)
    direct_synthetic_ids: list[str] = field(default_factory=list)
    active: bool = True
    deleted_round: int | None = None
    delete_reason: str | None = None


@dataclass
class SyntheticFilesystemState:
    raw_doc_root: Path
    visible_raw_doc_paths: list[str]
    question_id: str
    files_by_id: dict[str, SyntheticFileRecord] = field(default_factory=dict)
    next_cluster_idx: int = 1
    next_merge_idx: int = 1
    operation_round: int = 0

    def _slugify(self, text: str) -> str:
        chars: list[str] = []
        for ch in text.strip().lower():
            if ch.isalnum():
                chars.append(ch)
            else:
                chars.append("_")
        slug = "".join(chars).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug or "untitled"

    def _cluster_node_id(self) -> str:
        node_id = f"cluster_{self.next_cluster_idx:04d}"
        self.next_cluster_idx += 1
        return node_id

    def _merge_node_id(self) -> str:
        node_id = f"merge_{self.next_merge_idx:04d}"
        self.next_merge_idx += 1
        return node_id

    def _record_sort_key(self, rec: SyntheticFileRecord) -> tuple[int, str, str]:
        return (0 if rec.kind == "merge_summary" else 1, rec.title.lower(), rec.node_id)

    def _children(self, parent_id: str | None, *, active_only: bool = True) -> list[SyntheticFileRecord]:
        records = [
            rec
            for rec in self.files_by_id.values()
            if rec.parent_id == parent_id and (rec.active or not active_only)
        ]
        return sorted(records, key=self._record_sort_key)

    def _dir_name(self, rec: SyntheticFileRecord) -> str:
        return f"{rec.node_id}__{rec.slug}"

    def _file_name(self, rec: SyntheticFileRecord) -> str:
        return f"{rec.node_id}__{rec.slug}.txt"

    def _active_path_parts(self, node_id: str) -> list[str]:
        rec = self.files_by_id[node_id]
        parts: list[str] = []
        parent_id = rec.parent_id
        while parent_id is not None:
            parent = self.files_by_id[parent_id]
            parts.append(self._dir_name(parent))
            parent_id = parent.parent_id
        parts.reverse()
        return parts

    def active_path(self, node_id: str) -> str:
        rec = self.files_by_id[node_id]
        parts = ["synthetic_fs", *self._active_path_parts(node_id)]
        if rec.kind == "merge_summary":
            return "/".join(parts + [self._dir_name(rec), "summary.txt"])
        return "/".join(parts + [self._file_name(rec)])

    def history_path(self, node_id: str) -> str:
        rec = self.files_by_id[node_id]
        if rec.kind == "merge_summary":
            return f"history/{self._dir_name(rec)}/summary.txt"
        return f"history/{self._file_name(rec)}"

    def workspace_relative_path(self, node_id: str) -> str:
        active = self.active_path(node_id)
        if not active.startswith("synthetic_fs/"):
            raise ValueError(f"Unexpected active synthetic path: {active}")
        return active[len("synthetic_fs/") :]

    def active_records(self) -> list[SyntheticFileRecord]:
        return sorted([rec for rec in self.files_by_id.values() if rec.active], key=lambda rec: self.active_path(rec.node_id))

    def _active_path_index(self) -> dict[str, str]:
        return {self.active_path(rec.node_id): rec.node_id for rec in self.active_records()}

    def _history_path_index(self) -> dict[str, str]:
        return {self.history_path(rec.node_id): rec.node_id for rec in self.files_by_id.values()}

    def raw_doc_rel_paths(self) -> list[str]:
        return [f"raw_docs/{rel}" for rel in self.visible_raw_doc_paths]

    def _raw_handle_index(self) -> dict[str, str]:
        return {
            f"raw_{idx:04d}": path
            for idx, path in enumerate(self.raw_doc_rel_paths(), start=1)
        }

    def _raw_listing_metadata(self) -> dict[str, dict[str, str]]:
        return {
            path: {"handle": handle}
            for handle, path in self._raw_handle_index().items()
        }

    def _clean_tool_path(self, path: str) -> str:
        clean = str(path).strip()
        clean = clean.strip("\"'`")
        clean = clean.replace("\\", "/")
        while clean.startswith("./"):
            clean = clean[2:]
        clean = clean.lstrip("/")
        return clean

    def _path_match_key(self, path: str) -> str:
        return re.sub(r"[^0-9a-z]+", "", self._clean_tool_path(path).lower())

    def _unique_close_path_match(self, path: str, candidates: Sequence[str]) -> str | None:
        clean_key = self._path_match_key(PurePosixPath(self._clean_tool_path(path)).name)
        if len(clean_key) < 16:
            return None
        scored: list[tuple[float, str]] = []
        for candidate in candidates:
            candidate_key = self._path_match_key(PurePosixPath(candidate).name)
            if len(candidate_key) < 16:
                continue
            score = difflib.SequenceMatcher(None, clean_key, candidate_key).ratio()
            if score >= 0.92:
                scored.append((score, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.03:
            return None
        return scored[0][1]

    def _unique_path_match(self, path: str, candidates: Sequence[str]) -> str | None:
        clean = self._clean_tool_path(path)
        if clean in candidates:
            return clean
        lower_matches = [candidate for candidate in candidates if candidate.lower() == clean.lower()]
        if len(lower_matches) == 1:
            return lower_matches[0]
        suffix_matches = [candidate for candidate in candidates if candidate.endswith(clean)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        lower_suffix_matches = [candidate for candidate in candidates if candidate.lower().endswith(clean.lower())]
        if len(lower_suffix_matches) == 1:
            return lower_suffix_matches[0]
        active_suffix_matches = [candidate for candidate in suffix_matches if candidate.startswith("synthetic_fs/")]
        if len(active_suffix_matches) == 1:
            return active_suffix_matches[0]
        basename = PurePosixPath(clean).name
        basename_matches = [candidate for candidate in candidates if PurePosixPath(candidate).name == basename]
        if len(basename_matches) == 1:
            return basename_matches[0]
        lower_basename_matches = [
            candidate
            for candidate in candidates
            if PurePosixPath(candidate).name.lower() == basename.lower()
        ]
        if len(lower_basename_matches) == 1:
            return lower_basename_matches[0]
        active_basename_matches = [candidate for candidate in basename_matches if candidate.startswith("synthetic_fs/")]
        if len(active_basename_matches) == 1:
            return active_basename_matches[0]
        clean_match_key = self._path_match_key(clean)
        normalized_matches = [
            candidate for candidate in candidates if self._path_match_key(candidate) == clean_match_key
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0]
        clean_basename_key = self._path_match_key(basename)
        normalized_basename_matches = [
            candidate
            for candidate in candidates
            if self._path_match_key(PurePosixPath(candidate).name) == clean_basename_key
        ]
        if len(normalized_basename_matches) == 1:
            return normalized_basename_matches[0]
        close_match = self._unique_close_path_match(clean, candidates)
        if close_match is not None:
            return close_match
        return None

    def _node_id_from_path_text(self, path: str) -> str | None:
        import re

        match = re.search(r"(?<![A-Za-z0-9_])(?:cluster|merge)_\d{4}(?!\d)", path)
        return match.group(0) if match else None

    def _raw_handle_from_path_text(self, path: str) -> str | None:
        clean = self._clean_tool_path(path)
        if clean.startswith("raw_docs/"):
            clean = clean[len("raw_docs/") :]
        if re.fullmatch(r"raw_\d{4}\.txt", clean):
            clean = clean[:-4]
        return self._raw_handle_index().get(clean)

    def _looks_like_raw_handle_path(self, path: str) -> bool:
        clean = self._clean_tool_path(path)
        if clean.startswith("raw_docs/"):
            clean = clean[len("raw_docs/") :]
        return bool(re.fullmatch(r"raw_\d{4}(?:\.txt)?", clean))

    def path_suggestions(self, path: str, *, limit: int = 8) -> list[str]:
        clean = self._clean_tool_path(path).lower()
        candidates = [
            *self.raw_doc_rel_paths(),
            *self._active_path_index().keys(),
            *self._history_path_index().keys(),
        ]
        if not clean:
            return candidates[:limit]
        scored: list[tuple[int, str]] = []
        clean_base = PurePosixPath(clean).name
        clean_key = self._path_match_key(clean)
        clean_base_key = self._path_match_key(clean_base)
        for candidate in candidates:
            candidate_lower = candidate.lower()
            candidate_base = PurePosixPath(candidate_lower).name
            candidate_key = self._path_match_key(candidate)
            candidate_base_key = self._path_match_key(candidate_base)
            score = 0
            if candidate_lower == clean:
                score += 100
            if candidate_key == clean_key:
                score += 95
            if candidate_lower.endswith(clean):
                score += 70
            if clean_base and candidate_base == clean_base:
                score += 60
            if clean_base_key and candidate_base_key == clean_base_key:
                score += 55
            if clean in candidate_lower:
                score += 30
            if clean_base and clean_base in candidate_base:
                score += 20
            if clean_base_key and clean_base_key in candidate_base_key:
                score += 15
            if clean_base_key and len(clean_base_key) >= 16:
                close_score = difflib.SequenceMatcher(None, clean_base_key, candidate_base_key).ratio()
                if close_score >= 0.92:
                    score += int(close_score * 40)
            node_id = self._node_id_from_path_text(clean)
            if node_id and node_id in candidate_lower:
                score += 80
            if score > 0:
                scored.append((score, candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, candidate in scored[:limit]]

    def resolve_raw_doc(self, path: str) -> Path:
        path = self._clean_tool_path(path)
        handle_match = self._raw_handle_from_path_text(path)
        if handle_match is not None:
            path = handle_match
        elif self._looks_like_raw_handle_path(path):
            raise ValueError(f"Raw document handle not visible in this run: {path}")
        if not path.startswith("raw_docs/"):
            matched = self._unique_path_match(path, self.raw_doc_rel_paths())
            if matched is None:
                raise ValueError(f"Not a raw document path: {path}")
            path = matched
        rel = path[len("raw_docs/") :]
        rel_match = self._unique_path_match(rel, self.visible_raw_doc_paths)
        if rel_match is not None:
            rel = rel_match
            path = f"raw_docs/{rel}"
        if rel not in self.visible_raw_doc_paths:
            raise ValueError(f"Raw document not visible in this run: {path}")
        raw_doc_root = self.raw_doc_root.resolve()
        candidate = (raw_doc_root / rel).resolve()
        if candidate != raw_doc_root and raw_doc_root not in candidate.parents:
            raise ValueError(f"Path escapes raw document root: {path}")
        if not candidate.exists() or not candidate.is_file():
            raise ValueError(f"Raw document not found: {path}")
        return candidate

    def resolve_synthetic_node(self, path: str) -> SyntheticFileRecord:
        path = self._clean_tool_path(path)
        embedded_node_id = self._node_id_from_path_text(path)
        if embedded_node_id is not None and not path.startswith(("synthetic_fs/", "history/")):
            rec = self.files_by_id.get(embedded_node_id)
            if rec is not None and rec.active:
                return rec
        if path.startswith("synthetic_fs/"):
            path_index = self._active_path_index()
        elif path.startswith("history/"):
            path_index = self._history_path_index()
        else:
            all_paths = [*self._active_path_index().keys(), *self._history_path_index().keys()]
            matched = self._unique_path_match(path, all_paths)
            if matched is None:
                raise ValueError(f"Path is not in synthetic filesystem or history: {path}")
            path = matched
            path_index = self._active_path_index() if path.startswith("synthetic_fs/") else self._history_path_index()
        node_id = path_index.get(path)
        if node_id is None:
            embedded_node_id = self._node_id_from_path_text(path)
            if embedded_node_id is not None:
                rec = self.files_by_id.get(embedded_node_id)
                if rec is not None and (path.startswith("history/") or rec.active):
                    return rec
        if node_id is None:
            all_paths = [*self._active_path_index().keys(), *self._history_path_index().keys()]
            matched = self._unique_path_match(path, all_paths)
            if matched is not None:
                node_id = self._active_path_index().get(matched) or self._history_path_index().get(matched)
        if node_id is None:
            raise ValueError(f"Synthetic file not found: {path}")
        return self.files_by_id[node_id]

    def _render_header_block(self, rec: SyntheticFileRecord) -> list[str]:
        return [
            f"TITLE: {rec.title}",
            f"KIND: {rec.kind}",
            f"DIRECT_SOURCE_RAW_DOCS: {json.dumps(rec.direct_raw_doc_paths, ensure_ascii=False)}",
            f"DIRECT_SOURCE_SYNTHETIC_FILES: {json.dumps(rec.direct_synthetic_ids, ensure_ascii=False)}",
        ]

    def _render_final_workspace_header_block(self, rec: SyntheticFileRecord) -> list[str]:
        return [
            f"TITLE: {rec.title}",
            f"KIND: {rec.kind}",
            f"DIRECT_SOURCE_RAW_DOCS: {json.dumps(rec.direct_raw_doc_paths, ensure_ascii=False)}",
            f"DIRECT_SOURCE_SYNTHETIC_FILES: {json.dumps(rec.direct_synthetic_ids, ensure_ascii=False)}",
        ]

    def render_synthetic_file(self, rec: SyntheticFileRecord) -> str:
        header = self._render_header_block(rec)
        body_label = "MERGE_SUMMARY" if rec.kind == "merge_summary" else "CLUSTER_TEXT"
        lines = header + ["", f"{body_label}:", rec.content.strip()]
        return "\n".join(lines).strip() + "\n"

    def render_final_workspace_file(self, rec: SyntheticFileRecord) -> str:
        header = self._render_final_workspace_header_block(rec)
        body_label = "MERGE_SUMMARY" if rec.kind == "merge_summary" else "CLUSTER_TEXT"
        lines = header + ["", f"{body_label}:", rec.content.strip()]
        return "\n".join(lines).strip() + "\n"

    def read_lines(self, path: str, start_line: int = 1, num_lines: int = READ_FILE_MAX_LINES) -> dict[str, Any]:
        path = self._clean_tool_path(path)
        if start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {start_line}")
        if num_lines < 1:
            raise ValueError(f"num_lines must be >= 1, got {num_lines}")
        num_lines = min(num_lines, READ_FILE_MAX_LINES)
        raw_handle_match = self._raw_handle_from_path_text(path)
        if raw_handle_match is not None:
            path = raw_handle_match
        raw_path_match = path if path.startswith("raw_docs/") else self._unique_path_match(path, self.raw_doc_rel_paths())
        if raw_path_match is not None or self._looks_like_raw_handle_path(path):
            if raw_path_match is not None:
                path = raw_path_match
            raw_doc_path = self.resolve_raw_doc(path)
            path = f"raw_docs/{raw_doc_path.relative_to(self.raw_doc_root.resolve())}"
            text = raw_doc_path.read_text(encoding="utf-8")
        else:
            rec = self.resolve_synthetic_node(path)
            path = self.history_path(rec.node_id) if path.startswith("history/") or not rec.active else self.active_path(rec.node_id)
            text = self.render_synthetic_file(rec)
        all_lines = text.splitlines()
        start_index = start_line - 1
        chunk_lines = all_lines[start_index : start_index + num_lines]
        content = "\n".join(chunk_lines)
        truncated_by_chars = False
        if len(content) > READ_FILE_MAX_CHARS:
            content = content[:READ_FILE_MAX_CHARS].rstrip()
            truncated_by_chars = True
        end_line = start_line + len(chunk_lines) - 1 if chunk_lines else start_line - 1
        return {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": len(all_lines),
            "has_more": end_line < len(all_lines) or truncated_by_chars,
            "truncated_by_chars": truncated_by_chars,
            "content": content,
        }

    def _tree_lines(self, parent_id: str | None = None, indent: int = 0) -> list[str]:
        lines: list[str] = []
        prefix = "  " * indent
        for rec in self._children(parent_id, active_only=True):
            active_path = self.active_path(rec.node_id)
            if rec.kind == "merge_summary":
                lines.append(f"{prefix}- [MERGE] {rec.title} :: {active_path}")
                lines.extend(self._tree_lines(rec.node_id, indent + 1))
            else:
                lines.append(f"{prefix}- [CLUSTER] {rec.title} :: {active_path}")
        return lines

    def active_tree_string(self) -> str:
        lines = self._tree_lines()
        return "\n".join(lines) if lines else "(synthetic filesystem is currently empty)"

    def _transitive_raw_docs_for_node(self, node_id: str, memo: dict[str, set[str]] | None = None) -> set[str]:
        if memo is None:
            memo = {}
        if node_id in memo:
            return memo[node_id]
        rec = self.files_by_id[node_id]
        result = set(rec.direct_raw_doc_paths)
        for source_id in rec.direct_synthetic_ids:
            if source_id in self.files_by_id:
                result.update(self._transitive_raw_docs_for_node(source_id, memo))
        memo[node_id] = result
        return result

    def _virtual_list_from_paths(
        self,
        path: str,
        all_paths: Sequence[str],
        metadata_by_path: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        metadata_by_path = metadata_by_path or {}
        clean = path.strip().strip("/") or "."
        if clean in {"", "."}:
            clean = "."
        if clean == ".":
            return {
                "path": ".",
                "type": "dir",
                "entries": [
                    {"path": "raw_docs", "type": "dir"},
                    {"path": "synthetic_fs", "type": "dir"},
                    {"path": "history", "type": "dir"},
                ],
            }

        if clean in all_paths:
            payload = metadata_by_path.get(clean, {})
            return {
                "path": clean,
                "type": "file",
                **payload,
                "entries": [],
            }

        prefix = clean + "/"
        entries_by_path: dict[str, str] = {}
        has_descendants = False
        for item in all_paths:
            if not item.startswith(prefix):
                continue
            has_descendants = True
            remainder = item[len(prefix) :]
            if not remainder:
                continue
            child_name = remainder.split("/", 1)[0]
            child_path = str(PurePosixPath(clean) / child_name)
            child_type = "file" if "/" not in remainder else "dir"
            prev = entries_by_path.get(child_path)
            if prev == "dir":
                continue
            if prev == "file" and child_type == "dir":
                entries_by_path[child_path] = "dir"
                continue
            entries_by_path.setdefault(child_path, child_type)

        if not has_descendants:
            if clean in {"raw_docs", "synthetic_fs", "history"}:
                return {
                    "path": clean,
                    "type": "dir",
                    "entries": [],
                }
            raise ValueError(f"Path not found: {path}")

        entries = []
        for child_path, child_type in sorted(
            entries_by_path.items(), key=lambda item: (item[1] == "file", item[0].lower())
        ):
            entries.append(
                {
                    "path": child_path,
                    "type": child_type,
                    **metadata_by_path.get(child_path, {}),
                }
            )
        return {
            "path": clean,
            "type": "dir",
            "entries": entries,
        }

    def _record_listing_metadata(
        self,
        records: Sequence[SyntheticFileRecord],
        path_fn: Any,
    ) -> dict[str, dict[str, str]]:
        metadata: dict[str, dict[str, str]] = {}
        for rec in records:
            path = path_fn(rec.node_id)
            item = {
                "handle": rec.node_id,
                "node_id": rec.node_id,
                "title": rec.title,
                "kind": rec.kind,
            }
            metadata[path] = item
            if rec.kind == "merge_summary" and path.endswith("/summary.txt"):
                metadata[str(PurePosixPath(path).parent)] = item
        return metadata

    def list_entries(self, path: str = ".") -> dict[str, Any]:
        clean = self._clean_tool_path(path).strip("/") or "."
        if clean in {"", "."}:
            clean = "."

        if clean == ".":
            return self._virtual_list_from_paths(".", [])

        raw_handle_match = self._raw_handle_index().get(clean)
        if raw_handle_match is not None:
            clean = raw_handle_match
        if clean == "raw_docs" or clean.startswith("raw_docs/"):
            return self._virtual_list_from_paths(clean, self.raw_doc_rel_paths(), self._raw_listing_metadata())

        if clean == "synthetic_fs" or clean.startswith("synthetic_fs/"):
            active_records = self.active_records()
            active_paths = [self.active_path(rec.node_id) for rec in active_records]
            metadata = self._record_listing_metadata(active_records, self.active_path)
            return self._virtual_list_from_paths(clean, active_paths, metadata)

        if clean == "history" or clean.startswith("history/"):
            history_records = sorted(
                self.files_by_id.values(), key=lambda rec: (rec.created_round, rec.node_id)
            )
            history_paths = [self.history_path(rec.node_id) for rec in history_records]
            metadata = self._record_listing_metadata(history_records, self.history_path)
            return self._virtual_list_from_paths(clean, history_paths, metadata)

        raise ValueError(f"Path not found: {path}")

    def list_payload(self, view: Literal["active", "all"] = "active") -> dict[str, Any]:
        synthetic_entries: list[dict[str, Any]] = []
        if view == "active":
            records: Iterable[SyntheticFileRecord] = self.active_records()
        elif view == "all":
            records = sorted(self.files_by_id.values(), key=lambda rec: (rec.created_round, rec.node_id))
        else:
            raise ValueError(f"Unknown view: {view}")

        for rec in records:
            synthetic_entries.append(
                {
                    "node_id": rec.node_id,
                    "title": rec.title,
                    "kind": rec.kind,
                    "active": rec.active,
                    "active_path": self.active_path(rec.node_id) if rec.active else "",
                    "history_path": self.history_path(rec.node_id),
                    "created_round": rec.created_round,
                }
            )

        return {
            "question_id": self.question_id,
            "view": view,
            "raw_docs": self.raw_doc_rel_paths(),
            "synthetic_files": synthetic_entries,
            "synthetic_tree": self.active_tree_string(),
        }

    def _validate_title(self, title: str) -> str:
        cleaned = title.strip()
        if not cleaned:
            raise ValueError("title must be non-empty")
        return cleaned

    def _validate_synthetic_text(self, text: str, field_name: str) -> str:
        cleaned = str(text or "").strip()
        if cleaned.lower() in BAD_SYNTHETIC_TEXT_VALUES:
            raise ValueError(f"{field_name} must be non-empty and not a placeholder")
        return cleaned

    def _resolve_parent_id(self, parent_path: str | None) -> str | None:
        if not parent_path:
            return None
        parent = self.resolve_synthetic_node(parent_path)
        if not parent.active:
            raise ValueError(f"Parent synthetic file is not active: {parent_path}")
        if parent.kind != "merge_summary":
            raise ValueError(f"Parent path must point to an active merge summary: {parent_path}")
        return parent.node_id

    def _partition_input_paths(self, input_paths: Sequence[str]) -> tuple[list[str], list[str]]:
        if not input_paths:
            raise ValueError("input_paths must be non-empty")
        raw_doc_paths: list[str] = []
        synthetic_ids: list[str] = []
        seen_paths: set[str] = set()
        invalid_paths: list[str] = []
        for path in input_paths:
            path = self._clean_tool_path(path)
            raw_handle_match = self._raw_handle_index().get(path)
            if raw_handle_match is not None:
                path = raw_handle_match
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                if path.startswith("raw_docs/") or self._unique_path_match(path, self.raw_doc_rel_paths()):
                    resolved = self.resolve_raw_doc(path)
                    rel = str(resolved.relative_to(self.raw_doc_root.resolve()))
                    raw_doc_paths.append(f"raw_docs/{rel}")
                else:
                    rec = self.resolve_synthetic_node(path)
                    synthetic_ids.append(rec.node_id)
            except Exception as exc:
                invalid_paths.append(f"{path}: {exc}")
        if not raw_doc_paths and not synthetic_ids:
            details = "; ".join(invalid_paths[:3])
            raise ValueError(f"input_paths did not resolve to any visible files: {details}")
        return raw_doc_paths, synthetic_ids

    def create_cluster(
        self,
        *,
        title: str,
        input_paths: Sequence[str],
        cluster_text: str,
        parent_path: str | None = None,
    ) -> dict[str, Any]:
        title = self._validate_title(title)
        cluster_text = self._validate_synthetic_text(cluster_text, "cluster_text")
        raw_doc_paths, synthetic_ids = self._partition_input_paths(input_paths)
        self.operation_round += 1
        rec = SyntheticFileRecord(
            node_id=self._cluster_node_id(),
            kind="cluster",
            title=title,
            slug=self._slugify(title),
            content=cluster_text,
            created_round=self.operation_round,
            parent_id=self._resolve_parent_id(parent_path),
            direct_raw_doc_paths=raw_doc_paths,
            direct_synthetic_ids=synthetic_ids,
        )
        self.files_by_id[rec.node_id] = rec
        return {
            "created": {
                "handle": rec.node_id,
                "node_id": rec.node_id,
                "title": rec.title,
                "active_path": self.active_path(rec.node_id),
                "history_path": self.history_path(rec.node_id),
            },
        }

    def merge_clusters(
        self,
        *,
        title: str,
        child_paths: Sequence[str],
        merge_summary_text: str,
        parent_path: str | None = None,
    ) -> dict[str, Any]:
        title = self._validate_title(title)
        merge_summary_text = self._validate_synthetic_text(
            merge_summary_text, "merge_summary_text"
        )
        if not child_paths:
            raise ValueError("child_paths must be non-empty")
        child_ids: list[str] = []
        seen: set[str] = set()
        for path in child_paths:
            rec = self.resolve_synthetic_node(path)
            if not rec.active:
                raise ValueError(f"Cannot merge inactive synthetic file: {path}")
            if rec.node_id in seen:
                continue
            seen.add(rec.node_id)
            child_ids.append(rec.node_id)
        self.operation_round += 1
        merge_rec = SyntheticFileRecord(
            node_id=self._merge_node_id(),
            kind="merge_summary",
            title=title,
            slug=self._slugify(title),
            content=merge_summary_text,
            created_round=self.operation_round,
            parent_id=self._resolve_parent_id(parent_path),
            direct_raw_doc_paths=[],
            direct_synthetic_ids=child_ids.copy(),
        )
        self.files_by_id[merge_rec.node_id] = merge_rec
        for child_id in child_ids:
            self.files_by_id[child_id].parent_id = merge_rec.node_id
        return {
            "created": {
                "handle": merge_rec.node_id,
                "node_id": merge_rec.node_id,
                "title": merge_rec.title,
                "active_path": self.active_path(merge_rec.node_id),
                "history_path": self.history_path(merge_rec.node_id),
            },
            "moved_children": [self.active_path(child_id) for child_id in child_ids],
        }

    def delete(self, *, target_paths: Sequence[str], reason: str = "") -> dict[str, Any]:
        if not target_paths:
            raise ValueError("target_paths must be non-empty")
        deleted: list[dict[str, str]] = []
        seen: set[str] = set()
        for path in target_paths:
            rec = self.resolve_synthetic_node(path)
            if not rec.active:
                raise ValueError(f"Synthetic file is already inactive: {path}")
            if rec.node_id in seen:
                continue
            seen.add(rec.node_id)
            old_path = self.active_path(rec.node_id)
            if rec.kind == "merge_summary":
                children = self._children(rec.node_id, active_only=True)
                for child in children:
                    child.parent_id = rec.parent_id
            rec.active = False
            rec.deleted_round = self.operation_round + 1
            rec.delete_reason = reason.strip() or None
            deleted.append({
                "handle": rec.node_id,
                "node_id": rec.node_id,
                "title": rec.title,
                "old_active_path": old_path,
                "history_path": self.history_path(rec.node_id),
            })
        self.operation_round += 1
        return {
            "deleted": deleted,
        }

    def transitive_raw_docs_for_active_files(self) -> list[str]:
        raw_doc_paths: set[str] = set()
        memo: dict[str, set[str]] = {}
        for rec in self.active_records():
            raw_doc_paths.update(self._transitive_raw_docs_for_node(rec.node_id, memo))
        return sorted(raw_doc_paths)

    def raw_doc_fallback_order(self) -> list[str]:
        provenance_docs = self.transitive_raw_docs_for_active_files()
        remaining = [path for path in self.raw_doc_rel_paths() if path not in set(provenance_docs)]
        return provenance_docs + remaining

    def generate_readme(self) -> str:
        lines = [
            "# Final Synthetic Filesystem",
            "",
            "This workspace contains a synthetic filesystem built before answering.",
            "README.md is for navigation and workspace conventions only; it does not contain evidence for the answer.",
            "",
            "How to use it:",
            "1. The top-level synthetic entries are the highest-value entry points and are intended to be read first.",
            "2. If needed, browse into synthetic subdirectories and read their summary.txt files first.",
            "3. Do not guess: if the synthetic files read so far are insufficient to uniquely identify the answer, keep exploring the synthetic filesystem before answering.",
        ]
        lines.extend(
            [
            "",
            "Conventions:",
            "- Root-level synthetic entries are intended to be the first places to inspect and are directly loaded into context at the start of answering.",
            "- Merge-summary directories contain a summary.txt and then more specific child files underneath.",
            "- Synthetic files include provenance fields that point to relevant raw docs and prior synthetic files.",
            "- Use list_files to discover the available files and directories.",
            "- Do not answer from README.md alone; inspect synthetic files for evidence.",
            "",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def materialize_final_workspace(
        self, root_dir: Path, *, mode: AnswererWorkspaceMode = "synthetic_only"
    ) -> Path:
        root_dir.mkdir(parents=True, exist_ok=True)
        (root_dir / "README.md").write_text(
            self.generate_readme(), encoding="utf-8"
        )

        for rec in self.active_records():
            rel_path = self.workspace_relative_path(rec.node_id)
            target = root_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.render_final_workspace_file(rec), encoding="utf-8")
        return root_dir

    def root_active_records(self) -> list[SyntheticFileRecord]:
        return self._children(None, active_only=True)

    def workspace_path_to_node_id(self) -> dict[str, str]:
        return {self.workspace_relative_path(rec.node_id): rec.node_id for rec in self.active_records()}

    def answerer_bootstrap_entries(self) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for rec in self.root_active_records():
            entries.append(
                {
                    "path": self.workspace_relative_path(rec.node_id),
                    "content": self.render_final_workspace_file(rec),
                }
            )
        return entries


class ReadOnlyAnswererWorkspaceTools:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir.resolve()

    def _resolve(self, relative_path: str) -> Path:
        clean = relative_path.strip() or "."
        candidate = (self.root_dir / clean).resolve()
        if candidate != self.root_dir and self.root_dir not in candidate.parents:
            raise ValueError(f"Path escapes workspace root: {relative_path}")
        if not candidate.exists():
            raise ValueError(f"Path not found: {relative_path}")
        return candidate

    def list_files(self, path: str = ".") -> dict[str, Any]:
        target = self._resolve(path)
        if target.is_file():
            return {
                "path": path,
                "type": "file",
                "entries": [],
            }
        entries: list[dict[str, Any]] = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            rel = str(child.relative_to(self.root_dir))
            entries.append(
                {
                    "path": rel,
                    "type": "dir" if child.is_dir() else "file",
                }
            )
        return {
            "path": "." if target == self.root_dir else str(target.relative_to(self.root_dir)),
            "type": "dir",
            "entries": entries,
        }

    def read_file(self, path: str, start_line: int = 1, num_lines: int = READ_FILE_MAX_LINES) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise ValueError(f"Path is not a file: {path}")
        if start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {start_line}")
        if num_lines < 1:
            raise ValueError(f"num_lines must be >= 1, got {num_lines}")
        num_lines = min(num_lines, READ_FILE_MAX_LINES)
        all_lines = target.read_text(encoding="utf-8").splitlines()
        start_index = start_line - 1
        chunk_lines = all_lines[start_index : start_index + num_lines]
        content = "\n".join(chunk_lines)
        truncated_by_chars = False
        if len(content) > READ_FILE_MAX_CHARS:
            content = content[:READ_FILE_MAX_CHARS].rstrip()
            truncated_by_chars = True
        end_line = start_line + len(chunk_lines) - 1 if chunk_lines else start_line - 1
        return {
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": len(all_lines),
            "has_more": end_line < len(all_lines) or truncated_by_chars,
            "truncated_by_chars": truncated_by_chars,
            "content": content,
        }


@dataclass(frozen=True)
class FrozenSyntheticFileExecutor:
    """External frozen-model executor that writes synthetic files from planner-selected sources."""

    backend: JudgeBackend = "gemini"
    model: str = "gemini-3.1-flash-lite-preview"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"
    max_source_chars: int = 24000
    max_output_tokens: int = 768

    def _source_blocks(
        self,
        state: SyntheticFilesystemState,
        paths: Sequence[str],
    ) -> str:
        blocks: list[str] = []
        remaining_chars = max(1000, self.max_source_chars)
        for path in paths:
            if remaining_chars <= 0:
                break
            payload = state.read_lines(
                path=path,
                start_line=1,
                num_lines=READ_FILE_MAX_LINES,
            )
            content = str(payload.get("content", "")).strip()
            if len(content) > remaining_chars:
                content = content[:remaining_chars].rstrip()
            blocks.append(
                "\n".join(
                    [
                        f"[SOURCE] {payload.get('path', path)}",
                        f"LINES: {payload.get('start_line', 1)}-{payload.get('end_line', 0)} of {payload.get('total_lines', 0)}",
                        content,
                    ]
                ).strip()
            )
            remaining_chars -= len(content)
        return "\n\n".join(blocks)

    def generate_cluster_text(
        self,
        *,
        state: SyntheticFilesystemState,
        title: str,
        input_paths: Sequence[str],
        generation_instructions: str = "",
    ) -> str:
        source_blocks = self._source_blocks(state, input_paths)
        system_prompt = (
            "You are a frozen executor model that writes synthetic filesystem memory banks. "
            "Use only the provided source excerpts. Do not answer any particular downstream question. "
            "Write a concise, high-density cluster that preserves concrete facts useful for many future questions. "
            "Include names, aliases, dates, places, relationships, titles, and uncertainties when present. "
            "Return only the cluster text, with no JSON, no markdown fence, and no explanation."
        )
        user_prompt = (
            f"Cluster title: {title}\n"
            f"Planner instructions: {generation_instructions.strip() or 'Create one coherent reusable memory bank from these sources.'}\n\n"
            "Source excerpts:\n"
            f"{source_blocks}"
        )
        text = call_chat_model_messages(
            backend=self.backend,
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_json=False,
            max_output_tokens=self.max_output_tokens,
        ).strip()
        if text.lower() in BAD_SYNTHETIC_TEXT_VALUES:
            raise RuntimeError("Executor returned empty cluster text")
        return text

    def generate_merge_summary_text(
        self,
        *,
        state: SyntheticFilesystemState,
        title: str,
        child_paths: Sequence[str],
        generation_instructions: str = "",
    ) -> str:
        source_blocks = self._source_blocks(state, child_paths)
        system_prompt = (
            "You are a frozen executor model that writes parent summaries for a synthetic filesystem. "
            "Use only the provided child synthetic files. Build a navigation-oriented summary that explains "
            "the shared theme, important distinctions among children, and what future questions this parent helps route. "
            "Return only the merge summary text, with no JSON, no markdown fence, and no explanation."
        )
        user_prompt = (
            f"Parent summary title: {title}\n"
            f"Planner instructions: {generation_instructions.strip() or 'Summarize and organize these child synthetic files.'}\n\n"
            "Child synthetic files:\n"
            f"{source_blocks}"
        )
        text = call_chat_model_messages(
            backend=self.backend,
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_json=False,
            max_output_tokens=self.max_output_tokens,
        ).strip()
        if text.lower() in BAD_SYNTHETIC_TEXT_VALUES:
            raise RuntimeError("Executor returned empty merge summary text")
        return text


class BuilderFilesystemTools:
    def __init__(self, state: SyntheticFilesystemState, executor: FrozenSyntheticFileExecutor | None = None):
        self.state = state
        self.executor = executor

    def _error_code(self, tool_name: str, exc: Exception, *, directory_listing: dict[str, Any] | None = None) -> str:
        if directory_listing is not None:
            return "directory_path"
        message = str(exc).lower()
        if "start_line" in message or "num_lines" in message:
            return "bad_line_range"
        if "input_paths did not resolve" in message or "child_paths" in message:
            return "bad_input_paths"
        if "not active" in message or "inactive" in message:
            return "inactive_path"
        if "path not found" in message or "not found" in message:
            return "path_not_found"
        if "not a raw document path" in message or "not in synthetic filesystem" in message:
            return "wrong_namespace"
        if "not visible" in message:
            return "not_visible"
        if "must be non-empty" in message:
            return "empty_required_field"
        return f"{tool_name}_error"

    def _error_payload(
        self,
        *,
        tool_name: str,
        exc: Exception,
        path: str | None = None,
        paths: Sequence[str] | None = None,
        directory_listing: dict[str, Any] | None = None,
        hint: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        suggestion_paths = list(paths or ([] if path is None else [path]))
        suggestions = [
            suggestion
            for suggestion_path in suggestion_paths
            for suggestion in self.state.path_suggestions(suggestion_path, limit=3)
        ][:10]
        error_code = self._error_code(tool_name, exc, directory_listing=directory_listing)
        if error_code == "path_not_found" and not suggestions:
            suggestions = [self.state.active_path(rec.node_id) for rec in self.state.active_records()[:10]]
        if not suggestions and error_code == "not_visible":
            suggestions = self.state.raw_doc_rel_paths()[:10]
        payload: dict[str, Any] = {
            "ok": False,
            "tool": tool_name,
            "error_code": error_code,
            "error": str(exc),
            "suggestions": suggestions,
            "hint": hint,
        }
        if path is not None:
            payload["path"] = path
        if paths is not None:
            payload["paths"] = list(paths)
        if directory_listing is not None:
            payload["directory_listing"] = directory_listing
        if error_code == "not_visible":
            payload["available_raw_docs"] = self.state.raw_doc_rel_paths()[:25]
            payload["available_raw_doc_handles"] = [
                {"handle": handle, "path": path}
                for handle, path in list(self.state._raw_handle_index().items())[:25]
            ]
        if error_code == "path_not_found":
            payload["available_synthetic_files"] = [
                {
                    "handle": rec.node_id,
                    "path": self.state.active_path(rec.node_id),
                    "title": rec.title,
                    "kind": rec.kind,
                }
                for rec in self.state.active_records()[:25]
            ]
        if extra:
            payload.update(extra)
        return payload

    @tool
    async def list_files(
        self,
        path: Annotated[str, "Path under ., raw_docs/, synthetic_fs/, or history/"] = ".",
    ) -> ToolResult:
        try:
            payload = self.state.list_entries(path=path)
        except Exception as exc:
            payload = self._error_payload(
                tool_name="list_files",
                exc=exc,
                path=path,
                hint="Use one of the suggested paths, or list_files('.') to restart navigation.",
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def read_file(
        self,
        path: Annotated[str, "File path or handle returned by list_files/create_cluster under raw_docs/, synthetic_fs/, or history/"],
        start_line: Annotated[int, "1-based line number to start reading from"] = 1,
        num_lines: Annotated[int, "How many lines to read in this chunk (max 100); prefer focused reads on long files"] = READ_FILE_MAX_LINES,
    ) -> ToolResult:
        try:
            payload = self.state.read_lines(path=path, start_line=start_line, num_lines=num_lines)
        except Exception as exc:
            directory_listing: dict[str, Any] | None = None
            try:
                candidate_listing = self.state.list_entries(path=path)
                if candidate_listing.get("type") == "dir":
                    directory_listing = candidate_listing
            except Exception:
                directory_listing = None
            payload = self._error_payload(
                tool_name="read_file",
                exc=exc,
                path=path,
                directory_listing=directory_listing,
                hint=(
                    "read_file expects a file path. If this is a directory, use list_files on it; "
                    "otherwise retry with one of the suggested file paths."
                ),
                extra={"start_line": start_line, "num_lines": num_lines},
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def read_many(
        self,
        reads: Annotated[list[dict[str, Any]], "List of read specs: {path, optional start_line, optional num_lines}. At most 8 reads per call."],
    ) -> ToolResult:
        payload_reads: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for spec in list(reads or [])[:READ_MANY_MAX_FILES]:
            path = str(spec.get("path", ""))
            start_line = int(spec.get("start_line", 1))
            num_lines = int(spec.get("num_lines", READ_FILE_MAX_LINES))
            try:
                payload_reads.append(
                    self.state.read_lines(path=path, start_line=start_line, num_lines=num_lines)
                )
            except Exception as exc:
                errors.append(
                    self._error_payload(
                        tool_name="read_many",
                        exc=exc,
                        path=path,
                        hint="Each read_many item must use an exact file path or handle returned by list_files or a prior successful tool result.",
                        extra={"start_line": start_line, "num_lines": num_lines},
                    )
                )
        omitted = max(0, len(reads or []) - READ_MANY_MAX_FILES)
        payload: dict[str, Any] = {
            "ok": bool(payload_reads),
            "reads": payload_reads,
            "errors": errors,
            "omitted_specs": omitted,
        }
        if not payload_reads and errors:
            payload["error"] = "No read_many items succeeded"
            payload["error_code"] = "all_reads_failed"
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def create_cluster(
        self,
        title: Annotated[str, "Short descriptive title for the new synthetic memory bank"],
        input_paths: Annotated[list[str], "List of raw_docs/ or synthetic_fs/ or history/ paths/handles used to build one coherent synthetic memory bank"],
        cluster_text: Annotated[str, "Optional builder-written cluster text. Leave empty when frozen executor mode is enabled."] = "",
        generation_instructions: Annotated[str, "Short scope/focus instructions for the frozen executor when cluster_text is empty"] = "",
        parent_path: Annotated[str, "Optional active merge-summary path under synthetic_fs/ where the cluster should be placed"] = "",
    ) -> ToolResult:
        try:
            if self.executor is not None:
                cluster_text = await asyncio.to_thread(
                    self.executor.generate_cluster_text,
                    state=self.state,
                    title=title,
                    input_paths=input_paths,
                    generation_instructions=generation_instructions,
                )
            payload = self.state.create_cluster(
                title=title,
                input_paths=input_paths,
                cluster_text=cluster_text,
                parent_path=parent_path or None,
            )
            payload["executor_generated"] = self.executor is not None
            payload["created_count"] = 1.0
        except Exception as exc:
            payload = self._error_payload(
                tool_name="create_cluster",
                exc=exc,
                paths=input_paths,
                hint="Retry create_cluster with at least one valid raw_docs/, synthetic_fs/, or history/ input path. In executor mode, provide title plus short generation_instructions instead of long cluster_text.",
                extra={"title": title, "input_paths": input_paths},
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def create_clusters(
        self,
        clusters: Annotated[list[dict[str, Any]], "Batch of cluster specs. Each spec needs title, input_paths, and optional generation_instructions/parent_path/cluster_text."],
    ) -> ToolResult:
        created: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for idx, spec in enumerate(clusters or [], start=1):
            title = str(spec.get("title", "")).strip()
            input_paths = list(spec.get("input_paths") or [])
            parent_path = str(spec.get("parent_path", "") or "")
            generation_instructions = str(spec.get("generation_instructions", "") or "")
            cluster_text = str(spec.get("cluster_text", "") or "")
            try:
                if self.executor is not None:
                    cluster_text = await asyncio.to_thread(
                        self.executor.generate_cluster_text,
                        state=self.state,
                        title=title,
                        input_paths=input_paths,
                        generation_instructions=generation_instructions,
                    )
                payload = self.state.create_cluster(
                    title=title,
                    input_paths=input_paths,
                    cluster_text=cluster_text,
                    parent_path=parent_path or None,
                )
                created_payload = dict(payload["created"])
                created_payload["batch_index"] = idx
                created.append(created_payload)
            except Exception as exc:
                errors.append(
                    self._error_payload(
                        tool_name="create_clusters",
                        exc=exc,
                        paths=input_paths,
                        hint="Each create_clusters item needs a non-empty title and valid source paths. In executor mode, cluster_text is generated by the executor.",
                        extra={"batch_index": idx, "title": title, "input_paths": input_paths},
                    )
                )
        payload: dict[str, Any] = {
            "ok": bool(created),
            "created": created,
            "created_count": float(len(created)),
            "errors": errors,
            "executor_generated": self.executor is not None,
        }
        if not created and errors:
            payload["error"] = "No create_clusters items succeeded"
            payload["error_code"] = "all_creates_failed"
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def merge_clusters(
        self,
        title: Annotated[str, "Title for the new parent summary / directory"],
        child_paths: Annotated[list[str], "Active synthetic_fs/ file paths or handles that should be grouped under the new parent"],
        merge_summary_text: Annotated[str, "Optional builder-written parent summary. Leave empty when frozen executor mode is enabled."] = "",
        generation_instructions: Annotated[str, "Short scope/focus instructions for the frozen executor when merge_summary_text is empty"] = "",
        parent_path: Annotated[str, "Optional active merge-summary path under synthetic_fs/ where the new merge node should be placed"] = "",
    ) -> ToolResult:
        try:
            if self.executor is not None:
                merge_summary_text = await asyncio.to_thread(
                    self.executor.generate_merge_summary_text,
                    state=self.state,
                    title=title,
                    child_paths=child_paths,
                    generation_instructions=generation_instructions,
                )
            payload = self.state.merge_clusters(
                title=title,
                child_paths=child_paths,
                merge_summary_text=merge_summary_text,
                parent_path=parent_path or None,
            )
            payload["executor_generated"] = self.executor is not None
            payload["created_count"] = 1.0
        except Exception as exc:
            payload = self._error_payload(
                tool_name="merge_clusters",
                exc=exc,
                paths=child_paths,
                hint="Retry merge_clusters with active synthetic_fs/ child file paths. In executor mode, provide title plus short generation_instructions instead of long merge_summary_text.",
                extra={"title": title, "child_paths": child_paths},
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def merge_many(
        self,
        merges: Annotated[list[dict[str, Any]], "Batch of merge specs. Each spec needs title, child_paths, and optional generation_instructions/parent_path/merge_summary_text."],
    ) -> ToolResult:
        created: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        moved_children: list[str] = []
        for idx, spec in enumerate(merges or [], start=1):
            title = str(spec.get("title", "")).strip()
            child_paths = list(spec.get("child_paths") or [])
            parent_path = str(spec.get("parent_path", "") or "")
            generation_instructions = str(spec.get("generation_instructions", "") or "")
            merge_summary_text = str(spec.get("merge_summary_text", "") or "")
            try:
                if self.executor is not None:
                    merge_summary_text = await asyncio.to_thread(
                        self.executor.generate_merge_summary_text,
                        state=self.state,
                        title=title,
                        child_paths=child_paths,
                        generation_instructions=generation_instructions,
                    )
                payload = self.state.merge_clusters(
                    title=title,
                    child_paths=child_paths,
                    merge_summary_text=merge_summary_text,
                    parent_path=parent_path or None,
                )
                created_payload = dict(payload["created"])
                created_payload["batch_index"] = idx
                created.append(created_payload)
                moved_children.extend(list(payload.get("moved_children", [])))
            except Exception as exc:
                errors.append(
                    self._error_payload(
                        tool_name="merge_many",
                        exc=exc,
                        paths=child_paths,
                        hint="Each merge_many item needs a non-empty title and active synthetic_fs/ child paths.",
                        extra={"batch_index": idx, "title": title, "child_paths": child_paths},
                    )
                )
        payload = {
            "ok": bool(created),
            "created": created,
            "created_count": float(len(created)),
            "moved_children": moved_children,
            "errors": errors,
            "executor_generated": self.executor is not None,
        }
        if not created and errors:
            payload["error"] = "No merge_many items succeeded"
            payload["error_code"] = "all_merges_failed"
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    @tool
    async def delete(
        self,
        target_paths: Annotated[list[str], "Active synthetic_fs/ file paths or handles to remove from the final answerer-facing synthetic filesystem. Never pass raw_docs/ paths or history/ paths here."],
        reason: Annotated[str, "Optional short reason for deleting these files"] = "",
    ) -> ToolResult:
        try:
            payload = self.state.delete(target_paths=target_paths, reason=reason)
        except Exception as exc:
            payload = self._error_payload(
                tool_name="delete",
                exc=exc,
                paths=target_paths,
                hint="Retry delete only with active synthetic_fs/ paths, never raw_docs/ or history/.",
                extra={"target_paths": target_paths},
            )
        return simple_tool_result(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


@dataclass
class SyntheticFilesystemReward:
    state: SyntheticFilesystemState
    gold_answer: str
    question: str
    answerer_backend: JudgeBackend = "gemini"
    answerer_model: str = "gemini-3.1-flash-lite-preview"
    answerer_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    answerer_api_key_env: str = "GEMINI_API_KEY"
    reward_mode: str = "hybrid"
    judge_backend: JudgeBackend = "gemini"
    judge_model: str = "gemini-3.1-flash-lite-preview"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_api_key_env: str = "GEMINI_API_KEY"
    step_penalty: float = 0.0
    termination_penalty: float = 0.1
    raw_docs_penalty: float = 0.0
    empty_synthetic_penalty: float = 1.0
    synthetic_success_bonus: float = 0.0
    synthetic_usage_bonus: float = 0.0
    raw_usage_ratio_penalty: float = 0.0
    filesystem_maturity_scale: float = 0.5
    filesystem_coverage_weight: float = 0.35
    filesystem_expansion_weight: float = 0.3
    filesystem_organization_weight: float = 0.35
    filesystem_stop_weight: float = 0.0
    mature_stop_bonus: float = 0.0
    mature_stop_min_score: float = 0.8
    terminal_reward_clip_min: float = -1.0
    terminal_reward_clip_max: float = 3.0
    builder_max_turns: int = 32
    answerer_max_turns: int = 32
    answerer_workspace_mode: AnswererWorkspaceMode = "synthetic_only"
    answerer_final_answer_max_tokens: int = 128
    answerer_retrieval_cost_scale: float = 0.0
    answerer_retrieval_cost_token_unit: float = 1000.0
    answerer_retrieval_cost_correct_only: bool = True
    answerer_synthetic_read_cost_scale: float = 0.0
    answerer_synthetic_read_cost_unit: float = 10.0
    terminal_answerer_repeats: int = 4
    answerability_delta_reward_scale: float = 0.0
    answerability_delta_min_abs: float = 0.25
    answerability_delta_allow_negative: bool = True
    answerability_probe_repeats: int = 4
    judge_max_output_tokens: int = 64

    def _answerer_usage_metrics(self, answerer_result: dict[str, Any]) -> dict[str, float]:
        read_paths = [str(path) for path in answerer_result.get("read_paths", [])]
        synthetic_read_count = sum(
            1 for path in read_paths if path != "README.md" and not path.startswith("raw_docs/")
        )
        raw_doc_read_count = sum(1 for path in read_paths if path.startswith("raw_docs/"))
        synthetic_chars_read = float(answerer_result.get("synthetic_chars_read", 0) or 0)
        raw_chars_read = float(answerer_result.get("raw_chars_read", 0) or 0)
        readme_chars_read = float(answerer_result.get("readme_chars_read", 0) or 0)
        synthetic_tokens_read = synthetic_chars_read / float(APPROX_CHARS_PER_TOKEN)
        raw_tokens_read = raw_chars_read / float(APPROX_CHARS_PER_TOKEN)
        readme_tokens_read = readme_chars_read / float(APPROX_CHARS_PER_TOKEN)
        total_answerer_tokens_read = synthetic_tokens_read + raw_tokens_read + readme_tokens_read
        retrieval_token_unit = max(1.0, float(self.answerer_retrieval_cost_token_unit))
        retrieval_cost_score = (
            1.0 - (1.0 / ((1.0 + synthetic_tokens_read / retrieval_token_unit) ** 0.5))
            if synthetic_tokens_read > 0.0
            else 0.0
        )
        synthetic_read_unit = max(1.0, float(self.answerer_synthetic_read_cost_unit))
        synthetic_read_cost_score = (
            1.0 - (1.0 / ((1.0 + float(synthetic_read_count) / synthetic_read_unit) ** 0.5))
            if synthetic_read_count > 0
            else 0.0
        )
        return {
            "synthetic_read_count": float(synthetic_read_count),
            "raw_doc_read_count": float(raw_doc_read_count),
            "answerer_synthetic_chars_read": synthetic_chars_read,
            "answerer_raw_chars_read": raw_chars_read,
            "answerer_readme_chars_read": readme_chars_read,
            "answerer_synthetic_tokens_read": synthetic_tokens_read,
            "answerer_raw_tokens_read": raw_tokens_read,
            "answerer_readme_tokens_read": readme_tokens_read,
            "answerer_total_tokens_read": total_answerer_tokens_read,
            "answerer_retrieval_cost_score": retrieval_cost_score,
            "answerer_synthetic_read_cost_score": synthetic_read_cost_score,
        }

    async def _score_answerer_output(
        self,
        *,
        answerer_output: str,
        synthetic_read_count: float,
        answerer_evaluated: float,
        terminal_step: float,
    ) -> dict[str, float]:
        extracted = self._extract_answer(answerer_output)
        candidate = extracted if extracted is not None else answerer_output.strip()
        exact_match = 0.0
        judge_used = 0.0
        judge_score = 0.0
        judge_error = 0.0
        judge_skipped_no_candidate = 0.0
        judge_skipped_no_synthetic_reads = 0.0
        if terminal_step < 1.0:
            correct = 0.0
        elif self.answerer_workspace_mode == "synthetic_only" and synthetic_read_count <= 0:
            correct = 0.0
            judge_skipped_no_synthetic_reads = float(answerer_evaluated >= 1.0)
        elif candidate:
            exact_match = float(self._check_answer(candidate))
            if self.reward_mode == "exact":
                correct = exact_match
            elif self.reward_mode == "llm":
                judge_used = 1.0
                try:
                    judge_score = await self._llm_judge_score(candidate)
                except Exception:
                    LOGGER.warning("LLM judge call failed; scoring candidate as incorrect.", exc_info=True)
                    judge_error = 1.0
                    judge_score = 0.0
                correct = judge_score
            elif self.reward_mode == "hybrid":
                if exact_match >= 1.0:
                    correct = 1.0
                else:
                    judge_used = 1.0
                    try:
                        judge_score = await self._llm_judge_score(candidate)
                    except Exception:
                        LOGGER.warning("LLM judge call failed; scoring candidate as incorrect.", exc_info=True)
                        judge_error = 1.0
                        judge_score = 0.0
                    correct = judge_score
            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
        else:
            correct = 0.0
            judge_skipped_no_candidate = float(answerer_evaluated >= 1.0)
        return {
            "correct": correct,
            "exact_match": exact_match,
            "judge_used": judge_used,
            "judge_score": judge_score,
            "judge_error": judge_error,
            "answerer_error": 0.0,
            "judge_skipped_no_candidate": judge_skipped_no_candidate,
            "judge_skipped_no_synthetic_reads": judge_skipped_no_synthetic_reads,
        }

    def _empty_answerer_result(self) -> dict[str, Any]:
        return {
            "answer_text": "",
            "steps": 0.0,
            "read_paths": [],
            "used_raw_docs": False,
            "used_readme": False,
            "freeform_final_answer": False,
            "prefixed_final_answer": False,
            "synthetic_chars_read": 0,
            "raw_chars_read": 0,
            "readme_chars_read": 0,
        }

    async def _run_scored_answerer_repeats(
        self,
        *,
        repeats: int,
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, float], int]:
        """Run the answerer one or more times and average score/usage metrics.

        Repeats are sequential rather than parallel to avoid multiplying transient
        memory pressure during rollout batches.
        """
        repeat_count = max(1, int(repeats))
        scored: list[tuple[dict[str, Any], dict[str, float], dict[str, float]]] = []
        for repeat_idx in range(repeat_count):
            try:
                answerer_result = await self._run_answerer()
                usage = self._answerer_usage_metrics(answerer_result)
                score_metrics = await self._score_answerer_output(
                    answerer_output=str(answerer_result.get("answer_text", "")),
                    synthetic_read_count=usage["synthetic_read_count"],
                    answerer_evaluated=1.0,
                    terminal_step=1.0,
                )
                score_metrics.setdefault("answerer_error", 0.0)
                score_metrics.setdefault("judge_error", 0.0)
            except Exception:
                LOGGER.warning(
                    "Answerer repeat %s/%s failed; scoring this repeat as incorrect.",
                    repeat_idx + 1,
                    repeat_count,
                    exc_info=True,
                )
                answerer_result = self._empty_answerer_result()
                usage = self._answerer_usage_metrics(answerer_result)
                score_metrics = {
                    "correct": 0.0,
                    "exact_match": 0.0,
                    "judge_used": 0.0,
                    "judge_score": 0.0,
                    "judge_error": 0.0,
                    "answerer_error": 1.0,
                    "judge_skipped_no_candidate": 0.0,
                    "judge_skipped_no_synthetic_reads": 0.0,
                }
            scored.append((answerer_result, usage, score_metrics))

        primary_result = dict(
            next(
                (
                    result
                    for result, _, metrics in scored
                    if float(metrics.get("answerer_error", 0.0)) <= 0.0
                ),
                scored[0][0],
            )
        )

        def mean_float(values: Sequence[float]) -> float:
            return sum(float(v) for v in values) / float(len(values)) if values else 0.0

        usage_keys = sorted(set().union(*(usage.keys() for _, usage, _ in scored)))
        avg_usage = {
            key: mean_float([usage.get(key, 0.0) for _, usage, _ in scored])
            for key in usage_keys
        }
        score_keys = sorted(set().union(*(metrics.keys() for _, _, metrics in scored)))
        avg_score_metrics = {
            key: mean_float([score_metrics.get(key, 0.0) for _, _, score_metrics in scored])
            for key in score_keys
        }

        primary_result["steps"] = mean_float(
            [float(result.get("steps", 0.0) or 0.0) for result, _, _ in scored]
        )
        for key in [
            "used_raw_docs",
            "used_readme",
            "freeform_final_answer",
            "prefixed_final_answer",
        ]:
            primary_result[key] = mean_float(
                [float(bool(result.get(key, False))) for result, _, _ in scored]
            )
        return primary_result, avg_usage, avg_score_metrics, repeat_count

    async def answerability_progress_probe(
        self,
        *,
        previous_score: float,
        previous_best_score: float,
    ) -> tuple[float, float, dict[str, float]]:
        """Answerer-as-progress reward for the current partial filesystem.

        This is intentionally bounded by the caller. It rewards local progress
        over the previous probe score while also logging the best-so-far score
        for diagnostics.
        """
        active_records = self.state.active_records()
        if not active_records:
            return previous_best_score, 0.0, {
                "answerability_probe_evaluated": 0.0,
                "answerability_probe_reward": 0.0,
            }

        _, usage, score_metrics, repeat_count = await self._run_scored_answerer_repeats(
            repeats=self.answerability_probe_repeats,
        )
        score = score_metrics["correct"]
        raw_delta = score - previous_score
        if abs(raw_delta) < self.answerability_delta_min_abs:
            delta = 0.0
        elif self.answerability_delta_allow_negative:
            delta = raw_delta
        else:
            delta = max(0.0, raw_delta)
        new_best = max(previous_best_score, score)
        reward = self.answerability_delta_reward_scale * delta

        retrieval_penalty = 0.0
        synthetic_read_penalty = 0.0
        if score > 0.0:
            retrieval_penalty = (
                self.answerer_retrieval_cost_scale * usage["answerer_retrieval_cost_score"]
            )
            synthetic_read_penalty = (
                self.answerer_synthetic_read_cost_scale
                * usage["answerer_synthetic_read_cost_score"]
            )
            reward -= retrieval_penalty + synthetic_read_penalty

        metrics = {
            "answerability_probe_evaluated": 1.0,
            "answerability_probe_score": score,
            "answerability_probe_previous_score": previous_score,
            "answerability_probe_best_score": new_best,
            "answerability_probe_raw_delta": raw_delta,
            "answerability_probe_delta": delta,
            "answerability_probe_min_abs_delta": self.answerability_delta_min_abs,
            "answerability_probe_reward": reward,
            "answerability_probe_repeats": float(repeat_count),
            "answerability_probe_retrieval_cost_penalty": retrieval_penalty,
            "answerability_probe_synthetic_read_cost_penalty": synthetic_read_penalty,
            "answerability_probe_synthetic_read_count": usage["synthetic_read_count"],
            "answerability_probe_total_tokens_read": usage["answerer_total_tokens_read"],
            "answerability_probe_judge_used": score_metrics["judge_used"],
            "answerability_probe_exact_match": score_metrics["exact_match"],
            "answerability_probe_answerer_error_rate": score_metrics.get("answerer_error", 0.0),
            "answerability_probe_judge_error_rate": score_metrics.get("judge_error", 0.0),
        }
        return new_best, reward, metrics

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        assistant_messages = [msg for msg in history if msg.get("role") == "assistant"]
        assistant_turns = len(assistant_messages)
        final_message = assistant_messages[-1] if assistant_messages else None

        assistant_text = (get_text_content(final_message) or "").strip() if final_message is not None else ""
        stop_called = float(assistant_text == STOP_MESSAGE)
        final_has_tool_calls = bool(final_message is not None and final_message.get("tool_calls"))
        no_tool_terminal = float(final_message is not None and not final_has_tool_calls)
        max_turn_terminal = float(assistant_turns >= self.builder_max_turns)
        terminal_step = float(stop_called >= 1.0 or max_turn_terminal >= 1.0 or no_tool_terminal >= 1.0)
        active_records = self.state.active_records()

        answerer_result = self._empty_answerer_result()
        usage = self._answerer_usage_metrics(answerer_result)
        score_metrics = await self._score_answerer_output(
            answerer_output="",
            synthetic_read_count=usage["synthetic_read_count"],
            answerer_evaluated=0.0,
            terminal_step=terminal_step,
        )
        answerer_evaluated = 0.0
        answerer_repeats_used = 0
        if terminal_step >= 1.0:
            if not (self.answerer_workspace_mode == "synthetic_only" and not active_records):
                answerer_result, usage, score_metrics, answerer_repeats_used = (
                    await self._run_scored_answerer_repeats(
                        repeats=self.terminal_answerer_repeats,
                    )
                )
                answerer_evaluated = 1.0

        synthetic_read_count = usage["synthetic_read_count"]
        raw_doc_read_count = usage["raw_doc_read_count"]
        synthetic_chars_read = usage["answerer_synthetic_chars_read"]
        raw_chars_read = usage["answerer_raw_chars_read"]
        readme_chars_read = usage["answerer_readme_chars_read"]
        synthetic_tokens_read = usage["answerer_synthetic_tokens_read"]
        raw_tokens_read = usage["answerer_raw_tokens_read"]
        readme_tokens_read = usage["answerer_readme_tokens_read"]
        total_answerer_tokens_read = usage["answerer_total_tokens_read"]
        retrieval_cost_score = usage["answerer_retrieval_cost_score"]
        synthetic_read_cost_score = usage["answerer_synthetic_read_cost_score"]

        correct = score_metrics["correct"]
        exact_match = score_metrics["exact_match"]
        judge_used = score_metrics["judge_used"]
        judge_score = score_metrics["judge_score"]
        answerer_error_rate = score_metrics.get("answerer_error", 0.0)
        judge_error_rate = score_metrics.get("judge_error", 0.0)
        judge_skipped_no_candidate = score_metrics["judge_skipped_no_candidate"]
        judge_skipped_no_synthetic_reads = score_metrics["judge_skipped_no_synthetic_reads"]

        terminal_answer_score = correct if terminal_step >= 1.0 else 0.0
        reward = terminal_answer_score
        step_penalty_value = 0.0
        if self.step_penalty > 0.0:
            step_penalty_value = self.step_penalty * float(assistant_turns)
            reward -= step_penalty_value
        used_raw_docs = float(answerer_result["used_raw_docs"])
        num_active_files = float(len(active_records))
        empty_synthetic = float(num_active_files <= 0.0)
        raw_docs_penalty_value = 0.0
        if terminal_step >= 1.0 and used_raw_docs > 0.0:
            raw_docs_penalty_value = self.raw_docs_penalty
            reward -= raw_docs_penalty_value
        empty_synthetic_penalty_value = 0.0
        if terminal_step >= 1.0 and empty_synthetic > 0.0:
            empty_synthetic_penalty_value = self.empty_synthetic_penalty
            reward -= empty_synthetic_penalty_value
        synthetic_success_bonus_applied = 0.0
        synthetic_success_bonus_value = 0.0
        if terminal_step >= 1.0 and correct > 0.0 and used_raw_docs <= 0.0 and num_active_files > 0.0:
            synthetic_success_bonus_value = self.synthetic_success_bonus
            reward += synthetic_success_bonus_value
            synthetic_success_bonus_applied = 1.0
        num_active_clusters = sum(1 for rec in active_records if rec.kind == "cluster")
        num_active_merges = sum(1 for rec in active_records if rec.kind == "merge_summary")
        num_root_entries = sum(1 for rec in active_records if rec.parent_id is None)
        num_provenance_files = sum(1 for rec in active_records if rec.direct_raw_doc_paths)
        total_content_reads = synthetic_read_count + raw_doc_read_count
        synthetic_usage_ratio = (
            float(synthetic_read_count) / float(total_content_reads) if total_content_reads > 0 else 0.0
        )
        raw_usage_ratio = (
            float(raw_doc_read_count) / float(total_content_reads) if total_content_reads > 0 else 0.0
        )
        raw_usage_ratio_penalty_value = 0.0
        synthetic_usage_bonus_value = 0.0
        if terminal_step >= 1.0:
            raw_usage_ratio_penalty_value = self.raw_usage_ratio_penalty * raw_usage_ratio
            synthetic_usage_bonus_value = self.synthetic_usage_bonus * synthetic_usage_ratio
            reward -= raw_usage_ratio_penalty_value
            reward += synthetic_usage_bonus_value
        retrieval_cost_penalty = 0.0
        if terminal_step >= 1.0 and self.answerer_retrieval_cost_scale > 0.0:
            should_charge_retrieval = correct > 0.0 or not self.answerer_retrieval_cost_correct_only
            if should_charge_retrieval:
                retrieval_cost_penalty = self.answerer_retrieval_cost_scale * retrieval_cost_score
                reward -= retrieval_cost_penalty
        synthetic_read_cost_penalty = 0.0
        if terminal_step >= 1.0 and self.answerer_synthetic_read_cost_scale > 0.0:
            should_charge_synthetic_reads = correct > 0.0 or not self.answerer_retrieval_cost_correct_only
            if should_charge_synthetic_reads:
                synthetic_read_cost_penalty = (
                    self.answerer_synthetic_read_cost_scale * synthetic_read_cost_score
                )
                reward -= synthetic_read_cost_penalty
        maturity = self._filesystem_maturity_scores(
            active_records=active_records,
            num_active_files=num_active_files,
            num_active_merges=float(num_active_merges),
            stop_called=stop_called,
        )
        filesystem_maturity_reward = 0.0
        if terminal_step >= 1.0 and self.filesystem_maturity_scale > 0.0:
            filesystem_maturity_reward = self.filesystem_maturity_scale * maturity["filesystem_maturity_score"]
            reward += filesystem_maturity_reward
        mature_stop_bonus_applied = 0.0
        if (
            terminal_step >= 1.0
            and stop_called >= 1.0
            and self.mature_stop_bonus > 0.0
            and maturity["filesystem_maturity_score"] >= self.mature_stop_min_score
        ):
            reward += self.mature_stop_bonus
            mature_stop_bonus_applied = 1.0
        termination_penalty_value = 0.0
        if terminal_step >= 1.0 and stop_called < 1.0:
            termination_penalty_value = self.termination_penalty
            reward -= termination_penalty_value
        terminal_reward_unclipped = reward
        reward = max(self.terminal_reward_clip_min, min(self.terminal_reward_clip_max, reward))

        metrics = self._metrics_template(
            stop_called=stop_called,
            correct=correct,
            exact_match=exact_match,
            judge_used=judge_used,
            judge_score=judge_score,
            num_active_files=num_active_files,
            num_raw_docs_in_fallback=0.0,
            total_synthetic_files=float(len(self.state.files_by_id)),
            total_operations=float(self.state.operation_round),
            builder_turns_seen=float(assistant_turns),
            terminal_builder_step=terminal_step,
            max_turn_terminal=max_turn_terminal,
            no_tool_terminal=no_tool_terminal,
            answerer_evaluated=answerer_evaluated,
            answerer_steps=float(answerer_result["steps"]),
            answerer_used_raw_docs=used_raw_docs,
            answerer_used_readme=float(answerer_result["used_readme"]),
            answerer_freeform_final_answer=float(answerer_result.get("freeform_final_answer", False)),
            answerer_prefixed_final_answer=float(answerer_result.get("prefixed_final_answer", False)),
            judge_skipped_no_candidate=judge_skipped_no_candidate,
            judge_skipped_no_synthetic_reads=judge_skipped_no_synthetic_reads,
            step_penalty_applied=(
                self.step_penalty * float(assistant_turns) if self.step_penalty > 0.0 else 0.0
            ),
            termination_penalty_applied=float(terminal_step >= 1.0 and stop_called < 1.0),
            raw_docs_penalty_applied=float(terminal_step >= 1.0 and used_raw_docs > 0.0),
            empty_synthetic_penalty_applied=float(terminal_step >= 1.0 and empty_synthetic > 0.0),
            synthetic_success_bonus_applied=synthetic_success_bonus_applied,
            num_active_clusters=float(num_active_clusters),
            num_active_merges=float(num_active_merges),
            num_root_entries=float(num_root_entries),
            num_provenance_files=float(num_provenance_files),
            synthetic_read_count=float(synthetic_read_count),
            raw_doc_read_count=float(raw_doc_read_count),
            answerer_synthetic_chars_read=synthetic_chars_read,
            answerer_raw_chars_read=raw_chars_read,
            answerer_readme_chars_read=readme_chars_read,
            answerer_synthetic_tokens_read=synthetic_tokens_read,
            answerer_raw_tokens_read=raw_tokens_read,
            answerer_readme_tokens_read=readme_tokens_read,
            answerer_total_tokens_read=total_answerer_tokens_read,
            answerer_retrieval_cost_score=retrieval_cost_score,
            answerer_retrieval_cost_penalty=retrieval_cost_penalty,
            answerer_synthetic_read_cost_score=synthetic_read_cost_score,
            answerer_synthetic_read_cost_penalty=synthetic_read_cost_penalty,
            synthetic_usage_ratio=synthetic_usage_ratio,
            raw_usage_ratio=raw_usage_ratio,
            filesystem_maturity_reward=filesystem_maturity_reward,
            mature_stop_bonus_applied=mature_stop_bonus_applied,
            **maturity,
        )
        metrics["answerer_repeats"] = float(answerer_repeats_used)
        metrics["answerer_error_rate"] = answerer_error_rate
        metrics["judge_error_rate"] = judge_error_rate
        metrics.update(
            {
                "terminal_answer_score": terminal_answer_score,
                "terminal_filesystem_structure_reward": filesystem_maturity_reward,
                "terminal_retrieval_cost_penalty": retrieval_cost_penalty,
                "terminal_file_read_cost_penalty": synthetic_read_cost_penalty,
                "terminal_step_penalty": step_penalty_value,
                "terminal_raw_docs_penalty": raw_docs_penalty_value,
                "terminal_raw_usage_ratio_penalty": raw_usage_ratio_penalty_value,
                "terminal_empty_filesystem_penalty": empty_synthetic_penalty_value,
                "terminal_synthetic_success_bonus": synthetic_success_bonus_value,
                "terminal_synthetic_usage_bonus": synthetic_usage_bonus_value,
                "terminal_mature_stop_bonus": (
                    self.mature_stop_bonus * mature_stop_bonus_applied
                ),
                "terminal_termination_penalty": termination_penalty_value,
                "terminal_reward_unclipped": terminal_reward_unclipped,
                "terminal_reward": reward,
            }
        )
        return reward, metrics

    def _current_filesystem_maturity_score(self) -> float:
        active_records = self.state.active_records()
        num_active_merges = sum(1 for rec in active_records if rec.kind == "merge_summary")
        return self._filesystem_maturity_scores(
            active_records=active_records,
            num_active_files=float(len(active_records)),
            num_active_merges=float(num_active_merges),
            stop_called=0.0,
        )["filesystem_maturity_score"]

    def _filesystem_maturity_scores(
        self,
        *,
        active_records: Sequence[SyntheticFileRecord],
        num_active_files: float,
        num_active_merges: float,
        stop_called: float,
    ) -> dict[str, float]:
        visible_raw_docs = max(1, len(self.state.visible_raw_doc_paths))
        covered_raw_docs: set[str] = set()
        memo: dict[str, set[str]] = {}
        per_file_raw_counts: list[int] = []
        for rec in active_records:
            transitive = self.state._transitive_raw_docs_for_node(rec.node_id, memo)
            covered_raw_docs.update(transitive)
            if rec.kind == "cluster":
                per_file_raw_counts.append(len(transitive))

        covered_count = len(covered_raw_docs)
        coverage_score = min(1.0, float(covered_count) / float(visible_raw_docs))
        raw_doc_count = len(self.state.visible_raw_doc_paths)
        active_count = max(0.0, num_active_files)
        # Unbounded but diminishing: every useful synthetic file can help, without a fixed target count.
        expansion_score = 1.0 - (1.0 / ((1.0 + active_count) ** 0.5)) if active_count > 0.0 else 0.0
        root_entries = sum(1 for rec in active_records if rec.parent_id is None)
        root_ratio = float(root_entries) / active_count if active_count > 0.0 else 1.0
        branching_score = max(0.0, 1.0 - root_ratio) if active_count > 1.0 else 0.0
        merge_score = 1.0 - (1.0 / ((1.0 + num_active_merges) ** 0.5)) if num_active_merges > 0.0 else 0.0
        organization_score = 0.5 * branching_score + 0.5 * merge_score
        stop_maturity_bonus = float(
            stop_called >= 1.0
            and coverage_score >= 0.85
            and expansion_score >= 0.75
            and organization_score >= 0.25
        )
        maturity_score = (
            self.filesystem_coverage_weight * coverage_score
            + self.filesystem_expansion_weight * expansion_score
            + self.filesystem_organization_weight * organization_score
            + self.filesystem_stop_weight * stop_maturity_bonus
        )
        return {
            "filesystem_maturity_score": maturity_score,
            "filesystem_coverage_score": coverage_score,
            "filesystem_expansion_score": expansion_score,
            "filesystem_organization_score": organization_score,
            "filesystem_merge_score": merge_score,
            "filesystem_stop_maturity_bonus": stop_maturity_bonus,
            "filesystem_covered_raw_docs": float(covered_count),
            "filesystem_visible_raw_docs": float(raw_doc_count),
            "filesystem_root_ratio": root_ratio,
            "filesystem_branching_score": branching_score,
        }

    def _metrics_template(
        self,
        *,
        stop_called: float,
        correct: float = 0.0,
        exact_match: float = 0.0,
        judge_used: float = 0.0,
        judge_score: float = 0.0,
        num_active_files: float = 0.0,
        num_raw_docs_in_fallback: float = 0.0,
        total_synthetic_files: float = 0.0,
        total_operations: float = 0.0,
        builder_turns_seen: float = 0.0,
        terminal_builder_step: float = 0.0,
        max_turn_terminal: float = 0.0,
        no_tool_terminal: float = 0.0,
        answerer_evaluated: float = 0.0,
        answerer_steps: float = 0.0,
        answerer_used_raw_docs: float = 0.0,
        answerer_used_readme: float = 0.0,
        answerer_freeform_final_answer: float = 0.0,
        answerer_prefixed_final_answer: float = 0.0,
        judge_skipped_no_candidate: float = 0.0,
        judge_skipped_no_synthetic_reads: float = 0.0,
        step_penalty_applied: float = 0.0,
        termination_penalty_applied: float = 0.0,
        raw_docs_penalty_applied: float = 0.0,
        empty_synthetic_penalty_applied: float = 0.0,
        synthetic_success_bonus_applied: float = 0.0,
        num_active_clusters: float = 0.0,
        num_active_merges: float = 0.0,
        num_root_entries: float = 0.0,
        num_provenance_files: float = 0.0,
        synthetic_read_count: float = 0.0,
        raw_doc_read_count: float = 0.0,
        answerer_synthetic_chars_read: float = 0.0,
        answerer_raw_chars_read: float = 0.0,
        answerer_readme_chars_read: float = 0.0,
        answerer_synthetic_tokens_read: float = 0.0,
        answerer_raw_tokens_read: float = 0.0,
        answerer_readme_tokens_read: float = 0.0,
        answerer_total_tokens_read: float = 0.0,
        answerer_retrieval_cost_score: float = 0.0,
        answerer_retrieval_cost_penalty: float = 0.0,
        answerer_synthetic_read_cost_score: float = 0.0,
        answerer_synthetic_read_cost_penalty: float = 0.0,
        synthetic_usage_ratio: float = 0.0,
        raw_usage_ratio: float = 0.0,
        filesystem_maturity_reward: float = 0.0,
        mature_stop_bonus_applied: float = 0.0,
        filesystem_maturity_score: float = 0.0,
        filesystem_coverage_score: float = 0.0,
        filesystem_expansion_score: float = 0.0,
        filesystem_organization_score: float = 0.0,
        filesystem_merge_score: float = 0.0,
        filesystem_stop_maturity_bonus: float = 0.0,
        filesystem_covered_raw_docs: float = 0.0,
        filesystem_visible_raw_docs: float = 0.0,
        filesystem_root_ratio: float = 0.0,
        filesystem_branching_score: float = 0.0,
    ) -> dict[str, float]:
        return {
            "stop_called": stop_called,
            "correct": correct,
            "exact_match": exact_match,
            "judge_used": judge_used,
            "judge_score": judge_score,
            "num_active_files": num_active_files,
            "num_raw_docs_in_fallback": num_raw_docs_in_fallback,
            "total_synthetic_files": total_synthetic_files,
            "total_operations": total_operations,
            "builder_turns_seen": builder_turns_seen,
            "terminal_builder_step": terminal_builder_step,
            "max_turn_terminal": max_turn_terminal,
            "no_tool_terminal": no_tool_terminal,
            "answerer_evaluated": answerer_evaluated,
            "answerer_steps": answerer_steps,
            "answerer_used_raw_docs": answerer_used_raw_docs,
            "answerer_used_readme": answerer_used_readme,
            "answerer_freeform_final_answer": answerer_freeform_final_answer,
            "answerer_prefixed_final_answer": answerer_prefixed_final_answer,
            "judge_skipped_no_candidate": judge_skipped_no_candidate,
            "judge_skipped_no_synthetic_reads": judge_skipped_no_synthetic_reads,
            "step_penalty_applied": step_penalty_applied,
            "termination_penalty_applied": termination_penalty_applied,
            "raw_docs_penalty_applied": raw_docs_penalty_applied,
            "empty_synthetic_penalty_applied": empty_synthetic_penalty_applied,
            "synthetic_success_bonus_applied": synthetic_success_bonus_applied,
            "num_active_clusters": num_active_clusters,
            "num_active_merges": num_active_merges,
            "num_root_entries": num_root_entries,
            "num_provenance_files": num_provenance_files,
            "synthetic_read_count": synthetic_read_count,
            "raw_doc_read_count": raw_doc_read_count,
            "answerer_synthetic_chars_read": answerer_synthetic_chars_read,
            "answerer_raw_chars_read": answerer_raw_chars_read,
            "answerer_readme_chars_read": answerer_readme_chars_read,
            "answerer_synthetic_tokens_read": answerer_synthetic_tokens_read,
            "answerer_raw_tokens_read": answerer_raw_tokens_read,
            "answerer_readme_tokens_read": answerer_readme_tokens_read,
            "answerer_total_tokens_read": answerer_total_tokens_read,
            "answerer_retrieval_cost_score": answerer_retrieval_cost_score,
            "answerer_retrieval_cost_penalty": answerer_retrieval_cost_penalty,
            "answerer_synthetic_read_cost_score": answerer_synthetic_read_cost_score,
            "answerer_synthetic_read_cost_penalty": answerer_synthetic_read_cost_penalty,
            "synthetic_usage_ratio": synthetic_usage_ratio,
            "raw_usage_ratio": raw_usage_ratio,
            "filesystem_maturity_reward": filesystem_maturity_reward,
            "mature_stop_bonus_applied": mature_stop_bonus_applied,
            "filesystem_maturity_score": filesystem_maturity_score,
            "filesystem_coverage_score": filesystem_coverage_score,
            "filesystem_expansion_score": filesystem_expansion_score,
            "filesystem_organization_score": filesystem_organization_score,
            "filesystem_merge_score": filesystem_merge_score,
            "filesystem_stop_maturity_bonus": filesystem_stop_maturity_bonus,
            "filesystem_covered_raw_docs": filesystem_covered_raw_docs,
            "filesystem_visible_raw_docs": filesystem_visible_raw_docs,
            "filesystem_root_ratio": filesystem_root_ratio,
            "filesystem_branching_score": filesystem_branching_score,
        }

    def _extract_answer(self, text: str) -> str | None:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if len(lines) != 1:
            return None
        if not lines[0].startswith(ANSWER_PREFIX):
            return None
        answer = lines[0][len(ANSWER_PREFIX) :].strip()
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
        response_text = await asyncio.to_thread(
            self._call_chat_model,
            backend=self.judge_backend,
            model=self.judge_model,
            base_url=self.judge_base_url,
            api_key_env=self.judge_api_key_env,
            prompt=prompt,
            response_json=True,
            max_output_tokens=self.judge_max_output_tokens,
        )
        try:
            parsed = json.loads(response_text)
            val = float(parsed.get("correct", 0))
            return 1.0 if val >= 1.0 else 0.0
        except Exception:
            return 0.0

    async def _run_answerer(self) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="synthetic_fs_answerer_") as tmpdir:
            workspace_root = self.state.materialize_final_workspace(
                Path(tmpdir) / "final_workspace",
                mode=self.answerer_workspace_mode,
            )
            tools = ReadOnlyAnswererWorkspaceTools(workspace_root)
            root_bootstrap_entries = self.state.answerer_bootstrap_entries()

            readme_text = (workspace_root / "README.md").read_text(encoding="utf-8").strip()
            bootstrap_blocks = [
                "[README.md]",
                readme_text,
            ]
            for entry in root_bootstrap_entries:
                bootstrap_blocks.extend(
                    [
                        "",
                        f"[ROOT_SYNTHETIC_ENTRY] {entry['path']}",
                        entry["content"].strip(),
                    ]
                )

            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": build_answerer_system_prompt(self.answerer_workspace_mode),
                },
                {
                    "role": "user",
                    "content": build_answerer_user_prompt(
                        self.question, self.answerer_workspace_mode
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Workspace bootstrap:\n"
                        "README.md and the current root-level synthetic entries are already loaded below.\n\n"
                        + "\n".join(bootstrap_blocks)
                    ),
                },
            ]
            read_paths: list[str] = ["README.md", *[entry["path"] for entry in root_bootstrap_entries]]
            used_readme = True
            used_raw_docs = False
            used_synthetic = bool(root_bootstrap_entries)
            readme_chars_read = len(readme_text)
            synthetic_chars_read = sum(len(entry["content"]) for entry in root_bootstrap_entries)
            raw_chars_read = 0

            def answerer_result(
                *,
                answer_text: str,
                steps: int,
                freeform_final_answer: bool,
                prefixed_final_answer: bool,
            ) -> dict[str, Any]:
                return {
                    "answer_text": answer_text,
                    "steps": steps,
                    "read_paths": read_paths,
                    "used_raw_docs": used_raw_docs,
                    "used_readme": used_readme,
                    "freeform_final_answer": freeform_final_answer,
                    "prefixed_final_answer": prefixed_final_answer,
                    "synthetic_chars_read": synthetic_chars_read,
                    "raw_chars_read": raw_chars_read,
                    "readme_chars_read": readme_chars_read,
                }

            def list_files_payload(path: str) -> dict[str, Any]:
                clean = path.strip() or "."
                return tools.list_files(path=clean)

            for step in range(1, self.answerer_max_turns + 1):
                response_text = await asyncio.to_thread(
                    self._call_chat_model_messages,
                    backend=self.answerer_backend,
                    model=self.answerer_model,
                    base_url=self.answerer_base_url,
                    api_key_env=self.answerer_api_key_env,
                    messages=messages,
                    response_json=False,
                )
                response_text = response_text.strip()
                messages.append({"role": "assistant", "content": response_text})

                if response_text.startswith(ANSWER_PREFIX):
                    response_text = self._truncate_final_answer(response_text)
                    return answerer_result(
                        answer_text=response_text,
                        steps=step,
                        freeform_final_answer=False,
                        prefixed_final_answer=True,
                    )

                action = self._parse_answerer_action(response_text)
                if action is None:
                    return answerer_result(
                        answer_text=self._truncate_final_answer(response_text),
                        steps=step,
                        freeform_final_answer=True,
                        prefixed_final_answer=False,
                    )

                try:
                    if action["action"] == "list_files":
                        payload = list_files_payload(str(action.get("path", ".")))
                    elif action["action"] == "read_file":
                        path = str(action.get("path", ""))
                        payload = tools.read_file(
                            path=path,
                            start_line=int(action.get("start_line", 1)),
                            num_lines=int(action.get("num_lines", READ_FILE_MAX_LINES)),
                        )
                        content_chars = len(str(payload.get("content", "")))
                        read_paths.append(path)
                        if path != "README.md" and not path.startswith("raw_docs/"):
                            used_synthetic = True
                            synthetic_chars_read += content_chars
                        if path.startswith("raw_docs/"):
                            used_raw_docs = True
                            raw_chars_read += content_chars
                        if path == "README.md":
                            readme_chars_read += content_chars
                    else:
                        raise ValueError(f"Unknown action: {action['action']}")
                except Exception as e:
                    payload = {"error": str(e)}

                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}",
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the browsing/tool-call budget. "
                        "Do not call more tools. Based only on the workspace evidence already shown in this conversation, "
                        f"return exactly one non-empty final answer line in this format:\n{ANSWER_PREFIX} <final answer>"
                    ),
                }
            )
            response_text = await asyncio.to_thread(
                self._call_chat_model_messages,
                backend=self.answerer_backend,
                model=self.answerer_model,
                base_url=self.answerer_base_url,
                api_key_env=self.answerer_api_key_env,
                messages=messages,
                response_json=False,
            )
            response_text = self._truncate_final_answer(response_text.strip())
            if not response_text:
                response_text = f"{ANSWER_PREFIX} Unknown"
            return answerer_result(
                answer_text=response_text,
                steps=self.answerer_max_turns + 1,
                freeform_final_answer=not response_text.startswith(ANSWER_PREFIX),
                prefixed_final_answer=response_text.startswith(ANSWER_PREFIX),
            )

    def _truncate_final_answer(self, response_text: str) -> str:
        if self.answerer_final_answer_max_tokens <= 0:
            return response_text.strip()
        has_prefix = response_text.startswith(ANSWER_PREFIX)
        answer_body = response_text[len(ANSWER_PREFIX) :].strip() if has_prefix else response_text.strip()
        tokens = answer_body.split()
        if len(tokens) <= self.answerer_final_answer_max_tokens:
            return response_text.strip()
        truncated = " ".join(tokens[: self.answerer_final_answer_max_tokens]).strip()
        return f"{ANSWER_PREFIX} {truncated}".strip() if has_prefix else truncated

    def _parse_answerer_action(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        if not cleaned:
            return None
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:
            return None
        action = parsed.get("action")
        if action not in {"list_files", "read_file"}:
            return None
        return parsed

    def _call_chat_model(
        self,
        *,
        backend: JudgeBackend,
        model: str,
        base_url: str,
        api_key_env: str,
        prompt: str,
        response_json: bool,
        max_output_tokens: int | None = None,
    ) -> str:
        return self._call_chat_model_messages(
            backend=backend,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_json=response_json,
            max_output_tokens=max_output_tokens,
        )

    def _call_chat_model_messages(
        self,
        *,
        backend: JudgeBackend,
        model: str,
        base_url: str,
        api_key_env: str,
        messages: Sequence[dict[str, str]],
        response_json: bool,
        max_output_tokens: int | None = None,
    ) -> str:
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {api_key_env}")

        if backend == "openrouter":
            url = base_url.rstrip("/") + "/chat/completions"
            system_messages = [m for m in messages if m.get("role") == "system"]
            non_system_messages = [m for m in messages if m.get("role") != "system"]
            payload: dict[str, Any] = {
                "model": model,
                "messages": [],
                "temperature": 0,
            }
            if max_output_tokens is not None and max_output_tokens > 0:
                payload["max_tokens"] = max_output_tokens
            if response_json and not system_messages:
                payload["messages"].append(
                    {"role": "system", "content": "Return only compact JSON."}
                )
            payload["messages"].extend(system_messages)
            payload["messages"].extend(non_system_messages)
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            body = self._urlopen_json_with_retry(req)

            choices = body.get("choices") or []
            if not choices:
                raise RuntimeError(f"Response missing choices: {body}")
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, list):
                parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                return "".join(parts).strip()
            return str(content).strip()

        if backend == "gemini":
            model_path = model if model.startswith("models/") else f"models/{model}"
            url = f"{base_url.rstrip('/')}/{model_path}:generateContent"
            system_texts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
            contents = []
            for message in messages:
                role = message.get("role", "user")
                if role == "system":
                    continue
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": message.get("content", "")}]})
            payload: dict[str, Any] = {
                "contents": contents,
                "generationConfig": {
                    "temperature": 0,
                },
            }
            if max_output_tokens is not None and max_output_tokens > 0:
                payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
            if system_texts:
                payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
            if response_json:
                payload["generationConfig"]["responseMimeType"] = "application/json"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                method="POST",
            )
            body = self._urlopen_json_with_retry(req)

            candidates = body.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Response missing candidates: {body}")
            content = ((candidates[0].get("content") or {}).get("parts") or [])
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("text")]
            return "".join(text_parts).strip()

        raise ValueError(f"Unknown backend: {backend}")

    def _urlopen_json_with_retry(self, req: urllib.request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        retryable_http_codes = {429, 500, 502, 503, 504}

        for attempt in range(1, MODEL_HTTP_MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(req, timeout=MODEL_HTTP_TIMEOUT_SECONDS) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code in retryable_http_codes and attempt < MODEL_HTTP_MAX_ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS))
                    last_error = RuntimeError(f"HTTP error {e.code}: {body[:500]}")
                    continue
                raise RuntimeError(f"HTTP error {e.code}: {body[:500]}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < MODEL_HTTP_MAX_ATTEMPTS:
                    time.sleep(min(2 ** (attempt - 1), MODEL_HTTP_RETRY_SLEEP_CAP_SECONDS))
                    last_error = RuntimeError(f"URL/timeout error: {e}")
                    continue
                raise RuntimeError(f"URL/timeout error: {e}") from e

        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without a captured error")


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


def get_text_content(message: Message) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        return "".join(parts)
    return str(content)


RewardResult = tuple[float, dict[str, float]]
RewardFn = Callable[[list[Message]], Awaitable[RewardResult]]
ProxyScoreFn = Callable[[], float]


def _tool_result_payload(tool_result: ToolResult) -> dict[str, Any]:
    if not getattr(tool_result, "messages", None):
        return {}
    content = tool_result.messages[0].get("content", "")
    if not isinstance(content, str):
        return {}
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tool_result_text(tool_result: ToolResult, max_chars: int = 700) -> str:
    if not getattr(tool_result, "messages", None):
        return ""
    text = get_text_content(tool_result.messages[0])
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated {len(text) - max_chars} chars]"


def _tool_result_succeeded(tool_result: ToolResult) -> bool:
    metadata = getattr(tool_result, "metadata", {}) or {}
    if "error" in metadata:
        return False
    payload = _tool_result_payload(tool_result)
    if payload.get("ok") is False:
        return False
    if "error" in payload:
        return False
    return True


def _tool_result_error_text(tool_result: ToolResult) -> str:
    metadata = getattr(tool_result, "metadata", {}) or {}
    error = metadata.get("error", "")
    if error:
        return str(error)
    payload = _tool_result_payload(tool_result)
    payload_error = payload.get("error")
    if payload_error:
        error_code = payload.get("error_code", "")
        prefix = f"{error_code}: " if error_code else ""
        return prefix + str(payload_error)
    text = _tool_result_text(tool_result, max_chars=1200)
    if "error" in text.lower():
        return text
    return ""


def _tool_result_error_code(tool_result: ToolResult) -> str:
    metadata = getattr(tool_result, "metadata", {}) or {}
    metadata_error = metadata.get("error", "")
    if metadata_error:
        return "metadata_error"
    payload = _tool_result_payload(tool_result)
    error_code = payload.get("error_code", "")
    return str(error_code) if error_code else "unknown"


def _metric_key_part(text: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")
    return cleaned or "unknown"


def _tool_call_args(tool_call: Any) -> dict[str, Any] | None:
    try:
        args = json.loads(tool_call.function.arguments)
    except Exception:
        return None
    return args if isinstance(args, dict) else None


def _with_tool_call_args(tool_call: Any, args: dict[str, Any]) -> Any:
    encoded_args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    try:
        function = tool_call.function.model_copy(update={"arguments": encoded_args})
        return tool_call.model_copy(update={"function": function})
    except AttributeError:
        function = tool_call.function.__class__(
            name=tool_call.function.name,
            arguments=encoded_args,
        )
        return tool_call.__class__(
            function=function,
            id=getattr(tool_call, "id", None),
        )


def _redacted_tool_call(tool_call: Any, tool_result: ToolResult) -> Any:
    if not _tool_result_succeeded(tool_result):
        return tool_call

    tool_name = tool_call.function.name
    args = _tool_call_args(tool_call)
    if args is None:
        return tool_call

    payload = _tool_result_payload(tool_result)
    created = payload.get("created") if isinstance(payload.get("created"), dict) else {}
    active_path = str(created.get("active_path") or created.get("history_path") or "")
    node_id = str(created.get("node_id") or "")
    handle = active_path or node_id or "created synthetic file"

    if tool_name == "create_cluster" and "cluster_text" in args:
        original_chars = len(str(args.get("cluster_text", "")))
        args = dict(args)
        args["cluster_text"] = (
            f"[redacted after successful create_cluster; stored at {handle}; "
            f"original_chars={original_chars}; use read_file on that path if needed]"
        )
        return _with_tool_call_args(tool_call, args)

    if tool_name == "merge_clusters" and "merge_summary_text" in args:
        original_chars = len(str(args.get("merge_summary_text", "")))
        args = dict(args)
        args["merge_summary_text"] = (
            f"[redacted after successful merge_clusters; stored at {handle}; "
            f"original_chars={original_chars}; use read_file on that path if needed]"
        )
        return _with_tool_call_args(tool_call, args)

    return tool_call


def _redacted_tool_call_log(tool_call: Any, tool_result: ToolResult | None = None) -> str:
    if tool_result is not None:
        tool_call = _redacted_tool_call(tool_call, tool_result)
    args = getattr(tool_call.function, "arguments", "")
    return f"{tool_call.function.name}({args})"


def _message_debug_text(message: Message) -> str:
    role = str(message.get("role", ""))
    chunks = [f"role={role}"]
    text = get_text_content(message).strip()
    if text:
        chunks.append(text)
    tool_calls = list(message.get("tool_calls") or [])
    for tool_call in tool_calls:
        chunks.append(_redacted_tool_call_log(tool_call))
    return "\n".join(chunks).strip()


def _history_approx_tokens(history: Sequence[Message]) -> int:
    total_chars = 0
    for message in history:
        total_chars += len(_message_debug_text(message)) + 12
    return max(1, total_chars // APPROX_CHARS_PER_TOKEN)


def _trim_text_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    omitted = len(text) - max_chars
    return (
        text[:keep_head].rstrip()
        + f"\n\n[... omitted {omitted} chars from middle before compaction ...]\n\n"
        + text[-keep_tail:].lstrip()
    )


def _render_history_for_compaction(history: Sequence[Message], max_chars: int) -> str:
    blocks: list[str] = []
    for idx, message in enumerate(history, start=1):
        blocks.append(f"### MESSAGE {idx}\n{_message_debug_text(message)}")
    return _trim_text_middle("\n\n".join(blocks), max_chars)


def _is_compaction_summary_message(message: Message) -> bool:
    return (
        message.get("role") == "user"
        and get_text_content(message).lstrip().startswith(BUILDER_COMPACTION_SUMMARY_MARKER)
    )


def _compaction_summary_message(summary: str) -> Message:
    return {
        "role": "user",
        "content": (
            f"{BUILDER_COMPACTION_SUMMARY_MARKER}\n"
            "Older builder context has been summarized to keep the visible Tinker context small. "
            "Use this memory plus the recent verbatim turns below. Re-open exact files with read_file when needed.\n\n"
            f"{summary.strip()}"
        ),
    }


@dataclass
class RedactingAgentToolMessageEnv(MessageEnv):
    """Tool-use env that stores full filesystem state but compacts visible history."""

    tools: list[Tool]
    initial_messages: list[Message]
    max_turns: int
    reward_fn: RewardFn
    state: SyntheticFilesystemState | None = None
    filesystem_maturity_score_fn: ProxyScoreFn | None = None
    step_construction_action_bonus: float = 0.05
    step_filesystem_maturity_delta_scale: float = 0.5
    step_non_construction_turn_penalty: float = 0.005
    step_non_construction_streak_penalty: float = 0.0
    step_non_construction_streak_free: int = 3
    step_tool_error_penalty: float = 0.05
    answerability_delta_reward_scale: float = 0.0
    answerability_probe_max_per_episode: int = 4
    answerability_probe_interval_turns: int = 8
    answerability_probe_min_maturity: float = 0.45
    log_step_details: bool = False
    log_compaction_summaries: bool = False
    retain_reward_tool_messages: bool = False
    trim_terminal_history_for_memory: bool = True
    return_empty_terminal_observation: bool = True
    clear_state_on_terminal_for_memory: bool = True
    compaction_enabled: bool = True
    compaction_backend: JudgeBackend = "gemini"
    compaction_model: str = "gemini-3.1-flash-lite-preview"
    compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    compaction_api_key_env: str = "GEMINI_API_KEY"
    compaction_trigger_tokens: int = DEFAULT_BUILDER_COMPACTION_TRIGGER_TOKENS
    compaction_keep_recent_turns: int = DEFAULT_BUILDER_COMPACTION_KEEP_RECENT_TURNS
    compaction_max_output_tokens: int = DEFAULT_BUILDER_COMPACTION_MAX_OUTPUT_TOKENS
    compaction_input_max_chars: int = DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS
    history: list[Message] = field(default_factory=list)
    reward_history: list[Message] = field(default_factory=list)
    progress_metrics: dict[str, float] = field(default_factory=dict)

    _turn_count: int = 0
    _tool_dict: dict[str, Tool] = field(default_factory=dict, init=False)
    _should_stop: bool = field(default=False, init=False)
    _compacted_summary: str = field(default="", init=False)
    _compaction_count: int = field(default=0, init=False)
    _compaction_error_count: int = field(default=0, init=False)
    _last_step_filesystem_maturity_score: float = field(default=0.0, init=False)
    _non_construction_streak: int = field(default=0, init=False)
    _answerability_probe_count: int = field(default=0, init=False)
    _best_answerability_score: float = field(default=0.0, init=False)
    _last_answerability_score: float = field(default=0.0, init=False)
    _last_answerability_probe_turn: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._tool_dict = {tool_obj.name: tool_obj for tool_obj in self.tools}
        if self.filesystem_maturity_score_fn is not None:
            self._last_step_filesystem_maturity_score = self.filesystem_maturity_score_fn()

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        if not self.reward_history:
            self.reward_history = list(self.initial_messages)
        return self.history

    async def step(self, message: Message) -> MessageStepResult:
        self._turn_count += 1
        metrics: dict[str, float] = dict(self.progress_metrics)
        logs: dict[str, str] = {}

        assistant_idx = len(self.history)
        self.history.append(message)
        reward_assistant_idx = len(self.reward_history)
        self.reward_history.append(message)

        assistant_text = get_text_content(message)
        if assistant_text and self.log_step_details:
            logs["assistant_content"] = assistant_text

        tool_calls = list(message.get("tool_calls") or [])
        tool_results: list[ToolResult] = []
        if tool_calls:
            tool_results = await asyncio.gather(
                *[handle_tool_call(self._tool_dict, tool_call) for tool_call in tool_calls]
            )

            redacted_tool_calls = [
                _redacted_tool_call(tool_call, tool_result)
                for tool_call, tool_result in zip(tool_calls, tool_results, strict=True)
            ]
            if redacted_tool_calls != tool_calls:
                redacted_message = dict(message)
                redacted_message["tool_calls"] = redacted_tool_calls
                self.history[assistant_idx] = redacted_message
                self.reward_history[reward_assistant_idx] = redacted_message

            for idx, (tool_call, tool_result) in enumerate(zip(tool_calls, tool_results, strict=True)):
                if self.log_step_details:
                    logs[f"tool_call_{idx}"] = _redacted_tool_call_log(tool_call, tool_result)
                    logs[f"tool_result_{idx}"] = _tool_result_text(tool_result)
                if not _tool_result_succeeded(tool_result):
                    tool_name = getattr(tool_call.function, "name", "unknown_tool")
                    error_text = _tool_result_error_text(tool_result)
                    logs[f"tool_error_{idx}"] = f"{tool_name}: {error_text[:1000]}"
                for result_msg in tool_result.messages:
                    self.history.append(result_msg)
                    if self.retain_reward_tool_messages:
                        self.reward_history.append(result_msg)
                if tool_result.should_stop:
                    self._should_stop = True

        no_tool_calls = len(tool_calls) == 0
        max_turns_reached = self._turn_count >= self.max_turns
        done = no_tool_calls or max_turns_reached or self._should_stop

        reward, step_metrics = self._step_construction_reward(tool_calls, tool_results)
        metrics.update(step_metrics)
        answerability_reward, answerability_metrics = await self._maybe_answerability_progress_reward(
            done=done,
            step_metrics=step_metrics,
        )
        reward += answerability_reward
        metrics.update(answerability_metrics)
        if "step_reward" in metrics:
            step_process_reward = step_metrics.get("step_reward", 0.0)
            metrics["step_process_reward"] = step_process_reward
            metrics["step_reward"] = reward
            metrics.update(
                {
                    "intermediate_answerability_reward": answerability_reward,
                    "intermediate_process_reward": step_process_reward,
                    "intermediate_structure_reward": step_metrics.get(
                        "step_filesystem_maturity_reward", 0.0
                    ),
                    "intermediate_create_merge_action_bonus": step_metrics.get(
                        "step_construction_bonus", 0.0
                    ),
                    "intermediate_tool_error_penalty": step_metrics.get(
                        "step_tool_error_penalty", 0.0
                    ),
                    "intermediate_non_construction_penalty": (
                        step_metrics.get("step_non_construction_penalty", 0.0)
                        + step_metrics.get("step_non_construction_streak_penalty", 0.0)
                    ),
                    "intermediate_total_reward": reward,
                }
            )

        if not done:
            compaction_metrics, compaction_logs = await self._maybe_compact_visible_history()
            metrics.update(compaction_metrics)
            logs.update(compaction_logs)

        if max_turns_reached and not no_tool_calls:
            metrics["max_turns"] = 1.0
        if self._should_stop:
            metrics["tool_stopped"] = 1.0

        if done:
            terminal_reward, reward_metrics = await self.reward_fn(self.reward_history)
            reward += terminal_reward
            metrics.update(reward_metrics)
            metrics["builder_context_compactions"] = float(self._compaction_count)
            metrics["builder_context_compaction_errors"] = float(self._compaction_error_count)
            if self.trim_terminal_history_for_memory:
                self._trim_terminal_history()
            if not self.retain_reward_tool_messages:
                self.reward_history.clear()
            next_messages = [] if self.return_empty_terminal_observation else self.history
            if self.clear_state_on_terminal_for_memory and self.state is not None:
                self.state.files_by_id.clear()
                self.tools.clear()
                self._tool_dict.clear()
                self.initial_messages.clear()
            if self.return_empty_terminal_observation:
                self.history.clear()
            gc.collect()
        else:
            next_messages = self.history

        return MessageStepResult(
            reward=reward,
            episode_done=done,
            next_messages=next_messages,
            metrics=metrics,
            logs=logs,
        )

    async def _maybe_answerability_progress_reward(
        self,
        *,
        done: bool,
        step_metrics: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        metrics = {
            "answerability_probe_attempted": 0.0,
            "answerability_probe_evaluated": 0.0,
            "answerability_probe_reward": 0.0,
            "answerability_probe_interval_turns": float(self.answerability_probe_interval_turns),
        }
        if done:
            return 0.0, metrics
        if self.answerability_delta_reward_scale <= 0.0:
            return 0.0, metrics
        if self._answerability_probe_count >= max(0, self.answerability_probe_max_per_episode):
            return 0.0, metrics
        interval_turns = max(0, self.answerability_probe_interval_turns)
        if interval_turns > 0:
            turns_since_probe = self._turn_count - self._last_answerability_probe_turn
            if turns_since_probe < interval_turns:
                metrics["answerability_probe_skipped_interval"] = 1.0
                metrics["answerability_probe_turns_since_last"] = float(turns_since_probe)
                return 0.0, metrics
        if step_metrics.get("step_construction_actions", 0.0) <= 0.0:
            return 0.0, metrics
        maturity = step_metrics.get("step_filesystem_maturity_score", 0.0)
        if maturity < self.answerability_probe_min_maturity:
            return 0.0, metrics
        probe_fn = getattr(self.reward_fn, "answerability_progress_probe", None)
        if probe_fn is None:
            return 0.0, metrics

        metrics["answerability_probe_attempted"] = 1.0
        try:
            new_best, reward, probe_metrics = await probe_fn(
                previous_score=self._last_answerability_score,
                previous_best_score=self._best_answerability_score
            )
        except Exception as exc:
            metrics["answerability_probe_error"] = 1.0
            metrics["answerability_probe_reward"] = 0.0
            LOGGER.warning("Answerability progress probe failed: %s", exc)
            return 0.0, metrics

        self._answerability_probe_count += 1
        self._last_answerability_probe_turn = self._turn_count
        current_score = float(
            probe_metrics.get("answerability_probe_score", self._last_answerability_score)
        )
        self._best_answerability_score = max(self._best_answerability_score, float(new_best))
        self._last_answerability_score = current_score
        metrics.update(probe_metrics)
        return reward, metrics

    def _trim_terminal_history(self) -> None:
        last_assistant: Message | None = None
        for message in reversed(self.history):
            if message.get("role") == "assistant":
                last_assistant = message
                break
        trimmed = list(self.initial_messages)
        if self._compacted_summary.strip():
            trimmed.append(_compaction_summary_message(self._compacted_summary))
        if last_assistant is not None:
            trimmed.append(last_assistant)
        self.history = trimmed

    def _step_construction_reward(
        self,
        tool_calls: Sequence[Any],
        tool_results: Sequence[ToolResult],
    ) -> tuple[float, dict[str, float]]:
        if not tool_calls:
            return 0.0, {}

        successful_create = 0
        successful_merge = 0
        successful_read_file = 0
        successful_list_files = 0
        tool_errors = 0
        executor_generated_actions = 0
        tool_error_counts: dict[str, int] = {}
        tool_error_code_counts: dict[str, int] = {}
        for tool_call, tool_result in zip(tool_calls, tool_results, strict=True):
            tool_name = getattr(tool_call.function, "name", "")
            if not _tool_result_succeeded(tool_result):
                tool_errors += 1
                tool_error_counts[tool_name] = tool_error_counts.get(tool_name, 0) + 1
                error_code_key = f"{_metric_key_part(tool_name)}_{_metric_key_part(_tool_result_error_code(tool_result))}"
                tool_error_code_counts[error_code_key] = tool_error_code_counts.get(error_code_key, 0) + 1
                continue
            payload = _tool_result_payload(tool_result)
            if tool_name == "create_cluster":
                successful_create += 1
                if payload.get("executor_generated"):
                    executor_generated_actions += 1
            elif tool_name == "create_clusters":
                created_count = int(float(payload.get("created_count", 0.0) or 0.0))
                successful_create += created_count
                if payload.get("executor_generated"):
                    executor_generated_actions += created_count
            elif tool_name == "merge_clusters":
                successful_merge += 1
                if payload.get("executor_generated"):
                    executor_generated_actions += 1
            elif tool_name == "merge_many":
                created_count = int(float(payload.get("created_count", 0.0) or 0.0))
                successful_merge += created_count
                if payload.get("executor_generated"):
                    executor_generated_actions += created_count
            elif tool_name == "read_file":
                successful_read_file += 1
            elif tool_name == "read_many":
                reads = payload.get("reads", [])
                successful_read_file += len(reads) if isinstance(reads, list) else 1
            elif tool_name == "list_files":
                successful_list_files += 1

        construction_actions = successful_create + successful_merge
        non_construction_tool_calls = successful_read_file + successful_list_files
        reward = 0.0
        construction_bonus = self.step_construction_action_bonus * float(construction_actions)
        reward += construction_bonus

        filesystem_maturity_delta = 0.0
        filesystem_maturity_reward = 0.0
        filesystem_maturity_score = self._last_step_filesystem_maturity_score
        if construction_actions > 0 and self.filesystem_maturity_score_fn is not None:
            filesystem_maturity_score = self.filesystem_maturity_score_fn()
            filesystem_maturity_delta = max(
                0.0,
                filesystem_maturity_score - self._last_step_filesystem_maturity_score,
            )
            filesystem_maturity_reward = (
                self.step_filesystem_maturity_delta_scale * filesystem_maturity_delta
            )
            reward += filesystem_maturity_reward
            self._last_step_filesystem_maturity_score = filesystem_maturity_score

        non_construction_penalty = 0.0
        if non_construction_tool_calls > 0 and self.step_non_construction_turn_penalty > 0.0:
            non_construction_penalty = (
                self.step_non_construction_turn_penalty * float(non_construction_tool_calls)
            )
            reward -= non_construction_penalty

        if construction_actions > 0:
            self._non_construction_streak = 0
        elif non_construction_tool_calls > 0:
            self._non_construction_streak += non_construction_tool_calls

        non_construction_streak_penalty = 0.0
        if self.step_non_construction_streak_penalty > 0.0:
            chargeable_streak = max(
                0,
                self._non_construction_streak - max(0, self.step_non_construction_streak_free),
            )
            non_construction_streak_penalty = (
                self.step_non_construction_streak_penalty * float(chargeable_streak)
            )
            reward -= non_construction_streak_penalty

        tool_error_penalty = self.step_tool_error_penalty * float(tool_errors)
        reward -= tool_error_penalty

        metrics = {
            "step_successful_create_clusters": float(successful_create),
            "step_successful_merge_clusters": float(successful_merge),
            "step_successful_read_files": float(successful_read_file),
            "step_successful_list_files": float(successful_list_files),
            "step_construction_actions": float(construction_actions),
            "step_executor_generated_actions": float(executor_generated_actions),
            "step_construction_bonus": construction_bonus,
            "step_non_construction_tool_calls": float(non_construction_tool_calls),
            "step_filesystem_maturity_score": filesystem_maturity_score,
            "step_filesystem_maturity_delta": filesystem_maturity_delta,
            "step_filesystem_maturity_reward": filesystem_maturity_reward,
            "step_non_construction_penalty": non_construction_penalty,
            "step_non_construction_streak": float(self._non_construction_streak),
            "step_non_construction_streak_penalty": non_construction_streak_penalty,
            "step_tool_errors": float(tool_errors),
            "step_tool_errors_read_file": float(tool_error_counts.get("read_file", 0)),
            "step_tool_errors_read_many": float(tool_error_counts.get("read_many", 0)),
            "step_tool_errors_list_files": float(tool_error_counts.get("list_files", 0)),
            "step_tool_errors_create_cluster": float(tool_error_counts.get("create_cluster", 0)),
            "step_tool_errors_create_clusters": float(tool_error_counts.get("create_clusters", 0)),
            "step_tool_errors_merge_clusters": float(tool_error_counts.get("merge_clusters", 0)),
            "step_tool_errors_merge_many": float(tool_error_counts.get("merge_many", 0)),
            "step_tool_errors_delete": float(tool_error_counts.get("delete", 0)),
            "step_tool_error_penalty": tool_error_penalty,
            "step_reward": reward,
        }
        for error_code_key, count in tool_error_code_counts.items():
            metrics[f"step_tool_error_code_{error_code_key}"] = float(count)
        return reward, metrics

    async def _maybe_compact_visible_history(self) -> tuple[dict[str, float], dict[str, str]]:
        metrics: dict[str, float] = {}
        logs: dict[str, str] = {}
        if not self.compaction_enabled:
            return metrics, logs
        if self.compaction_trigger_tokens <= 0:
            return metrics, logs

        visible_tokens_before = _history_approx_tokens(self.history)
        metrics["builder_context_visible_tokens"] = float(visible_tokens_before)
        if visible_tokens_before < self.compaction_trigger_tokens:
            return metrics, logs

        prefix_len = len(self.initial_messages)
        if len(self.history) <= prefix_len:
            return metrics, logs

        mutable_history = list(self.history[prefix_len:])
        if mutable_history and _is_compaction_summary_message(mutable_history[0]):
            mutable_history = mutable_history[1:]

        assistant_indices = [
            idx for idx, item in enumerate(mutable_history) if item.get("role") == "assistant"
        ]
        keep_recent_turns = max(1, self.compaction_keep_recent_turns)
        if len(assistant_indices) <= keep_recent_turns:
            return metrics, logs

        tail_start = assistant_indices[-keep_recent_turns]
        history_to_compact = mutable_history[:tail_start]
        recent_history = mutable_history[tail_start:]
        if not history_to_compact:
            return metrics, logs

        transcript = _render_history_for_compaction(
            history_to_compact,
            max_chars=self.compaction_input_max_chars,
        )
        try:
            summary = await self._summarize_builder_context(transcript)
        except Exception as exc:
            self._compaction_error_count += 1
            summary = self._fallback_compaction_summary(transcript, error=str(exc))
            logs["builder_context_compaction_error"] = str(exc)[:500]

        self._compacted_summary = summary.strip()
        self.history = (
            list(self.initial_messages)
            + [_compaction_summary_message(self._compacted_summary)]
            + recent_history
        )
        self._compaction_count += 1

        visible_tokens_after = _history_approx_tokens(self.history)
        metrics.update(
            {
                "builder_context_compaction_applied": 1.0,
                "builder_context_compactions": float(self._compaction_count),
                "builder_context_tokens_before_compaction": float(visible_tokens_before),
                "builder_context_tokens_after_compaction": float(visible_tokens_after),
                "builder_context_messages_compacted": float(len(history_to_compact)),
                "builder_context_recent_messages_kept": float(len(recent_history)),
            }
        )
        if self.log_compaction_summaries:
            logs["builder_context_compaction_summary"] = self._compacted_summary[:2000]
        return metrics, logs

    async def _summarize_builder_context(self, latest_transcript: str) -> str:
        filesystem_snapshot = self._filesystem_snapshot()
        existing_summary = self._compacted_summary.strip() or "(none yet)"
        system_prompt = (
            "You compact the visible context of a tool-using agent that is constructing a synthetic filesystem. "
            "Write an updated memory summary only from the provided transcript, previous summary, and filesystem snapshot. "
            "Do not add facts that are not present. Preserve concrete source paths, line ranges, names, dates, aliases, "
            "relationships, and any evidence-to-synthetic-file links. The builder is optimizing for broad corpus "
            "coverage and reusable structure, not for a particular query, so do not infer answers or introduce "
            "query-specific conclusions."
        )
        user_prompt = (
            "Update the rolling builder memory for filesystem construction.\n\n"
            "The summary should be concise but useful for the next builder turns. Include:\n"
            "- raw files/chunks already read and the important evidence extracted from them\n"
            "- synthetic files or merge summaries already created, with paths/titles and what evidence they incorporated\n"
            "- current active filesystem structure and any deleted/obsolete state if visible\n"
            "- unresolved useful next actions, such as files/chunks that may still need reading or clusters that may need merging\n"
            "- warnings about uncertainty or incomplete coverage\n\n"
            "Keep enough detail that the builder can keep constructing without re-reading every old chunk. "
            "If exact wording is needed later, say which path/line range should be re-opened with read_file.\n\n"
            f"### EXISTING ROLLING SUMMARY\n{existing_summary}\n\n"
            f"### CURRENT FILESYSTEM SNAPSHOT\n{filesystem_snapshot}\n\n"
            f"### LATEST BUILDER TRANSCRIPT TO INCORPORATE\n{latest_transcript}\n\n"
            "Return only the updated summary. Do not include preamble."
        )
        response = await asyncio.to_thread(
            call_chat_model_messages,
            backend=self.compaction_backend,
            model=self.compaction_model,
            base_url=self.compaction_base_url,
            api_key_env=self.compaction_api_key_env,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_json=False,
            max_output_tokens=self.compaction_max_output_tokens,
        )
        return response.strip()

    def _filesystem_snapshot(self) -> str:
        if self.state is None:
            return "(filesystem state unavailable)"
        active_records = self.state.active_records()
        lines = [
            f"operation_round: {self.state.operation_round}",
            "active_tree:",
            self.state.active_tree_string(),
            "active_files:",
        ]
        if not active_records:
            lines.append("- (none)")
        for rec in active_records:
            lines.append(
                "- "
                + json.dumps(
                    {
                        "path": self.state.active_path(rec.node_id),
                        "history_path": self.state.history_path(rec.node_id),
                        "title": rec.title,
                        "kind": rec.kind,
                        "direct_raw_doc_paths": rec.direct_raw_doc_paths,
                        "direct_synthetic_ids": rec.direct_synthetic_ids,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        return "\n".join(lines)

    def _fallback_compaction_summary(self, latest_transcript: str, *, error: str) -> str:
        transcript_excerpt = _trim_text_middle(latest_transcript, 6000)
        existing = self._compacted_summary.strip()
        pieces = []
        if existing:
            pieces.append(existing)
        pieces.append(
            "Automatic model compaction failed, so this is a conservative fallback summary. "
            f"Compaction error: {error[:300]}"
        )
        pieces.append("Current filesystem snapshot:\n" + self._filesystem_snapshot())
        pieces.append(
            "Recent older transcript excerpt that was compacted; re-open exact files with read_file if needed:\n"
            + transcript_excerpt
        )
        return "\n\n".join(pieces).strip()


def build_redacting_agent_tool_env(
    *,
    renderer: Renderer,
    tools: list[Tool],
    initial_messages: list[Message],
    reward_fn: RewardFn,
    max_turns: int,
    state: SyntheticFilesystemState | None = None,
    filesystem_maturity_score_fn: ProxyScoreFn | None = None,
    step_construction_action_bonus: float = 0.05,
    step_filesystem_maturity_delta_scale: float = 0.5,
    step_non_construction_turn_penalty: float = 0.005,
    step_non_construction_streak_penalty: float = 0.0,
    step_non_construction_streak_free: int = 3,
    step_tool_error_penalty: float = 0.05,
    answerability_delta_reward_scale: float = 0.0,
    answerability_probe_max_per_episode: int = 4,
    answerability_probe_interval_turns: int = 8,
    answerability_probe_min_maturity: float = 0.45,
    log_step_details: bool = False,
    log_compaction_summaries: bool = False,
    retain_reward_tool_messages: bool = False,
    trim_terminal_history_for_memory: bool = True,
    return_empty_terminal_observation: bool = True,
    clear_state_on_terminal_for_memory: bool = True,
    compaction_enabled: bool = True,
    compaction_backend: JudgeBackend = "gemini",
    compaction_model: str = "gemini-3.1-flash-lite-preview",
    compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    compaction_api_key_env: str = "GEMINI_API_KEY",
    compaction_trigger_tokens: int = DEFAULT_BUILDER_COMPACTION_TRIGGER_TOKENS,
    compaction_keep_recent_turns: int = DEFAULT_BUILDER_COMPACTION_KEEP_RECENT_TURNS,
    compaction_max_output_tokens: int = DEFAULT_BUILDER_COMPACTION_MAX_OUTPUT_TOKENS,
    compaction_input_max_chars: int = DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS,
    failed_parse_reward: float = -0.1,
    max_trajectory_tokens: int | None = 140000,
    context_overflow_reward: float = -0.1,
    progress_metrics: dict[str, float] | None = None,
) -> Env:
    message_env = RedactingAgentToolMessageEnv(
        tools=tools,
        initial_messages=initial_messages,
        max_turns=max_turns,
        reward_fn=reward_fn,
        state=state,
        filesystem_maturity_score_fn=filesystem_maturity_score_fn,
        step_construction_action_bonus=step_construction_action_bonus,
        step_filesystem_maturity_delta_scale=step_filesystem_maturity_delta_scale,
        step_non_construction_turn_penalty=step_non_construction_turn_penalty,
        step_non_construction_streak_penalty=step_non_construction_streak_penalty,
        step_non_construction_streak_free=step_non_construction_streak_free,
        step_tool_error_penalty=step_tool_error_penalty,
        answerability_delta_reward_scale=answerability_delta_reward_scale,
        answerability_probe_max_per_episode=answerability_probe_max_per_episode,
        answerability_probe_interval_turns=answerability_probe_interval_turns,
        answerability_probe_min_maturity=answerability_probe_min_maturity,
        log_step_details=log_step_details,
        log_compaction_summaries=log_compaction_summaries,
        retain_reward_tool_messages=retain_reward_tool_messages,
        trim_terminal_history_for_memory=trim_terminal_history_for_memory,
        return_empty_terminal_observation=return_empty_terminal_observation,
        clear_state_on_terminal_for_memory=clear_state_on_terminal_for_memory,
        compaction_enabled=compaction_enabled,
        compaction_backend=compaction_backend,
        compaction_model=compaction_model,
        compaction_base_url=compaction_base_url,
        compaction_api_key_env=compaction_api_key_env,
        compaction_trigger_tokens=compaction_trigger_tokens,
        compaction_keep_recent_turns=compaction_keep_recent_turns,
        compaction_max_output_tokens=compaction_max_output_tokens,
        compaction_input_max_chars=compaction_input_max_chars,
        progress_metrics=dict(progress_metrics or {}),
    )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=message_env,
        failed_parse_reward=failed_parse_reward,
        max_trajectory_tokens=max_trajectory_tokens,
        context_overflow_reward=context_overflow_reward,
    )


def initial_messages_builder(
    datum: SyntheticFilesystemDatum,
    renderer: Renderer,
    fs_tools: BuilderFilesystemTools,
    *,
    executor_enabled: bool = False,
    batch_tools_enabled: bool = False,
) -> list[Message]:
    tool_schemas = [
        fs_tools.list_files.to_spec(),
        fs_tools.read_file.to_spec(),
    ]
    if batch_tools_enabled:
        tool_schemas.append(fs_tools.read_many.to_spec())
    tool_schemas.extend(
        [
            fs_tools.create_cluster.to_spec(),
            fs_tools.merge_clusters.to_spec(),
        ]
    )
    if batch_tools_enabled:
        tool_schemas.extend(
            [
                fs_tools.create_clusters.to_spec(),
                fs_tools.merge_many.to_spec(),
            ]
        )
    tool_schemas.append(fs_tools.delete.to_spec())
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=build_builder_system_prompt(
            executor_enabled=executor_enabled,
            batch_tools_enabled=batch_tools_enabled,
        ),
    )
    if executor_enabled:
        user_prompt = (
            "Use the filesystem tools as a planner/controller. Inspect useful files, then propose concise operations; "
            "the frozen executor will write cluster and merge content from your selected sources.\n"
            "Build a broad, tree-like synthetic filesystem of reusable memory banks that will help later answering agents answer as many plausible future questions as possible.\n"
            "Raw documents are available under raw_docs/. Use list_files/read_many to inspect only what is needed to choose good sources and grouping.\n"
            "Prefer cycles of batched reads, batched focused leaf-cluster creation, merge related clusters into parent summaries, then STOP once the filesystem is mature.\n"
            f"When you are done, return exactly one line: {STOP_MESSAGE}"
        )
    else:
        user_prompt = (
            "Use the filesystem tools to inspect whichever files you think are useful. "
            "Build a broad, tree-like synthetic filesystem of reusable memory banks that will help later answering agents answer as many plausible future questions as possible.\n"
            "Raw documents are available under raw_docs/. Use list_files to inspect only the files needed to build useful synthetic memory banks.\n"
            "Do not treat reading every raw document as a goal; the goal is the final synthetic filesystem. "
            "Prefer cycles of read a few files, create focused leaf clusters, merge related clusters into parent summaries, then STOP once the filesystem is mature.\n"
            f"When you are done, return exactly one line: {STOP_MESSAGE}"
        )
    return prefix + [{"role": "user", "content": user_prompt}]


@dataclass(frozen=True)
class SyntheticFilesystemEnvGroupBuilder(EnvGroupBuilder):
    datum: SyntheticFilesystemDatum
    model_name: str
    renderer_name: str | None
    group_size: int
    reward_mode: str = "hybrid"
    answerer_backend: JudgeBackend = "gemini"
    answerer_model: str = "gemini-3.1-flash-lite-preview"
    answerer_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    answerer_api_key_env: str = "GEMINI_API_KEY"
    judge_backend: JudgeBackend = "gemini"
    judge_model: str = "gemini-3.1-flash-lite-preview"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_api_key_env: str = "GEMINI_API_KEY"
    max_turns: int = 32
    max_trajectory_tokens: int | None = 140000
    max_generation_tokens: int | None = None
    step_penalty: float = 0.0
    termination_penalty: float = 0.1
    raw_docs_penalty: float = 0.0
    empty_synthetic_penalty: float = 1.0
    synthetic_success_bonus: float = 0.0
    synthetic_usage_bonus: float = 0.0
    raw_usage_ratio_penalty: float = 0.0
    filesystem_maturity_scale: float = 0.5
    filesystem_coverage_weight: float = 0.35
    filesystem_expansion_weight: float = 0.3
    filesystem_organization_weight: float = 0.35
    filesystem_stop_weight: float = 0.0
    mature_stop_bonus: float = 0.0
    mature_stop_min_score: float = 0.8
    terminal_reward_clip_min: float = -1.0
    terminal_reward_clip_max: float = 3.0
    answerer_max_turns: int = 32
    answerer_workspace_mode: AnswererWorkspaceMode = "synthetic_only"
    answerer_final_answer_max_tokens: int = 128
    answerer_retrieval_cost_scale: float = 0.0
    answerer_retrieval_cost_token_unit: float = 1000.0
    answerer_retrieval_cost_correct_only: bool = True
    answerer_synthetic_read_cost_scale: float = 0.0
    answerer_synthetic_read_cost_unit: float = 10.0
    terminal_answerer_repeats: int = 4
    answerability_delta_reward_scale: float = 0.0
    answerability_delta_min_abs: float = 0.25
    answerability_delta_allow_negative: bool = True
    answerability_probe_max_per_episode: int = 4
    answerability_probe_interval_turns: int = 8
    answerability_probe_min_maturity: float = 0.45
    answerability_probe_repeats: int = 4
    judge_max_output_tokens: int = 64
    log_step_details: bool = False
    log_compaction_summaries: bool = False
    retain_reward_tool_messages: bool = False
    trim_terminal_history_for_memory: bool = True
    return_empty_terminal_observation: bool = True
    clear_state_on_terminal_for_memory: bool = True
    builder_compaction_enabled: bool = True
    builder_compaction_backend: JudgeBackend = "gemini"
    builder_compaction_model: str = "gemini-3.1-flash-lite-preview"
    builder_compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    builder_compaction_api_key_env: str = "GEMINI_API_KEY"
    builder_compaction_trigger_tokens: int = 3000
    builder_compaction_keep_recent_turns: int = 1
    builder_compaction_max_output_tokens: int = 800
    builder_compaction_input_max_chars: int = DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS
    builder_executor_enabled: bool = True
    builder_batch_tools_enabled: bool = True
    builder_executor_backend: JudgeBackend = "openrouter"
    builder_executor_model: str = "qwen/qwen3.5-35b-a3b"
    builder_executor_base_url: str = "https://openrouter.ai/api/v1"
    builder_executor_api_key_env: str = "OPENROUTER_API_KEY"
    builder_executor_max_source_chars: int = 16000
    builder_executor_max_output_tokens: int = 512
    step_construction_action_bonus: float = 0.05
    step_filesystem_maturity_delta_scale: float = 0.5
    step_non_construction_turn_penalty: float = 0.005
    step_non_construction_streak_penalty: float = 0.0
    step_non_construction_streak_free: int = 3
    step_tool_error_penalty: float = 0.05
    training_progress_metrics: tuple[tuple[str, float], ...] = ()

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(renderer_name, tokenizer)
        envs: list[Env] = []
        raw_doc_root = Path(self.datum["agent_query_dir"])
        visible_raw_doc_paths = [
            str(file_info.get("relative_path", ""))
            for file_info in self.datum.get("files", [])
            if str(file_info.get("relative_path", ""))
        ]
        for _ in range(self.group_size):
            state = SyntheticFilesystemState(
                raw_doc_root=raw_doc_root,
                visible_raw_doc_paths=visible_raw_doc_paths,
                question_id=self.datum["question_id"],
            )
            executor = (
                FrozenSyntheticFileExecutor(
                    backend=self.builder_executor_backend,
                    model=self.builder_executor_model,
                    base_url=self.builder_executor_base_url,
                    api_key_env=self.builder_executor_api_key_env,
                    max_source_chars=self.builder_executor_max_source_chars,
                    max_output_tokens=self.builder_executor_max_output_tokens,
                )
                if self.builder_executor_enabled
                else None
            )
            fs_tools = BuilderFilesystemTools(state, executor=executor)
            tools = [
                fs_tools.list_files,
                fs_tools.read_file,
            ]
            if self.builder_batch_tools_enabled or self.builder_executor_enabled:
                tools.append(fs_tools.read_many)
            tools.extend(
                [
                    fs_tools.create_cluster,
                    fs_tools.merge_clusters,
                ]
            )
            if self.builder_batch_tools_enabled or self.builder_executor_enabled:
                tools.extend(
                    [
                        fs_tools.create_clusters,
                        fs_tools.merge_many,
                    ]
                )
            tools.append(fs_tools.delete)
            reward_fn = SyntheticFilesystemReward(
                state=state,
                gold_answer=self.datum["gold_answer"],
                question=self.datum["question"],
                answerer_backend=self.answerer_backend,
                answerer_model=self.answerer_model,
                answerer_base_url=self.answerer_base_url,
                answerer_api_key_env=self.answerer_api_key_env,
                reward_mode=self.reward_mode,
                judge_backend=self.judge_backend,
                judge_model=self.judge_model,
                judge_base_url=self.judge_base_url,
                judge_api_key_env=self.judge_api_key_env,
                step_penalty=self.step_penalty,
                termination_penalty=self.termination_penalty,
                raw_docs_penalty=self.raw_docs_penalty,
                empty_synthetic_penalty=self.empty_synthetic_penalty,
                synthetic_success_bonus=self.synthetic_success_bonus,
                synthetic_usage_bonus=self.synthetic_usage_bonus,
                raw_usage_ratio_penalty=self.raw_usage_ratio_penalty,
                filesystem_maturity_scale=self.filesystem_maturity_scale,
                filesystem_coverage_weight=self.filesystem_coverage_weight,
                filesystem_expansion_weight=self.filesystem_expansion_weight,
                filesystem_organization_weight=self.filesystem_organization_weight,
                filesystem_stop_weight=self.filesystem_stop_weight,
                mature_stop_bonus=self.mature_stop_bonus,
                mature_stop_min_score=self.mature_stop_min_score,
                terminal_reward_clip_min=self.terminal_reward_clip_min,
                terminal_reward_clip_max=self.terminal_reward_clip_max,
                builder_max_turns=self.max_turns,
                answerer_max_turns=self.answerer_max_turns,
                answerer_workspace_mode=self.answerer_workspace_mode,
                answerer_final_answer_max_tokens=self.answerer_final_answer_max_tokens,
                answerer_retrieval_cost_scale=self.answerer_retrieval_cost_scale,
                answerer_retrieval_cost_token_unit=self.answerer_retrieval_cost_token_unit,
                answerer_retrieval_cost_correct_only=self.answerer_retrieval_cost_correct_only,
                answerer_synthetic_read_cost_scale=self.answerer_synthetic_read_cost_scale,
                answerer_synthetic_read_cost_unit=self.answerer_synthetic_read_cost_unit,
                terminal_answerer_repeats=self.terminal_answerer_repeats,
                answerability_delta_reward_scale=self.answerability_delta_reward_scale,
                answerability_delta_min_abs=self.answerability_delta_min_abs,
                answerability_delta_allow_negative=self.answerability_delta_allow_negative,
                answerability_probe_repeats=self.answerability_probe_repeats,
                judge_max_output_tokens=self.judge_max_output_tokens,
            )
            envs.append(
                build_redacting_agent_tool_env(
                    renderer=renderer,
                    tools=tools,
                    initial_messages=initial_messages_builder(
                        self.datum,
                        renderer,
                        fs_tools,
                        executor_enabled=self.builder_executor_enabled,
                        batch_tools_enabled=self.builder_batch_tools_enabled or self.builder_executor_enabled,
                    ),
                    reward_fn=reward_fn,
                    max_turns=self.max_turns,
                    state=state,
                    filesystem_maturity_score_fn=reward_fn._current_filesystem_maturity_score,
                    step_construction_action_bonus=self.step_construction_action_bonus,
                    step_filesystem_maturity_delta_scale=self.step_filesystem_maturity_delta_scale,
                    step_non_construction_turn_penalty=self.step_non_construction_turn_penalty,
                    step_non_construction_streak_penalty=self.step_non_construction_streak_penalty,
                    step_non_construction_streak_free=self.step_non_construction_streak_free,
                    step_tool_error_penalty=self.step_tool_error_penalty,
                    answerability_delta_reward_scale=self.answerability_delta_reward_scale,
                    answerability_probe_max_per_episode=self.answerability_probe_max_per_episode,
                    answerability_probe_interval_turns=self.answerability_probe_interval_turns,
                    answerability_probe_min_maturity=self.answerability_probe_min_maturity,
                    log_step_details=self.log_step_details,
                    log_compaction_summaries=self.log_compaction_summaries,
                    retain_reward_tool_messages=self.retain_reward_tool_messages,
                    trim_terminal_history_for_memory=self.trim_terminal_history_for_memory,
                    return_empty_terminal_observation=self.return_empty_terminal_observation,
                    clear_state_on_terminal_for_memory=self.clear_state_on_terminal_for_memory,
                    compaction_enabled=self.builder_compaction_enabled,
                    compaction_backend=self.builder_compaction_backend,
                    compaction_model=self.builder_compaction_model,
                    compaction_base_url=self.builder_compaction_base_url,
                    compaction_api_key_env=self.builder_compaction_api_key_env,
                    compaction_trigger_tokens=self.builder_compaction_trigger_tokens,
                    compaction_keep_recent_turns=self.builder_compaction_keep_recent_turns,
                    compaction_max_output_tokens=self.builder_compaction_max_output_tokens,
                    compaction_input_max_chars=self.builder_compaction_input_max_chars,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    progress_metrics=dict(self.training_progress_metrics),
                    # Intentionally do not pass max_generation_tokens here. The builder should
                    # not be method-limited by a per-turn generation cap; only the backend-level
                    # sampler ceiling remains outside this environment.
                )
            )
        return envs

    def logging_tags(self) -> list[str]:
        return [self.datum.get("dataset_type", "support_only"), "synthetic_filesystem_builder"]


class SyntheticFilesystemRLDataset(RLDataset):
    def __init__(self, env_group_builders: list[SyntheticFilesystemEnvGroupBuilder], batch_size: int):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batches_per_epoch = max(1, len(self))
        batch_index_in_epoch = index % batches_per_epoch
        start = batch_index_in_epoch * self.batch_size
        end = start + self.batch_size
        batch_builders = self.env_group_builders[start:end]
        examples_this_batch = len(batch_builders)
        examples_seen = index * self.batch_size + examples_this_batch
        progress_metrics = tuple(
            sorted(
                {
                    "training_batch_index": float(index + 1),
                    "training_batch_in_epoch": float(batch_index_in_epoch + 1),
                    "training_batches_per_epoch": float(batches_per_epoch),
                    "training_epoch": float(index + 1) / float(batches_per_epoch),
                    "training_epoch_index": float(index // batches_per_epoch),
                    "training_examples_per_batch": float(self.batch_size),
                    "training_examples_seen": float(examples_seen),
                    "training_rollouts_seen": float(examples_seen * self.env_group_builders[0].group_size)
                    if self.env_group_builders
                    else 0.0,
                    "training_train_rows": float(len(self.env_group_builders)),
                }.items()
            )
        )
        return [
            replace(builder, training_progress_metrics=progress_metrics)
            for builder in batch_builders
        ]

    def __len__(self) -> int:
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class SyntheticFilesystemDatasetBuilder(RLDatasetBuilder):
    index_jsonl: str
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    reward_mode: str = "hybrid"
    answerer_backend: JudgeBackend = "gemini"
    answerer_model: str = "gemini-3.1-flash-lite-preview"
    answerer_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    answerer_api_key_env: str = "GEMINI_API_KEY"
    judge_backend: JudgeBackend = "gemini"
    judge_model: str = "gemini-3.1-flash-lite-preview"
    judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    judge_api_key_env: str = "GEMINI_API_KEY"
    max_turns: int = 32
    max_trajectory_tokens: int | None = 140000
    max_generation_tokens: int | None = None
    step_penalty: float = 0.0
    termination_penalty: float = 0.1
    raw_docs_penalty: float = 0.0
    empty_synthetic_penalty: float = 1.0
    synthetic_success_bonus: float = 0.0
    synthetic_usage_bonus: float = 0.0
    raw_usage_ratio_penalty: float = 0.0
    filesystem_maturity_scale: float = 0.5
    filesystem_coverage_weight: float = 0.35
    filesystem_expansion_weight: float = 0.3
    filesystem_organization_weight: float = 0.35
    filesystem_stop_weight: float = 0.0
    mature_stop_bonus: float = 0.0
    mature_stop_min_score: float = 0.8
    terminal_reward_clip_min: float = -1.0
    terminal_reward_clip_max: float = 3.0
    answerer_max_turns: int = 32
    answerer_workspace_mode: AnswererWorkspaceMode = "synthetic_only"
    answerer_final_answer_max_tokens: int = 128
    answerer_retrieval_cost_scale: float = 0.0
    answerer_retrieval_cost_token_unit: float = 1000.0
    answerer_retrieval_cost_correct_only: bool = True
    answerer_synthetic_read_cost_scale: float = 0.0
    answerer_synthetic_read_cost_unit: float = 10.0
    terminal_answerer_repeats: int = 4
    answerability_delta_reward_scale: float = 0.0
    answerability_delta_min_abs: float = 0.25
    answerability_delta_allow_negative: bool = True
    answerability_probe_max_per_episode: int = 4
    answerability_probe_interval_turns: int = 8
    answerability_probe_min_maturity: float = 0.45
    answerability_probe_repeats: int = 4
    judge_max_output_tokens: int = 64
    log_step_details: bool = False
    log_compaction_summaries: bool = False
    retain_reward_tool_messages: bool = False
    trim_terminal_history_for_memory: bool = True
    return_empty_terminal_observation: bool = True
    clear_state_on_terminal_for_memory: bool = True
    builder_compaction_enabled: bool = True
    builder_compaction_backend: JudgeBackend = "gemini"
    builder_compaction_model: str = "gemini-3.1-flash-lite-preview"
    builder_compaction_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    builder_compaction_api_key_env: str = "GEMINI_API_KEY"
    builder_compaction_trigger_tokens: int = 3000
    builder_compaction_keep_recent_turns: int = 1
    builder_compaction_max_output_tokens: int = 800
    builder_compaction_input_max_chars: int = DEFAULT_BUILDER_COMPACTION_INPUT_MAX_CHARS
    builder_executor_enabled: bool = True
    builder_batch_tools_enabled: bool = True
    builder_executor_backend: JudgeBackend = "openrouter"
    builder_executor_model: str = "qwen/qwen3.5-35b-a3b"
    builder_executor_base_url: str = "https://openrouter.ai/api/v1"
    builder_executor_api_key_env: str = "OPENROUTER_API_KEY"
    builder_executor_max_source_chars: int = 16000
    builder_executor_max_output_tokens: int = 512
    step_construction_action_bonus: float = 0.05
    step_filesystem_maturity_delta_scale: float = 0.5
    step_non_construction_turn_penalty: float = 0.005
    step_non_construction_streak_penalty: float = 0.0
    step_non_construction_streak_free: int = 3
    step_tool_error_penalty: float = 0.05
    excluded_qids_jsonl: str = ""
    eval_index_jsonl: str = ""
    seed: int = 0
    eval_size: int = 0
    limit: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        data = load_index(Path(self.index_jsonl))
        excluded = load_excluded_qids(Path(self.excluded_qids_jsonl)) if self.excluded_qids_jsonl else set()
        if excluded:
            data = [datum for datum in data if datum["question_id"] not in excluded]

        eval_rows: list[SyntheticFilesystemDatum] = []
        if self.eval_index_jsonl:
            eval_rows = load_index(Path(self.eval_index_jsonl))
            if excluded:
                eval_rows = [datum for datum in eval_rows if datum["question_id"] not in excluded]
            eval_qids = {datum["question_id"] for datum in eval_rows}
            data = [datum for datum in data if datum["question_id"] not in eval_qids]

        rng = random.Random(self.seed)
        rng.shuffle(data)
        if self.limit > 0:
            data = data[: self.limit]

        train_rows = data
        if not eval_rows and self.eval_size > 0:
            eval_rows = data[: self.eval_size]
            train_rows = data[self.eval_size :]
        if not train_rows:
            raise ValueError(
                "No training rows remain after applying excluded_qids_jsonl/eval_index_jsonl/limit. "
                "Use a training index that does not fully overlap the held-out eval index."
            )
        LOGGER.info(
            "Synthetic FS dataset split: train_rows=%d eval_rows=%d excluded_qids=%d eval_index_jsonl=%s",
            len(train_rows),
            len(eval_rows),
            len(excluded),
            self.eval_index_jsonl or "(none; eval_size split)" if self.eval_size > 0 else self.eval_index_jsonl or "(none)",
        )

        train_builders = [
            SyntheticFilesystemEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                reward_mode=self.reward_mode,
                answerer_backend=self.answerer_backend,
                answerer_model=self.answerer_model,
                answerer_base_url=self.answerer_base_url,
                answerer_api_key_env=self.answerer_api_key_env,
                judge_backend=self.judge_backend,
                judge_model=self.judge_model,
                judge_base_url=self.judge_base_url,
                judge_api_key_env=self.judge_api_key_env,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                step_penalty=self.step_penalty,
                termination_penalty=self.termination_penalty,
                raw_docs_penalty=self.raw_docs_penalty,
                empty_synthetic_penalty=self.empty_synthetic_penalty,
                synthetic_success_bonus=self.synthetic_success_bonus,
                synthetic_usage_bonus=self.synthetic_usage_bonus,
                raw_usage_ratio_penalty=self.raw_usage_ratio_penalty,
                filesystem_maturity_scale=self.filesystem_maturity_scale,
                filesystem_coverage_weight=self.filesystem_coverage_weight,
                filesystem_expansion_weight=self.filesystem_expansion_weight,
                filesystem_organization_weight=self.filesystem_organization_weight,
                filesystem_stop_weight=self.filesystem_stop_weight,
                mature_stop_bonus=self.mature_stop_bonus,
                mature_stop_min_score=self.mature_stop_min_score,
                terminal_reward_clip_min=self.terminal_reward_clip_min,
                terminal_reward_clip_max=self.terminal_reward_clip_max,
                answerer_max_turns=self.answerer_max_turns,
                answerer_workspace_mode=self.answerer_workspace_mode,
                answerer_final_answer_max_tokens=self.answerer_final_answer_max_tokens,
                answerer_retrieval_cost_scale=self.answerer_retrieval_cost_scale,
                answerer_retrieval_cost_token_unit=self.answerer_retrieval_cost_token_unit,
                answerer_retrieval_cost_correct_only=self.answerer_retrieval_cost_correct_only,
                answerer_synthetic_read_cost_scale=self.answerer_synthetic_read_cost_scale,
                answerer_synthetic_read_cost_unit=self.answerer_synthetic_read_cost_unit,
                terminal_answerer_repeats=self.terminal_answerer_repeats,
                answerability_delta_reward_scale=self.answerability_delta_reward_scale,
                answerability_delta_min_abs=self.answerability_delta_min_abs,
                answerability_delta_allow_negative=self.answerability_delta_allow_negative,
                answerability_probe_max_per_episode=self.answerability_probe_max_per_episode,
                answerability_probe_interval_turns=self.answerability_probe_interval_turns,
                answerability_probe_min_maturity=self.answerability_probe_min_maturity,
                answerability_probe_repeats=self.answerability_probe_repeats,
                judge_max_output_tokens=self.judge_max_output_tokens,
                log_step_details=self.log_step_details,
                log_compaction_summaries=self.log_compaction_summaries,
                retain_reward_tool_messages=self.retain_reward_tool_messages,
                trim_terminal_history_for_memory=self.trim_terminal_history_for_memory,
                return_empty_terminal_observation=self.return_empty_terminal_observation,
                clear_state_on_terminal_for_memory=self.clear_state_on_terminal_for_memory,
                builder_compaction_enabled=self.builder_compaction_enabled,
                builder_compaction_backend=self.builder_compaction_backend,
                builder_compaction_model=self.builder_compaction_model,
                builder_compaction_base_url=self.builder_compaction_base_url,
                builder_compaction_api_key_env=self.builder_compaction_api_key_env,
                builder_compaction_trigger_tokens=self.builder_compaction_trigger_tokens,
                builder_compaction_keep_recent_turns=self.builder_compaction_keep_recent_turns,
                builder_compaction_max_output_tokens=self.builder_compaction_max_output_tokens,
                builder_compaction_input_max_chars=self.builder_compaction_input_max_chars,
                builder_executor_enabled=self.builder_executor_enabled,
                builder_batch_tools_enabled=self.builder_batch_tools_enabled,
                builder_executor_backend=self.builder_executor_backend,
                builder_executor_model=self.builder_executor_model,
                builder_executor_base_url=self.builder_executor_base_url,
                builder_executor_api_key_env=self.builder_executor_api_key_env,
                builder_executor_max_source_chars=self.builder_executor_max_source_chars,
                builder_executor_max_output_tokens=self.builder_executor_max_output_tokens,
                step_construction_action_bonus=self.step_construction_action_bonus,
                step_filesystem_maturity_delta_scale=self.step_filesystem_maturity_delta_scale,
                step_non_construction_turn_penalty=self.step_non_construction_turn_penalty,
                step_non_construction_streak_penalty=self.step_non_construction_streak_penalty,
                step_non_construction_streak_free=self.step_non_construction_streak_free,
                step_tool_error_penalty=self.step_tool_error_penalty,
            )
            for datum in train_rows
        ]
        train_dataset = SyntheticFilesystemRLDataset(train_builders, batch_size=self.batch_size)

        if not eval_rows:
            return train_dataset, None

        eval_builders = [
            SyntheticFilesystemEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                group_size=self.group_size,
                reward_mode=self.reward_mode,
                answerer_backend=self.answerer_backend,
                answerer_model=self.answerer_model,
                answerer_base_url=self.answerer_base_url,
                answerer_api_key_env=self.answerer_api_key_env,
                judge_backend=self.judge_backend,
                judge_model=self.judge_model,
                judge_base_url=self.judge_base_url,
                judge_api_key_env=self.judge_api_key_env,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                step_penalty=self.step_penalty,
                termination_penalty=self.termination_penalty,
                raw_docs_penalty=self.raw_docs_penalty,
                empty_synthetic_penalty=self.empty_synthetic_penalty,
                synthetic_success_bonus=self.synthetic_success_bonus,
                synthetic_usage_bonus=self.synthetic_usage_bonus,
                raw_usage_ratio_penalty=self.raw_usage_ratio_penalty,
                filesystem_maturity_scale=self.filesystem_maturity_scale,
                filesystem_coverage_weight=self.filesystem_coverage_weight,
                filesystem_expansion_weight=self.filesystem_expansion_weight,
                filesystem_organization_weight=self.filesystem_organization_weight,
                filesystem_stop_weight=self.filesystem_stop_weight,
                mature_stop_bonus=self.mature_stop_bonus,
                mature_stop_min_score=self.mature_stop_min_score,
                terminal_reward_clip_min=self.terminal_reward_clip_min,
                terminal_reward_clip_max=self.terminal_reward_clip_max,
                answerer_max_turns=self.answerer_max_turns,
                answerer_workspace_mode=self.answerer_workspace_mode,
                answerer_final_answer_max_tokens=self.answerer_final_answer_max_tokens,
                answerer_retrieval_cost_scale=self.answerer_retrieval_cost_scale,
                answerer_retrieval_cost_token_unit=self.answerer_retrieval_cost_token_unit,
                answerer_retrieval_cost_correct_only=self.answerer_retrieval_cost_correct_only,
                answerer_synthetic_read_cost_scale=self.answerer_synthetic_read_cost_scale,
                answerer_synthetic_read_cost_unit=self.answerer_synthetic_read_cost_unit,
                terminal_answerer_repeats=self.terminal_answerer_repeats,
                answerability_delta_reward_scale=self.answerability_delta_reward_scale,
                answerability_delta_min_abs=self.answerability_delta_min_abs,
                answerability_delta_allow_negative=self.answerability_delta_allow_negative,
                answerability_probe_max_per_episode=self.answerability_probe_max_per_episode,
                answerability_probe_interval_turns=self.answerability_probe_interval_turns,
                answerability_probe_min_maturity=self.answerability_probe_min_maturity,
                answerability_probe_repeats=self.answerability_probe_repeats,
                judge_max_output_tokens=self.judge_max_output_tokens,
                log_step_details=self.log_step_details,
                log_compaction_summaries=self.log_compaction_summaries,
                retain_reward_tool_messages=self.retain_reward_tool_messages,
                trim_terminal_history_for_memory=self.trim_terminal_history_for_memory,
                return_empty_terminal_observation=self.return_empty_terminal_observation,
                clear_state_on_terminal_for_memory=self.clear_state_on_terminal_for_memory,
                builder_compaction_enabled=self.builder_compaction_enabled,
                builder_compaction_backend=self.builder_compaction_backend,
                builder_compaction_model=self.builder_compaction_model,
                builder_compaction_base_url=self.builder_compaction_base_url,
                builder_compaction_api_key_env=self.builder_compaction_api_key_env,
                builder_compaction_trigger_tokens=self.builder_compaction_trigger_tokens,
                builder_compaction_keep_recent_turns=self.builder_compaction_keep_recent_turns,
                builder_compaction_max_output_tokens=self.builder_compaction_max_output_tokens,
                builder_compaction_input_max_chars=self.builder_compaction_input_max_chars,
                builder_executor_enabled=self.builder_executor_enabled,
                builder_batch_tools_enabled=self.builder_batch_tools_enabled,
                builder_executor_backend=self.builder_executor_backend,
                builder_executor_model=self.builder_executor_model,
                builder_executor_base_url=self.builder_executor_base_url,
                builder_executor_api_key_env=self.builder_executor_api_key_env,
                builder_executor_max_source_chars=self.builder_executor_max_source_chars,
                builder_executor_max_output_tokens=self.builder_executor_max_output_tokens,
                step_construction_action_bonus=self.step_construction_action_bonus,
                step_filesystem_maturity_delta_scale=self.step_filesystem_maturity_delta_scale,
                step_non_construction_turn_penalty=self.step_non_construction_turn_penalty,
                step_non_construction_streak_penalty=self.step_non_construction_streak_penalty,
                step_non_construction_streak_free=self.step_non_construction_streak_free,
                step_tool_error_penalty=self.step_tool_error_penalty,
            )
            for datum in eval_rows
        ]
        eval_dataset = SyntheticFilesystemRLDataset(eval_builders, batch_size=self.batch_size)
        return train_dataset, eval_dataset


def load_excluded_qids(path: Path) -> set[str]:
    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", "")).strip()
            if qid:
                qids.add(qid)
    return qids


def load_index(path: Path) -> list[SyntheticFilesystemDatum]:
    rows: list[SyntheticFilesystemDatum] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ["question_id", "agent_query_dir", "privileged_query_dir", "num_docs"]
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_num} missing required keys: {missing}")
            rows.append(hydrate_privileged_record(row))
    if not rows:
        raise ValueError(f"No rows found in index: {path}")
    return rows


def hydrate_privileged_record(row: dict[str, Any]) -> SyntheticFilesystemDatum:
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
