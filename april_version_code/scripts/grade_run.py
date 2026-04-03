#!/usr/bin/env python3
"""Convenience wrapper for grading any result JSONL in the April package."""

from __future__ import annotations

import argparse

from _script_utils import PACKAGE_ROOT, run_script_main
from april_version_code.common import grading as grading_core
from april_version_code.common.llm_backends import load_local_env, require_backend_credentials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Grade a result JSONL produced by april_version_code.')
    parser.add_argument('--in-jsonl', required=True)
    parser.add_argument('--out-jsonl', required=True)
    parser.add_argument('--dataset-jsonl', default=str(PACKAGE_ROOT / 'data' / 'derived' / 'support_only' / 'support_only_q50_manydocs.jsonl'))
    parser.add_argument('--judge-backend', choices=['gemini', 'openrouter'], default='openrouter')
    parser.add_argument('--judge-model', default='qwen/qwen3.5-35b-a3b')
    parser.add_argument('--openrouter-base-url', default='https://openrouter.ai/api/v1')
    parser.add_argument('--openrouter-app-title', default='AprilVersionCode')
    parser.add_argument('--progress-every', type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env(PACKAGE_ROOT)
    require_backend_credentials([args.judge_backend])
    run_script_main(
        'grading',
        grading_core.main,
        [
            '--in_jsonl', args.in_jsonl,
            '--out_jsonl', args.out_jsonl,
            '--dataset_jsonl', args.dataset_jsonl,
            '--judge_backend', args.judge_backend,
            '--judge_model', args.judge_model,
            '--openrouter_base_url', args.openrouter_base_url,
            '--openrouter_app_title', args.openrouter_app_title,
            '--progress_every', str(args.progress_every),
        ],
    )


if __name__ == '__main__':
    main()
