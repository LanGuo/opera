#!/usr/bin/env bash
# scripts/run_eval.sh — launch an agent run with the "pilot" harness profile.
# Usage: ./scripts/run_eval.sh [model] [max_turns] [task_id]
# Examples:
#   ./scripts/run_eval.sh claude-sonnet-4-6 10                  # 10-turn smoke test
#   ./scripts/run_eval.sh claude-opus-4-7 250                   # full benchmark run
#   ./scripts/run_eval.sh claude-opus-4-7 250 my_task           # different task
# For matched, isolated runs use: uv run python -m evaluator.run <model> --profile matched
#
# This delegates to evaluator.run, which records the run in ledger.jsonl (tool calls included)
# and appends the ledger section generated from the task spec to the agent's prompt.
set -euo pipefail

MODEL="${1:-claude-opus-4-7}"
MAX_TURNS="${2:-250}"
TASK_ID="${3:-sle_biomarker_discovery}"

exec uv run python -m evaluator.run "$MODEL" --profile pilot --max-turns "$MAX_TURNS" --task "$TASK_ID"
