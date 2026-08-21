#!/usr/bin/env bash
# scripts/run_eval.sh
# Usage: ./scripts/run_eval.sh [model] [max_turns] [task_id]
# Examples:
#   ./scripts/run_eval.sh                                        # full run, opus, sle_biomarker_discovery
#   ./scripts/run_eval.sh claude-sonnet-4-6 10                  # 10-turn smoke test
#   ./scripts/run_eval.sh claude-opus-4-7 250                   # full benchmark run
#   ./scripts/run_eval.sh claude-opus-4-7 250 my_task           # different task
set -euo pipefail

MODEL="${1:-claude-opus-4-7}"
MAX_TURNS="${2:-250}"
TASK_ID="${3:-sle_biomarker_discovery}"
TASK_DIR="tasks/${TASK_ID}"
RUN_ID="${MODEL}_$(date +%Y%m%d_%H%M%S)"
EVAL_RUN_DIR="${WORKSPACE_ROOT:-$HOME/bioeval}/runs/${RUN_ID}"

if [[ ! -f "${TASK_DIR}/agent_task.md" ]]; then
  echo "Error: ${TASK_DIR}/agent_task.md not found. Run bioagent-propose to create a task first." >&2
  exit 1
fi

mkdir -p "$EVAL_RUN_DIR"
[[ -f "${TASK_DIR}/task_spec.yaml" ]] && cp "${TASK_DIR}/task_spec.yaml" "$EVAL_RUN_DIR/task_spec.yaml"
echo "Task:    $TASK_ID"
echo "Model:   $MODEL"
echo "Turns:   $MAX_TURNS"
echo "Run dir: $EVAL_RUN_DIR"

# Detect harness
if [[ "$MODEL" == gemini-* ]]; then
  HARNESS="gemini"
  # EVAL_RUN_DIR and any tooluniverse API keys (NCBI_API_KEY etc.) flow through
  # the inline env assignment below; gemini passes its env to MCP subprocesses.
  HARNESS_ARGS=(
    --approval-mode yolo
    --skip-trust
    --model "$MODEL"
    --output-format stream-json
    -p "Instruction: $(cat "${TASK_DIR}/agent_task.md")\n\nBegin the task."
  )
else
  HARNESS="claude"
  HARNESS_ARGS=(
    --dangerously-skip-permissions
    --model "$MODEL"
    --max-turns "$MAX_TURNS"
    --output-format stream-json
    --verbose
    --system-prompt "$(cat "${TASK_DIR}/agent_task.md")"
    -p "Begin the task."
  )
fi

echo "Harness: $HARNESS"

# Safe here because the agent runs inside an isolated Docker container or user-managed bioeval dir.
EVAL_RUN_DIR="$EVAL_RUN_DIR" \
$HARNESS "${HARNESS_ARGS[@]}" \
  > "$EVAL_RUN_DIR/transcript.jsonl"

echo "Run complete. Analyzing..."
uv run python -m reviewer.log_replay "$EVAL_RUN_DIR"
