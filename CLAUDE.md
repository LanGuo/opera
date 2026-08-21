# BioAgent-Eval

Open-world evaluation framework for LLM agents on long-horizon biomedical discovery tasks.
Three personas: Task Proposer → Evaluator → Reviewer. See `README.md` for full architecture.

## Running Tests

```bash
uv run pytest -v                          # all tests
uv run pytest tests/test_review.py        # reviewer tests only
uv run pytest -m "not integration"        # skip MCP integration tests
```

## Key Conventions

- `hypothesis_ledger/workspace.py` — all run artifact I/O goes through `Workspace`
- `reviewer/log_replay.py` — offline metrics; call `analyze_run(run_dir)` for `RunMetrics`
- `reviewer/score.py` — LLM scoring; reads `task_spec.yaml` from run_dir to build rubric-driven prompts; two-pass scoring (evidence trail only → then agent rationale comparison); produces `CriterionScore` per rubric item; `score_run(run_dir, client)` for one backend, `score_run_all_backends(run_dir)` for all configured
- `reviewer/review.py` — HTML generation; writes `review_<timestamp>.html` (never overwrites); renders per-criterion score table in blinded section; flags contested criteria (≥2 backends differ by ≥2 points)
- Tests use `tmp_path` for isolation; no shared state between tests
- `uv run` for all Python commands

## Adding a New Eval Task

1. Run `uv run bioagent-propose` (wizard generates `task_spec.yaml` + `agent_task.md` from plain-text description)
   — or hand-author `tasks/<task_id>/task_spec.yaml` with `rubric:` block (see `tasks/sle_biomarker_discovery/task_spec.yaml`)
2. The `rubric:` list is the source of truth for scoring — same `guidance` text goes to LLM scorer and human reviewer HTML form
   - Agent tasks must keep `evidence[].description` factual (observed data, numbers, sources) and `update_history[].rationale` interpretive (why confidence changed) — the scorer uses this separation for its two-pass design
3. Launch: `./scripts/run_eval.sh <model> <max_turns> <task_id>`

## Common Commands

```bash
./scripts/run_eval.sh claude-sonnet-4-6 10          # smoke-test run (10 turns)
./scripts/run_eval.sh claude-opus-4-7 250            # full benchmark run
uv run python -m reviewer.log_replay <run_dir>                         # analyze run metrics
uv run python -m reviewer.review <run_dir>                             # generate review_<timestamp>.html (no LLM scores)
ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... uv run bioagent-review <run_dir>  # all configured backends score in parallel
```

## MCP Servers (eval runs)

- `hypothesis-ledger`: custom MCP server in `hypothesis_ledger/`
- `tooluniverse`: external, 600+ biomedical tools (see `.mcp.json`)
