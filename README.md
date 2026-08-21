# OPERA

**Open-World Performance Evaluation for Biomedical Research Agents**

OPERA implements a CRUX-style evaluation paradigm: rather than testing against fixed answers, it captures the agent's full reasoning trace, hypothesis evolution, and tool usage, then scores the quality of its scientific conclusions. The pilot domain is SLE/lupus biomarker discovery using the Perez et al. 2022 scRNA-seq PBMC cohort.

---

## Architecture

Three personas with clear responsibility boundaries and explicit artifact handoffs:

```
┌──────────────────────────┐        ┌──────────────────────────┐        ┌──────────────────────────┐
│     TASK PROPOSER        │        │       EVALUATOR          │        │    REVIEWER / SCORER     │
│  task_proposer/  tasks/  │        │  hypothesis_ledger/      │        │      reviewer/           │
│                          │        │  scripts/run_eval.sh     │        │                          │
│  Define what to          │──────► │  Dockerfile              │──────► │  log_replay.py  metrics  │
│  investigate.            │        │                          │        │  score.py       LLM judge│
│  Output: task_spec.yaml  │        │  Claude Code (headless)  │        │  review.py   HTML+rubric │
│  + agent_task.md             │        │  + MCP servers           │        │                          │
└──────────────────────────┘        └──────────────────────────┘        └──────────────────────────┘
   task_spec.yaml                      runs/<run_id>/                      review_<timestamp>.html
   agent_task.md                           ledger.json                         *_review.json
                                       step_log.jsonl
                                       final_report.json
                                       transcript.jsonl
                                       data/  results/
```

---

## Three Personas

### Task Proposer (`task_proposer/`, `tasks/`)

Defines what the agent should investigate. Produces two artifacts per task:
- `task_spec.yaml` — machine-readable config: goal, termination criteria, model, and a `rubric` list that drives both LLM scoring prompts and the human-reviewer HTML form
- `agent_task.md` — agent-facing task prompt (goal-oriented; does not prescribe methodology); its "Success Criteria" section is consistent with the `rubric` items

The agent never reads `task_spec.yaml`. `run_eval.sh` reads it for `model` and `max_turns`, then copies it into the run directory so the reviewer can read the rubric without needing the source repo.

### Evaluator (`hypothesis_ledger/`, `scripts/`, `Dockerfile`)

Runs the agent. Claude Code (headless) connects to two MCP servers:
- `hypothesis-ledger`: custom MCP server for hypothesis state tracking (6 tools)
- `tooluniverse`: 600+ pre-built biomedical tools

The agent reads `agent_task.md`, runs the task, and writes all artifacts to `runs/<run_id>/`.

### Reviewer (`reviewer/`)

Scores and presents the completed run:
- `log_replay.py` — computes automated metrics from run artifacts: belief revisions, refutation rate, tool diversity, modality coverage, **martingale score** (Pearson r between b_t and Δb_t — near-zero means evidence-driven updates, positive means belief entrenchment), and **directional flip count** (0.5-threshold confidence crossings)
- `falsification.py` — post-run observer LLM module: a separate Claude Haiku instance reads only each terminal hypothesis's evidence (never the agent's stated confidence) and independently assesses it via e-value accumulation from tool output statistics (p-values, effect sizes), producing `observer_confidence`, `e_value_confidence`, `confidence_delta`, and falsification criteria. Also cross-checks the LLM's self-reported `observer_confidence` against the deterministic e-value-only estimate, flagging (`confidence_flagged`, `deterministic_delta`) when they diverge by more than 0.2 — a sanity check on the LLM's arithmetic, not a correctness oracle
- `score.py` — LLM-as-judge scoring driven by the task's `rubric` list (reads `task_spec.yaml` from run dir); runs all configured backends (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OLLAMA_MODEL`) in one pass; uses a **two-pass approach** — pass 1 scores each rubric criterion using only observed evidence (evidence trail from `update_history`, open questions, refuted hypothesis details, proposed validation); pass 2 shows the agent's own rationale and asks the judge to note agreement/divergence without changing scores; produces structured `CriterionScore` objects (`criterion_id`, `rationale`, `score`, `agent_note`) per rubric item
- `review.py` — generates a timestamped `review_<timestamp>.html` for a domain scientist; each run creates a new file so prior reviews are never overwritten; shows a task context panel (goal, termination criteria, rubric dimensions) at the top; LLM scores are shown as a **per-criterion table** in the blinded section (revealed only after the human submits their rubric); when 2+ backends score the same criterion and their numeric scores differ by ≥2, the cell is flagged as **contested**; falls back to raw text if structured scores are unavailable

---

## Artifact Handoffs

### Task Proposer → Evaluator

| Artifact | Path | Consumer |
|---|---|---|
| `agent_task.md` | `tasks/<task_id>/agent_task.md` | Claude Code agent (task prompt) |
| `task_spec.yaml` | `tasks/<task_id>/task_spec.yaml` | `run_eval.sh` (model, max_turns); copied to run dir for reviewer |

### Evaluator → Reviewer

| Artifact | Path | Consumer |
|---|---|---|
| `task_spec.yaml` | `runs/<run_id>/task_spec.yaml` | `score.py` (rubric-driven prompts), `review.py` (task context panel + rubric form) |
| `ledger.json` | `runs/<run_id>/ledger.json` | `log_replay.py`, `score.py`, `review.py` |
| `step_log.jsonl` | `runs/<run_id>/step_log.jsonl` | `review.py` |
| `run_log.jsonl` | `runs/<run_id>/run_log.jsonl` | `log_replay.py` |
| `final_report.json` | `runs/<run_id>/final_report.json` | `score.py`, `review.py` |
| `transcript.jsonl` | `runs/<run_id>/transcript.jsonl` | `log_replay.py` (tool diversity) |
| `data/`, `results/` | `runs/<run_id>/data/`, `.../results/` | `review.py` (supporting data) |

### Reviewer → Archive

| Artifact | Path | Consumer |
|---|---|---|
| `review_<timestamp>.html` | `runs/<run_id>/review_<timestamp>.html` | Human reviewer (open in browser) |
| `*_review.json` | `runs/<run_id>/*_review.json` | Future: aggregate scoring |

---

## Project Structure

```
opera/
├── tasks/                          # Task library — one subdir per evaluation task
│   └── sle_biomarker_discovery/
│       ├── task_spec.yaml          # Task config (goal, termination criteria, model, rubric)
│       └── agent_task.md               # Agent-facing task prompt
├── task_proposer/                  # LLM-assisted task authoring wizard (bioagent-propose CLI)
├── hypothesis_ledger/              # Evaluator MCP server (6 hypothesis-tracking tools)
├── reviewer/                       # Reviewer tools
│   ├── log_replay.py               # Offline metrics (martingale score, flips, revisions, tool diversity)
│   ├── falsification.py            # Observer LLM + e-value module for terminal hypothesis assessment
│   ├── score.py                    # LLM-as-judge scoring
│   └── review.py                   # HTML review page generator (bioagent-review CLI)
├── scripts/
│   └── run_eval.sh                 # Headless eval launcher
├── tests/                          # Unit + integration tests
├── .mcp.json                       # MCP server registry for Claude Code
├── .gemini/settings.json           # MCP server registry for Gemini CLI
├── CLAUDE.md                       # Developer notes (this repo)
├── Dockerfile                      # Isolated eval container
└── pyproject.toml
```

---

## Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), and one of:
- [Claude Code](https://claude.ai/code) — for Claude models in the Evaluator stage
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) — for Gemini models in the Evaluator stage (`gemini-*` model names)

```bash
# 1a. Install Claude Code CLI (for Claude models)
npm install -g @anthropic-ai/claude-code
# 1b. Or install Gemini CLI (for Gemini models)
npm install -g @google/gemini-cli

# 2. Install Python dependencies
uv sync --extra dev

# 3. Verify
uv run pytest -v
```

**API keys:**

```bash
# Evaluator stage: pick one
export ANTHROPIC_API_KEY=...   # Claude.ai Pro/Max subscription — for claude-* models
export GOOGLE_API_KEY=...      # Google AI Studio (free tier) — for gemini-* models

# Reviewer LLM scoring: runs ALL keys set below in parallel, shows scores side-by-side.
# Set any combination — each configured backend produces its own scores.
export OLLAMA_MODEL=gemma4:e4b   # local, no API key needed

# Optional: raise rate limits on biomedical data tools
export NCBI_API_KEY=...        # ncbi.nlm.nih.gov/account (free)
export NVIDIA_API_KEY=...      # build.nvidia.com (free)
export BIOGRID_API_KEY=...     # thebiogrid.org (free)
export DISGENET_API_KEY=...    # disgenet.com (free tier)
```

**No API key?** The Task Proposer and Reviewer can run fully locally with [Ollama](https://ollama.com):
```bash
ollama pull gemma4:e4b          # or any model you prefer
export OLLAMA_MODEL=gemma4:e4b
```

---

## Running an Eval

### Full Pipeline (local)

**Step 1 — Task Proposer** (opens wizard in browser at `http://localhost:7464`)

```bash
ANTHROPIC_API_KEY=... uv run bioagent-propose
# or with a local Ollama model (no API key needed):
OLLAMA_MODEL=gemma4:e4b uv run bioagent-propose
```

The wizard is a 3-step linear flow:
1. **Describe** — type your task in plain language; the LLM extracts structured fields
2. **Review & Edit** — adjust goal, termination criteria, evaluator model, and rubric items; click "Generate & Check" to produce the `agent_task.md` preview and run a quality check
3. **Save** — writes `tasks/<task_id>/task_spec.yaml` and `tasks/<task_id>/agent_task.md`

**Step 2 — Evaluator** (requires Claude.ai subscription)

```bash
mkdir -p ~/bioeval/runs

# Smoke test (10 turns, sonnet, default task)
./scripts/run_eval.sh claude-sonnet-4-6 10

# Full benchmark run (250 turns, opus, default task)
./scripts/run_eval.sh claude-opus-4-7 250

# Gemini model (uses Gemini CLI harness automatically)
./scripts/run_eval.sh gemini-2.5-pro 250

# Different task
./scripts/run_eval.sh claude-opus-4-7 250 my_task_id
```

**Step 3 — Reviewer** (run by you, not the domain expert)

```bash
# Automated metrics only (free, offline — martingale score, flip count, tool diversity)
uv run python -m reviewer.log_replay ~/bioeval/runs/<run_id>

# Generate self-contained review page — bakes in LLM scoring + falsification observer.
# Set any combination of keys; all configured backends run and scores appear side-by-side.
ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... uv run bioagent-review ~/bioeval/runs/<run_id>
# local only (no API key):
OLLAMA_MODEL=gemma4:e4b uv run bioagent-review ~/bioeval/runs/<run_id>
# all three at once:
ANTHROPIC_API_KEY=... GOOGLE_API_KEY=... OLLAMA_MODEL=gemma3:12b uv run bioagent-review ~/bioeval/runs/<run_id>
```

Each run writes a new `review_<timestamp>.html` — previous reviews are never overwritten.
All LLM inference runs here; the generated file is fully self-contained (no server, no
Python, no API key required to open it).

**Sharing with a domain expert (no setup required on their end):**
```bash
# Share the latest review file (or the whole run dir)
ls -t ~/bioeval/runs/<run_id>/review_*.html | head -1   # find latest
zip -r run_for_review.zip ~/bioeval/runs/<run_id>/
```
The expert opens the `.html` file in any browser, fills the rubric, clicks Submit — their
browser downloads `review.json`. They send that file back to you; drop it in the run directory.

### Docker (isolated environment)

```bash
docker build -t bioagent-eval .
docker run --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v ~/bioeval:/workspace \
  bioagent-eval bash scripts/run_eval.sh claude-opus-4-7 250
```

---

## Adding a New Task

1. Create `tasks/<task_id>/task_spec.yaml` (or use the wizard — `uv run bioagent-propose`):
   ```yaml
   task_id: my_new_task
   goal: >
     Describe the scientific question in goal-oriented terms.
     Do not prescribe methodology.
   termination_criteria:
     min_hypotheses_with_multimodal_evidence: 3
     max_tool_calls: 200
   evaluator:
     model: claude-opus-4-7
     max_turns: 250
   rubric:
     - id: novelty
       label: "Scientific Novelty"
       scope: per_hypothesis
       scale: "1-5"
       guidance: >
         Task-specific scoring guidance — shown verbatim to both the LLM scorer and
         the human reviewer. Make it concrete and tied to the task's success criteria.
     - id: overall_quality
       label: "Overall Scientific Quality"
       scope: overall
       scale: "1-5"
       guidance: >
         Overall rigor of the run — appropriate methods, evidence integration, depth.
   ```
   Each rubric item needs `id` (snake_case), `label`, `scope` (`per_hypothesis` or `overall`), `scale` (`1-5` or `yes/no/partial`), and `guidance`. The wizard generates these from your plain-text task description.

2. Write `tasks/<task_id>/agent_task.md` — include goal, required agent behaviors (ledger maintenance, logging, multi-modal evidence), and termination criteria. See `tasks/sle_biomarker_discovery/agent_task.md` as reference.

3. Launch: `./scripts/run_eval.sh <model> <max_turns> <task_id>`

## License

MIT — see [LICENSE](LICENSE).
