"""Plan and record agent runs under a named harness profile.

Two profiles:

- "pilot" reproduces scripts/run_eval.sh exactly, the setup behind the paper's pilot runs. The
  agent runs from the repo root and inherits the launching user's CLI configuration (plugins,
  hooks, connectors, extra MCP servers). Claude's task replaces its built-in system prompt and
  runs under --max-turns; Gemini gets the task as a user message with no turn cap.
- "matched" runs each CLI as a fresh install would: an empty config home per CLI, a clean
  per-run working directory, only OPERA's two MCP servers, the task appended to each CLI's own
  system prompt, the same "Begin the task." message, and a turn cap for both. Subagents are off
  in both CLIs, and so are Claude Code's background tasks, which can wake the agent after the
  cap and keep `claude -p` running. Each run gets its own Python environment and temp directory,
  so packages and files one agent leaves behind don't reach the next. The task spec, which holds
  the rubric, is copied into the run directory only after the agent finishes.

Every run gets a manifest recording what actually ran, including the models the provider
reported, which can differ from the model requested.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from reviewer.log_replay import gemini_response_count

PROFILES = ("pilot", "matched")
USER_PROMPT = "Begin the task."
# In scripts/run_eval.sh the "\n" sits inside bash double quotes, so it stays a literal
# backslash-n; the pilot Gemini prompt contained those characters verbatim.
PILOT_GEMINI_PROMPT_TEMPLATE = "Instruction: {task}\\n\\nBegin the task."
TOOLUNIVERSE_KEYS = ("NCBI_API_KEY", "BIOGRID_API_KEY", "DISGENET_API_KEY", "NVIDIA_API_KEY")
# Claude Code's init record lists the subagent tool as "Task"; the model calls it as "Agent".
CLAUDE_SUBAGENT_TOOLS = ("Agent", "Task")
GEMINI_SUBAGENT_TOOL = "invoke_agent"
# Python for each run's own environment. cellxgene-census's tiledbsoma doesn't build on 3.14.
RUN_PYTHON = "3.12"


@dataclass
class HarnessPlan:
    profile: str
    harness: str
    model: str
    run_dir: Path
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    unset_env: list[str] = field(default_factory=list)
    files: dict[Path, str] = field(default_factory=dict)
    dirs: list[Path] = field(default_factory=list)
    copy_before_run: dict[Path, Path] = field(default_factory=dict)
    copy_after_run: dict[Path, Path] = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    # Stop the CLI once it writes its first result record, so nothing can extend the run past it.
    stop_on_result: bool = False
    venv: Path | None = None   # created with RUN_PYTHON before the run; first on the agent's PATH
    sweep_tmp: bool = False    # move files the agent left in the shared /tmp into the run dir


def harness_for(model: str) -> str:
    return "gemini" if model.startswith("gemini") else "claude"


def _mcp_servers(repo_root: Path, run_dir: Path, environ: dict, trust: bool) -> dict:
    ledger = {
        "command": "uv",
        "args": ["run", "--project", str(repo_root), "python", "-m", "hypothesis_ledger.server"],
        "env": {"ALLOW_HUMAN_INPUT": "false", "EVAL_RUN_DIR": str(run_dir)},
    }
    # Reference keys by name so no secret is written into run artifacts; both CLIs expand ${VAR}.
    tool_env = {"PYTHONIOENCODING": "utf-8"}
    tool_env.update({k: "${%s}" % k for k in TOOLUNIVERSE_KEYS if environ.get(k)})
    tools = {"command": "uvx", "args": ["tooluniverse"], "env": tool_env}
    if trust:
        ledger["trust"] = tools["trust"] = True
    return {"hypothesis-ledger": ledger, "tooluniverse": tools}


def plan_run(
    *, profile: str, model: str, max_turns: int, task_dir, run_dir, repo_root, environ: dict,
    gemini_max_session_turns: int | None = None,
) -> HarnessPlan:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
    task_dir, run_dir, repo_root = Path(task_dir), Path(run_dir), Path(repo_root)
    task_text = (task_dir / "agent_task.md").read_text(encoding="utf-8")
    spec = task_dir / "task_spec.yaml"
    spec_copy = {spec: run_dir / "task_spec.yaml"} if spec.exists() else {}
    harness = harness_for(model)
    base = dict(profile=profile, harness=harness, model=model, run_dir=run_dir)

    if profile == "pilot":
        task_bash = task_text.rstrip("\n")  # $(cat file) strips trailing newlines
        if harness == "claude":
            argv = ["claude", "--dangerously-skip-permissions", "--model", model,
                    "--max-turns", str(max_turns), "--output-format", "stream-json", "--verbose",
                    "--system-prompt", task_bash, "-p", USER_PROMPT]
            delivery = "replaces the CLI's built-in system prompt (--system-prompt)"
            cap = f"--max-turns {max_turns}"
        else:
            argv = ["gemini", "--approval-mode", "yolo", "--skip-trust", "--model", model,
                    "--output-format", "stream-json",
                    "-p", PILOT_GEMINI_PROMPT_TEMPLATE.format(task=task_bash)]
            delivery = "user message, prefixed 'Instruction: '"
            cap = "none"
        return HarnessPlan(
            **base, argv=argv, cwd=repo_root, env={"EVAL_RUN_DIR": str(run_dir)},
            copy_before_run=spec_copy,
            details={"task_delivery": delivery, "turn_cap": cap,
                     "isolation": "none: runs from the repo root with the user's CLI config"},
        )

    ws, hdir = run_dir / "workspace", run_dir / "harness"
    venv, tmp = hdir / "venv", hdir / "tmp"
    path = environ.get("PATH", "")
    # Agents install packages and write scratch files. A per-run environment, with the user's
    # site-packages hidden, and a per-run TMPDIR keep one run's leftovers from reaching the next.
    run_env = {"PATH": f"{venv / 'bin'}:{path}" if path else str(venv / "bin"),
               "VIRTUAL_ENV": str(venv), "PYTHONNOUSERSITE": "1", "TMPDIR": str(tmp)}
    isolation = dict(venv=venv, sweep_tmp=True)
    env_details = {"python_env": f"fresh venv (Python {RUN_PYTHON}) first on PATH, user site hidden",
                   "tmpdir": "per-run TMPDIR; new /tmp entries the agent named are moved into the run dir"}
    if harness == "claude":
        home, task_file, mcp_file = hdir / "claude-home", hdir / "agent_task.md", hdir / "mcp.json"
        servers = _mcp_servers(repo_root, run_dir, environ, trust=False)
        argv = ["claude", "-p", USER_PROMPT, "--model", model, "--max-turns", str(max_turns),
                "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions",
                "--strict-mcp-config", "--mcp-config", str(mcp_file),
                "--append-system-prompt-file", str(task_file),
                "--disallowedTools", *CLAUDE_SUBAGENT_TOOLS]
        return HarnessPlan(
            **base, argv=argv, cwd=ws,
            env={"CLAUDE_CONFIG_DIR": str(home), "EVAL_RUN_DIR": str(run_dir),
                 # A background command finishing after the cap starts a new invocation with a
                 # fresh turn allowance, and one that never exits keeps `claude -p` alive.
                 "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1", **run_env,
                 # Claude Code gives its shell TMPDIR=<this>/claude-<uid> (default base /tmp,
                 # shared by every Claude Code session on the machine).
                 "CLAUDE_CODE_TMPDIR": str(tmp)},
            files={task_file: task_text, mcp_file: json.dumps({"mcpServers": servers}, indent=2)},
            dirs=[ws, home, tmp], copy_after_run=spec_copy, stop_on_result=True, **isolation,
            details={"task_delivery": "appended to the CLI's built-in system prompt "
                                      "(--append-system-prompt-file)",
                     "turn_cap": f"--max-turns {max_turns}",
                     "subagents": "off (--disallowedTools " + " ".join(CLAUDE_SUBAGENT_TOOLS) + ")",
                     "background_tasks": "off (CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1)",
                     **env_details,
                     "isolation": "empty CLAUDE_CONFIG_DIR, clean workspace, --strict-mcp-config"},
        )

    cap = gemini_max_session_turns or max_turns
    home = hdir / "gemini-home"
    servers = _mcp_servers(repo_root, run_dir, environ, trust=True)
    # experimental.enableAgents only gates user-defined agents; the built-in ones stay reachable
    # through invoke_agent unless the tool itself is excluded.
    settings = {"model": {"maxSessionTurns": cap}, "tools": {"exclude": [GEMINI_SUBAGENT_TOOL]},
                "mcpServers": servers}
    argv = ["gemini", "-p", USER_PROMPT, "--model", model, "--output-format", "stream-json",
            "--approval-mode", "yolo", "--skip-trust"]
    return HarnessPlan(
        **base, argv=argv, cwd=ws,
        env={"GEMINI_CLI_HOME": str(home), "GEMINI_CLI_TRUST_WORKSPACE": "true",
             "EVAL_RUN_DIR": str(run_dir), **run_env},
        files={ws / "GEMINI.md": task_text,
               ws / ".gemini" / "settings.json": json.dumps(settings, indent=2)},
        dirs=[ws, home, tmp], copy_after_run=spec_copy, stop_on_result=True, **isolation,
        details={"task_delivery": "appended to the CLI's built-in system prompt (workspace GEMINI.md)",
                 "turn_cap": f"model.maxSessionTurns {cap}",
                 "subagents": f"off (tools.exclude {GEMINI_SUBAGENT_TOOL})",
                 "background_tasks": "on (background shell processes don't wake the agent; "
                                     "maxSessionTurns is a hard stop)",
                 **env_details,
                 "isolation": "empty GEMINI_CLI_HOME, clean workspace, only OPERA's MCP servers"},
    )


def prepare_run(plan: HarnessPlan) -> None:
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    for d in plan.dirs:
        d.mkdir(parents=True, exist_ok=True)
    for path, content in plan.files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for src, dst in plan.copy_before_run.items():
        shutil.copy2(src, dst)
    if plan.venv is not None:
        subprocess.run(["uv", "venv", "--quiet", "--seed", "--python", RUN_PYTHON, str(plan.venv)],
                       check=True, stdin=subprocess.DEVNULL)


def finish_run(plan: HarnessPlan) -> None:
    for src, dst in plan.copy_after_run.items():
        shutil.copy2(src, dst)


def _records(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summarize_transcript(path, harness: str, requested_model: str) -> dict:
    """What the transcript says actually ran: segments, turns, tool calls, and models used.

    A transcript with more than one segment means the session was resumed, so its total budget
    exceeds the per-invocation turn cap.
    """
    recs = _records(path)
    models: dict[str, int] = {}
    if harness == "claude":
        inits = [r for r in recs if r.get("type") == "system" and r.get("subtype") == "init"]
        results = [r for r in recs if r.get("type") == "result"]
        turns = [r.get("num_turns") for r in results if r.get("num_turns") is not None]
        tool_calls = sum(
            1 for r in recs if r.get("type") == "assistant"
            for c in (r.get("message") or {}).get("content", []) or []
            if isinstance(c, dict) and c.get("type") == "tool_use"
        )
        # modelUsage is cumulative within a session (a resumed session repeats the running total),
        # so take each session's last result and add sessions together.
        last_by_session = {}
        for r in results:
            last_by_session[r.get("session_id")] = r
        for r in last_by_session.values():
            for m, u in (r.get("modelUsage") or {}).items():
                models[m] = models.get(m, 0) + sum(
                    u.get(k, 0) or 0 for k in
                    ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens")
                )
        segments = len(results) or len(inits)
        turns_total = sum(turns) if turns else None
        cap_hit = any(r.get("subtype") == "error_max_turns" for r in results)
        # stream-json emits one assistant event per content block; blocks of one response share an
        # id. Subagent messages carry parent_tool_use_id; the turn cap limits only the main agent.
        ids = {True: set(), False: set()}
        for r in recs:
            mid = (r.get("message") or {}).get("id") if r.get("type") == "assistant" else None
            if mid:
                ids[bool(r.get("parent_tool_use_id"))].add(mid)
        responses, subagent_responses = len(ids[False]), len(ids[True])
        tool_errors = {"error": sum(
            1 for r in recs if r.get("type") == "user"
            for c in (r.get("message") or {}).get("content", []) or []
            if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error")
        )}
    else:
        results = [r for r in recs if r.get("type") == "result"]
        tool_calls = sum(1 for r in recs if r.get("type") == "tool_use")
        # Per-model total_tokens includes cached reads, which Gemini CLI 0.50 over-reports for
        # subagent models (more cached than input tokens), so count input plus output.
        for r in results:
            for m, s in ((r.get("stats") or {}).get("models") or {}).items():
                s = s or {}
                models[m] = models.get(m, 0) + (s.get("input_tokens") or 0) + (s.get("output_tokens") or 0)
        segments = len(results) or sum(1 for r in recs if r.get("type") == "init")
        turns, turns_total, subagent_responses = [], None, 0
        cap_hit = any(((r.get("error") or {}).get("type") == "FatalTurnLimitedError") for r in results)
        responses = gemini_response_count(recs)
        # Failed calls by cause. Gemini CLI adds a wait_for_previous parameter to every tool schema,
        # MCP tools included, and passes it on; MCP servers that validate arguments reject it.
        tool_errors = {}
        for r in recs:
            err = r.get("error") if r.get("type") == "tool_result" else None
            if err:
                cause = err.get("type") or "unknown"
                if "wait_for_previous" in (err.get("message") or ""):
                    cause = "wait_for_previous_rejected"
                tool_errors[cause] = tool_errors.get(cause, 0) + 1
    primary = max(models, key=models.get) if models else None
    return {
        "segments": segments,
        "turns_per_segment": turns,
        "turns_total": turns_total,
        "cap_hit": cap_hit,
        "model_responses": responses,
        "subagent_responses": subagent_responses,
        "tool_calls": tool_calls,
        "tool_errors": {k: v for k, v in tool_errors.items() if v},
        "models_used": models,
        "primary_model": primary,
        "primary_model_matches_request": None if primary is None else primary.startswith(requested_model),
    }
