"""Run an agent on a task under a harness profile and record what actually ran.

  python -m evaluator.run claude-sonnet-4-6 --profile matched --max-turns 30
  python -m evaluator.run gemini-2.5-pro --profile matched --max-turns 30 --gemini-max-session-turns 30
  python -m evaluator.run claude-sonnet-4-6 --profile matched --dry-run

Load API keys first (e.g. `set -a; source .env; set +a`). See evaluator.harness for the profiles.
The run directory gets transcript.jsonl, run_manifest.json, and the ledger written by the
hypothesis-ledger server; reviewer.log_replay runs afterwards unless --skip-log-replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluator.harness import PROFILES, HarnessPlan, finish_run, plan_run, prepare_run, summarize_transcript
from reviewer.variability import code_version

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_GRACE_S = 15  # time for the CLI to exit on its own after its result record
POLL_S = 0.5
SHARED_TMP = Path("/tmp")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_argv(plan: HarnessPlan, task_text: str) -> list[str]:
    """The command as run, with the task text (embedded in pilot commands) replaced by a placeholder."""
    stripped = task_text.rstrip("\n")
    placeholder = f"<agent_task.md: {len(task_text)} chars>"
    return [a.replace(stripped, placeholder) if stripped and stripped in a else a for a in plan.argv]


def cli_version(cli: str) -> str | None:
    try:
        r = subprocess.run([cli, "--version"], capture_output=True, text=True, timeout=60)
        return (r.stdout or r.stderr).strip().splitlines()[0] if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def _is_result(line: str) -> bool:
    if '"result"' not in line:
        return False
    try:
        return json.loads(line).get("type") == "result"
    except (json.JSONDecodeError, AttributeError):
        return False


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _stop_group(proc: subprocess.Popen) -> None:
    """Stop the CLI and everything it started (MCP servers, shell commands)."""
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    _signal_group(proc, signal.SIGKILL)


def _run_cli(plan: HarnessPlan, env: dict, out, err, timeout_s: float | None) -> tuple[int, str | None]:
    """Run the CLI with its output going to `out`. Returns (exit code, why the launcher stopped it).

    The launcher stops it after `timeout_s`, or, when plan.stop_on_result, if it is still running
    RESULT_GRACE_S after its first result record (the run is over; anything later would extend it).
    Output goes straight to the file, not through a pipe: Node writes to pipes asynchronously on
    macOS, and Gemini CLI exits before its final result record is flushed.
    """
    proc = subprocess.Popen(plan.argv, cwd=plan.cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=out, stderr=err, start_new_session=True)
    stopped: list[str] = []
    started = time.monotonic()
    result_seen_at = None
    with open(out.name, encoding="utf-8", errors="replace") as watch:
        pending = ""
        while proc.poll() is None:
            time.sleep(POLL_S)
            pending += watch.read()
            *lines, pending = pending.split("\n")
            if plan.stop_on_result and result_seen_at is None and any(_is_result(l) for l in lines):
                result_seen_at = time.monotonic()
            if timeout_s and time.monotonic() - started > timeout_s:
                stopped.append("timeout")
            elif result_seen_at is not None and time.monotonic() - result_seen_at > RESULT_GRACE_S:
                stopped.append("after_first_result")
            if stopped:
                _stop_group(proc)
                break
    _signal_group(proc, signal.SIGKILL)  # leftover MCP servers or shell commands
    return proc.wait(), (stopped[0] if stopped else None)


def _tmp_entries(tmp: Path) -> set[str]:
    try:
        return set(os.listdir(tmp))
    except OSError:
        return set()


def sweep_tmp(tmp: Path, before: set[str], transcript: Path, dest: Path) -> dict:
    """Move entries that appeared in `tmp` during the run, and that the transcript names, to `dest`.

    Agents write scratch files to literal /tmp paths, which a per-run TMPDIR doesn't catch, and a
    later run could find them. Entries the transcript doesn't name may belong to other programs, so
    they are only listed.
    """
    text = transcript.read_text(encoding="utf-8", errors="replace") if transcript.exists() else ""
    moved, unattributed = [], []
    for name in sorted(_tmp_entries(tmp) - before):
        if re.search(r"/tmp/" + re.escape(name) + r"(?![\w.-])", text):
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp / name), str(dest / name))
            moved.append(name)
        else:
            unattributed.append(name)
    return {"moved_to": str(dest) if moved else None, "moved": moved, "unattributed": unattributed}


def execute(plan: HarnessPlan, *, task_dir, cli_version: str | None, environ: dict | None = None,
            timeout_s: float | None = None) -> dict:
    if (plan.run_dir / "transcript.jsonl").exists():
        raise FileExistsError(f"run directory already has a transcript: {plan.run_dir}")
    task_text = (Path(task_dir) / "agent_task.md").read_text(encoding="utf-8")
    prepare_run(plan)
    manifest = {
        "profile": plan.profile,
        "harness": plan.harness,
        "cli_version": cli_version,
        "model_requested": plan.model,
        **plan.details,
        "user_message": "Begin the task." if plan.profile == "matched" or plan.harness == "claude" else None,
        "task_dir": str(task_dir),
        "task_sha256": hashlib.sha256(task_text.encode("utf-8")).hexdigest(),
        "cwd": str(plan.cwd),
        "env_set": plan.env,
        "argv": manifest_argv(plan, task_text),
        "timeout_minutes": timeout_s / 60 if timeout_s else None,
        "stop_on_result": plan.stop_on_result,
        **code_version(),
        "started_at": _now(),
    }
    path = plan.run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = {k: v for k, v in (os.environ if environ is None else environ).items() if k not in plan.unset_env}
    env.update(plan.env)
    tmp_before = _tmp_entries(SHARED_TMP) if plan.sweep_tmp else set()
    with open(plan.run_dir / "transcript.jsonl", "w", encoding="utf-8") as out, \
         open(plan.run_dir / "harness_stderr.log", "w", encoding="utf-8") as err:
        exit_code, stopped_by = _run_cli(plan, env, out, err, timeout_s)
    finish_run(plan)
    if plan.sweep_tmp:
        manifest["tmp_sweep"] = sweep_tmp(SHARED_TMP, tmp_before, plan.run_dir / "transcript.jsonl",
                                          plan.run_dir / "harness" / "tmp_leftovers")
    if plan.venv is not None:
        cfg = plan.venv / "pyvenv.cfg"
        manifest["python_version"] = next(
            (l.split("=", 1)[1].strip() for l in cfg.read_text().splitlines()
             if l.split("=", 1)[0].strip() in ("version", "version_info")), None) if cfg.exists() else None

    manifest.update(
        finished_at=_now(),
        exit_code=exit_code,
        stopped_by_launcher=stopped_by,
        outcome=summarize_transcript(plan.run_dir / "transcript.jsonl", plan.harness, plan.model),
    )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--gemini-max-session-turns", type=int, default=None,
                        help="matched Gemini runs only; default: --max-turns")
    parser.add_argument("--task", default="sle_biomarker_discovery")
    parser.add_argument("--runs-root", default=os.environ.get("WORKSPACE_ROOT", str(Path.home() / "bioeval")) + "/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    parser.add_argument("--timeout-minutes", type=float, default=120,
                        help="stop the run after this long; 0 for no limit (default: 120)")
    parser.add_argument("--skip-log-replay", action="store_true")
    args = parser.parse_args(argv)

    task_dir = REPO_ROOT / "tasks" / args.task
    if not (task_dir / "agent_task.md").is_file():
        sys.exit(f"no agent_task.md in {task_dir}")
    run_id = args.run_id or f"{args.model}_{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plan = plan_run(
        profile=args.profile, model=args.model, max_turns=args.max_turns, task_dir=task_dir,
        run_dir=Path(args.runs_root) / run_id, repo_root=REPO_ROOT, environ=dict(os.environ),
        gemini_max_session_turns=args.gemini_max_session_turns,
    )
    if args.dry_run:
        task_text = (task_dir / "agent_task.md").read_text(encoding="utf-8")
        print(json.dumps({
            "run_dir": str(plan.run_dir), "cwd": str(plan.cwd), "env_set": plan.env,
            "argv": manifest_argv(plan, task_text), "files": sorted(str(p) for p in plan.files),
            "copy_before_run": {str(k): str(v) for k, v in plan.copy_before_run.items()},
            "copy_after_run": {str(k): str(v) for k, v in plan.copy_after_run.items()},
            "stop_on_result": plan.stop_on_result, "timeout_minutes": args.timeout_minutes or None,
            **plan.details,
        }, indent=2))
        return

    m = execute(plan, task_dir=task_dir, cli_version=cli_version(plan.argv[0]),
                timeout_s=args.timeout_minutes * 60 or None)
    o = m["outcome"]
    if m["stopped_by_launcher"]:
        print(f"stopped by the launcher: {m['stopped_by_launcher']}", file=sys.stderr)
    print(f"run dir: {plan.run_dir}\nexit code: {m['exit_code']}\n"
          f"segments: {o['segments']}  turns: {o['turns_total']}  tool calls: {o['tool_calls']}  "
          f"cap hit: {o['cap_hit']}\nmodels used: {o['models_used']}", file=sys.stderr)
    if o["primary_model_matches_request"] is False:
        print(f"WARNING: requested {args.model} but most tokens went to {o['primary_model']}", file=sys.stderr)
    if not args.skip_log_replay:
        subprocess.run([sys.executable, "-m", "reviewer.log_replay", str(plan.run_dir)])


if __name__ == "__main__":
    main()
