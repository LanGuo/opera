import json

import pytest

from evaluator.harness import (
    PILOT_GEMINI_PROMPT_TEMPLATE, PROFILES, plan_run, prepare_run, summarize_transcript,
)


@pytest.fixture
def setup(tmp_path):
    repo = tmp_path / "repo"
    task_dir = repo / "tasks" / "t1"
    task_dir.mkdir(parents=True)
    (task_dir / "agent_task.md").write_text("TASK-BODY line 1\nline 2\n\n")
    (task_dir / "task_spec.yaml").write_text("rubric: [secret-rubric]\n")
    run_dir = tmp_path / "runs" / "run_x"
    return repo, task_dir, run_dir


def _plan(setup, profile, model, environ=None, **kw):
    repo, task_dir, run_dir = setup
    return plan_run(profile=profile, model=model, max_turns=30, task_dir=task_dir,
                    run_dir=run_dir, repo_root=repo, environ=environ or {}, **kw)


def test_profiles():
    assert PROFILES == ("pilot", "matched")


def test_unknown_profile_rejected(setup):
    with pytest.raises(ValueError, match="profile"):
        _plan(setup, "fancy", "claude-sonnet-4-6")


# --- pilot: must reproduce scripts/run_eval.sh exactly ---

def test_pilot_claude_matches_run_eval_sh(setup):
    repo, _, run_dir = setup
    p = _plan(setup, "pilot", "claude-sonnet-4-6")
    assert p.harness == "claude" and p.cwd == repo
    assert p.argv == [
        "claude", "--dangerously-skip-permissions", "--model", "claude-sonnet-4-6",
        "--max-turns", "30", "--output-format", "stream-json", "--verbose",
        # $(cat file) in bash strips trailing newlines
        "--system-prompt", "TASK-BODY line 1\nline 2", "-p", "Begin the task.",
    ]
    assert p.env == {"EVAL_RUN_DIR": str(run_dir)} and p.unset_env == []
    assert p.files == {}


def test_pilot_gemini_matches_run_eval_sh(setup):
    p = _plan(setup, "pilot", "gemini-2.5-pro")
    # In bash double quotes "\n" stays a literal backslash-n, so the pilot prompt contained it verbatim.
    assert PILOT_GEMINI_PROMPT_TEMPLATE == "Instruction: {task}\\n\\nBegin the task."
    assert p.argv == [
        "gemini", "--approval-mode", "yolo", "--skip-trust", "--model", "gemini-2.5-pro",
        "--output-format", "stream-json",
        "-p", "Instruction: TASK-BODY line 1\nline 2\\n\\nBegin the task.",
    ]
    assert "--max-turns" not in p.argv  # the pilot Gemini runs had no turn cap


# --- matched ---

def test_matched_claude_isolates_and_appends(setup):
    repo, _, run_dir = setup
    p = _plan(setup, "matched", "claude-sonnet-4-6")
    assert p.cwd == run_dir / "workspace"
    assert p.env["CLAUDE_CONFIG_DIR"] == str(run_dir / "harness" / "claude-home")
    assert p.env["EVAL_RUN_DIR"] == str(run_dir)
    assert p.argv[:3] == ["claude", "-p", "Begin the task."]
    for flag in ("--strict-mcp-config", "--dangerously-skip-permissions", "--verbose"):
        assert flag in p.argv
    assert p.argv[p.argv.index("--max-turns") + 1] == "30"
    task_file = p.argv[p.argv.index("--append-system-prompt-file") + 1]
    assert task_file == str(run_dir / "harness" / "agent_task.md")
    assert p.files[run_dir / "harness" / "agent_task.md"] == "TASK-BODY line 1\nline 2\n\n"
    mcp = json.loads(p.files[run_dir / "harness" / "mcp.json"])["mcpServers"]
    assert set(mcp) == {"hypothesis-ledger", "tooluniverse"}
    assert mcp["hypothesis-ledger"]["args"][:3] == ["run", "--project", str(repo)]
    assert "--system-prompt" not in p.argv


def test_matched_gemini_isolates_appends_and_caps(setup):
    _, _, run_dir = setup
    p = _plan(setup, "matched", "gemini-2.5-pro", gemini_max_session_turns=40)
    ws = run_dir / "workspace"
    assert p.cwd == ws
    assert p.env["GEMINI_CLI_HOME"] == str(run_dir / "harness" / "gemini-home")
    assert p.env["GEMINI_CLI_TRUST_WORKSPACE"] == "true"
    assert p.argv[:3] == ["gemini", "-p", "Begin the task."]
    assert p.files[ws / "GEMINI.md"] == "TASK-BODY line 1\nline 2\n\n"
    settings = json.loads(p.files[ws / ".gemini" / "settings.json"])
    assert settings["model"]["maxSessionTurns"] == 40
    assert set(settings["mcpServers"]) == {"hypothesis-ledger", "tooluniverse"}
    assert all(s["trust"] for s in settings["mcpServers"].values())


def test_matched_gemini_turn_cap_defaults_to_max_turns(setup):
    _, _, run_dir = setup
    p = _plan(setup, "matched", "gemini-2.5-pro")
    settings = json.loads(p.files[run_dir / "workspace" / ".gemini" / "settings.json"])
    assert settings["model"]["maxSessionTurns"] == 30


def test_matched_configs_reference_tool_keys_without_writing_secrets(setup):
    _, _, run_dir = setup
    env = {"NCBI_API_KEY": "real-secret-value", "OTHER": "x"}
    for model, path in (("claude-sonnet-4-6", run_dir / "harness" / "mcp.json"),
                        ("gemini-2.5-pro", run_dir / "workspace" / ".gemini" / "settings.json")):
        p = _plan(setup, "matched", model, environ=env)
        text = p.files[path]
        assert "real-secret-value" not in text
        tu = json.loads(text)["mcpServers"]["tooluniverse"]["env"]
        assert tu["NCBI_API_KEY"] == "${NCBI_API_KEY}" and "BIOGRID_API_KEY" not in tu


def test_matched_keeps_rubric_out_of_reach_during_run(setup):
    _, _, run_dir = setup
    for model in ("claude-sonnet-4-6", "gemini-2.5-pro"):
        p = _plan(setup, "matched", model)
        assert not any("task_spec" in str(f) for f in p.files)
        assert p.copy_after_run == {setup[1] / "task_spec.yaml": run_dir / "task_spec.yaml"}


def test_pilot_copies_rubric_before_run(setup):
    _, task_dir, run_dir = setup
    p = _plan(setup, "pilot", "claude-sonnet-4-6")
    assert p.copy_before_run == {task_dir / "task_spec.yaml": run_dir / "task_spec.yaml"}


def test_prepare_run_writes_files_and_dirs(setup):
    _, _, run_dir = setup
    p = _plan(setup, "matched", "gemini-2.5-pro")
    p.venv = None  # venv creation is tested on its own
    prepare_run(p)
    assert (run_dir / "workspace" / "GEMINI.md").is_file()
    assert (run_dir / "harness" / "gemini-home").is_dir()
    assert (run_dir / "harness" / "tmp").is_dir()
    assert not (run_dir / "task_spec.yaml").exists()


# --- transcript summary ---

def _write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_summarize_claude_counts_resumed_segments(tmp_path):
    t = tmp_path / "transcript.jsonl"
    seg = lambda turns, sub, sonnet, haiku: {  # noqa: E731
        "type": "result", "subtype": sub, "num_turns": turns,
        "modelUsage": {"claude-sonnet-4-6": {"inputTokens": sonnet, "outputTokens": 0},
                       "claude-haiku-4-5-20251001": {"inputTokens": haiku, "outputTokens": 0}},
    }
    tool = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}}
    _write(t, [{"type": "system", "subtype": "init", "claude_code_version": "2.1.160"}, tool,
               seg(31, "error_max_turns", 100, 10),
               {"type": "system", "subtype": "init"}, tool, tool, seg(17, "success", 300, 5)])
    s = summarize_transcript(t, "claude", requested_model="claude-sonnet-4-6")
    assert s["segments"] == 2 and s["turns_per_segment"] == [31, 17] and s["turns_total"] == 48
    assert s["cap_hit"] is True and s["tool_calls"] == 3
    assert s["primary_model"] == "claude-sonnet-4-6" and s["primary_model_matches_request"] is True
    assert set(s["models_used"]) == {"claude-sonnet-4-6", "claude-haiku-4-5-20251001"}


def test_summarize_gemini_flags_model_substitution(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write(t, [{"type": "init", "model": "gemini-2.5-flash"},
               {"type": "tool_use", "tool_name": "x"},
               {"type": "result", "status": "success",
                "stats": {"tool_calls": 1, "models": {"gemini-3.6-flash": {"total_tokens": 900}}}}])
    s = summarize_transcript(t, "gemini", requested_model="gemini-2.5-flash")
    assert s["segments"] == 1 and s["tool_calls"] == 1
    assert s["primary_model"] == "gemini-3.6-flash"
    assert s["primary_model_matches_request"] is False
    assert s["turns_total"] is None  # Gemini CLI reports no turn count


def test_summarize_tolerates_non_json_lines(tmp_path):
    t = tmp_path / "transcript.jsonl"
    t.write_text('noise\n{"type": "result", "status": "success", "stats": {"models": {}}}\n')
    assert summarize_transcript(t, "gemini", requested_model="m")["segments"] == 1


# --- execute (with a fake CLI standing in for claude/gemini) ---

import sys

from evaluator.run import execute


def test_execute_runs_records_manifest_and_copies_rubric_after(setup):
    repo, task_dir, run_dir = setup
    plan = plan_run(profile="matched", model="claude-sonnet-4-6", max_turns=30, task_dir=task_dir,
                    run_dir=run_dir, repo_root=repo, environ={})
    plan.venv, plan.sweep_tmp = None, False
    fake = ('import json, os, pathlib\n'
            'seen = pathlib.Path(os.environ["EVAL_RUN_DIR"], "task_spec.yaml").exists()\n'
            'print(json.dumps({"type": "system", "subtype": "init", "cwd": os.getcwd(), "rubric_visible": seen}))\n'
            'print(json.dumps({"type": "result", "subtype": "success", "num_turns": 4, '
            '"modelUsage": {"claude-sonnet-4-6": {"inputTokens": 5}}}))\n')
    plan.argv = [sys.executable, "-c", fake]
    m = execute(plan, task_dir=task_dir, cli_version="test-cli 1.0", environ={"PATH": "/usr/bin"})
    init = json.loads((run_dir / "transcript.jsonl").read_text().splitlines()[0])
    assert init["cwd"] == str(run_dir / "workspace")
    assert init["rubric_visible"] is False                 # not in the run dir during the run
    assert (run_dir / "task_spec.yaml").exists()           # copied in afterwards
    saved = json.loads((run_dir / "run_manifest.json").read_text())
    assert saved == m
    assert m["profile"] == "matched" and m["harness"] == "claude" and m["exit_code"] == 0
    assert m["cli_version"] == "test-cli 1.0"
    assert m["task_sha256"] and len(m["task_sha256"]) == 64
    assert m["outcome"]["turns_total"] == 4 and m["outcome"]["primary_model_matches_request"] is True
    assert "code_commit" in m and m["started_at"] <= m["finished_at"]


def test_manifest_elides_task_text_from_argv(setup):
    repo, task_dir, run_dir = setup
    plan = plan_run(profile="pilot", model="claude-sonnet-4-6", max_turns=30, task_dir=task_dir,
                    run_dir=run_dir, repo_root=repo, environ={})
    from evaluator.run import manifest_argv
    shown = manifest_argv(plan, task_text=(task_dir / "agent_task.md").read_text())
    assert not any("TASK-BODY" in a for a in shown)
    assert any(a.startswith("<agent_task.md") for a in shown)


def test_execute_refuses_existing_run_dir(setup):
    repo, task_dir, run_dir = setup
    run_dir.mkdir(parents=True)
    (run_dir / "transcript.jsonl").write_text("{}\n")
    plan = plan_run(profile="matched", model="gemini-2.5-pro", max_turns=30, task_dir=task_dir,
                    run_dir=run_dir, repo_root=repo, environ={})
    with pytest.raises(FileExistsError):
        execute(plan, task_dir=task_dir, cli_version=None, environ={})


def test_summarize_counts_model_responses_claude(tmp_path):
    t = tmp_path / "transcript.jsonl"
    a = lambda mid, block: {"type": "assistant", "message": {"id": mid, "content": [block]}}  # noqa: E731
    _write(t, [a("m1", {"type": "thinking"}), a("m1", {"type": "tool_use", "name": "X"}),
               a("m2", {"type": "text"}), a("m2", {"type": "tool_use", "name": "Y"}),
               {"type": "result", "subtype": "error_max_turns", "num_turns": 3, "modelUsage": {}}])
    s = summarize_transcript(t, "claude", requested_model="m")
    assert s["model_responses"] == 2 and s["turns_total"] == 3 and s["tool_calls"] == 2


def test_summarize_gemini_counts_responses_and_detects_cap(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write(t, [{"type": "init"}, {"type": "message", "role": "user", "content": "Begin"},
               {"type": "tool_use", "tool_name": "a"}, {"type": "tool_use", "tool_name": "b"},
               {"type": "tool_result"}, {"type": "tool_result"},
               {"type": "message", "role": "assistant", "content": "x"}, {"type": "tool_use", "tool_name": "c"},
               {"type": "tool_result"},
               {"type": "result", "status": "error",
                "error": {"type": "FatalTurnLimitedError", "message": "Reached max session turns"},
                "stats": {"models": {"gemini-2.5-pro": {"total_tokens": 5}}}}])
    s = summarize_transcript(t, "gemini", requested_model="gemini-2.5-pro")
    assert s["model_responses"] == 2 and s["tool_calls"] == 3 and s["cap_hit"] is True


def test_summarize_gemini_cap_not_hit(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _write(t, [{"type": "tool_use", "tool_name": "a"}, {"type": "tool_result"},
               {"type": "result", "status": "success", "stats": {"models": {}}}])
    assert summarize_transcript(t, "gemini", requested_model="m")["cap_hit"] is False


def test_summarize_separates_subagent_responses(tmp_path):
    t = tmp_path / "transcript.jsonl"
    main = {"type": "assistant", "parent_tool_use_id": None, "message": {"id": "m1", "content": [{"type": "text"}]}}
    sub = {"type": "assistant", "parent_tool_use_id": "tu1", "message": {"id": "s1", "content": [{"type": "text"}]}}
    _write(t, [main, sub, {"type": "result", "subtype": "success", "num_turns": 1, "modelUsage": {}}])
    s = summarize_transcript(t, "claude", requested_model="m")
    assert s["model_responses"] == 1 and s["subagent_responses"] == 1


def test_summarize_claude_usage_is_cumulative_within_a_session(tmp_path):
    # Claude Code reports modelUsage cumulatively within a session, so a resumed session's
    # last result already holds the total; separate sessions add up.
    t = tmp_path / "transcript.jsonl"
    res = lambda sid, tokens: {"type": "result", "session_id": sid, "subtype": "success",  # noqa: E731
                               "num_turns": 1, "modelUsage": {"m": {"inputTokens": tokens}}}
    _write(t, [res("s1", 100), res("s1", 250), res("s2", 40)])
    s = summarize_transcript(t, "claude", requested_model="m")
    assert s["models_used"] == {"m": 290}


# --- subagents, background tasks, and stopping runs ---

def test_matched_claude_turns_off_subagents_and_background_tasks(setup):
    p = _plan(setup, "matched", "claude-sonnet-4-6")
    i = p.argv.index("--disallowedTools")
    assert p.argv[i + 1:i + 3] == ["Agent", "Task"]
    assert p.env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert p.stop_on_result is True
    assert p.details["subagents"].startswith("off")


def test_matched_gemini_excludes_subagent_tool(setup):
    _, _, run_dir = setup
    p = _plan(setup, "matched", "gemini-2.5-pro")
    settings = json.loads(p.files[run_dir / "workspace" / ".gemini" / "settings.json"])
    assert settings["tools"]["exclude"] == ["invoke_agent"]
    assert p.stop_on_result is True


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "gemini-2.5-pro"])
def test_pilot_keeps_cli_defaults(setup, model):
    p = _plan(setup, "pilot", model)
    assert "--disallowedTools" not in p.argv
    assert "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS" not in p.env
    assert p.stop_on_result is False


def _fake_plan(setup, script):
    repo, task_dir, run_dir = setup
    plan = plan_run(profile="matched", model="claude-sonnet-4-6", max_turns=30, task_dir=task_dir,
                    run_dir=run_dir, repo_root=repo, environ={})
    plan.argv = [sys.executable, "-c", script]
    plan.venv, plan.sweep_tmp = None, False  # tested on their own below
    return plan, task_dir, run_dir


_RESULT = 'print(json.dumps({"type": "result", "subtype": "success", "num_turns": 3}), flush=True)\n'


def test_execute_records_natural_exit(setup):
    plan, task_dir, _ = _fake_plan(setup, "import json\n" + _RESULT)
    m = execute(plan, task_dir=task_dir, cli_version=None, environ={"PATH": "/usr/bin"}, timeout_s=30)
    assert m["stopped_by_launcher"] is None and m["exit_code"] == 0
    assert m["timeout_minutes"] == 0.5 and m["stop_on_result"] is True


def test_execute_stops_cli_that_keeps_running_after_its_result(setup, monkeypatch):
    import time
    import evaluator.run as run_mod
    monkeypatch.setattr(run_mod, "RESULT_GRACE_S", 0.3)
    script = ("import json, time\n" + _RESULT +
              "time.sleep(30)\nprint(json.dumps({'type': 'result', 'subtype': 'late'}))\n")
    plan, task_dir, run_dir = _fake_plan(setup, script)
    t0 = time.monotonic()
    m = execute(plan, task_dir=task_dir, cli_version=None, environ={"PATH": "/usr/bin"}, timeout_s=60)
    assert time.monotonic() - t0 < 15
    assert m["stopped_by_launcher"] == "after_first_result"
    assert "late" not in (run_dir / "transcript.jsonl").read_text()
    assert m["outcome"]["segments"] == 1


def test_execute_timeout_stops_cli(setup):
    import time
    plan, task_dir, _ = _fake_plan(setup, "import time\ntime.sleep(60)\n")
    t0 = time.monotonic()
    m = execute(plan, task_dir=task_dir, cli_version=None, environ={"PATH": "/usr/bin"}, timeout_s=1)
    assert time.monotonic() - t0 < 15
    assert m["stopped_by_launcher"] == "timeout"


def test_execute_kills_processes_the_cli_leaves_behind(setup):
    import os
    import time
    # The CLI exits but leaves a child running, like a stuck background shell.
    script = ("import subprocess, sys, os\n"
              "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
              "print(c.pid, flush=True)\n")
    plan, task_dir, run_dir = _fake_plan(setup, script)
    t0 = time.monotonic()
    m = execute(plan, task_dir=task_dir, cli_version=None, environ={"PATH": "/usr/bin"}, timeout_s=60)
    assert time.monotonic() - t0 < 15
    assert m["stopped_by_launcher"] is None
    child = int((run_dir / "transcript.jsonl").read_text().split()[0])
    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("leftover child still running")


def test_execute_keeps_output_written_just_before_exit(setup):
    # Gemini CLI writes its result record and exits at once; it must reach the transcript.
    script = ("import os\n"
              "os.write(1, b'x' * 200000 + b'\\n{\"type\": \"result\", \"subtype\": \"end\"}\\n')\n"
              "os._exit(53)\n")
    plan, task_dir, run_dir = _fake_plan(setup, script)
    m = execute(plan, task_dir=task_dir, cli_version=None, environ={"PATH": "/usr/bin"}, timeout_s=60)
    assert m["exit_code"] == 53
    assert (run_dir / "transcript.jsonl").read_text().splitlines()[-1] == '{"type": "result", "subtype": "end"}'


# --- per-run Python environment and temp files ---

@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "gemini-2.5-pro"])
def test_matched_runs_get_own_python_and_tmpdir(setup, model):
    _, _, run_dir = setup
    p = _plan(setup, "matched", model, environ={"PATH": "/usr/bin:/bin"})
    venv = run_dir / "harness" / "venv"
    assert p.venv == venv and p.sweep_tmp is True
    assert p.env["PATH"] == f"{venv / 'bin'}:/usr/bin:/bin"
    assert p.env["VIRTUAL_ENV"] == str(venv)
    assert p.env["PYTHONNOUSERSITE"] == "1"
    assert p.env["TMPDIR"] == str(run_dir / "harness" / "tmp")
    if model.startswith("claude"):
        assert p.env["CLAUDE_CODE_TMPDIR"] == str(run_dir / "harness" / "tmp")


def test_pilot_runs_keep_user_python(setup):
    p = _plan(setup, "pilot", "claude-sonnet-4-6", environ={"PATH": "/usr/bin"})
    assert p.venv is None and p.sweep_tmp is False and "PATH" not in p.env


def test_prepare_run_creates_venv_with_run_python(setup):
    import shutil as _sh
    if not _sh.which("uv"):
        pytest.skip("uv not installed")
    from evaluator.harness import RUN_PYTHON
    _, _, run_dir = setup
    p = _plan(setup, "matched", "claude-sonnet-4-6")
    try:
        prepare_run(p)
    except Exception as e:  # e.g. no Python RUN_PYTHON available offline
        pytest.skip(f"uv venv failed: {e}")
    cfg = (run_dir / "harness" / "venv" / "pyvenv.cfg").read_text()
    assert f"version_info = {RUN_PYTHON}." in cfg or f"version = {RUN_PYTHON}." in cfg
    assert (run_dir / "harness" / "venv" / "bin" / "pip").exists()


def test_sweep_tmp_moves_only_new_entries_the_agent_named(tmp_path):
    from evaluator.run import sweep_tmp
    tmp = tmp_path / "shared"
    tmp.mkdir()
    (tmp / "old.py").write_text("before the run")
    before = {"old.py"}
    for name in ("sle_x.py", "sle_x.py.bak", "other_app.sock"):
        (tmp / name).write_text("new")
    (tmp / "scratch").mkdir()
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"command": "python3 /tmp/sle_x.py; ls /private/tmp/scratch/ ; cat /tmp/old.py"}\n')
    out = sweep_tmp(tmp, before, transcript, tmp_path / "leftovers")
    assert out["moved"] == ["scratch", "sle_x.py"]
    assert out["unattributed"] == ["other_app.sock", "sle_x.py.bak"]
    assert (tmp_path / "leftovers" / "sle_x.py").is_file() and (tmp / "old.py").exists()
    assert not (tmp / "sle_x.py").exists() and (tmp / "other_app.sock").exists()


def test_summarize_counts_tool_errors_by_cause(tmp_path):
    path = tmp_path / "t.jsonl"
    _write(path, [
        {"type": "tool_use", "tool_name": "mcp_tooluniverse_find_tools", "tool_id": "1"},
        {"type": "tool_result", "tool_id": "1", "status": "error", "error": {
            "type": "mcp_tool_error", "message": "wait_for_previous  Unexpected keyword argument"}},
        {"type": "tool_use", "tool_name": "mcp_hypothesis_ledger_x", "tool_id": "2"},
        {"type": "tool_result", "tool_id": "2", "status": "error",
         "error": {"type": "tool_not_registered", "message": "not found"}},
        {"type": "tool_use", "tool_name": "read_file", "tool_id": "3"},
        {"type": "tool_result", "tool_id": "3", "status": "success"},
    ])
    s = summarize_transcript(path, "gemini", "gemini-2.5-pro")
    assert s["tool_errors"] == {"wait_for_previous_rejected": 1, "tool_not_registered": 1}
