"""Turn-budget warning hook: counting responses, when it warns, what it records."""
import io
import json

from evaluator import turn_budget
from evaluator.harness import plan_run
from hypothesis_ledger.ledger import Ledger


def _claude_transcript(path, responses, subagent=0):
    recs = []
    for i in range(responses):  # two content blocks per response share one message id
        recs += [{"type": "assistant", "message": {"id": f"m{i}"}}] * 2
    recs += [{"type": "assistant", "message": {"id": f"s{i}"}, "parent_tool_use_id": "t"}
             for i in range(subagent)]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n{\"type\": \"assist", encoding="utf-8")


def test_count_responses_claude_ignores_blocks_subagents_and_partial_line(tmp_path):
    _claude_transcript(tmp_path / "transcript.jsonl", 4, subagent=3)
    assert turn_budget.count_responses(tmp_path / "transcript.jsonl", "claude") == 4
    assert turn_budget.count_responses(tmp_path / "missing.jsonl", "claude") == 0


def test_count_responses_gemini(tmp_path):
    recs = [{"type": "message", "role": "assistant"}, {"type": "tool_use"}, {"type": "tool_use"},
            {"type": "tool_result"}, {"type": "tool_use"}, {"type": "tool_result"}]
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    assert turn_budget.count_responses(tmp_path / "t.jsonl", "gemini") == 2


def test_check_warns_only_inside_window_and_records_once_per_turn(tmp_path):
    _claude_transcript(tmp_path / "transcript.jsonl", 24)
    assert turn_budget.check(tmp_path, "claude", 30, 5) is None          # 6 left
    _claude_transcript(tmp_path / "transcript.jsonl", 26)
    text = turn_budget.check(tmp_path, "claude", 30, 5)                  # 4 left
    assert "4 turns left of 30" in text and "declare_done" in text
    assert turn_budget.check(tmp_path, "claude", 30, 5) == text           # second tool, same turn
    _claude_transcript(tmp_path / "transcript.jsonl", 29)
    assert "1 turn left" in turn_budget.check(tmp_path, "claude", 30, 5)
    _claude_transcript(tmp_path / "transcript.jsonl", 30)
    assert turn_budget.check(tmp_path, "claude", 30, 5) is None          # no response left to read it
    assert turn_budget.check(tmp_path, "claude", 30, 0) is None
    msgs = Ledger(tmp_path).entries("harness_message")
    assert [(m["turn"], m["turns_left"]) for m in msgs] == [(26, 4), (29, 1)]
    assert all(m["written_by"] == "harness" and m["purpose"] == "turn_budget" for m in msgs)


def test_main_prints_hook_json(tmp_path, monkeypatch, capsys):
    _claude_transcript(tmp_path / "transcript.jsonl", 28)
    monkeypatch.setenv("EVAL_RUN_DIR", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    assert turn_budget.main(["--harness", "claude", "--cap", "30", "--warn-at", "5"]) == 0
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert out["hookEventName"] == "PostToolUse" and "2 turns left" in out["additionalContext"]
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    _claude_transcript(tmp_path / "transcript.jsonl", 3)
    turn_budget.main(["--harness", "claude", "--cap", "30", "--warn-at", "5"])
    assert capsys.readouterr().out == ""


def _task(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "agent_task.md").write_text("Do science.\n", encoding="utf-8")
    return task


def _plan(tmp_path, model, profile="matched", **kw):
    return plan_run(profile=profile, model=model, max_turns=30, task_dir=_task(tmp_path),
                    run_dir=tmp_path / "run", repo_root=tmp_path, environ={}, **kw)


def test_matched_claude_plan_registers_hook(tmp_path):
    plan = _plan(tmp_path, "claude-sonnet-4-6", turn_warning=5)
    settings = json.loads(plan.files[tmp_path / "run/harness/claude-home/settings.json"])
    hook = settings["hooks"]["PostToolUse"][0]
    assert hook["matcher"] == "*"
    assert "evaluator.turn_budget --harness claude --cap 30 --warn-at 5" in hook["hooks"][0]["command"]
    assert "**Turn budget:** this run stops after 30 turns" in plan.task_text
    assert plan.details["turn_warning"].startswith("from 5 turns")


def test_matched_gemini_plan_registers_hook_with_session_cap(tmp_path):
    plan = _plan(tmp_path, "gemini-2.5-pro", turn_warning=5, gemini_max_session_turns=40)
    settings = json.loads(plan.files[tmp_path / "run/workspace/.gemini/settings.json"])
    cmd = settings["hooks"]["AfterTool"][0]["hooks"][0]["command"]
    assert "--harness gemini --cap 40" in cmd and "stops after 40 turns" in plan.task_text


def test_no_hook_without_warning_or_in_pilot(tmp_path):
    plan = _plan(tmp_path, "claude-sonnet-4-6")
    assert not any(p.name == "settings.json" for p in plan.files)
    assert "Turn budget" not in plan.task_text and plan.details["turn_warning"] == "none"
    pilot = plan_run(profile="pilot", model="claude-sonnet-4-6", max_turns=30,
                     task_dir=tmp_path / "task", run_dir=tmp_path / "run2", repo_root=tmp_path,
                     environ={}, turn_warning=5)
    assert "Turn budget" not in pilot.task_text


def test_gemini_key_mapped_from_google_key_only_for_gemini():
    from evaluator.run import gemini_auth_from_google_key
    env = {"GOOGLE_API_KEY": "g"}
    gemini_auth_from_google_key("claude", env)
    assert "GEMINI_API_KEY" not in env
    gemini_auth_from_google_key("gemini", env)
    assert env["GEMINI_API_KEY"] == "g"
    env = {"GOOGLE_API_KEY": "g", "GEMINI_API_KEY": "own"}
    gemini_auth_from_google_key("gemini", env)
    assert env["GEMINI_API_KEY"] == "own"
