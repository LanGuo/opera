# tests/test_workspace.py
import json
import pytest
import yaml
from pathlib import Path
from hypothesis_ledger.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace(str(tmp_path / "run_001"))


def test_init_creates_directories(ws, tmp_path):
    run_dir = tmp_path / "run_001"
    assert (run_dir / "data").exists()
    assert (run_dir / "results").exists()
    assert (run_dir / "ledger.json").exists()


def test_init_ledger_is_empty(ws):
    data = ws.read_ledger()
    assert data["hypotheses"] == {}
    assert data["tool_call_count"] == 0


def test_update_ledger_mutates(ws):
    def add_entry(data):
        data["hypotheses"]["H001"] = {"claim": "test"}
    ws.update_ledger(add_entry)
    assert ws.read_ledger()["hypotheses"]["H001"]["claim"] == "test"


def test_increment_tool_call_count(ws):
    assert ws.increment_tool_call_count() == 1
    assert ws.increment_tool_call_count() == 2
    assert ws.read_ledger()["tool_call_count"] == 2


def test_log_run_appends_jsonl(ws, tmp_path):
    ws.log_run({"tool": "test_tool", "result": "ok"})
    ws.log_run({"tool": "test_tool2", "result": "ok"})
    lines = (tmp_path / "run_001" / "run_log.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["tool"] == "test_tool"


def test_log_step_appends_jsonl(ws, tmp_path):
    ws.log_step({"step_type": "analysis", "interpretation": "found DEGs"})
    lines = (tmp_path / "run_001" / "step_log.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["step_type"] == "analysis"


def test_write_final_report(ws, tmp_path):
    ws.write_final_report({"top_hypotheses": ["H001"]})
    report = json.loads((tmp_path / "run_001" / "final_report.json").read_text())
    assert report["top_hypotheses"] == ["H001"]


@pytest.fixture
def ws_with_task_spec(tmp_path):
    ws = Workspace(str(tmp_path / "run_001"))
    spec = {
        "task_id": "test_task",
        "goal": "Find something interesting.",
        "termination_criteria": {"max_tool_calls": 50},
        "rubric": [
            {"id": "novelty", "label": "Novelty", "scope": "per_hypothesis",
             "scale": "1-5", "guidance": "Is it novel?"},
            {"id": "quality", "label": "Quality", "scope": "overall",
             "scale": "1-5", "guidance": "Is it high quality?"},
        ],
    }
    ws.task_spec_path.write_text(yaml.dump(spec), encoding="utf-8")
    return ws, spec


def test_read_task_spec_returns_dict(ws_with_task_spec):
    ws, spec = ws_with_task_spec
    result = ws.read_task_spec()
    assert isinstance(result, dict)
    assert result["task_id"] == "test_task"
    assert result["goal"] == "Find something interesting."


def test_read_task_spec_rubric_items(ws_with_task_spec):
    ws, spec = ws_with_task_spec
    result = ws.read_task_spec()
    assert len(result["rubric"]) == 2
    assert result["rubric"][0]["id"] == "novelty"
    assert result["rubric"][1]["scope"] == "overall"


def test_read_task_spec_returns_none_when_absent(tmp_path):
    ws = Workspace(str(tmp_path / "run_no_spec"))
    assert ws.read_task_spec() is None


def test_task_spec_path_property(tmp_path):
    ws = Workspace(str(tmp_path / "run_001"))
    assert ws.task_spec_path == ws.run_dir / "task_spec.yaml"
