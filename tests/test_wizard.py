# tests/test_wizard.py
import json
import pytest
from fastapi.testclient import TestClient
from task_proposer.wizard import app
from unittest.mock import patch
import task_proposer.wizard as wizard_module
from task_proposer.wizard import _build_yaml, SaveRequest

client = TestClient(app)


def test_get_index_returns_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "<!DOCTYPE html>" in resp.text


def test_get_index_contains_phase_divs():
    resp = client.get("/")
    assert 'id="phase1"' in resp.text
    assert 'id="phase2"' in resp.text


def test_get_index_contains_form_fields():
    resp = client.get("/")
    for field_id in ("task-id", "goal", "min-hypotheses", "max-tool-calls",
                     "evaluator-model", "max-turns", "rubric-items",
                     "agent-task-preview"):
        assert f'id="{field_id}"' in resp.text, f"Missing field id: {field_id}"


def test_get_index_contains_buttons():
    resp = client.get("/")
    assert "Generate open-world eval task" in resp.text
    assert "Generate &amp; Check" in resp.text
    assert "Save Task" in resp.text


MOCK_EXTRACT_RESPONSE = json.dumps({
    "task_id": "cancer_biomarker_discovery",
    "goal": "Identify top 5 candidate biomarkers for pancreatic cancer using multi-omics data.",
    "termination_criteria": {
        "min_hypotheses_with_multimodal_evidence": 3,
        "max_tool_calls": 200,
        "custom": []
    },
    "evaluator": {"model": "claude-opus-4-7", "max_turns": 250},
    "rubric_overrides": {}
})


def test_extract_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    resp = client.post("/extract", json={"free_text": "find cancer biomarkers"})
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_extract_calls_claude_with_free_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def mock_call(system, user):
        calls.append((system, user))
        return MOCK_EXTRACT_RESPONSE

    with patch("task_proposer.wizard._call_llm", mock_call):
        resp = client.post("/extract", json={"free_text": "find cancer biomarkers"})
    assert resp.status_code == 200
    assert len(calls) == 1
    assert "find cancer biomarkers" in calls[0][1]


def test_extract_returns_expected_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("task_proposer.wizard._call_llm", return_value=MOCK_EXTRACT_RESPONSE):
        resp = client.post("/extract", json={"free_text": "find cancer biomarkers"})
    data = resp.json()
    assert data["task_id"] == "cancer_biomarker_discovery"
    assert "goal" in data
    assert "termination_criteria" in data
    assert "custom" in data["termination_criteria"]
    assert "evaluator" in data


MOCK_CLAUDEMD = (
    "# Task\n\nYour task is to identify cancer biomarkers.\n\n"
    "## Termination Criteria\n- [ ] 3 hypotheses with multimodal evidence\n"
)
MOCK_ISSUES = json.dumps([
    {"field": "goal", "severity": "warning",
     "issue": "Slightly vague", "suggestion": "Add dataset pointer"}
])
MOCK_ISSUES_EMPTY = json.dumps([])


def test_generate_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    resp = client.post("/generate", json={
        "task_id": "cancer_biomarker",
        "goal": "Find biomarkers",
        "termination_criteria": {"min_hypotheses_with_multimodal_evidence": 3, "max_tool_calls": 200},
        "evaluator": {"model": "claude-opus-4-7", "max_turns": 250},
        "rubric_overrides": {}
    })
    assert "error" in resp.json()


def test_generate_makes_two_claude_calls(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    call_count = [0]

    def mock_call(system, user):
        call_count[0] += 1
        return MOCK_CLAUDEMD if call_count[0] == 1 else MOCK_ISSUES

    with patch("task_proposer.wizard._call_llm", mock_call):
        resp = client.post("/generate", json={
            "task_id": "cancer_biomarker",
            "goal": "Find biomarkers",
            "termination_criteria": {"min_hypotheses_with_multimodal_evidence": 3, "max_tool_calls": 200},
            "evaluator": {"model": "claude-opus-4-7", "max_turns": 250},
            "rubric_overrides": {}
        })
    assert call_count[0] == 2
    data = resp.json()
    assert data["agent_task_md"] == MOCK_CLAUDEMD
    assert len(data["issues"]) == 1
    assert data["issues"][0]["field"] == "goal"


def test_generate_returns_empty_issues(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    call_count = [0]

    def mock_call(system, user):
        call_count[0] += 1
        return MOCK_CLAUDEMD if call_count[0] == 1 else MOCK_ISSUES_EMPTY

    with patch("task_proposer.wizard._call_llm", mock_call):
        resp = client.post("/generate", json={
            "task_id": "t", "goal": "g",
            "termination_criteria": {"max_tool_calls": 100},
            "evaluator": {"model": "claude-opus-4-7", "max_turns": 250},
            "rubric_overrides": {}
        })
    assert resp.json()["issues"] == []


SAVE_PAYLOAD = {
    "task_id": "test_task",
    "goal": "Find biomarkers for lupus.",
    "termination_criteria": {
        "min_hypotheses_with_multimodal_evidence": 3,
        "max_tool_calls": 200,
        "requires_validation": True,
    },
    "evaluator": {"model": "claude-opus-4-7", "max_turns": 250},
    "rubric_overrides": {},
    "agent_task_md": "# Task\n\nYour task is to find biomarkers.\n"
}


def test_build_yaml_standard_fields():
    req = SaveRequest(**SAVE_PAYLOAD)
    yaml = _build_yaml(req)
    assert "task_id: test_task" in yaml
    assert "goal: >" in yaml
    assert "Find biomarkers for lupus." in yaml
    assert "min_hypotheses_with_multimodal_evidence: 3" in yaml
    assert "max_tool_calls: 200" in yaml
    assert "requires_validation: true" in yaml
    assert "model: claude-opus-4-7" in yaml
    assert "max_turns: 250" in yaml


def test_build_yaml_quantitative_custom_criterion():
    req = SaveRequest(
        task_id="t", goal="g",
        termination_criteria={
            "min_hypotheses_with_multimodal_evidence": 3,
            "max_tool_calls": 200,
            "min_confidence_score": 0.8,
        },
        evaluator={"model": "claude-opus-4-7", "max_turns": 250},
        rubric_overrides={},
        agent_task_md=""
    )
    yaml = _build_yaml(req)
    assert "min_confidence_score: 0.8" in yaml


def test_save_writes_files(monkeypatch, tmp_path):
    monkeypatch.setattr(wizard_module, "REPO_ROOT", tmp_path)
    with patch("task_proposer.wizard.threading.Thread"):
        resp = client.post("/save", json=SAVE_PAYLOAD)
    assert resp.json()["status"] == "ok"
    task_dir = tmp_path / "tasks" / "test_task"
    assert (task_dir / "task_spec.yaml").exists()
    assert (task_dir / "agent_task.md").exists()
    assert "test_task" in (task_dir / "task_spec.yaml").read_text()
    assert "find biomarkers" in (task_dir / "agent_task.md").read_text()


def test_save_task_exists_error(monkeypatch, tmp_path):
    monkeypatch.setattr(wizard_module, "REPO_ROOT", tmp_path)
    (tmp_path / "tasks" / "test_task").mkdir(parents=True)
    resp = client.post("/save", json=SAVE_PAYLOAD)
    assert "error" in resp.json()
    assert "already exists" in resp.json()["error"]


def test_save_returns_path(monkeypatch, tmp_path):
    monkeypatch.setattr(wizard_module, "REPO_ROOT", tmp_path)
    with patch("task_proposer.wizard.threading.Thread"):
        resp = client.post("/save", json=SAVE_PAYLOAD)
    assert resp.json()["path"] == "tasks/test_task/"


def test_js_contains_extract_function():
    resp = client.get("/")
    assert "async function extractTask()" in resp.text


def test_js_contains_generate_function():
    resp = client.get("/")
    assert "async function generateAndCheck()" in resp.text


def test_js_contains_save_function():
    resp = client.get("/")
    assert "async function saveTask()" in resp.text


def test_js_contains_criterion_helpers():
    resp = client.get("/")
    assert "function addCriterion()" in resp.text
    assert "function addCriterionRow(" in resp.text
    assert "function toggleCriterionType(" in resp.text


def test_js_contains_apply_fix():
    resp = client.get("/")
    assert "function applyFix(" in resp.text


def test_js_contains_collect_spec():
    resp = client.get("/")
    assert "function collectSpec()" in resp.text


def test_build_yaml_with_rubric_list():
    import yaml
    from task_proposer.wizard import _build_yaml, SaveRequest
    req = SaveRequest(
        task_id="test_task",
        goal="Find something.",
        termination_criteria={"max_tool_calls": 50},
        evaluator={"model": "claude-opus-4-7", "max_turns": 250},
        rubric_overrides=[
            {"id": "novelty", "label": "Novelty", "scope": "per_hypothesis",
             "scale": "1-5", "guidance": "Is it novel?"},
            {"id": "quality", "label": "Quality", "scope": "overall",
             "scale": "yes/no/partial", "guidance": "Is it good?"},
        ],
        agent_task_md="# Task\nDo stuff.",
    )
    yaml_str = _build_yaml(req)
    parsed = yaml.safe_load(yaml_str)
    assert parsed["task_id"] == "test_task"
    assert isinstance(parsed["rubric"], list)
    assert len(parsed["rubric"]) == 2
    assert parsed["rubric"][0]["id"] == "novelty"
    assert parsed["rubric"][0]["scope"] == "per_hypothesis"
    assert parsed["rubric"][1]["scale"] == "yes/no/partial"
    assert parsed["rubric"][0]["guidance"] == "Is it novel?"


def test_right_panel_fields_disabled_on_load():
    resp = client.get("/")
    # All right-panel inputs should start disabled
    assert 'id="task-id" type="text"' in resp.text
    assert 'id="right-panel-form" disabled' in resp.text


def test_evaluator_model_dropdown_has_correct_haiku_id():
    resp = client.get("/")
    assert "claude-haiku-4-5-20251001" not in resp.text
    assert "claude-haiku-4-5" in resp.text


def test_step_indicator_present():
    resp = client.get("/")
    assert 'id="step-indicator"' in resp.text


def test_generate_button_is_secondary_initially():
    resp = client.get("/")
    # Generate & Check starts as secondary (less prominent) until step 2
    assert 'id="generate-btn"' in resp.text


def test_start_over_link_in_phase2():
    resp = client.get("/")
    assert 'startOver()' in resp.text


def test_save_button_starts_disabled():
    resp = client.get("/")
    assert 'id="save-btn"' in resp.text
    assert 'disabled' in resp.text


def test_left_panel_has_instruction_area():
    resp = client.get("/")
    assert 'id="step-instruction"' in resp.text
