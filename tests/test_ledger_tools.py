import json
import os
import pytest

# Each test gets a fresh tmp workspace
@pytest.fixture(autouse=True)
def set_run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_RUN_DIR", str(tmp_path / "run_001"))
    # Reset the module-level _workspace singleton between tests
    import hypothesis_ledger.server as srv
    srv._workspace = None


def call_tool(name: str, **kwargs) -> dict:
    """Helper: import and call a tool function directly (bypassing MCP wire protocol)."""
    import hypothesis_ledger.server as srv
    fn = getattr(srv, f"_tool_{name}")
    result = fn(**kwargs)
    return json.loads(result)


def test_write_hypothesis_returns_id():
    result = call_tool(
        "write_hypothesis",
        cell_type="pDC",
        candidate_gene="SIGLEC1",
        claim="Upregulated in SLE pDCs",
        evidence=[{"source": "DEG_analysis", "strength": "strong", "log2fc": 2.1}],
        confidence=0.7,
        open_questions=["Is it elevated in remission?"],
    )
    assert "hypothesis_id" in result
    assert result["status"] == "recorded"
    assert len(result["hypothesis_id"]) == 6


def test_write_hypothesis_persists_to_ledger():
    result = call_tool(
        "write_hypothesis",
        cell_type="pDC",
        candidate_gene="SIGLEC1",
        claim="Test claim",
        evidence=[],
        confidence=0.5,
        open_questions=[],
    )
    h_id = result["hypothesis_id"]

    from hypothesis_ledger.workspace import Workspace
    ws = Workspace(os.environ["EVAL_RUN_DIR"])
    ledger = ws.read_ledger()
    assert h_id in ledger["hypotheses"]
    assert ledger["hypotheses"][h_id]["candidate_gene"] == "SIGLEC1"
    assert ledger["hypotheses"][h_id]["status"] == "active"


def test_read_hypotheses_filters_by_status():
    call_tool(
        "write_hypothesis",
        cell_type="pDC", candidate_gene="GENE_A", claim="c1",
        evidence=[], confidence=0.5, open_questions=[],
    )
    result = call_tool("read_hypotheses", status_filter="active")
    assert result["filtered_count"] == 1

    result_all = call_tool("read_hypotheses", status_filter="all")
    assert result_all["total_count"] == 1


def test_read_hypotheses_increments_count():
    call_tool(
        "write_hypothesis",
        cell_type="pDC", candidate_gene="GENE_B", claim="c2",
        evidence=[], confidence=0.4, open_questions=[],
    )
    result = call_tool("read_hypotheses", status_filter="active")
    # write_hypothesis (1) + read_hypotheses (1) = 2
    assert result["tool_call_count"] == 2


def test_update_hypothesis_changes_confidence():
    r = call_tool(
        "write_hypothesis",
        cell_type="pDC", candidate_gene="IRF7", claim="elevated",
        evidence=[], confidence=0.5, open_questions=[],
    )
    h_id = r["hypothesis_id"]
    call_tool("update_hypothesis", hypothesis_id=h_id,
              updates={"confidence": 0.85}, rationale="GWAS hit found")

    from hypothesis_ledger.workspace import Workspace
    ws = Workspace(os.environ["EVAL_RUN_DIR"])
    h = ws.read_ledger()["hypotheses"][h_id]
    assert h["confidence"] == 0.85
    assert len(h["update_history"]) == 1
    assert h["update_history"][0]["rationale"] == "GWAS hit found"


def test_update_hypothesis_unknown_id_raises():
    with pytest.raises(ValueError, match="not found"):
        call_tool("update_hypothesis", hypothesis_id="ZZZZZZ",
                  updates={"status": "refuted"}, rationale="n/a")


def test_log_analysis_step_writes_to_step_log(tmp_path):
    call_tool(
        "log_analysis_step",
        step_type="DEG_analysis",
        tool_used="run_deg_analysis",
        inputs_summary="SLE vs control in pDCs",
        outputs_summary="412 upregulated genes, top: SIGLEC1",
        interpretation="SIGLEC1 is a strong candidate",
        next_action="Search GWAS catalog for SIGLEC1 loci",
    )
    log_path = tmp_path / "run_001" / "step_log.jsonl"
    lines = log_path.read_text().strip().split("\n")
    entry = json.loads(lines[0])
    assert entry["step_type"] == "DEG_analysis"
    assert entry["tool_used"] == "run_deg_analysis"


def test_declare_done_writes_final_report(tmp_path):
    r = call_tool(
        "write_hypothesis",
        cell_type="pDC", candidate_gene="SIGLEC1", claim="Top candidate",
        evidence=[{"source": "DEG", "strength": "strong"}], confidence=0.8, open_questions=[],
    )
    h_id = r["hypothesis_id"]

    result = call_tool(
        "declare_done",
        top_hypotheses=[h_id],
        rationale="Converged on 1 strong candidate",
        confidence_summary="0.8 for SIGLEC1",
        suggested_validation="FACS sort pDCs from SLE donors; measure SIGLEC1 by flow cytometry",
    )
    assert result["status"] == "completed"
    assert result["hypothesis_count"] == 1

    report_path = tmp_path / "run_001" / "final_report.json"
    report = json.loads(report_path.read_text())
    assert len(report["top_hypotheses"]) == 1
    assert report["top_hypotheses"][0]["candidate_gene"] == "SIGLEC1"


def test_declare_done_marks_hypotheses_terminal():
    r = call_tool(
        "write_hypothesis",
        cell_type="pDC", candidate_gene="MX1", claim="IFN marker",
        evidence=[], confidence=0.6, open_questions=[],
    )
    h_id = r["hypothesis_id"]
    call_tool(
        "declare_done",
        top_hypotheses=[h_id],
        rationale="done", confidence_summary="0.6", suggested_validation="flow",
    )
    from hypothesis_ledger.workspace import Workspace
    ws = Workspace(os.environ["EVAL_RUN_DIR"])
    assert ws.read_ledger()["hypotheses"][h_id]["status"] == "terminal"


def test_request_human_input_headless(monkeypatch):
    monkeypatch.setenv("ALLOW_HUMAN_INPUT", "false")
    result = call_tool(
        "request_human_input",
        question="Is SIGLEC1 druggable?",
        context="Evaluating therapeutic potential",
    )
    assert result["status"] == "headless_mode"
    assert "SIGLEC1" in result["question"]
