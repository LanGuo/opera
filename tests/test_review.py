# tests/test_review.py
import base64
import csv
import json
import pytest
from pathlib import Path
from hypothesis_ledger.workspace import Workspace
from reviewer.review import build_review_data, _render_file, _collect_supporting_files, DEFAULT_RUBRIC


@pytest.fixture
def run_with_data(tmp_path):
    ws = Workspace(str(tmp_path / "run_001"))

    def add(data):
        data["hypotheses"] = {
            "H002": {
                "hypothesis_id": "H002", "candidate_gene": "KLRB1",
                "cell_type": "NK", "status": "refuted",
                "evidence": [{"type": "DEG", "description": "not significant"}],
                "confidence": 0.2, "open_questions": [],
                "update_history": [],
                "created_at": "2026-01-01T00:01:00+00:00",
            },
            "H001": {
                "hypothesis_id": "H001", "candidate_gene": "SIGLEC1",
                "cell_type": "pDC", "status": "active",
                "evidence": [{"source": "DEG", "description": "strong signal"}],
                "confidence": 0.8, "open_questions": [],
                "update_history": [{"timestamp": "2026-01-01T00:05:00+00:00",
                                    "rationale": "GWAS confirmed",
                                    "changes": {"from": {"confidence": 0.6}, "to": {"confidence": 0.8}}}],
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        }
    ws.update_ledger(add)
    ws.log_step({"step_type": "deg_analysis", "tool_used": "scanpy",
                 "inputs_summary": "Perez 2022", "outputs_summary": "IFN genes",
                 "interpretation": "IFN signature dominant", "next_action": "write hypothesis"})
    ws.write_final_report({"rationale": "converged", "confidence_summary": "0.8",
                           "suggested_validation": "FACS sort", "top_hypotheses": []})

    # Add a CSV to data/
    csv_path = tmp_path / "run_001" / "data" / "deg.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("gene,logfc,pval\nSIGLEC1,2.1,0.001\nIFI44L,3.5,0.0001\n")

    return str(tmp_path / "run_001")


def test_build_review_data_has_required_keys(run_with_data):
    d = build_review_data(run_with_data)
    for key in ("run_id", "run_dir", "metrics", "steps", "supporting_files",
                "hypotheses", "locked", "task_spec", "rubric"):
        assert key in d


def test_build_review_data_hypotheses_sorted_by_created_at(run_with_data):
    d = build_review_data(run_with_data)
    ids = [h["hypothesis_id"] for h in d["hypotheses"]]
    assert ids == ["H001", "H002"]


def test_build_review_data_steps_parsed(run_with_data):
    d = build_review_data(run_with_data)
    assert len(d["steps"]) == 1
    assert d["steps"][0]["step_type"] == "deg_analysis"


def test_build_review_data_locked_from_report(run_with_data):
    d = build_review_data(run_with_data)
    assert d["locked"]["rationale"] == "converged"
    assert d["locked"]["suggested_validation"] == "FACS sort"
    assert d["locked"]["llm_scores"] is None
    assert d["locked"]["overall_raw_by_backend"] is None


def test_build_review_data_metrics_coverage_is_lists(run_with_data):
    d = build_review_data(run_with_data)
    for srcs in d["metrics"]["modality_coverage"].values():
        assert isinstance(srcs, list)


def test_collect_supporting_files_empty(tmp_path):
    ws = Workspace(str(tmp_path / "empty"))
    files = _collect_supporting_files(ws.run_dir)
    assert files == []


def test_collect_supporting_files_finds_csv(run_with_data):
    from hypothesis_ledger.workspace import Workspace as WS
    ws = WS(run_with_data)
    files = _collect_supporting_files(ws.run_dir)
    assert len(files) == 1
    assert files[0]["name"] == "deg.csv"
    assert files[0]["render_type"] == "table"
    assert files[0]["subdir"] == "data"


def test_render_file_csv_preview(tmp_path):
    p = tmp_path / "test.csv"
    rows = [["gene", "lfc"]] + [[f"G{i}", str(i * 0.1)] for i in range(25)]
    p.write_text("\n".join(",".join(r) for r in rows))
    result = _render_file(p)
    assert result["render_type"] == "table"
    parsed = json.loads(result["content"])
    assert len(parsed) == 21  # header + 20 rows
    assert result["row_count"] == 26  # header + 25 data rows


def test_render_file_text_truncated(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("\n".join(f"line {i}" for i in range(150)))
    result = _render_file(p)
    assert result["render_type"] == "text"
    assert "50 more lines" in result["content"]


def test_render_file_binary(tmp_path):
    p = tmp_path / "data.h5ad"
    p.write_bytes(b"\x00" * 100)
    result = _render_file(p)
    assert result["render_type"] == "binary"
    assert result["content"] == ""
    assert result["size_bytes"] == 100


def test_render_file_png(tmp_path):
    # Minimal 1x1 PNG
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    p = tmp_path / "plot.png"
    p.write_bytes(png_bytes)
    result = _render_file(p)
    assert result["render_type"] == "image"
    assert result["content"].startswith("data:image/png;base64,")




def test_render_file_svg(tmp_path):
    p = tmp_path / "plot.svg"
    p.write_bytes(b"<svg></svg>")
    result = _render_file(p)
    assert result["render_type"] == "image"
    assert result["content"].startswith("data:image/svg+xml;base64,")


def test_render_file_jpg(tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header
    result = _render_file(p)
    assert result["render_type"] == "image"
    assert result["content"].startswith("data:image/jpeg;base64,")


def test_render_file_json_200_line_limit(tmp_path):
    p = tmp_path / "big.json"
    p.write_text("\n".join(f'{{"k": {i}}}' for i in range(250)))
    result = _render_file(p)
    assert result["render_type"] == "text"
    assert "50 more lines" in result["content"]


def test_render_file_tsv(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("gene\tlogfc\nSIGLEC1\t2.1\nKLRB1\t0.5\n")
    result = _render_file(p)
    assert result["render_type"] == "table"
    rows = json.loads(result["content"])
    assert rows[0] == ["gene", "logfc"]
    assert rows[1] == ["SIGLEC1", "2.1"]


def test_build_review_data_llm_scores(run_with_data):
    from reviewer.score import RunScores, HypothesisScores

    hs = HypothesisScores(
        hypothesis_id="H001",
        candidate_gene="SIGLEC1",
        per_hypothesis_raw="per hyp evaluation text",
    )
    run_scores = RunScores(
        hypothesis_scores=[hs],
        overall_raw="overall evaluation text",
        run_dir=run_with_data,
    )

    llm_scores_dict = {"anthropic": run_scores}
    d = build_review_data(run_with_data, llm_scores=llm_scores_dict)
    assert d["locked"]["llm_scores"] is not None
    assert len(d["locked"]["llm_scores"]) == 1
    entry = d["locked"]["llm_scores"][0]
    assert entry["hypothesis_id"] == "H001"
    assert entry["gene"] == "SIGLEC1"
    assert "scores_by_backend" in entry
    assert "anthropic" in entry["scores_by_backend"]
    assert entry["scores_by_backend"]["anthropic"]["per_hypothesis_raw"] == "per hyp evaluation text"
    assert d["locked"]["overall_raw_by_backend"]["anthropic"] == "overall evaluation text"


def test_build_review_data_llm_scores_legacy_single_backend(run_with_data):
    """Legacy single RunScores (no dict wrapping) still works."""
    from reviewer.score import RunScores, HypothesisScores

    hs = HypothesisScores(
        hypothesis_id="H001",
        candidate_gene="SIGLEC1",
        per_hypothesis_raw="per hyp text",
    )
    run_scores = RunScores(hypothesis_scores=[hs], overall_raw="", run_dir=run_with_data)

    d = build_review_data(run_with_data, llm_scores=run_scores)
    assert d["locked"]["llm_scores"] is not None
    entry = d["locked"]["llm_scores"][0]
    assert "default" in entry["scores_by_backend"]
    assert entry["scores_by_backend"]["default"]["per_hypothesis_raw"] == "per hyp text"


def test_build_review_data_tool_diversity_top5(run_with_data):
    d = build_review_data(run_with_data)
    top5 = d["metrics"]["tool_diversity_top5"]
    assert isinstance(top5, list)
    # Should be sorted descending by count
    if len(top5) >= 2:
        counts = [c for _, c in top5]
        assert counts == sorted(counts, reverse=True)


# HTML rendering tests
from reviewer.review import render_html


@pytest.fixture
def minimal_data():
    return {
        "run_id": "test-run-001",
        "run_dir": "/tmp/test-run-001",
        "run_model": "claude-sonnet-4-6",
        "run_date": "2026-01-01",
        "total_tool_calls": 87,
        "metrics": {
            "hypothesis_count": 2,
            "belief_revision_count": 1,
            "refutation_rate": 0.5,
            "modality_coverage": {"H001": ["DEG", "GWAS"]},
            "tool_diversity_top5": [("Bash", 10), ("write_hypothesis", 3)],
            "completed": True,
        },
        "steps": [
            {"step_type": "deg_analysis", "tool_used": "scanpy",
             "inputs_summary": "Perez 2022", "outputs_summary": "IFN genes top",
             "interpretation": "IFN dominant", "next_action": "write hypothesis",
             "timestamp": "2026-01-01T00:00:00Z"},
        ],
        "supporting_files": [],
        "hypotheses": [
            {"hypothesis_id": "H001", "candidate_gene": "SIGLEC1", "cell_type": "pDC",
             "status": "active", "evidence": [{"type": "DEG", "description": "strong"}],
             "update_history": [], "created_at": "2026-01-01T00:00:00Z"},
            {"hypothesis_id": "H002", "candidate_gene": "KLRB1", "cell_type": "NK",
             "status": "refuted", "evidence": [], "update_history": [],
             "created_at": "2026-01-01T00:01:00Z"},
        ],
        "locked": {
            "rationale": "done", "confidence_summary": "0.8",
            "suggested_validation": "FACS sort",
            "llm_scores": None,
            "overall_raw_by_backend": None,
        },
        "task_spec": None,
        "rubric": DEFAULT_RUBRIC,
    }


def test_render_html_has_doctype(minimal_data):
    html = render_html(minimal_data)
    assert html.strip().startswith("<!DOCTYPE html>")


def test_render_html_contains_run_id(minimal_data):
    html = render_html(minimal_data)
    assert "test-run-001" in html


def test_render_html_shows_model_and_date(minimal_data):
    html = render_html(minimal_data)
    assert "claude-sonnet-4-6" in html
    assert "2026-01-01" in html
    assert "87" in html


def test_render_html_contains_metrics(minimal_data):
    html = render_html(minimal_data)
    assert "50%" in html or "50" in html   # refutation rate
    assert "2" in html                      # hypothesis count


def test_render_html_has_submit_button(minimal_data):
    html = render_html(minimal_data)
    assert "submitReview" in html


def test_render_html_locked_section_blurred(minimal_data):
    html = render_html(minimal_data)
    assert "locked-section" in html
    assert "blur" in html


def test_render_html_hypothesis_genes_present(minimal_data):
    html = render_html(minimal_data)
    assert "SIGLEC1" in html
    assert "KLRB1" in html


def test_render_html_rubric_radio_inputs(minimal_data):
    html = render_html(minimal_data)
    assert 'name="ovr_overall_quality"' in html
    assert 'name="ovr_wet_lab_worthy"' in html
    assert 'name="hyp_H001_novelty"' in html
    assert 'name="hyp_H001_evidence_quality"' in html


def test_render_html_status_label_active(minimal_data):
    html = render_html(minimal_data)
    assert "Should this hypothesis have been refuted" in html


def test_render_html_status_label_refuted(minimal_data):
    html = render_html(minimal_data)
    assert "Was this refutation/weakening warranted" in html


def test_render_html_js_injects_run_id(minimal_data):
    html = render_html(minimal_data)
    assert '"test-run-001"' in html


def test_render_html_no_agent_confidence_in_hypothesis_header(minimal_data):
    import re
    html = render_html(minimal_data)
    # Extract just the hypothesis-header divs (from <div class="hypothesis-header"> to </div>)
    header_blocks = re.findall(r'<div class="hypothesis-header">.*?</div>', html, re.DOTALL)
    for block in header_blocks:
        assert "0.8" not in block
        assert "0.2" not in block


def test_render_html_escapes_special_chars_in_gene(minimal_data):
    minimal_data["hypotheses"][0]["candidate_gene"] = 'IFI<44>&L"gene'
    html = render_html(minimal_data)
    # Check HTML body (before <script>) has escaped version
    script_start = html.index("<script>")
    html_body = html[:script_start]
    assert "IFI<44>" not in html_body
    assert "IFI&lt;44&gt;&amp;L" in html_body


def test_render_html_safe_json_escapes_script_tag(minimal_data):
    minimal_data["run_id"] = "run</script>alert"
    html = render_html(minimal_data)
    # The </script> sequence must not appear raw inside the <script> block
    script_start = html.index("<script>")
    script_end = html.index("</script>", script_start + 8)
    js_block = html[script_start:script_end]
    assert "</script>" not in js_block


@pytest.fixture
def minimal_data_with_spec(minimal_data):
    minimal_data["task_spec"] = {
        "task_id": "test_task",
        "goal": "Find the best biomarkers for disease X using multi-modal evidence.",
        "termination_criteria": {"min_hypotheses": 3, "max_tool_calls": 200},
        "rubric": [
            {"id": "novelty", "label": "Scientific Novelty", "scope": "per_hypothesis",
             "scale": "1-5", "guidance": "Is it novel?"},
            {"id": "quality", "label": "Overall Quality", "scope": "overall",
             "scale": "1-5", "guidance": "Is it high quality?"},
        ],
    }
    minimal_data["rubric"] = minimal_data["task_spec"]["rubric"]
    return minimal_data


def test_render_html_task_context_shown_when_spec_present(minimal_data_with_spec):
    html = render_html(minimal_data_with_spec)
    assert "Find the best biomarkers for disease X" in html
    assert "min_hypotheses" in html
    assert "max_tool_calls" in html


def test_render_html_no_task_context_when_spec_absent(minimal_data):
    html = render_html(minimal_data)
    assert "Find the best biomarkers" not in html


def test_render_html_dynamic_per_hypothesis_rubric(minimal_data_with_spec):
    html = render_html(minimal_data_with_spec)
    assert 'name="hyp_H001_novelty"' in html
    assert 'name="hyp_H002_novelty"' in html


def test_render_html_dynamic_overall_rubric(minimal_data_with_spec):
    html = render_html(minimal_data_with_spec)
    assert 'name="ovr_quality"' in html


def test_render_html_rubric_guidance_shown(minimal_data_with_spec):
    html = render_html(minimal_data_with_spec)
    assert "Is it novel?" in html
    assert "Is it high quality?" in html


def test_render_html_default_rubric_items_shown_without_spec(minimal_data):
    html = render_html(minimal_data)
    assert 'name="ovr_overall_quality"' in html
    assert 'name="ovr_wet_lab_worthy"' in html


def test_render_html_locked_shows_per_hypothesis_raw(minimal_data_with_spec):
    minimal_data_with_spec["locked"]["llm_scores"] = [{
        "hypothesis_id": "H001",
        "gene": "SIGLEC1",
        "scores_by_backend": {
            "anthropic": {"per_hypothesis_raw": "Detailed per-hypothesis evaluation from LLM."}
        },
    }]
    minimal_data_with_spec["locked"]["overall_raw_by_backend"] = {
        "anthropic": "Overall run evaluation."
    }
    html = render_html(minimal_data_with_spec)
    assert "Detailed per-hypothesis evaluation from LLM." in html
    assert "Overall run evaluation." in html


# CLI integration tests
import sys
import pytest
from reviewer.review import main as review_main


def test_main_writes_review_html(run_with_data, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["review", run_with_data])
    review_main()
    html_files = list(Path(run_with_data).glob("review_*.html"))
    assert len(html_files) == 1
    content = html_files[0].read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "SIGLEC1" in content
    assert "deg_analysis" in content


def test_main_creates_new_file_each_run(run_with_data, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["review", run_with_data])
    review_main()
    review_main()
    html_files = list(Path(run_with_data).glob("review_*.html"))
    assert len(html_files) == 2


def test_main_missing_arg_exits(monkeypatch):
    monkeypatch.setattr("sys.argv", ["review"])
    with pytest.raises(SystemExit) as exc:
        review_main()
    assert exc.value.code != 0


# Falsification section tests
from unittest.mock import patch
from reviewer.falsification import FalsificationResult, EvidenceAssessment
from reviewer.review import _render_falsification_section

MOCK_FALSIFICATION_RESULTS = [
    FalsificationResult(
        hypothesis_id="ABCD12",
        agent_confidence=0.85,
        observer_confidence=0.62,
        e_value_product=10.0,
        e_value_confidence=0.91,
        confidence_delta=-0.23,
        falsification_criteria=["Would effect disappear in controls?"],
        evidence_assessments=[
            EvidenceAssessment(source="DEG", e_value=5.0, reasoning="adj.p<0.01"),
            EvidenceAssessment(source="GWAS", e_value=2.0, reasoning="moderate"),
        ],
    )
]

def test_render_falsification_section_returns_html():
    html = _render_falsification_section(MOCK_FALSIFICATION_RESULTS)
    assert "<table" in html or "<div" in html

def test_render_falsification_section_contains_hypothesis_id():
    html = _render_falsification_section(MOCK_FALSIFICATION_RESULTS)
    assert "ABCD12" in html

def test_render_falsification_section_shows_confidence_values():
    html = _render_falsification_section(MOCK_FALSIFICATION_RESULTS)
    assert "0.85" in html  # agent
    assert "0.62" in html  # observer

def test_render_falsification_section_shows_delta():
    html = _render_falsification_section(MOCK_FALSIFICATION_RESULTS)
    assert "-0.23" in html

def test_generate_review_html_includes_falsification_when_results_provided():
    from reviewer.review import generate_review_html
    from reviewer.log_replay import RunMetrics
    metrics = RunMetrics(
        belief_revision_count=1, refutation_rate=0.0, modality_coverage={},
        tool_diversity={}, hypothesis_count=1, tool_call_count=5,
        completed=True, martingale_score=0.12, martingale_n=5, directional_flip_count=1,
        cost_usd=None, turn_count=10, revisions_per_hypothesis=1.0,
        tool_diversity_entropy=None, tool_diversity_entropy_ledger=None, tool_diversity_entropy_universe=None,
    )
    html = generate_review_html(
        run_dir="/fake",
        metrics=metrics,
        falsification_results=MOCK_FALSIFICATION_RESULTS,
    )
    assert "ABCD12" in html
    assert "observer" in html.lower()


def test_generate_review_html_propagates_overall_raw():
    from reviewer.review import generate_review_html
    from reviewer.log_replay import RunMetrics
    from reviewer.score import RunScores, HypothesisScores

    metrics = RunMetrics(
        belief_revision_count=1, refutation_rate=0.0, modality_coverage={},
        tool_diversity={}, hypothesis_count=1, tool_call_count=5,
        completed=True, martingale_score=0.0, martingale_n=3, directional_flip_count=0,
        cost_usd=None, turn_count=10, revisions_per_hypothesis=1.0,
        tool_diversity_entropy=None, tool_diversity_entropy_ledger=None, tool_diversity_entropy_universe=None,
    )
    hs = HypothesisScores(
        hypothesis_id="H001", candidate_gene="GENE1",
        per_hypothesis_raw="per hyp text",
    )
    run_scores = RunScores(hypothesis_scores=[hs], overall_raw="overall text", run_dir="/fake")
    html = generate_review_html(run_dir="/fake", metrics=metrics, scores={"anthropic": run_scores})
    assert "overall text" in html


def test_locked_section_renders_criterion_table(minimal_data):
    minimal_data["locked"]["llm_scores"] = [{
        "hypothesis_id": "H001",
        "gene": "SIGLEC1",
        "scores_by_backend": {
            "anthropic": {
                "per_hypothesis_raw": "",
                "criteria": [{"criterion_id": "novelty", "rationale": "Good", "score": "4/5", "agent_note": ""}],
            }
        },
    }]
    html = render_html(minimal_data)
    assert '<table class="criterion-table">' in html
    assert "novelty" in html


def test_locked_section_falls_back_to_raw(minimal_data):
    minimal_data["locked"]["llm_scores"] = [{
        "hypothesis_id": "H001",
        "gene": "SIGLEC1",
        "scores_by_backend": {
            "anthropic": {
                "per_hypothesis_raw": "raw text fallback",
                "criteria": [],
            }
        },
    }]
    html = render_html(minimal_data)
    assert "<pre" in html
    assert '<table class="criterion-table">' not in html


def test_contested_score_flagged(minimal_data):
    minimal_data["locked"]["llm_scores"] = [{
        "hypothesis_id": "H001",
        "gene": "SIGLEC1",
        "scores_by_backend": {
            "anthropic": {
                "per_hypothesis_raw": "",
                "criteria": [{"criterion_id": "novelty", "rationale": "High", "score": "4/5", "agent_note": ""}],
            },
            "google": {
                "per_hypothesis_raw": "",
                "criteria": [{"criterion_id": "novelty", "rationale": "Low", "score": "2/5", "agent_note": ""}],
            },
        },
    }]
    html = render_html(minimal_data)
    assert "score-contested" in html


def test_render_falsification_section_none_agent_confidence():
    result_no_conf = FalsificationResult(
        hypothesis_id="XY99",
        agent_confidence=None,
        observer_confidence=0.55,
        e_value_product=2.0,
        e_value_confidence=0.67,
        confidence_delta=0.05,
        falsification_criteria=[],
        evidence_assessments=[],
    )
    html = _render_falsification_section([result_no_conf])
    assert "N/A" in html
