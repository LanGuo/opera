import pytest
from unittest.mock import patch
from hypothesis_ledger.workspace import Workspace
from reviewer.llm import MockLLMClient
from reviewer.score import score_run, score_run_all_backends, RunScores, _parse_criterion_scores, CriterionScore

MOCK_RESPONSE = "Score: 4/5\nOverall score: 16/20"


@pytest.fixture
def completed_run(tmp_path):
    ws = Workspace(str(tmp_path / "run_001"))
    ws.write_final_report({
        "completed_at": "2026-01-01T00:00:00+00:00",
        "top_hypotheses": [
            {
                "hypothesis_id": "H001",
                "cell_type": "LILRA4+ plasmacytoid dendritic cells",
                "candidate_gene": "SIGLEC1",
                "claim": "Upregulated in SLE pDCs; correlates with IFN score; druggable target",
                "evidence": [
                    {"source": "DEG_analysis", "strength": "strong", "log2fc": 2.1},
                    {"source": "GWAS", "strength": "moderate"},
                ],
                "confidence": 0.82,
                "open_questions": ["Elevated in remission?"],
                "status": "terminal",
                "update_history": [{"rationale": "GWAS hit raised confidence"}],
            }
        ],
        "rationale": "Converged after 3 modalities of evidence",
        "confidence_summary": "0.82 for SIGLEC1",
        "suggested_validation": (
            "FACS-sort LILRA4+ pDCs from 10 SLE donors and 10 healthy controls. "
            "Measure SIGLEC1 surface expression by flow cytometry. "
            "Correlate with IFN score from bulk RNA-seq."
        ),
        "total_tool_calls": 87,
        "total_hypotheses_explored": 5,
    })
    return str(tmp_path / "run_001")


def test_score_run_returns_scores(completed_run):
    mock_llm = MockLLMClient(MOCK_RESPONSE)
    scores = score_run(completed_run, client=mock_llm)
    assert isinstance(scores, RunScores)
    assert len(scores.hypothesis_scores) == 1
    assert scores.hypothesis_scores[0].hypothesis_id == "H001"


def test_score_run_calls_llm_twice_per_hypothesis(completed_run):
    mock_llm = MockLLMClient(MOCK_RESPONSE)
    score_run(completed_run, client=mock_llm)
    assert mock_llm.call_count == 2


def test_score_run_design_prompt_contains_gene(completed_run):
    mock_llm = MockLLMClient(MOCK_RESPONSE)
    score_run(completed_run, client=mock_llm)
    all_content = " ".join(m["content"] for call in mock_llm.calls for m in call)
    assert "SIGLEC1" in all_content


def test_score_run_novelty_prompt_contains_cell_type(completed_run):
    mock_llm = MockLLMClient(MOCK_RESPONSE)
    score_run(completed_run, client=mock_llm)
    all_content = " ".join(m["content"] for call in mock_llm.calls for m in call)
    assert "plasmacytoid" in all_content


def test_score_run_no_report_raises(tmp_path):
    Workspace(str(tmp_path / "empty_run"))
    with pytest.raises(FileNotFoundError):
        score_run(str(tmp_path / "empty_run"))


def test_score_run_falls_back_to_ledger_when_top_hypotheses_empty(tmp_path):
    ws = Workspace(str(tmp_path / "run_empty_top"))
    def add(data):
        data["hypotheses"] = {
            "H001": {
                "hypothesis_id": "H001",
                "cell_type": "monocyte", "candidate_gene": "SIGLEC1",
                "claim": "elevated", "status": "active",
                "evidence": [{"type": "scrnaseq_deg"}],
                "confidence": 0.7, "open_questions": [],
                "update_history": [], "created_at": "2026-01-01T00:00:00+00:00",
            },
        }
    ws.update_ledger(add)
    ws.write_final_report({"top_hypotheses": [], "suggested_validation": ""})
    mock_llm = MockLLMClient(MOCK_RESPONSE)
    scores = score_run(str(tmp_path / "run_empty_top"), client=mock_llm)
    assert len(scores.hypothesis_scores) == 1
    assert scores.hypothesis_scores[0].hypothesis_id == "H001"


def test_score_run_stores_raw_text(completed_run):
    mock_llm = MockLLMClient("Overall score: 18/20")
    scores = score_run(completed_run, client=mock_llm)
    assert "Overall score: 18/20" in scores.hypothesis_scores[0].per_hypothesis_raw


def test_score_run_has_overall_raw(completed_run):
    mock_llm = MockLLMClient(MOCK_RESPONSE)
    scores = score_run(completed_run, client=mock_llm)
    assert hasattr(scores, "overall_raw")
    assert isinstance(scores.overall_raw, str)


# ----- score_run_all_backends tests -----

def test_score_run_all_backends_returns_empty_when_no_env(completed_run, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    result = score_run_all_backends(completed_run)
    assert result == {}


def test_score_run_all_backends_returns_one_entry_per_backend(completed_run, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    mock_llm = MockLLMClient(MOCK_RESPONSE)

    # Patch LLMClient constructor to return our mock for anthropic
    with patch("reviewer.score.LLMClient", return_value=mock_llm):
        result = score_run_all_backends(completed_run)

    assert "anthropic" in result
    assert isinstance(result["anthropic"], RunScores)
    assert len(result["anthropic"].hypothesis_scores) == 1
    assert hasattr(result["anthropic"], "overall_raw")


def test_score_run_all_backends_multiple_backends(completed_run, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    mock_llm = MockLLMClient(MOCK_RESPONSE)

    with patch("reviewer.score.LLMClient", return_value=mock_llm):
        result = score_run_all_backends(completed_run)

    assert "anthropic" in result
    assert "google" in result
    assert len(result) == 2


def test_score_run_all_backends_skips_failed_backend(completed_run, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    mock_llm_good = MockLLMClient(MOCK_RESPONSE)
    mock_llm_bad = MockLLMClient(MOCK_RESPONSE)

    call_count = {"n": 0}

    def make_client(backend, model, **kwargs):
        call_count["n"] += 1
        if backend == "anthropic":
            return mock_llm_good
        # google backend construction succeeds, but chat raises
        raise RuntimeError("google backend construction failure")

    with patch("reviewer.score.LLMClient", side_effect=make_client):
        result = score_run_all_backends(completed_run)

    # anthropic should succeed; google construction failed so it's skipped
    assert "anthropic" in result
    assert "google" not in result


def test_score_run_all_backends_ollama(completed_run, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")

    mock_llm = MockLLMClient(MOCK_RESPONSE)

    with patch("reviewer.score.LLMClient", return_value=mock_llm):
        result = score_run_all_backends(completed_run)

    assert "ollama:llama3" in result
    assert isinstance(result["ollama:llama3"], RunScores)


# ----- task_spec-driven prompt path tests -----

@pytest.fixture
def completed_run_with_spec(tmp_path):
    import yaml
    ws = Workspace(str(tmp_path / "run_with_spec"))
    ws.write_final_report({
        "completed_at": "2026-01-01T00:00:00+00:00",
        "top_hypotheses": [
            {
                "hypothesis_id": "H001",
                "cell_type": "pDC",
                "candidate_gene": "SIGLEC1",
                "claim": "Upregulated in SLE pDCs",
                "evidence": [{"source": "DEG_analysis", "strength": "strong"}],
                "confidence": 0.8,
                "open_questions": [],
                "status": "terminal",
                "update_history": [],
            }
        ],
        "suggested_validation": "FACS sort pDCs",
        "total_tool_calls": 42,
        "total_hypotheses_explored": 3,
    })
    spec = {
        "task_id": "test_task",
        "goal": "Find biomarkers for disease X.",
        "termination_criteria": {"max_tool_calls": 50},
        "rubric": [
            {"id": "novelty", "label": "Novelty", "scope": "per_hypothesis",
             "scale": "1-5", "guidance": "Is it novel?"},
            {"id": "quality", "label": "Quality", "scope": "overall",
             "scale": "1-5", "guidance": "Is it high quality?"},
        ],
    }
    ws.task_spec_path.write_text(yaml.dump(spec), encoding="utf-8")
    return str(tmp_path / "run_with_spec")


def test_score_run_with_task_spec_uses_goal_in_prompt(completed_run_with_spec):
    mock_llm = MockLLMClient(MOCK_RESPONSE)
    scores = score_run(completed_run_with_spec, client=mock_llm)
    all_content = " ".join(m["content"] for call in mock_llm.calls for m in call)
    assert "Find biomarkers for disease X" in all_content


def test_score_run_with_task_spec_makes_two_calls(completed_run_with_spec):
    mock_llm = MockLLMClient(MOCK_RESPONSE)
    score_run(completed_run_with_spec, client=mock_llm)
    # 1 per-hypothesis call + 1 overall call = 2 total
    assert mock_llm.call_count == 2


def test_score_run_with_task_spec_overall_raw_populated(completed_run_with_spec):
    mock_llm = MockLLMClient("Quality score response")
    scores = score_run(completed_run_with_spec, client=mock_llm)
    assert "Quality score response" in scores.overall_raw


# ----- _parse_criterion_scores tests -----

RUBRIC_2 = [
    {"id": "novelty", "label": "Novelty", "scale": "1-5"},
    {"id": "feasibility", "label": "Feasibility", "scale": "1-5"},
]

WELL_FORMED_RAW = """\
1. Novelty
This gene has not been previously reported in this context. Multi-modal evidence strengthens the claim. The finding is genuinely surprising.
Score: 4/5

2. Feasibility
The proposed experiment uses standard FACS protocols available in most labs. Cohort sizes are realistic given typical SLE biobank availability. Reagents are commercially available.
Score: 3/5
"""


def test_parse_criterion_scores_basic():
    result = _parse_criterion_scores(WELL_FORMED_RAW, RUBRIC_2)
    assert len(result) == 2

    novelty = result[0]
    assert isinstance(novelty, CriterionScore)
    assert novelty.criterion_id == "novelty"
    assert novelty.score == "4/5"
    assert "not been previously reported" in novelty.rationale
    assert novelty.agent_note == ""

    feasibility = result[1]
    assert feasibility.criterion_id == "feasibility"
    assert feasibility.score == "3/5"
    assert "FACS protocols" in feasibility.rationale


def test_parse_criterion_scores_with_agent_note():
    raw = """\
1. Novelty
Strong evidence from three independent modalities. Largely agrees with prior literature. The gene is moderately novel.
Score: 3/5
Agent self-assessment note: Agree with human reviewer; slight underestimation possible.

2. Feasibility
Standard protocols apply here. Cohort is accessible. Timeline is reasonable.
Score: 4/5
"""
    result = _parse_criterion_scores(raw, RUBRIC_2)
    assert len(result) == 2
    assert "Agree with human reviewer" in result[0].agent_note
    assert result[1].agent_note == ""


def test_parse_criterion_scores_returns_empty_on_bad_input():
    assert _parse_criterion_scores("", RUBRIC_2) == []
    assert _parse_criterion_scores(WELL_FORMED_RAW, []) == []


def test_parse_preamble_before_numbered_blocks():
    raw = """\
Here is my evaluation of the two criteria below.

1. Novelty
This gene has not been previously reported. Evidence is strong.
Score: 4/5

2. Feasibility
Standard FACS protocols apply. Reagents available.
Score: 3/5
"""
    result = _parse_criterion_scores(raw, RUBRIC_2)
    assert len(result) == 2
    assert result[0].criterion_id == "novelty"
    assert result[0].score == "4/5"
    assert result[1].criterion_id == "feasibility"
    assert result[1].score == "3/5"


def test_parse_double_newline_fallback():
    raw = """\
This gene has not been previously reported. Multi-modal evidence.
Score: 4/5

Standard FACS protocols apply. Cohort sizes are realistic.
Score: 3/5
"""
    result = _parse_criterion_scores(raw, RUBRIC_2)
    assert len(result) == 2
    assert all(isinstance(cs, CriterionScore) for cs in result)
    assert result[0].score == "4/5"
    assert result[1].score == "3/5"


def test_parse_markdown_bold_score_line():
    raw = """\
1. Novelty
This gene has not been previously reported. Multi-modal evidence strengthens the claim.
**Score: 4/5**

2. Feasibility
Standard FACS protocols apply. Cohort sizes are realistic.
**Score:** 3/5
"""
    result = _parse_criterion_scores(raw, RUBRIC_2)
    assert len(result) == 2
    assert result[0].score == "4/5"
    assert result[1].score == "3/5"


def test_parse_markdown_header_numbered_blocks():
    raw = """\
## 1. Novelty
This gene has not been previously reported. Evidence is strong.
Score: 4/5

## 2. Feasibility
Standard FACS protocols apply. Reagents available.
Score: 3/5
"""
    result = _parse_criterion_scores(raw, RUBRIC_2)
    assert len(result) == 2
    assert result[0].criterion_id == "novelty"
    assert result[0].score == "4/5"
    assert "not been previously reported" in result[0].rationale
    assert result[1].criterion_id == "feasibility"
    assert result[1].score == "3/5"


def test_parse_markdown_agent_note_header():
    raw = """\
1. Novelty
Strong evidence from three independent modalities. Largely agrees with prior literature.
Score: 3/5
**Agent self-assessment note:** Agree with human reviewer; slight underestimation possible.

2. Feasibility
Standard protocols apply here. Cohort is accessible.
Score: 4/5
"""
    result = _parse_criterion_scores(raw, RUBRIC_2)
    assert len(result) == 2
    assert result[0].score == "3/5"
    assert "Agree with human reviewer" in result[0].agent_note
    assert result[1].agent_note == ""


def test_parse_missing_score_line_returns_partial():
    raw = """\
1. Novelty
This gene has not been previously reported. Multi-modal evidence.
Score: 4/5

2. Feasibility
Standard FACS protocols apply but no score provided here.
"""
    result = _parse_criterion_scores(raw, RUBRIC_2)
    assert len(result) == 2
    novelty = next(cs for cs in result if cs.criterion_id == "novelty")
    feasibility = next(cs for cs in result if cs.criterion_id == "feasibility")
    assert novelty.score == "4/5"
    assert feasibility.score == ""
    assert feasibility.rationale != ""
