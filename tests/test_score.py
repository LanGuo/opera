import pytest
from unittest.mock import patch
from hypothesis_ledger.workspace import Workspace
from reviewer.llm import MockLLMClient
from reviewer.score import (
    score_run, score_run_all_backends, RunScores, _parse_json_criteria,
    _parse_json_agent_notes, _evidence_text, _format_evidence_trail,
    _format_refuted_hypotheses, CriterionScore,
)

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


# ----- string-format evidence tests -----
# Gemini agents store evidence items as plain strings; Claude agents store dicts.
# Regression coverage for a real crash: score_run_all_backends silently swallowed
# 'str' object has no attribute 'get' for every backend on a Gemini-authored run,
# because _format_evidence_trail / _format_refuted_hypotheses assumed dicts.

def test_evidence_text_handles_string_item():
    assert _evidence_text("Upregulated in SLE monocytes") == ("unknown", "Upregulated in SLE monocytes")


def test_evidence_text_handles_dict_item():
    assert _evidence_text({"type": "scRNA-seq", "description": "DEG in monocytes"}) == (
        "scRNA-seq", "DEG in monocytes",
    )


def test_format_evidence_trail_handles_string_evidence():
    update_history = [{
        "changes": {
            "from": {"evidence": []},
            "to": {"evidence": ["Upregulated in SLE B cells (scRNA-seq)"]},
        },
    }]
    result = _format_evidence_trail(update_history)
    assert "Upregulated in SLE B cells" in result


def test_format_refuted_hypotheses_handles_string_evidence():
    all_hypotheses = {
        "H1": {
            "status": "refuted", "candidate_gene": "GENE1", "cell_type": "B cell",
            "evidence": ["No association found in follow-up cohort"],
        },
    }
    result = _format_refuted_hypotheses(all_hypotheses)
    assert "No association found" in result


def test_score_run_with_string_evidence_does_not_crash(tmp_path):
    """End-to-end: a Gemini-style run (string evidence items) must not crash score_run."""
    ws = Workspace(str(tmp_path / "gemini_style_run"))
    ws.write_final_report({
        "completed_at": "2026-01-01T00:00:00+00:00",
        "top_hypotheses": [
            {
                "hypothesis_id": "H001",
                "cell_type": "B cell",
                "candidate_gene": "IFI44L",
                "claim": "Upregulated in SLE B cells",
                "evidence": ["Upregulated in B cells of SLE patients (scRNA-seq)"],
                "confidence": 0.8,
                "open_questions": [],
                "status": "terminal",
                "update_history": [{
                    "rationale": "Literature support found",
                    "changes": {
                        "from": {"evidence": []},
                        "to": {"evidence": ["Upregulated in B cells of SLE patients (scRNA-seq)"]},
                    },
                }],
            }
        ],
        "suggested_validation": "Flow cytometry validation",
        "total_tool_calls": 12,
        "total_hypotheses_explored": 3,
    })
    mock_llm = MockLLMClient(WELL_FORMED_JSON)
    scores = score_run(str(tmp_path / "gemini_style_run"), client=mock_llm)
    assert len(scores.hypothesis_scores) == 1


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


@pytest.fixture
def completed_run_with_spec_and_rationale(tmp_path):
    """Like completed_run_with_spec, but with a non-empty update_history rationale
    (triggers pass-2) and a two-item per_hypothesis rubric matching RUBRIC_2's ids,
    with no overall-scope items (keeps call count to exactly pass-1 + pass-2)."""
    import yaml
    ws = Workspace(str(tmp_path / "run_with_rationale"))
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
                "update_history": [{"rationale": "GWAS hit raised confidence"}],
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
            {"id": "feasibility", "label": "Feasibility", "scope": "per_hypothesis",
             "scale": "1-5", "guidance": "Is it feasible?"},
        ],
    }
    ws.task_spec_path.write_text(yaml.dump(spec), encoding="utf-8")
    return str(tmp_path / "run_with_rationale")


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


# ----- _parse_json_criteria / _parse_json_agent_notes tests -----

RUBRIC_2 = [
    {"id": "novelty", "label": "Novelty", "scale": "1-5"},
    {"id": "feasibility", "label": "Feasibility", "scale": "1-5"},
]

WELL_FORMED_JSON = """\
{"criteria": [
  {"criterion_id": "novelty", "score": "4/5", "rationale": "This gene has not been previously reported in this context. Multi-modal evidence strengthens the claim."},
  {"criterion_id": "feasibility", "score": "3/5", "rationale": "The proposed experiment uses standard FACS protocols available in most labs."}
]}
"""


def test_parse_json_criteria_basic():
    result = _parse_json_criteria(WELL_FORMED_JSON, RUBRIC_2)
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


def test_parse_json_criteria_strips_markdown_code_fence():
    raw = "```json\n" + WELL_FORMED_JSON + "\n```"
    result = _parse_json_criteria(raw, RUBRIC_2)
    assert len(result) == 2
    assert result[0].score == "4/5"


def test_parse_json_criteria_skips_unknown_criterion_id():
    raw = """\
{"criteria": [
  {"criterion_id": "novelty", "score": "4/5", "rationale": "Real criterion."},
  {"criterion_id": "hallucinated_extra_id", "score": "2/5", "rationale": "Not in rubric."}
]}
"""
    result = _parse_json_criteria(raw, RUBRIC_2)
    assert len(result) == 1
    assert result[0].criterion_id == "novelty"


def test_parse_json_criteria_returns_empty_on_bad_input():
    assert _parse_json_criteria("", RUBRIC_2) == []
    assert _parse_json_criteria(WELL_FORMED_JSON, []) == []
    assert _parse_json_criteria("not json at all", RUBRIC_2) == []
    # The exact real-world failure this replaces: pass-2 restructures its entire
    # response into an unrelated numbered critique list instead of JSON.
    restructured_prose = """\
1. **Source validation:** The agent does not critically evaluate this claim.
2. **Novelty framing:** The core finding is pre-established literature.
"""
    assert _parse_json_criteria(restructured_prose, RUBRIC_2) == []


def test_parse_json_agent_notes_basic():
    raw = '{"agent_notes": [{"criterion_id": "novelty", "agent_note": "Agree with human reviewer."}]}'
    notes = _parse_json_agent_notes(raw)
    assert notes == {"novelty": "Agree with human reviewer."}


def test_parse_json_agent_notes_strips_markdown_code_fence():
    raw = '```json\n{"agent_notes": [{"criterion_id": "novelty", "agent_note": "Agree."}]}\n```'
    assert _parse_json_agent_notes(raw) == {"novelty": "Agree."}


def test_parse_json_agent_notes_returns_empty_on_bad_input():
    assert _parse_json_agent_notes("") == {}
    assert _parse_json_agent_notes("not json") == {}
    assert _parse_json_agent_notes('{"agent_notes": [{"agent_note": "no id given"}]}') == {}


# ----- score_run two-pass integration tests -----
# Regression coverage for the real failure mode this rewrite fixes: pass-2 is
# prompted to add an agreement note without touching pass-1's scores, but the old
# free-text design re-derived everything from pass-2's own (sometimes restructured)
# text. Now pass-2's JSON can only ever attach agent_note to pass-1's already-parsed
# CriterionScore objects -- it has no way to replace score or rationale at all.

def test_score_run_two_pass_survives_pass2_restructuring(completed_run_with_spec_and_rationale):
    pass2_restructured_response = """\
1. **Source validation:** The agent does not critically evaluate this claim.
My independent assessment differs on this point -- not valid JSON, no agent_notes key.
"""
    mock_llm = MockLLMClient([WELL_FORMED_JSON, pass2_restructured_response, WELL_FORMED_JSON])
    scores = score_run(completed_run_with_spec_and_rationale, client=mock_llm)
    criteria = scores.hypothesis_scores[0].criteria
    assert len(criteria) == 2
    assert criteria[0].score == "4/5"  # pass-1's score survives pass-2's malformed response
    assert criteria[0].agent_note == ""  # no valid note to attach, but nothing corrupted


def test_score_run_two_pass_attaches_agent_note_on_success(completed_run_with_spec_and_rationale):
    pass2_notes = '{"agent_notes": [{"criterion_id": "novelty", "agent_note": "Agrees with agent."}]}'
    mock_llm = MockLLMClient([WELL_FORMED_JSON, pass2_notes, WELL_FORMED_JSON])
    scores = score_run(completed_run_with_spec_and_rationale, client=mock_llm)
    criteria = scores.hypothesis_scores[0].criteria
    novelty = next(c for c in criteria if c.criterion_id == "novelty")
    assert novelty.score == "4/5"
    assert novelty.agent_note == "Agrees with agent."
