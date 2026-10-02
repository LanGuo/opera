import json
import pytest
from reviewer.llm import MockLLMClient
from reviewer.falsification import (
    _extract_e_values_from_text,
    assess_hypothesis,
    assess_terminal_hypotheses,
    FalsificationResult,
    EvidenceAssessment,
)

# --- _extract_e_values_from_text ---

def test_extract_e_values_p_less_than_001():
    evidence = [{"description": "adj.p = 0.0005, log2FC = 2.3", "source": "DEG"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(10.0)

def test_extract_e_values_p_less_than_005():
    evidence = [{"description": "p < 0.03", "source": "GWAS"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(2.0)

def test_extract_e_values_supporting_text():
    evidence = [{"description": "strongly supports the hypothesis", "source": "literature"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(1.5)

def test_extract_e_values_contradicting_text():
    evidence = [{"description": "contradicts prior findings", "source": "literature"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(0.5)

def test_extract_e_values_neutral():
    evidence = [{"description": "gene was observed in the dataset", "source": "census"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(1.0)

def test_extract_e_values_product_across_items():
    evidence = [
        {"description": "adj.p = 0.0005", "source": "DEG"},   # e=10
        {"description": "p < 0.03", "source": "GWAS"},          # e=2
    ]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(20.0)

def test_extract_e_values_caps_product_at_100():
    evidence = [
        {"description": "adj.p = 0.0001", "source": "DEG"},   # e=10
        {"description": "adj.p = 0.0001", "source": "GWAS"},  # e=10
        {"description": "adj.p = 0.0001", "source": "PPI"},   # e=10 → product would be 1000, capped at 100
    ]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(100.0)

# --- assess_hypothesis ---

MOCK_OBSERVER_RESPONSE = json.dumps({
    "observer_confidence": 0.72,
    "evidence_assessments": [
        {"source": "DEG", "e_value": 5.0, "reasoning": "adj.p < 0.01 supports"},
        {"source": "GWAS", "e_value": 2.0, "reasoning": "moderate association"},
    ],
    "falsification_criteria": [
        "Does the effect disappear in healthy controls?",
        "Is the gene expression cell-type-specific?",
    ],
})

MOCK_HYPOTHESIS = {
    "hypothesis_id": "ABCD12",
    "cell_type": "pDC",
    "candidate_gene": "SIGLEC1",
    "claim": "elevated in SLE vs. control",
    "confidence": 0.85,
    "evidence": [
        {"source": "DEG", "description": "adj.p = 0.005, log2FC = 1.8"},
        {"source": "GWAS", "description": "p < 0.04, OR = 1.6"},
    ],
    "status": "terminal",
}

def test_assess_hypothesis_returns_falsification_result():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE))
    assert isinstance(result, FalsificationResult)
    assert result.hypothesis_id == "ABCD12"


def test_assess_hypothesis_observer_confidence():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE))
    assert result.observer_confidence == pytest.approx(0.72)


def test_assess_hypothesis_confidence_delta():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE))
    # agent=0.85, observer=0.72 → delta = -0.13
    assert result.confidence_delta == pytest.approx(-0.13, abs=0.01)


def test_assess_hypothesis_e_value_product_from_text():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE))
    # evidence: adj.p=0.005→e=5, p<0.04→e=2 → product=10
    assert result.e_value_product == pytest.approx(10.0)
    assert result.e_value_confidence == pytest.approx(10 / 11, abs=0.01)


def test_assess_hypothesis_falsification_criteria():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE))
    assert len(result.falsification_criteria) == 2


def test_assess_hypothesis_evidence_assessments():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE))
    assert len(result.evidence_assessments) == 2
    assert isinstance(result.evidence_assessments[0], EvidenceAssessment)
    assert result.evidence_assessments[0].source == "DEG"
    assert result.evidence_assessments[0].e_value == pytest.approx(5.0)


# --- observer vs. e-value confidence cross-check ---

MOCK_OBSERVER_RESPONSE_AGREEING = json.dumps({
    # evidence e_value_product = 10 (adj.p=0.005->5, p<0.04->2) -> e_value_confidence = 10/11 = 0.909
    # observer_confidence close to that -> should NOT be flagged
    "observer_confidence": 0.85,
    "evidence_assessments": [
        {"source": "DEG", "e_value": 5.0, "reasoning": "adj.p < 0.01 supports"},
        {"source": "GWAS", "e_value": 2.0, "reasoning": "moderate association"},
    ],
    "falsification_criteria": [],
})

MOCK_OBSERVER_RESPONSE_DIVERGING = json.dumps({
    # same evidence -> e_value_confidence = 0.909, but observer reports a wildly
    # different number -> should be flagged as arithmetically inconsistent
    "observer_confidence": 0.15,
    "evidence_assessments": [
        {"source": "DEG", "e_value": 5.0, "reasoning": "adj.p < 0.01 supports"},
        {"source": "GWAS", "e_value": 2.0, "reasoning": "moderate association"},
    ],
    "falsification_criteria": [],
})


def test_assess_hypothesis_confidence_not_flagged_when_agreeing():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_AGREEING))
    assert result.deterministic_delta == pytest.approx(0.85 - 10 / 11, abs=0.01)
    assert result.confidence_flagged is False


def test_assess_hypothesis_confidence_flagged_when_diverging():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_DIVERGING))
    assert result.deterministic_delta == pytest.approx(0.15 - 10 / 11, abs=0.01)
    assert result.confidence_flagged is True


def test_assess_hypothesis_diverging_prints_warning(capsys):
    assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_DIVERGING))
    captured = capsys.readouterr()
    assert "[falsification] WARNING" in captured.err
    assert "diverge" in captured.err


def test_assess_hypothesis_agreeing_prints_no_divergence_warning(capsys):
    assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_AGREEING))
    captured = capsys.readouterr()
    assert "diverge" not in captured.err


def test_assess_hypothesis_does_not_override_observer_confidence_when_flagged():
    """The cross-check is a flag only — it must never mutate observer_confidence."""
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_DIVERGING))
    assert result.confidence_flagged is True
    assert result.observer_confidence == pytest.approx(0.15)


# --- assess_terminal_hypotheses ---

def test_assess_terminal_hypotheses_includes_active_with_evidence(tmp_path):
    """Observer runs on terminal and active hypotheses; refuted and evidence-less are excluded."""
    from hypothesis_ledger.workspace import Workspace
    ws = Workspace(str(tmp_path / "run_001"))
    def add(data):
        data["hypotheses"] = {
            "T1": dict(MOCK_HYPOTHESIS, hypothesis_id="T1", status="terminal"),
            "A1": dict(MOCK_HYPOTHESIS, hypothesis_id="A1", status="active"),
            "R1": dict(MOCK_HYPOTHESIS, hypothesis_id="R1", status="refuted"),
            "E0": dict(MOCK_HYPOTHESIS, hypothesis_id="E0", status="active", evidence=[]),
        }
    ws.update_ledger(add)
    client = MockLLMClient(MOCK_OBSERVER_RESPONSE)
    results = assess_terminal_hypotheses(str(tmp_path / "run_001"), client)
    assessed_ids = {r.hypothesis_id for r in results}
    assert "T1" in assessed_ids
    assert "A1" in assessed_ids
    assert "R1" not in assessed_ids  # refuted excluded
    assert "E0" not in assessed_ids  # no evidence excluded


# --- aggregation self-consistency check (LLM's own e-values vs. its own confidence) ---
# Distinct from the AGREEING/DIVERGING fixtures above, which compare observer_confidence
# against an *independently, regex-derived* e-value read of the raw text. These fixtures
# instead ask a narrower, purely arithmetic question: given the e-values the model itself
# assigned in evidence_assessments, does E=prod(e_i), C=E/(1+E) match the confidence it
# itself reported?

MOCK_OBSERVER_RESPONSE_SELF_CONSISTENT = json.dumps({
    # own e-values multiply to E=10 -> C=10/11=0.909; self-reported confidence matches
    "observer_confidence": 0.91,
    "evidence_assessments": [
        {"source": "DEG", "e_value": 5.0, "reasoning": "adj.p < 0.01 supports"},
        {"source": "GWAS", "e_value": 2.0, "reasoning": "moderate association"},
    ],
    "falsification_criteria": [],
})

MOCK_OBSERVER_RESPONSE_SELF_INCONSISTENT = json.dumps({
    # same own e-values (E=10 -> C=0.909), but self-reported confidence is disconnected
    # from them entirely -- the model's final number doesn't follow from its own math
    "observer_confidence": 0.30,
    "evidence_assessments": [
        {"source": "DEG", "e_value": 5.0, "reasoning": "adj.p < 0.01 supports"},
        {"source": "GWAS", "e_value": 2.0, "reasoning": "moderate association"},
    ],
    "falsification_criteria": [],
})

MOCK_OBSERVER_RESPONSE_NO_ASSESSMENTS = json.dumps({
    "observer_confidence": 0.95,
    "evidence_assessments": [],
    "falsification_criteria": [],
})


def test_assess_hypothesis_llm_e_value_product_from_own_assessments():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_SELF_CONSISTENT))
    assert result.llm_e_value_product == pytest.approx(10.0)
    assert result.llm_recomputed_confidence == pytest.approx(10 / 11, abs=0.01)


def test_assess_hypothesis_not_flagged_when_self_consistent():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_SELF_CONSISTENT))
    assert result.aggregation_inconsistent is False
    assert abs(result.aggregation_delta) < 0.05


def test_assess_hypothesis_flagged_when_self_inconsistent():
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_SELF_INCONSISTENT))
    assert result.aggregation_inconsistent is True
    # 0.30 reported vs 0.909 implied by its own e-values
    assert result.aggregation_delta == pytest.approx(0.30 - 10 / 11, abs=0.01)


def test_assess_hypothesis_self_inconsistent_prints_warning(capsys):
    assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_SELF_INCONSISTENT))
    captured = capsys.readouterr()
    assert "arithmetically inconsistent" in captured.err


def test_assess_hypothesis_no_assessments_not_flagged():
    """Missing per-item breakdown is a data gap, not an arithmetic inconsistency."""
    result = assess_hypothesis(MOCK_HYPOTHESIS, MockLLMClient(MOCK_OBSERVER_RESPONSE_NO_ASSESSMENTS))
    assert result.aggregation_inconsistent is False


# --- broadened deterministic e-value extraction (percentages, effect sizes, caveats) ---

def test_extract_e_values_sensitivity_specificity_percentage():
    evidence = [{"description": "Sensitivity 98.7%, specificity 82.1% for diagnosis", "source": "clinical"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(3.0)  # both >=90... actually first match (sensitivity 98.7) wins


def test_extract_e_values_moderate_percentage():
    evidence = [{"description": "accuracy 75% in validation cohort", "source": "clinical"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(2.0)


def test_extract_e_values_interaction_score():
    evidence = [{"description": "STRING combined score 0.83, strong interaction", "source": "STRING"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(1.5)


def test_extract_e_values_effect_size_without_pvalue():
    evidence = [{"description": "odds ratio 2.4 for disease association", "source": "GWAS"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(2.0)


def test_extract_e_values_caution_caveat_not_caught_by_old_keywords():
    """Explicit caveats phrased as caution/non-specificity, not the original narrow wording."""
    evidence = [{"description": "CAUTION: marker is not SLE-specific, also elevated in COVID-19", "source": "lit"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(0.5)


def test_extract_e_values_orphan_druggability_caveat():
    evidence = [{"description": "orphan protein with no known natural substrate", "source": "GeneCards"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(0.5)


# --- negation-prefix substring bug: "insignificant" must not match "significant" ---

def test_extract_e_values_insignificant_not_treated_as_supporting():
    evidence = [{"description": "effect was insignificant across all cohorts", "source": "lit"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(0.5)  # correctly contradicting, not 1.5


def test_extract_e_values_unsupported_not_treated_as_supporting():
    evidence = [{"description": "the association was unsupported by replication", "source": "lit"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(0.5)


def test_extract_e_values_invalidated_not_treated_as_supporting():
    evidence = [{"description": "the prior finding was invalidated by a larger cohort", "source": "lit"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(0.5)


def test_extract_e_values_unconfirmed_not_treated_as_supporting():
    evidence = [{"description": "the interaction remains unconfirmed in vivo", "source": "lit"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(0.5)


def test_extract_e_values_suffix_still_matches_root_keyword():
    """The fix must not break legitimate suffix matches like support -> supports/supported."""
    evidence = [{"description": "this result strongly supports the hypothesis", "source": "lit"}]
    E = _extract_e_values_from_text(evidence)
    assert E == pytest.approx(1.5)


def test_assess_terminal_hypotheses_raises_without_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    from hypothesis_ledger.workspace import Workspace
    Workspace(str(tmp_path / "run_001"))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        assess_terminal_hypotheses(str(tmp_path / "run_001"))
