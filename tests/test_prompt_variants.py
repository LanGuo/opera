import json
from dataclasses import fields

import pytest
import yaml

from hypothesis_ledger.workspace import Workspace
from reviewer.falsification import assess_hypothesis
from reviewer.llm import MockLLMClient
from reviewer.prompt_variants import (
    OBSERVER_METHOD, OBSERVER_SHAPE, VARIANTS, PromptVariant, get_variant, observer_system,
)
from reviewer.score import score_run

RUBRIC = [
    {"id": "novelty", "label": "Novelty", "scope": "per_hypothesis", "scale": "1-5", "guidance": "NOVELTY-GUIDANCE"},
    {"id": "overall_quality", "label": "Overall", "scope": "overall", "scale": "1-5", "guidance": "OVERALL-GUIDANCE"},
]
JUDGE_RESPONSE = json.dumps({"criteria": [
    {"criterion_id": "novelty", "score": "3/5", "rationale": "r"},
    {"criterion_id": "overall_quality", "score": "3/5", "rationale": "r"},
]})
HYP = {
    "hypothesis_id": "H1", "cell_type": "pDC", "candidate_gene": "SIGLEC1", "claim": "up in SLE",
    "confidence": 0.8, "status": "terminal",
    "evidence": [{"source": "DEG", "description": "EVIDENCE-TEXT adj.p = 0.005"}],
    "update_history": [{"rationale": "AGENT-RATIONALE-TEXT"}],
}


@pytest.fixture
def run_dir(tmp_path):
    ws = Workspace(str(tmp_path / "run"))
    ws.write_final_report({"top_hypotheses": [HYP], "rationale": "TERMINATION-TEXT",
                           "confidence_summary": "CONFIDENCE-TEXT"})
    ws.update_ledger(lambda d: d.__setitem__("hypotheses", {"H1": HYP}))
    (tmp_path / "run" / "task_spec.yaml").write_text(yaml.safe_dump({"goal": "GOAL-TEXT", "rubric": RUBRIC}))
    return str(tmp_path / "run")


def _judge_prompts(run_dir, variant):
    client = MockLLMClient(JUDGE_RESPONSE)
    score_run(run_dir, client=client, prompt_variant=variant)
    # call order: per-hypothesis pass 1, pass 2, overall pass 1, pass 2
    return [msgs[0]["content"] for msgs in client.calls]


def test_variants_present_and_complete():
    assert set(VARIANTS) == {"original", "P0", "P1", "P2"}
    for v in VARIANTS.values():
        assert all(getattr(v, f.name) for f in fields(PromptVariant))


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="original"):
        get_variant("P9")


def test_original_keeps_the_doubled_brace_defect():
    # Production wording is preserved exactly, defect included, so the baseline matches the paper.
    assert '{{"criteria"' in get_variant("original").judge_pass1_format


def test_p0_only_fixes_the_braces():
    orig, p0 = get_variant("original"), get_variant("P0")
    for f in fields(PromptVariant):
        if f.name == "judge_pass1_format":
            assert p0.judge_pass1_format == orig.judge_pass1_format.replace("{{", "{").replace("}}", "}")
        else:
            assert getattr(p0, f.name) == getattr(orig, f.name), f.name


@pytest.mark.parametrize("name", ["P0", "P1", "P2"])
def test_variants_keep_json_contract_and_single_braces(name):
    v = get_variant(name)
    for key in ('"criteria"', '"criterion_id"', '"score"', '"rationale"', "2-3 sentences", '"4/5"'):
        assert key in v.judge_pass1_format
    for key in ('"agent_notes"', '"criterion_id"', '"agent_note"', "1-2 sentences"):
        assert key in v.judge_pass2_format
    assert "{{" not in v.judge_pass1_format and "}}" not in v.judge_pass1_format


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_judge_prompts_keep_fixed_content(run_dir, name):
    v = get_variant(name)
    p1_hyp, p2_hyp, p1_overall, p2_overall = _judge_prompts(run_dir, name)
    for p in (p1_hyp, p1_overall):
        assert "Task goal: GOAL-TEXT" in p
        assert v.judge_role in p and v.judge_pass1_format in p
    assert "EVIDENCE-TEXT" in p1_hyp and "criterion_id: novelty" in p1_hyp and "NOVELTY-GUIDANCE" in p1_hyp
    assert "criterion_id: overall_quality" in p1_overall and "OVERALL-GUIDANCE" in p1_overall
    assert p2_hyp.startswith(v.judge_pass2_hyp_lead)
    assert v.judge_pass2_hyp_task in p2_hyp
    assert "Agent's rationale for the final belief update:\nAGENT-RATIONALE-TEXT" in p2_hyp
    assert p2_hyp.endswith(v.judge_pass2_format)
    assert p2_overall.startswith(v.judge_pass2_overall_lead) and v.judge_pass2_overall_task in p2_overall
    assert "Agent's termination rationale:\nTERMINATION-TEXT" in p2_overall
    assert "Agent's confidence summary:\nCONFIDENCE-TEXT" in p2_overall


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_observer_prompt_keeps_method_block(name):
    client = MockLLMClient(json.dumps({"observer_confidence": 0.7, "evidence_assessments": [],
                                       "falsification_criteria": ["a", "b", "c"]}))
    assess_hypothesis(HYP, client, prompt_variant=name)
    system, user = client.calls[0]
    assert system["content"] == observer_system(get_variant(name))
    assert system["content"].endswith(OBSERVER_SHAPE + " " + OBSERVER_METHOD)
    assert user["content"].startswith("Hypothesis: SIGLEC1 in pDC")


def test_default_variant_is_original(run_dir):
    assert _judge_prompts(run_dir, "original") == [
        m[0]["content"] for m in (lambda c: (score_run(run_dir, client=c), c.calls)[1])(MockLLMClient(JUDGE_RESPONSE))
    ]
