"""Judge input fixes: evidence text from any key set, evidence dropped by later
updates kept, weakened/refuted hypotheses scored, run metrics over all hypotheses, and the
overall prompt listing the hypotheses."""
import yaml

from tests.v1_run import V1Workspace as Workspace
from reviewer.evidence import all_evidence, evidence_parts, format_evidence_list
from reviewer.falsification import assess_hypothesis
from reviewer.llm import MockLLMClient
from reviewer.score import _format_evidence_trail, _run_metrics, score_run
from reviewer.variability import _rankings

# --- evidence_parts ---


def test_evidence_parts_string():
    assert evidence_parts("Upregulated in B cells") == ("unknown", "Upregulated in B cells")


def test_evidence_parts_reads_modality_and_detail_keys():
    kind, text = evidence_parts({"modality": "GWAS", "detail": "rs123 p=1e-9", "source": "GWAS Catalog"})
    assert kind == "GWAS"
    assert text.startswith("rs123 p=1e-9")
    assert "source: GWAS Catalog" in text


def test_evidence_parts_without_text_key_renders_other_fields():
    # Previously rendered as an empty description.
    kind, text = evidence_parts({"source": "DEG_analysis", "strength": "strong", "log2fc": 2.1})
    assert kind == "unknown"
    assert "source: DEG_analysis" in text and "strength: strong" in text and "log2fc: 2.1" in text


def test_evidence_trail_no_longer_blank_for_finding_key():
    history = [{"changes": {"from": {"evidence": []},
                            "to": {"evidence": [{"type": "literature", "finding": "Known autoantigen"}]}}}]
    assert "literature: Known autoantigen" in _format_evidence_trail(history)


# --- all_evidence ---

A = {"type": "scRNA-seq", "description": "DEG in monocytes"}
B = {"type": "literature", "description": "Prior cohort"}
C = {"type": "GWAS", "description": "Locus hit"}


def _hyp_with_overwrite(status="active"):
    # Creation evidence [A, B]; one update resent only [A, C], dropping B.
    return {
        "hypothesis_id": "H1", "cell_type": "monocyte", "candidate_gene": "G1", "claim": "c",
        "status": status, "confidence": 0.6, "open_questions": [],
        "evidence": [A, C],
        "update_history": [{"rationale": "added GWAS",
                            "changes": {"from": {"evidence": [A, B]}, "to": {"evidence": [A, C]}}}],
    }


def test_all_evidence_keeps_item_dropped_by_update():
    items = all_evidence(_hyp_with_overwrite())
    assert [i for i, _ in items] == [A, B, C]
    assert dict((i["description"], s) for i, s in items) == {
        "DEG in monocytes": "final", "Prior cohort": "dropped", "Locus hit": "final"}


def test_all_evidence_without_history_is_final_list():
    h = {"evidence": ["x", "y"], "update_history": []}
    assert all_evidence(h) == [("x", "final"), ("y", "final")]


def test_reworded_item_is_superseded_not_shown_twice():
    # Same kind, same facts reworded (pattern seen in saved runs).
    old = {"type": "gwas", "description": "IRF5 (OR~1.8, p~5e-56) and IRF7 (p~2e-9) are SLE GWAS risk genes "
                                          "driving IFN, inducing SIGLEC1 via ISRE promoter elements."}
    new = {"type": "gwas", "description": "IRF5 (p~5e-56, OR~1.8) and IRF7 (p~2e-9) are SLE GWAS risk genes "
                                          "driving type I IFN -> SIGLEC1 via ISRE promoter elements."}
    h = {"evidence": [new], "update_history": [
        {"changes": {"from": {"evidence": [old]}, "to": {"evidence": [new]}}}]}
    assert [s for _, s in all_evidence(h)] == ["superseded", "final"]
    text = format_evidence_list(h)
    assert text.count("IRF5") == 1 and "dropped" not in text


def test_different_kind_with_overlap_is_still_dropped():
    old = {"type": "scRNA-seq", "description": "G1 up in monocytes p=1e-5"}
    new = {"type": "literature", "description": "G1 up in monocytes p=1e-5 per review"}
    h = {"evidence": [new], "update_history": [
        {"changes": {"from": {"evidence": [old]}, "to": {"evidence": [new]}}}]}
    assert [s for _, s in all_evidence(h)] == ["dropped", "final"]


def test_format_evidence_list_marks_dropped_items():
    text = format_evidence_list(_hyp_with_overwrite())
    assert "Prior cohort [recorded earlier; not in the agent's final evidence list]" in text
    assert "1 of 3 items were dropped" in text


# --- run metrics ---


def _ledger(*hyps):
    return {h["hypothesis_id"]: h for h in hyps}


def _h(hid, status, n_updates=0):
    return {"hypothesis_id": hid, "cell_type": "t", "candidate_gene": hid, "claim": f"claim {hid}",
            "status": status, "evidence": ["e"], "confidence": 0.5, "open_questions": [],
            "update_history": [{"rationale": "r", "changes": {}}] * n_updates}


def test_run_metrics_counts_refuted_over_all_hypotheses():
    led = _ledger(_h("A", "terminal", 2), _h("B", "terminal"), _h("C", "weakened", 1), _h("D", "refuted", 1))
    m = _run_metrics(led, [])
    assert m["hypothesis_count"] == 4
    assert m["final_count"] == 2 and m["weakened_count"] == 1 and m["refuted_count"] == 1
    assert m["refutation_rate"] == 0.25 and m["weakened_rate"] == 0.25
    assert m["belief_revision_count"] == 4


# --- score_run end to end ---

SPEC = {
    "task_id": "t", "goal": "Find candidates",
    "rubric": [
        {"id": "evidence_strength", "label": "E", "scope": "per_hypothesis", "scale": "1-5", "guidance": "g"},
        {"id": "reasoning_quality", "label": "R", "scope": "overall", "scale": "1-5", "guidance": "g"},
    ],
}


def _run(tmp_path):
    ws = Workspace(str(tmp_path / "run"))
    final = _hyp_with_overwrite(status="terminal")
    led = _ledger(final, _h("WK", "weakened", 1), _h("RF", "refuted", 1))
    ws.update_ledger(lambda d: d.update({"hypotheses": led}))
    ws.write_final_report({"top_hypotheses": [final], "rationale": "", "confidence_summary": "",
                           "suggested_validation": ""})
    ws.task_spec_path.write_text(yaml.dump(SPEC), encoding="utf-8")
    return str(tmp_path / "run")


def _prompts(mock):
    return [" ".join(m["content"] for m in call) for call in mock.calls]


def test_score_run_scores_weakened_and_refuted_with_status(tmp_path):
    mock = MockLLMClient('{"criteria": []}')
    scores = score_run(_run(tmp_path), client=mock)
    got = {hs.hypothesis_id: (hs.status, hs.final_candidate) for hs in scores.hypothesis_scores}
    assert got == {"H1": ("terminal", True), "WK": ("weakened", False), "RF": ("refuted", False)}
    wk_prompt = next(p for p in _prompts(mock) if "claim WK" in p and "Hypothesis:" in p)
    assert "Status at end of run: weakened" in wk_prompt


def test_score_run_per_hypothesis_prompt_includes_dropped_evidence(tmp_path):
    mock = MockLLMClient('{"criteria": []}')
    score_run(_run(tmp_path), client=mock)
    h1_prompt = next(p for p in _prompts(mock) if "monocyte / G1" in p)
    assert "Prior cohort [recorded earlier" in h1_prompt


def test_score_run_overall_prompt_has_true_counts_and_hypothesis_list(tmp_path):
    mock = MockLLMClient('{"criteria": []}')
    score_run(_run(tmp_path), client=mock)
    overall = next(p for p in _prompts(mock) if "Run summary:" in p)
    assert "refuted: 1" in overall and "weakened: 1" in overall
    assert "Refutation rate: 33%" in overall
    assert "Hypotheses at the end of the run:" in overall
    assert "[final candidate]" in overall and "[weakened]" in overall and "[refuted]" in overall
    assert "0.6" not in overall  # no agent confidence numbers in pass 1


# --- observer and rankings ---


def test_observer_sees_evidence_dropped_by_update():
    mock = MockLLMClient('{"evidence_assessments": [], "observer_confidence": 0.5}')
    assess_hypothesis(_hyp_with_overwrite(), mock)
    assert "Prior cohort" in _prompts(mock)[0]


def test_rankings_ignore_non_final_hypotheses():
    def rec(final_flags):
        return {"run_id": "r", "arm": "a", "backend": "b", "result": {"hypotheses": [
            {"hypothesis_id": h, "candidate_gene": h, "final_candidate": f,
             "criteria": [{"criterion_id": "x", "score": f"{s}/5"}]}
            for h, s, f in final_flags]}}
    out = _rankings([rec([("A", 3, True), ("B", 4, True), ("Z", 5, False)])])
    assert out and all("Z" not in str(o) for o in out)


# --- length-neutral rule and recorded input size ---


def test_pass1_prompts_include_evidence_weighing_rule(tmp_path):
    mock = MockLLMClient('{"criteria": []}')
    score_run(_run(tmp_path), client=mock)
    pass1 = [p for p in _prompts(mock) if "Hypothesis:" in p or "Run summary:" in p]
    assert pass1 and all("not how many items there are or how long they are" in p for p in pass1)


def test_input_stats_recorded_per_hypothesis_and_overall(tmp_path):
    mock = MockLLMClient('{"criteria": []}')
    scores = score_run(_run(tmp_path), client=mock)
    h1 = next(hs for hs in scores.hypothesis_scores if hs.hypothesis_id == "H1")
    st = h1.input_stats
    assert st["evidence_items"] == 3 and st["dropped_items"] == 1 and st["superseded_items"] == 0
    assert st["evidence_chars"] > 0 and st["pass1_prompt_chars"] > st["evidence_chars"]
    assert st["rationale_chars"] == len("added GWAS")
    assert st["pass2_prompt_chars"] == 0  # pass 1 returned no criteria, so pass 2 was skipped
    o = scores.overall_input_stats
    assert o["hypotheses_listed"] == 3 and o["pass1_prompt_chars"] > o["hypothesis_table_chars"] > 0


def test_input_size_summary_reports_correlations_and_skips_old_records():
    from reviewer.variability import _input_size

    def rec(run, rows, with_stats=True):
        return {"run_id": run, "arm": "a", "backend": "b", "result": {"hypotheses": [
            {"hypothesis_id": hid, "criteria": [{"criterion_id": "x", "score": f"{sc}/5"}],
             **({"input_stats": {"evidence_items": it, "evidence_chars": ch, "rationale_chars": 10}}
                if with_stats else {})}
            for hid, sc, it, ch in rows]}}
    rows = [("h1", 1, 1, 100), ("h2", 2, 2, 250), ("h3", 3, 3, 300), ("h4", 4, 4, 500), ("h5", 5, 5, 600)]
    out = _input_size([rec("r1", rows), rec("old", rows, with_stats=False)])
    assert len(out) == 1 and out[0]["n_hypotheses"] == 5
    assert out[0]["score_vs_items"] > 0.99 and out[0]["score_vs_chars"] > 0.99
    assert out[0]["score_vs_rationale_chars"] is None  # constant input, undefined
