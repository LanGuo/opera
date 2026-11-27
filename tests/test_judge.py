"""The ledger judge (judging v2): inputs, rubric declarations, passes, metrics, results file.

Every test runs on the synthetic runs of tests/synthetic_run.py (with and without a conclusion),
and also on any real shake-down runs present in tests/fixtures/ (kept in the private repo only).
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from reviewer import judge as J
from reviewer.judge_inputs import (build_run_view, render_evidence_input, render_reasoning_input,
                                   render_run_input)
from reviewer.rubric import RubricError, load_criteria, missing_inputs

FIXTURES = Path(__file__).parent / "fixtures"
TASKS = Path(__file__).resolve().parent.parent / "tasks"
REAL_RUNS = sorted(p.name for p in FIXTURES.iterdir() if (p / "ledger.jsonl").exists()) if FIXTURES.is_dir() else []


@pytest.fixture(params=["synthetic_concluded", "synthetic_unconcluded", *REAL_RUNS])
def run(request, tmp_path):
    from tests.synthetic_run import write_synthetic_run
    dst = tmp_path / request.param
    if request.param.startswith("synthetic_"):
        return write_synthetic_run(dst, concluded=request.param == "synthetic_concluded")
    shutil.copytree(FIXTURES / request.param, dst)
    return dst


class FakeClient:
    """Answers each pass from its prompt: scores for the listed criteria; for pass R, assertions
    quoting the start of every rationale shown, plus one invented quote."""
    model, backend = "fake", "fake"

    def __init__(self, score=3, tag="E-RATIONALE-TOKEN"):
        self.score, self.tag, self.prompts, self.last_model = score, tag, [], "fake-1"

    def chat(self, messages, schema=None):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "Audit the reasoning" in prompt:
            entries = re.findall(r"Entry (L\d+) .*?\n\s+Rationale: (.+)", prompt)
            if not entries:
                m = re.search(r"Audit only the Conclusion rationale \(entry (L\d+)\)", prompt)
                rationale = re.search(r"\n  Rationale: (.+)", prompt)
                entries = [(m.group(1), rationale.group(1))] if m and rationale else []
            assertions = [{"entry_id": eid, "quote": text[:60], "label": "supported",
                           "evidence_ids": [], "reason": "shown"} for eid, text in entries]
            assertions.append({"entry_id": entries[0][0] if entries else "L1",
                               "quote": "a sentence the agent never wrote", "label": "overstated"})
            dirs = [{"entry_id": eid, "evidence_direction": "up"} for eid, _ in entries[1:]]
            return json.dumps({"assertions": assertions, "entry_directions": dirs, "contradictions": []})
        ids = re.findall(r"- criterion_id: (\S+)", prompt)
        known = re.findall(r"\[(E\d+)\]", prompt)[:2]
        return json.dumps({"criteria": [{"criterion_id": c, "score": str(self.score), "rationale": self.tag,
                                         "evidence_ids": known + ["E999"]} for c in ids]})


# --- rubric ---------------------------------------------------------------------------------------

def test_task_specs_declare_pass_and_inputs():
    for spec_path in TASKS.glob("*/task_spec.yaml"):
        crits, warnings = load_criteria(yaml.safe_load(spec_path.read_text()))
        assert warnings == [] and crits, spec_path


def test_rubric_defaults_warn_and_bad_declarations_fail():
    crits, warnings = load_criteria({"rubric": [{"id": "a", "scope": "per_hypothesis"},
                                                {"id": "reasoning_quality", "scope": "overall"},
                                                {"id": "b", "scope": "overall"}]})
    assert [(c.judge_pass, c.defaulted) for c in crits] == [("evidence", True), ("reasoning", True), ("run", True)]
    assert len(warnings) == 3
    with pytest.raises(RubricError, match="doesn't read"):
        load_criteria({"rubric": [{"id": "a", "pass": "evidence", "inputs": ["rationales"]}]})
    with pytest.raises(RubricError, match="unknown input"):
        load_criteria({"rubric": [{"id": "a", "pass": "run", "scope": "overall", "inputs": ["vibes"]}]})
    with pytest.raises(RubricError, match="overall criteria only"):
        load_criteria({"rubric": [{"id": "a", "pass": "run", "scope": "per_hypothesis", "inputs": []}]})


def test_old_run_rubric_takes_current_declarations(run):
    # Runs made before criteria declared pass/inputs carry a rubric copy without them.
    spec = yaml.safe_load((run / "task_spec.yaml").read_text())
    for c in spec["rubric"]:
        c.pop("pass", None), c.pop("inputs", None)
    (run / "task_spec.yaml").write_text(yaml.safe_dump(spec))
    merged, notes = J._read_spec(run, TASKS)
    assert all("pass" in c and "inputs" in c for c in merged["rubric"]) and "taken from" in notes[0]
    current = {c["id"]: c for c in yaml.safe_load((TASKS / spec["task_id"] / "task_spec.yaml").read_text())["rubric"]}
    assert all(c["inputs"] == current[c["id"]]["inputs"] for c in merged["rubric"])
    spec["rubric"][0]["guidance"] = "changed"
    (run / "task_spec.yaml").write_text(yaml.safe_dump(spec))
    merged, notes = J._read_spec(run, TASKS)
    assert not any("pass" in c for c in merged["rubric"]) and "differs" in notes[0]


def test_optional_inputs_do_not_make_a_criterion_unassessable(run):
    crits, _ = load_criteria({"rubric": [
        {"id": "q", "scope": "overall", "pass": "run", "inputs": ["evidence", "conclusion?"]},
        {"id": "w", "scope": "overall", "pass": "run", "inputs": ["conclusion"]}]})
    view = build_run_view(run)
    view.conclusion = None
    assert missing_inputs(crits[0], view) == [] and crits[0].optional == ["conclusion"]
    assert missing_inputs(crits[1], view) == ["conclusion (the agent never called declare_done)"]


# --- inputs ----------------------------------------------------------------------------------------

def test_evidence_input_has_no_interpretation_and_marks_cuts(run):
    view = build_run_view(run)
    for hid, h in view.hypotheses.items():
        text = render_evidence_input(view, hid, include_validation=True)
        for e in h.entries:
            assert e.rationale[:80] not in text
        assert "Rationale:" not in text and not re.search(r"confidence \d|status (active|supported|weakened)", text)
        for ev in h.evidence.values():
            assert f"[{ev.evidence_id}]" in text and ev.finding in text
    joined = "\n".join(render_evidence_input(view, h) for h in view.hypotheses)
    long_outputs = [c for c in view.tool_calls.values() if c["tool_call_id"] in view.cited_call_ids
                    and len(c.get("output") or "") > 2000]
    assert bool(re.search(r"\[output cut at 2,000 of [\d,]+ characters\]", joined)) == bool(long_outputs)
    from reviewer.judge_inputs import cut
    assert cut("x" * 2500, 2000).endswith("[output cut at 2,000 of 2,500 characters]") and cut("ab", 2000) == "ab"


def test_reasoning_input_shows_every_rationale_beside_its_evidence(run):
    view = build_run_view(run)
    for hid, h in view.hypotheses.items():
        text = render_reasoning_input(view, hid)
        for e in h.entries:
            assert f"Entry {e.entry_id}" in text and e.rationale in text
            for eid, _ in e.cited:
                assert f"[{eid}]" in text


def test_run_input_and_not_assessable(run):
    view = build_run_view(run)
    text = render_run_input(view, ["notes", "tool_calls", "validation_plan"])
    assert all(hid in text for hid in view.hypotheses) and "Tool calls not cited" in text
    crits, _ = load_criteria(J._read_spec(run, TASKS)[0])
    design = next(c for c in crits if c.id == "experimental_design")
    view.conclusion = None
    assert missing_inputs(design, view, "H1") == ["validation_plan for H1 (no declare_done)"]


# --- passes ----------------------------------------------------------------------------------------

def test_judge_end_to_end_with_fake_backends(run):
    ledger_before = hashlib.sha256((run / "ledger.jsonl").read_bytes()).hexdigest()
    clients = {"a": FakeClient(score=2), "b": FakeClient(score=4)}
    out = J.judge_run(run, clients)
    assert hashlib.sha256((run / "ledger.jsonl").read_bytes()).hexdigest() == ledger_before
    assert Path(out["path"]).parent == run and Path(out["path"]).name.startswith("judge_")
    saved = json.loads(Path(out["path"]).read_text())
    r = saved["backends"]["a"]
    view = build_run_view(run)
    # pass E: every hypothesis, every per-hypothesis evidence criterion; unknown ids separated
    for hid in view.hypotheses:
        res = r["evidence"]["per_hypothesis"][hid]
        assert set(res) == {"novelty", "experimental_design", "evidence_strength", "clinical_relevance"}
        for cid, v in res.items():
            if cid == "experimental_design" and not view.validation_for(hid):
                assert v["status"] == "not_assessable" and "validation_plan" in v["missing"][0]
                continue
            assert v["status"] == "scored" and v["unknown_evidence_ids"] == ["E999"]
            assert set(v["evidence_ids"]) <= set(view.hypotheses[hid].evidence)
    # pass R: verbatim check rejects the invented quote; metrics computed
    for hid, rr in r["reasoning"]["per_hypothesis"].items():
        assert rr["status"] == "ok"
        assert [a["quote"] for a in rr["rejected_assertions"]] == ["a sentence the agent never wrote"]
        assert rr["metrics"]["supported_rate"] == 1.0
        assert rr["metrics"]["n_assertions"] == len(view.hypotheses[hid].entries)
    # independence: no reasoning-pass or reasoning-score prompt contains pass-E output
    for call in r["calls"]:
        if call["pass"] in ("reasoning", "reasoning_score"):
            assert "E-RATIONALE-TOKEN" not in call["prompt"]
        assert call["prompt"] and call["response"]
    concluded = view.conclusion is not None
    assert r["reasoning_scores"]["overall"]["reasoning_quality"]["status"] == "scored"   # conclusion?
    assert r["reasoning"]["run"]["status"] == ("ok" if concluded else "no_conclusion_rationale")
    assert r["run"]["overall_quality"]["status"] == "scored"
    assert r["run"]["wet_lab_worthy"]["status"] == ("scored" if concluded else "not_assessable")
    # contested: 2 vs 4 on every numeric criterion, in each pass
    assert {f["pass"] for f in saved["contested"]} == {"evidence", "reasoning", "run"}
    assert any(f["criterion_id"] == "novelty" and f["scores"] == {"a": 2.0, "b": 4.0} for f in saved["contested"])


def test_reasoning_metrics_calibration_and_disconfirmation(run):
    j = J.Judge(run, FakeClient())
    res = j.reasoning_pass()
    view = j.run
    for hid, h in view.hypotheses.items():
        m = res["per_hypothesis"][hid]["metrics"]
        moved = [c["moved"] for c in m["confidence_changes"]]
        assert len(moved) == len(h.entries) - 1
        # the fake says evidence always points up; calibration is the share that moved up
        if moved:
            assert m["calibration"] == pytest.approx(moved.count("up") / len(moved))
        contra = {eid for e in h.entries for eid, rel in e.cited if rel == "contradicts"}
        assert {c["evidence_id"] for c in m["contradicting_items"]} == contra
        for c in m["contradicting_items"]:
            assert c["handled"] == (c["acknowledged"] or c["followed_by_lower_confidence"]
                                    or c["followed_by_status_change"])


def test_no_reasoning_recorded_and_failed_calls_are_explicit(run):
    class Broken(FakeClient):
        def chat(self, messages, schema=None):
            raise ConnectionError("down")
    j = J.Judge(run, Broken())
    r = j.judge()
    assert all(v["status"] in ("no_score", "not_assessable")
               for h in r["evidence"]["per_hypothesis"].values() for v in h.values())
    assert any(v["status"] == "no_score" for h in r["evidence"]["per_hypothesis"].values() for v in h.values())
    assert all(rr["status"] == "call_failed" for rr in r["reasoning"]["per_hypothesis"].values())
    assert all("error" in c for c in r["calls"])
    j2 = J.Judge(run, FakeClient())
    for h in j2.run.hypotheses.values():
        for e in h.entries:
            e.rationale = ""
        h.notes = []
    assert all(v["status"] == "no_reasoning_recorded"
               for v in j2.reasoning_pass()["per_hypothesis"].values())


# --- falsification observer ----------------------------------------------------------------------

def test_observer_reads_the_pass_e_input_and_quantities(run):
    from reviewer.falsification import _ledger_item_e, assess_terminal_hypotheses
    from reviewer.llm import MockLLMClient
    client = MockLLMClient(json.dumps({"observer_confidence": 0.6, "evidence_assessments": [],
                                       "falsification_criteria": ["x"]}))
    results = assess_terminal_hypotheses(str(run), client)
    view = build_run_view(run)
    expected = [hid for hid, h in view.hypotheses.items() if h.final.status not in ("refuted", "abandoned")]
    assert [r.hypothesis_id for r in results] == expected
    for r, messages in zip(results, client.calls):
        assert messages[-1]["content"] == render_evidence_input(view, r.hypothesis_id)
        assert r.agent_confidence == view.hypotheses[r.hypothesis_id].final.confidence
    ev = next(iter(view.hypotheses.values())).evidence
    item = next(iter(ev.values()))
    item.quantities, item.finding = [{"name": "p_value", "value": 1e-12}], "no p in the text"
    assert _ledger_item_e(item) == 10.0
    item.quantities = {"padj": 0.2}
    assert _ledger_item_e(item) == 0.5


def test_observer_skips_abandoned_hypotheses(tmp_path):
    from reviewer.falsification import assess_terminal_hypotheses
    from reviewer.llm import MockLLMClient
    from tests.synthetic_run import write_synthetic_run
    run = write_synthetic_run(tmp_path / "r", concluded=True)
    lines = (run / "ledger.jsonl").read_text().splitlines()
    last = max(i for i, l in enumerate(lines)
               if json.loads(l).get("kind") == "hypothesis" and json.loads(l)["hypothesis_id"] == "H3")
    e = json.loads(lines[last])
    e["status"], e["confidence"] = "abandoned", 0.0
    lines[last] = json.dumps(e)
    (run / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    client = MockLLMClient(json.dumps({"observer_confidence": 0.6, "evidence_assessments": [],
                                       "falsification_criteria": ["x"]}))
    ids = [r.hypothesis_id for r in assess_terminal_hypotheses(str(run), client)]
    assert "H3" not in ids and "H1" in ids


# --- review page ---------------------------------------------------------------------------------

def test_review_page_shows_the_judge_inputs_and_results(run, monkeypatch, capsys):
    import sys
    from reviewer import review
    from reviewer.review_v2 import render_judge_inputs, render_judge_results
    out = J.judge_run(run, {"a": FakeClient(score=2), "b": FakeClient(score=4)})
    view = build_run_view(run)
    open_part = render_judge_inputs(run, "evidence")
    for h in view.hypotheses.values():
        assert all(e.rationale[:60] not in open_part for e in h.entries if e.rationale)
    results = render_judge_results(out["path"])
    assert "Contested" in results and all(hid in results for hid in view.hypotheses)
    first_prompt = out["backends"]["a"]["calls"][0]["prompt"]
    import html as _html
    assert _html.escape(first_prompt[:200]) in results          # saved prompts are on the page
    for k in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OLLAMA_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sys, "argv", ["review", str(run)])
    review.main()
    page = sorted(run.glob("review_*.html"))[-1].read_text()
    assert "What the evidence judge reads" in page and "What the reasoning judge reads" in page
    assert "Judge results (v2.1)" in page
    assert page.index("What the evidence judge reads") < page.index("locked-section")
    assert page.index("What the reasoning judge reads") > page.index("locked-section")


def test_responses_parsed_from_bare_lists_and_bad_json_retried(run):
    assert J.parse_response('```json\n[{"criterion_id": "a", "score": "3"}]\n```', "criteria") == {
        "criteria": [{"criterion_id": "a", "score": "3"}]}

    class Flaky(FakeClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        def chat(self, messages, schema=None):
            self.n += 1
            if self.n <= 2:
                assert (schema is not None) == (self.n == 1)  # the retry goes without the schema
            return '{"criteria": [{"criterion_id": ' if self.n == 1 else super().chat(messages, schema)
    j = J.Judge(run, Flaky())
    res = j.evidence_pass()
    first = j.calls[0]
    assert "error" not in first and len(first["unparsed_responses"]) == 1
    assert all(v["status"] in ("scored", "not_assessable") for v in next(iter(res["per_hypothesis"].values())).values())
    assert any(v["status"] == "scored" for v in next(iter(res["per_hypothesis"].values())).values())


def test_long_output_excerpt_shows_the_passage_with_the_items_values():
    """D-60: past the start, the judges see where the item's numbers and symbols occur."""
    from reviewer.judge_inputs import excerpt
    warnings = "Warning: transforming to str index\n" * 100
    table = "".join(f"GENE{i}: logFC=+0.{i:03d}\n" for i in range(200))
    out = warnings + table + "IRF7: SLE=4.043 vs normal=2.451, logFC=+1.592\n" + table
    x = excerpt(out, 2000, "IRF7 logFC=+1.592 (SLE=4.043 vs 2.451)")
    assert "IRF7: SLE=4.043 vs normal=2.451, logFC=+1.592" in x
    assert x.startswith("[excerpt of") and "not shown" in x and len(x) < 2300
    assert out.startswith(x.split("\n", 1)[1][:400])                     # the start is kept
    assert excerpt("short output", 2000, "IRF7 1.592") == "short output"
    # a fraction written as a percentage, and a symbol with a weak number
    out2 = warnings + "row SSB classical monocyte pct 0.112 p 0.0\n" + "x 0.0\n" * 500
    assert "SSB classical monocyte pct 0.112" in excerpt(out2, 2000, "SSB up in monocytes, 11.2% of cells")
    # nothing matched past the start: cut at the budget, as before
    assert excerpt(warnings * 2, 2000, "IRF7 1.592").endswith("[output cut at 2,000 of 7,000 characters]")
