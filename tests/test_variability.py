import json
import pytest
import yaml

from hypothesis_ledger.workspace import Workspace
from reviewer.llm import MockLLMClient
from reviewer.variability import (
    RecordingClient, run_repeats, summarize, load_records, load_human_scores, _numeric, _ordering,
)


RUBRIC = [
    {"id": "novelty", "label": "Novelty", "scope": "per_hypothesis", "scale": "1-5", "guidance": "g"},
    {"id": "overall_quality", "label": "Overall", "scope": "overall", "scale": "1-5", "guidance": "g"},
]


def _judge_response(score):
    return json.dumps({"criteria": [
        {"criterion_id": "novelty", "score": f"{score}/5", "rationale": "r"},
        {"criterion_id": "overall_quality", "score": f"{score}/5", "rationale": "r"},
    ]})


OBSERVER_RESPONSE = json.dumps({
    "observer_confidence": 0.7,
    "evidence_assessments": [{"source": "DEG", "e_value": 2.0, "reasoning": "x"}],
    "falsification_criteria": ["c1"],
})


@pytest.fixture
def run_dir(tmp_path):
    ws = Workspace(str(tmp_path / "run_A"))
    hyp = {
        "hypothesis_id": "H1", "cell_type": "pDC", "candidate_gene": "SIGLEC1",
        "claim": "up in SLE", "confidence": 0.8, "status": "terminal",
        "evidence": [{"source": "DEG", "description": "adj.p = 0.005"}],
        "update_history": [{"rationale": "DEG raised confidence"}],
    }
    ws.write_final_report({"top_hypotheses": [hyp], "rationale": "done", "confidence_summary": "0.8"})
    ws.update_ledger(lambda d: d.__setitem__("hypotheses", {"H1": hyp}))
    (tmp_path / "run_A" / "task_spec.yaml").write_text(yaml.safe_dump({"goal": "g", "rubric": RUBRIC}))
    return str(tmp_path / "run_A")


def _factory(scores_by_backend=None):
    scores_by_backend = scores_by_backend or {"anthropic": 3, "google": 4}

    def make(component, backend, temperature):
        if component == "observer":
            return MockLLMClient(OBSERVER_RESPONSE)
        return MockLLMClient(_judge_response(scores_by_backend[backend]))
    return make


# --- RecordingClient ---

def test_recording_client_records_each_call():
    rec = RecordingClient(MockLLMClient("hello"))
    assert rec.chat([{"role": "user", "content": "a"}]) == "hello"
    rec.chat([{"role": "user", "content": "b"}])
    assert len(rec.calls) == 2
    first = rec.calls[0]
    assert first["response"] == "hello"
    assert first["model_returned"] == "mock"
    assert len(first["prompt_sha256"]) == 64
    assert rec.calls[0]["prompt_sha256"] != rec.calls[1]["prompt_sha256"]


def test_recording_client_same_prompt_same_hash():
    rec = RecordingClient(MockLLMClient("x"))
    rec.chat([{"role": "user", "content": "same"}])
    rec.chat([{"role": "user", "content": "same"}])
    assert rec.calls[0]["prompt_sha256"] == rec.calls[1]["prompt_sha256"]


# --- run_repeats ---

def test_run_repeats_writes_one_record_per_unit(run_dir, tmp_path):
    out = tmp_path / "out.jsonl"
    n = run_repeats([run_dir], repeats=2, out_path=str(out), client_factory=_factory(), workers=1)
    records = load_records(str(out))
    # 2 judge backends x 2 repeats + 1 observer x 2 repeats
    assert n == 6 and len(records) == 6
    judge = [r for r in records if r["component"] == "judge"]
    assert {r["backend"] for r in judge} == {"anthropic", "google"}
    assert all(r["ok"] for r in records)
    assert all(r["arm"] == "default" and r["temperature"] is None for r in records)
    assert judge[0]["result"]["hypotheses"][0]["criteria"][0]["criterion_id"] == "novelty"
    assert judge[0]["calls"], "raw calls should be recorded"
    assert judge[0]["models_returned"] == ["mock"]


def test_run_repeats_is_resumable(run_dir, tmp_path):
    out = str(tmp_path / "out.jsonl")
    run_repeats([run_dir], repeats=2, out_path=out, client_factory=_factory(), workers=1)
    assert run_repeats([run_dir], repeats=2, out_path=out, client_factory=_factory(), workers=1) == 0
    assert run_repeats([run_dir], repeats=3, out_path=out, client_factory=_factory(), workers=1) == 3
    assert len(load_records(out)) == 9


def test_run_repeats_records_failures_and_retries_them(run_dir, tmp_path):
    out = str(tmp_path / "out.jsonl")

    class Boom:
        last_model = None
        def chat(self, messages, schema=None):
            raise RuntimeError("api down")

    def failing(component, backend, temperature):
        return Boom() if backend == "google" and component == "judge" else _factory()(component, backend, temperature)

    run_repeats([run_dir], repeats=1, out_path=out, client_factory=failing, workers=1)
    bad = [r for r in load_records(out) if not r["ok"]]
    assert len(bad) == 1 and "api down" in bad[0]["error"]
    # A later invocation retries only the failed unit.
    assert run_repeats([run_dir], repeats=1, out_path=out, client_factory=_factory(), workers=1) == 1


def test_run_repeats_arm_name_reflects_temperature(run_dir, tmp_path):
    out = str(tmp_path / "out.jsonl")
    run_repeats([run_dir], repeats=1, out_path=out, client_factory=_factory(), workers=1, temperature=0.0)
    assert {r["arm"] for r in load_records(out)} == {"temp0.0"}


def test_run_repeats_observer_only(run_dir, tmp_path):
    out = str(tmp_path / "out.jsonl")
    run_repeats([run_dir], repeats=2, out_path=out, client_factory=_factory(), workers=1,
                components=("observer",))
    recs = load_records(out)
    assert len(recs) == 2 and all(r["component"] == "observer" for r in recs)
    assert recs[0]["result"][0]["observer_confidence"] == pytest.approx(0.7)


# --- helpers ---

@pytest.mark.parametrize("raw, expected", [("4/5", 4), ("4", 4), (" 2 / 5", 2), ("yes", None), ("", None)])
def test_numeric(raw, expected):
    assert _numeric(raw) == expected


def test_ordering_groups_ties_alphabetically():
    assert _ordering({"IFI44L": 14, "SIGLEC1": 10, "GZMH": 10}) == "IFI44L>GZMH=SIGLEC1"
    assert _ordering({"A": 3, "B": 5}) == "B>A"


def test_load_human_scores(tmp_path):
    (tmp_path / "human_review_x.json").write_text(json.dumps({
        "per_hypothesis": {"H1": {"novelty": 2, "status_decision": "partial", "notes": "n"}},
    }))
    assert load_human_scores(str(tmp_path)) == {"H1": {"novelty": 2}}
    assert load_human_scores(str(tmp_path / "missing")) == {}


# --- summarize ---

def _judge_rec(backend, repeat, scores, run_id="run_A", arm="default"):
    """scores: {gene: novelty_score}"""
    return {
        "run_id": run_id, "run_dir": "/x", "component": "judge", "backend": backend,
        "arm": arm, "repeat": repeat, "ok": True, "models_returned": ["m"],
        "result": {
            "hypotheses": [
                {"hypothesis_id": g, "candidate_gene": g,
                 "criteria": [{"criterion_id": "novelty", "score": f"{s}/5", "rationale": "", "agent_note": ""}]}
                for g, s in scores.items()
            ],
            "overall_criteria": [],
        },
    }


def _obs_rec(repeat, observer, agent=0.8, recomputed=0.9, hid="H1"):
    return {
        "run_id": "run_A", "run_dir": "/x", "component": "observer", "backend": "anthropic",
        "arm": "default", "repeat": repeat, "ok": True, "models_returned": ["m"],
        "result": [{"hypothesis_id": hid, "agent_confidence": agent, "observer_confidence": observer,
                    "llm_recomputed_confidence": recomputed}],
    }


def test_summarize_criterion_spread():
    recs = [_judge_rec("anthropic", i, {"G1": s}) for i, s in enumerate([3, 3, 5])]
    row = summarize(recs)["criterion_spread"][0]
    assert row["n"] == 3 and row["mean"] == pytest.approx(11 / 3)
    assert row["min"] == 3 and row["max"] == 5


def test_summarize_rankings_report_frequency_and_stability():
    recs = [_judge_rec("anthropic", i, {"A": 5, "B": 3}) for i in range(9)]
    recs.append(_judge_rec("anthropic", 9, {"A": 3, "B": 3}))
    row = summarize(recs)["rankings"][0]
    assert row["orderings"] == {"A>B": 9, "A=B": 1}
    assert row["top_share"] == pytest.approx(0.9) and row["stable"] is True


def test_summarize_leniency_pairs_by_repeat():
    recs = []
    for i in range(10):
        recs.append(_judge_rec("anthropic", i, {"G": 3}))
        recs.append(_judge_rec("google", i, {"G": 4 if i < 8 else 2}))
    row = summarize(recs)["leniency"][0]
    assert row["n_pairs"] == 10
    assert row["frac_google_higher"] == pytest.approx(0.8) and row["stable"] is False


def test_summarize_contested_rate():
    recs = []
    for i in range(4):
        recs.append(_judge_rec("anthropic", i, {"G": 2}))
        recs.append(_judge_rec("google", i, {"G": 4 if i < 3 else 3}))
    row = summarize(recs)["contested"][0]
    assert row["flag_rate"] == pytest.approx(0.75)


def test_summarize_human_mad():
    recs = [_judge_rec("anthropic", 0, {"H1": 3}), _judge_rec("google", 0, {"H1": 5})]
    human = {"run_A": {"H1": {"novelty": 2}}}
    s = summarize(recs, human_scores=human)
    by_backend = {r["backend"]: r for r in s["human_mad"]}
    assert by_backend["anthropic"]["mean"] == pytest.approx(1.0)
    assert by_backend["google"]["mean"] == pytest.approx(3.0)
    assert s["human_mad_gap"][0]["frac_anthropic_closer"] == pytest.approx(1.0)


def test_summarize_observer_sign_and_conservatism():
    recs = [_obs_rec(i, observer=0.9 if i < 9 else 0.7) for i in range(10)]
    row = summarize(recs)["observer"][0]
    assert row["n"] == 10
    assert row["frac_observer_above_agent"] == pytest.approx(0.9) and row["sign_stable"] is True
    assert row["frac_conservative"] == pytest.approx(0.1)


def test_summarize_ignores_failed_and_keeps_latest_duplicate():
    a = _judge_rec("anthropic", 0, {"G": 2})
    failed = dict(_judge_rec("anthropic", 0, {"G": 5}), ok=False)
    b = _judge_rec("anthropic", 0, {"G": 4})
    row = summarize([a, failed, b])["criterion_spread"][0]
    assert row["n"] == 1 and row["mean"] == 4


def test_stability_verdict_needs_minimum_repeats():
    recs = [_judge_rec("anthropic", i, {"A": 5, "B": 3}) for i in range(2)]
    row = summarize(recs)["rankings"][0]
    assert row["top_share"] == 1.0 and row["stable"] is None  # too few repeats to judge
    obs = summarize([_obs_rec(i, observer=0.9) for i in range(2)])["observer"][0]
    assert obs["sign_stable"] is None


def test_summarize_labels_hypotheses_with_gene_names():
    recs = []
    for i in range(2):
        a = _judge_rec("anthropic", i, {"G": 2})
        g = _judge_rec("google", i, {"G": 5})
        for r in (a, g):
            r["result"]["hypotheses"][0].update(hypothesis_id="H1", candidate_gene="SIGLEC1")
        recs += [a, g]
    recs.append(_obs_rec(0, observer=0.9, hid="H1"))
    s = summarize(recs)
    assert s["contested"][0]["candidate_gene"] == "SIGLEC1"
    assert s["observer"][0]["candidate_gene"] == "SIGLEC1"


def test_run_repeats_rejects_paths_that_are_not_run_dirs(run_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="ledger.json"):
        run_repeats([run_dir, str(tmp_path / "nope")], repeats=1, out_path=str(tmp_path / "o.jsonl"),
                    client_factory=_factory(), workers=1)
    assert not (tmp_path / "o.jsonl").exists()


def test_summarize_excludes_observer_parse_failures():
    good = _obs_rec(0, observer=0.9)
    bad = _obs_rec(1, observer=0.5)
    bad["result"][0]["parse_failed"] = True
    s = summarize([good, bad])
    assert s["observer"][0]["n"] == 1 and s["observer"][0]["mean"] == pytest.approx(0.9)
    assert s["health"][0]["observer_parse_failures"] == 1


# --- prompt variants ---

def test_run_repeats_records_prompt_variant_and_arm(run_dir, tmp_path):
    out = str(tmp_path / "out.jsonl")
    seen = []

    def factory(component, backend, temperature):
        c = _factory()(component, backend, temperature)
        seen.append(c)
        return c

    run_repeats([run_dir], repeats=1, out_path=out, client_factory=factory, workers=1,
                temperature=0.0, prompt_variant="P1")
    recs = load_records(out)
    assert {r["arm"] for r in recs} == {"temp0.0-P1"}
    assert {r["prompt_variant"] for r in recs} == {"P1"}
    from reviewer.prompt_variants import get_variant
    judge_prompt = seen[0].calls[0][0]["content"] if seen[0].calls[0][0]["role"] == "user" else ""
    assert any(get_variant("P1").judge_role in m[0]["content"] for c in seen for m in c.calls
               if m[0]["role"] == "user")


def test_original_variant_keeps_existing_arm_names(run_dir, tmp_path):
    out = str(tmp_path / "out.jsonl")
    run_repeats([run_dir], repeats=1, out_path=out, client_factory=_factory(), workers=1)
    assert {(r["arm"], r["prompt_variant"]) for r in load_records(out)} == {("default", "original")}


def test_unknown_prompt_variant_rejected_before_any_call(run_dir, tmp_path):
    with pytest.raises(ValueError, match="P9"):
        run_repeats([run_dir], repeats=1, out_path=str(tmp_path / "o.jsonl"),
                    client_factory=_factory(), workers=1, prompt_variant="P9")
    assert not (tmp_path / "o.jsonl").exists()


# --- compare ---

from reviewer.variability import compare


def test_compare_flags_scores_outside_baseline_range():
    base = [_judge_rec("anthropic", i, {"A": s, "B": 3}, arm="temp0.0") for i, s in enumerate([3, 4, 3])]
    same = [_judge_rec("anthropic", 0, {"A": 4, "B": 3}, arm="temp0.0-P1")]
    shifted = [_judge_rec("anthropic", 0, {"A": 5, "B": 1}, arm="temp0.0-P2")]
    c = compare(base + same + shifted, baseline_arm="temp0.0")
    rows = {r["arm"]: r for r in c["judge"]}
    assert rows["temp0.0-P1"]["n_criteria"] == 2 and rows["temp0.0-P1"]["n_outside"] == 0
    assert rows["temp0.0-P2"]["n_outside"] == 2
    outside = {(o["hypothesis_id"], o["criterion_id"]): o for o in rows["temp0.0-P2"]["outside"]}
    assert outside[("A", "novelty")]["baseline_range"] == [3, 4]
    assert outside[("A", "novelty")]["variant_values"] == [5]


def test_compare_observer_against_baseline_range():
    base = [_obs_rec(i, observer=v) for i, v in enumerate([0.70, 0.74])]
    for r in base:
        r["arm"] = "temp0.0"
    v = _obs_rec(0, observer=0.9)
    v["arm"] = "temp0.0-P1"
    rows = compare(base + [v], baseline_arm="temp0.0")["observer"]
    assert rows[0]["arm"] == "temp0.0-P1" and rows[0]["n_outside"] == 1


def test_records_carry_code_version(run_dir, tmp_path):
    import re
    out = str(tmp_path / "out.jsonl")
    run_repeats([run_dir], repeats=1, out_path=out, client_factory=_factory(), workers=1,
                components=("observer",))
    rec = load_records(out)[0]
    assert "code_commit" in rec and "code_dirty" in rec
    assert rec["code_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", rec["code_commit"])
