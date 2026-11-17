"""The run ledger (ledger.jsonl): recording, the agent's ledger tools, and readers."""
import json
import multiprocessing

import pytest

from hypothesis_ledger import server
from hypothesis_ledger.instructions import ledger_instructions
from hypothesis_ledger.ledger import (Ledger, StreamParser, artifact_entries, link_source,
                                      split_tool_name)
from hypothesis_ledger.workspace import Workspace

SLE_RULES = {
    "hypothesis_attributes": [{"name": "gene", "description": "HGNC symbol"},
                              {"name": "cell_type", "description": "cell type(s)"}],
    "hypothesis_constraints": [{"unique": ["gene"], "among": ["active", "supported", "weakened"]}],
}


def _w(path, records):
    with open(path, "a") as f:
        for r in records:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")


def claude_call(cid, name, args, output, is_error=False):
    return [
        {"type": "assistant", "message": {"id": "m" + cid, "content": [
            {"type": "tool_use", "id": cid, "name": name, "input": args}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": cid, "is_error": is_error,
             "content": [{"type": "text", "text": output}]}]}},
    ]


@pytest.fixture
def run(tmp_path):
    """A run dir with a run_info start entry declaring the SLE rules, and the server pointed at it."""
    led = Ledger(tmp_path)
    led.append([{"kind": "run_info", "phase": "start", "task_rules": SLE_RULES}], "harness")
    server._reset_for_tests(tmp_path)
    return tmp_path


def call(fn, *a, **kw):
    return json.loads(fn(*a, **kw))


def ev(finding, tool=None, match=None, relation="supports", **kw):
    src = {}
    if tool:
        src["tool"] = tool
    if match:
        src["match"] = match
    return {"finding": finding, "source": src, "relation": relation, **kw}


# --- stream recording ---

def test_split_tool_name_both_clis():
    assert split_tool_name("mcp__tooluniverse__execute_tool") == ("tooluniverse", "execute_tool")
    assert split_tool_name("mcp_hypothesis-ledger_write_hypothesis") == ("hypothesis-ledger", "write_hypothesis")
    assert split_tool_name("Bash") == (None, "Bash")


def test_claude_stream_records_calls_messages_and_skips_ledger_tools(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _w(t, [{"type": "system", "subtype": "init"},
           {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "plan it"}]}},
           {"type": "assistant", "message": {"content": [{"type": "text", "text": "Starting."}]}}])
    _w(t, claude_call("t1", "mcp__tooluniverse__execute_tool",
                      {"tool_name": "OpenTargets_x", "arguments": {"q": "GZMH"}}, "score 0.9"))
    _w(t, claude_call("t2", "mcp__hypothesis-ledger__log_note", {"kind": "plan"}, "ok"))
    _w(t, claude_call("t3", "Bash", {"command": "python deg.py"}, "boom", is_error=True))
    led = Ledger(tmp_path)
    assert led.sync_stream() == 4
    calls = led.entries("tool_call")
    assert [c["tool_call_id"] for c in calls] == ["T1", "T2"]
    assert calls[0]["tool"] == "OpenTargets_x" and calls[0]["via"] == "execute_tool"
    assert calls[0]["output"] == "score 0.9" and calls[0]["error"] is None
    assert calls[1]["tool"] == "Bash" and calls[1]["error"]
    msgs = led.entries("agent_message")
    assert [(m["visibility"], m["text"]) for m in msgs] == [("thinking", "plan it"), ("visible", "Starting.")]
    assert all(e["written_by"] == "harness" for e in calls + msgs)


def test_gemini_stream_joins_text_and_keeps_error_type(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _w(t, [{"type": "message", "role": "assistant", "content": "Hel", "delta": True},
           {"type": "message", "role": "assistant", "content": "lo", "delta": True},
           {"type": "tool_use", "tool_name": "run_shell_command", "tool_id": "g1",
            "parameters": {"command": "ls"}, "timestamp": "t0"},
           {"type": "tool_result", "tool_id": "g1", "status": "success", "output": "a b", "timestamp": "t1"},
           {"type": "tool_use", "tool_name": "Nope", "tool_id": "g2", "parameters": {}},
           {"type": "tool_result", "tool_id": "g2", "status": "error", "output": "not found",
            "error": {"type": "tool_not_registered", "message": "Tool \"Nope\" not found"}}])
    led = Ledger(tmp_path)
    led.sync_stream()
    assert [m["text"] for m in led.entries("agent_message")] == ["Hello"]
    c1, c2 = led.entries("tool_call")
    assert (c1["output"], c1["started"], c1["finished"]) == ("a b", "t0", "t1")
    assert c2["error_type"] == "tool_not_registered"


def test_partial_last_line_waits_and_unfinished_call_recorded_at_end(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _w(t, claude_call("t1", "Bash", {"command": "ls"}, "x")[:1])
    with open(t, "a") as f:
        f.write('{"type": "assist')          # being written
    led = Ledger(tmp_path)
    assert led.sync_stream() == 0           # no result yet, partial line ignored
    assert led.sync_stream(final=True) == 1
    [c] = led.entries("tool_call")
    assert c["output"] is None and "run ended" in c["error"]


def test_two_writers_never_record_a_call_twice(tmp_path):
    t = tmp_path / "transcript.jsonl"
    _w(t, claude_call("t1", "Bash", {"command": "a"}, "1"))
    harness, ledger_server = Ledger(tmp_path), Ledger(tmp_path)
    harness.sync_stream()
    _w(t, claude_call("t2", "Bash", {"command": "b"}, "2"))
    ledger_server.sync_stream()
    harness.sync_stream()
    assert [c["cli_call_id"] for c in Ledger(tmp_path).entries("tool_call")] == ["t1", "t2"]
    assert [c["tool_call_id"] for c in Ledger(tmp_path).entries("tool_call")] == ["T1", "T2"]


def _append_many(run_dir, n, tag):
    led = Ledger(run_dir)
    for i in range(n):
        led.append([{"kind": "note", "note_kind": "other", "text": f"{tag}{i}"}], "agent")


def test_concurrent_appends_from_processes_keep_ids_unique(tmp_path):
    procs = [multiprocessing.Process(target=_append_many, args=(tmp_path, 40, t)) for t in "ab"]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    entries = Ledger(tmp_path).entries()
    assert len(entries) == 80
    assert len({e["entry_id"] for e in entries}) == 80


# --- hypothesis entries ---

HYP_FIELDS = {"entry_id", "time", "kind", "written_by", "hypothesis_id", "action", "hypothesis",
              "attributes", "derived_from", "confidence", "status", "evidence",
              "evidence_withdrawn", "rationale", "open_questions"}


def test_create_and_update_entries_have_the_same_fields(run):
    _w(run / "transcript.jsonl", claude_call("t1", "Bash", {"command": "python deg.py GZMH"}, "lfc 2.1"))
    r = call(server._tool_write_hypothesis, "GZMH marks expanded CD8 clones",
             {"gene": "GZMH", "cell_type": "CD8 T"}, 0.6,
             [ev("GZMH lfc 2.1 in CD8", tool="Bash", match="GZMH")], "Strong DE, one modality")
    assert r["status"] == "recorded" and r["hypothesis_id"] == "H1"
    assert r["evidence"] == [{"evidence_id": "E1", "link": "exact", "tool_call_ids": ["T1"]}]
    r2 = call(server._tool_update_hypothesis, "H1", 0.7, "supported",
              [{"ref": "E1", "relation": "supports"}, ev("GWAS hit", citation_note=1)],
              "Second modality")
    assert r2["status"] == "recorded"
    create, update = Ledger(run).entries("hypothesis")
    assert set(create) == HYP_FIELDS and set(update) == HYP_FIELDS
    assert update["hypothesis"] == create["hypothesis"] and update["attributes"] == create["attributes"]
    assert update["evidence"][0] == {"ref": "E1", "relation": "supports"}
    assert update["evidence"][1]["evidence_id"] == "E2" and update["evidence"][1]["link"] == "none"
    assert update["evidence"][1]["extra"] == {"citation_note": 1}


def test_update_has_no_way_to_change_hypothesis_text():
    import inspect
    params = inspect.signature(server.update_hypothesis).parameters
    assert "hypothesis" not in params and "attributes" not in params and "updates" not in params


@pytest.mark.parametrize("kwargs, msg", [
    ({"evidence": []}, "evidence is required"),
    ({"rationale": "  "}, "rationale is required"),
    ({"confidence": 1.5}, "between 0 and 1"),
    ({"attributes": {"gene": "X"}}, "missing required attribute"),
    ({"attributes": {"gene": "X", "cell_type": "B", "tissue": "y"}}, "unknown attribute"),
    ({"evidence": [{"finding": "x", "source": {}}]}, "relation must be one of"),
    ({"evidence": [{"source": {}, "relation": "supports"}]}, "'finding'"),
    ({"evidence": [{"ref": "E9", "relation": "supports"}]}, "no evidence 'E9'"),
])
def test_write_hypothesis_rejects_with_fixable_message(run, kwargs, msg):
    args = dict(hypothesis="h", attributes={"gene": "X", "cell_type": "B"}, confidence=0.5,
                evidence=[ev("f")], rationale="r")
    args.update(kwargs)
    r = call(server._tool_write_hypothesis, **args)
    assert r["status"] == "rejected" and msg in r["error"]
    assert Ledger(run).entries("hypothesis") == []


def test_update_validation(run):
    call(server._tool_write_hypothesis, "h", {"gene": "X", "cell_type": "B"}, 0.5, [ev("f")], "r")
    bad = [("H9", 0.5, "active", [ev("f")], "r", None, "no hypothesis 'H9'"),
           ("H1", 0.5, "terminal", [ev("f")], "r", None, "status must be one of"),
           ("H1", 0.5, "active", [], "r", None, "evidence is required"),
           ("H1", 0.5, "active", [ev("f")], "r", ["E7"], "never cited")]
    for hid, c, st, e, why, wd, msg in bad:
        r = call(server._tool_update_hypothesis, hid, c, st, e, why, wd)
        assert r["status"] == "rejected" and msg in r["error"], msg
    ok = call(server._tool_update_hypothesis, "H1", 0.2, "weakened",
              [ev("contradicting result", relation="contradicts")], "artefact", ["E1"])
    assert ok["status"] == "recorded"
    assert Ledger(run).entries("hypothesis")[-1]["evidence_withdrawn"] == ["E1"]


def test_unique_gene_constraint_and_derived_from(run):
    attrs = {"gene": "GZMH", "cell_type": "CD8 T"}
    call(server._tool_write_hypothesis, "h1", attrs, 0.5, [ev("f")], "r")
    r = call(server._tool_write_hypothesis, "h2", {"gene": "gzmh", "cell_type": "NK"}, 0.5, [ev("f")], "r")
    assert r["status"] == "rejected" and "H1 is already" in r["error"] and "abandoned" in r["error"]
    call(server._tool_update_hypothesis, "H1", 0.3, "abandoned", [{"ref": "E1", "relation": "context"}],
         "re-scoping to NK cells")
    r = call(server._tool_write_hypothesis, "h2", {"gene": "GZMH", "cell_type": "NK"}, 0.5,
             [{"ref": "E1", "relation": "supports"}], "r", derived_from="H1")
    assert r["status"] == "recorded" and r["hypothesis_id"] == "H2"
    assert Ledger(run).entries("hypothesis")[-1]["derived_from"] == "H1"
    # reviving H1 would break the rule again
    r = call(server._tool_update_hypothesis, "H1", 0.5, "active", [{"ref": "E1", "relation": "supports"}], "r")
    assert r["status"] == "rejected" and "H2 is already" in r["error"]


def test_task_without_attributes_takes_only_text(tmp_path):
    Ledger(tmp_path).append([{"kind": "run_info", "phase": "start",
                              "task_rules": {"hypothesis_attributes": [], "hypothesis_constraints": []}}],
                            "harness")
    server._reset_for_tests(tmp_path)
    assert call(server._tool_write_hypothesis, "h", {}, 0.5, [ev("f")], "r")["status"] == "recorded"
    r = call(server._tool_write_hypothesis, "h", {"gene": "X"}, 0.5, [ev("f")], "r")
    assert r["status"] == "rejected" and "declares no hypothesis attributes" in r["error"]


def test_rules_fall_back_to_task_spec_in_run_dir(tmp_path):
    (tmp_path / "task_spec.yaml").write_text("hypothesis_attributes:\n  - name: protein\n")
    server._reset_for_tests(tmp_path)
    r = call(server._tool_write_hypothesis, "h", {}, 0.5, [ev("f")], "r")
    assert "missing required attribute(s) ['protein']" in r["error"]


# --- evidence linking ---

def test_link_source():
    calls = [{"tool_call_id": "T1", "tool": "Bash", "cli_tool": "Bash", "arguments": {"command": "python a.py GZMH"}},
             {"tool_call_id": "T2", "tool": "OT_x", "cli_tool": "mcp__tooluniverse__execute_tool",
              "via": "execute_tool", "arguments": {"tool_name": "OT_x", "arguments": {"gene": "GZMH"}}},
             {"tool_call_id": "T3", "tool": "Bash", "cli_tool": "Bash", "arguments": {"command": "python b.py"}}]
    assert link_source({"tool": "bash", "match": "a.py"}, calls)[:2] == (["T1"], "exact")
    assert link_source({"tool": "Bash"}, calls)[:2] == (["T3"], "probable")
    assert link_source({"tool": "OT_x"}, calls)[:2] == (["T2"], "exact")
    assert link_source({"tool": "execute_tool", "match": "gzmh"}, calls)[:2] == (["T2"], "exact")
    assert link_source({"tool": "cellxgene", "match": "a.py"}, calls)[:2] == (["T1"], "exact")
    assert link_source({"tool_call_ids": ["T3"]}, calls)[:2] == (["T3"], "exact")
    assert link_source({"citation": "PMID 1"}, calls)[:2] == ([], "citation")
    assert link_source({"tool": "nothing"}, calls)[:2] == ([], "none")
    assert link_source({}, calls)[:2] == ([], "none")


def test_unlinked_evidence_warns_the_agent(run):
    r = call(server._tool_write_hypothesis, "h", {"gene": "X", "cell_type": "B"}, 0.5,
             [ev("f", tool="Bash", match="nothing ran")], "r")
    assert r["status"] == "recorded" and "no recorded source" in r["warnings"][0]


def test_server_syncs_stream_before_linking(run):
    # The call is only in the transcript (the harness hasn't polled yet).
    _w(run / "transcript.jsonl", claude_call("t1", "Bash", {"command": "run deg"}, "out"))
    r = call(server._tool_write_hypothesis, "h", {"gene": "X", "cell_type": "B"}, 0.5,
             [ev("f", tool="Bash", match="deg")], "r")
    assert r["evidence"][0]["link"] == "exact"


def test_recent_tool_calls_lists_ids(run):
    _w(run / "transcript.jsonl", claude_call("t1", "Bash", {"command": "ls"}, "out"))
    [c] = json.loads(server._tool_recent_tool_calls(5))
    assert c["tool_call_id"] == "T1" and c["tool"] == "Bash"


# --- notes, human input, conclusion ---

def test_note_human_input_and_conclusion(run):
    call(server._tool_write_hypothesis, "h", {"gene": "X", "cell_type": "B"}, 0.5, [ev("f")], "r")
    assert call(server._tool_log_note, "interpretation", "means X", ["H1"])["status"] == "recorded"
    assert call(server._tool_log_note, "musing", "x")["status"] == "rejected"
    assert call(server._tool_log_note, "plan", "x", ["H5"])["status"] == "rejected"
    assert call(server._tool_request_human_input, "Q?", "ctx")["status"] == "headless_mode"
    assert call(server._tool_declare_done, [], "r", "c", {})["status"] == "rejected"
    assert call(server._tool_declare_done, ["H7"], "r", "c", {})["status"] == "rejected"
    r = call(server._tool_declare_done, ["H1"], "best", "high", "one run-level plan")
    assert r["status"] == "completed" and "one plan per hypothesis" in r["warnings"][0]
    led = Ledger(run)
    assert led.entries("note")[0]["hypothesis_ids"] == ["H1"]
    assert led.entries("human_input")[0]["question"] == "Q?"
    assert led.conclusion()["validation"] == {"run": "one run-level plan"}


def test_registered_tools():
    import asyncio
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert names == {"write_hypothesis", "update_hypothesis", "read_hypotheses", "recent_tool_calls",
                     "log_note", "request_human_input", "declare_done"}


def test_read_hypotheses_shows_latest_state(run):
    call(server._tool_write_hypothesis, "h", {"gene": "X", "cell_type": "B"}, 0.5, [ev("f")], "r")
    call(server._tool_update_hypothesis, "H1", 0.1, "refuted", [ev("g", relation="contradicts")], "no")
    assert json.loads(server._tool_read_hypotheses("live"))["hypotheses"] == []
    [h] = json.loads(server._tool_read_hypotheses("all"))["hypotheses"]
    assert h["status"] == "refuted" and h["evidence_ids"] == ["E1", "E2"] and h["entries"] == 2


# --- prompt section ---

def test_instructions_state_task_attributes_and_rule():
    text = ledger_instructions({**SLE_RULES})
    assert "`gene` (required): HGNC symbol" in text and "at most one hypothesis per `gene`" in text
    assert "attributes={}" in ledger_instructions({})
    assert "log_analysis_step" not in text


# --- readers ---

def test_workspace_view_and_metrics_for_a_ledger_run(run):
    from reviewer.log_replay import analyze_run
    _w(run / "transcript.jsonl", claude_call("t1", "mcp__tooluniverse__execute_tool",
                                             {"tool_name": "OT_x", "arguments": {}}, "out"))
    call(server._tool_write_hypothesis, "h", {"gene": "X", "cell_type": "B"}, 0.4,
         [ev("f", tool="OT_x")], "first")
    call(server._tool_update_hypothesis, "H1", 0.7, "active", [ev("g")], "more")
    call(server._tool_log_note, "debug", "retry")
    call(server._tool_declare_done, ["H1"], "best", "c", {"H1": "qPCR"})
    ws = Workspace(run)
    assert ws.is_ledger_run and not (run / "ledger.json").exists()
    h = ws.read_ledger()["hypotheses"]["H1"]
    assert h["candidate_gene"] == "X" and h["claim"] == "h" and h["status"] == "terminal"
    assert [e["evidence_id"] for e in h["evidence"]] == ["E1", "E2"]
    assert h["update_history"][0]["changes"]["from"]["confidence"] == 0.4
    assert h["creation_rationale"] == "first"
    report = ws.read_final_report()
    assert report["suggested_validation"] == "H1: qPCR" and report["top_hypotheses"][0]["hypothesis_id"] == "H1"
    assert ws.read_step_log()[0]["step_type"] == "debug"
    m = analyze_run(str(run))
    assert m.completed and m.hypothesis_count == 1 and m.belief_revision_count == 1
    assert m.universe_tool_counts == {"OT_x": 1}
    assert m.tool_diversity["write_hypothesis"] == 1 and m.tool_diversity["declare_done"] == 1


def test_artifact_entries(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n")
    (tmp_path / "plot.png").write_bytes(b"\x89PNG\x00\xff")
    (tmp_path / ".gemini").mkdir()
    (tmp_path / ".gemini" / "settings.json").write_text("{}")
    arts = {a["path"]: a for a in artifact_entries(tmp_path, skip=(".gemini",), prefix="ws/")}
    assert set(arts) == {"ws/a.py", "ws/plot.png"}
    assert arts["ws/a.py"]["content"] == "print(1)\n" and "content" not in arts["ws/plot.png"]
    assert len(arts["ws/plot.png"]["sha256"]) == 64


# --- old runs ---

def test_convert_v1_run(tmp_path):
    from hypothesis_ledger.legacy import convert_v1
    old = tmp_path / "old"
    old.mkdir()
    (old / "ledger.json").write_text(json.dumps({"hypotheses": {"AB12": {
        "hypothesis_id": "AB12", "cell_type": "Mono", "candidate_gene": "SIGLEC1", "claim": "new claim",
        "evidence": [{"type": "lit", "description": "paper"}], "confidence": 0.8, "status": "terminal",
        "open_questions": [], "created_at": "2026-01-01",
        "update_history": [{"timestamp": "t1", "rationale": "DE found",
                            "changes": {"from": {"confidence": 0.5, "claim": "old claim",
                                                 "evidence": [{"type": "scRNA", "description": "lfc 3"}]},
                                        "to": {"confidence": 0.8, "claim": "new claim",
                                               "evidence": [{"type": "lit", "description": "paper"}]}}}]}}}))
    (old / "step_log.jsonl").write_text(json.dumps({"step_type": "Debugging", "interpretation": "fixed"}) + "\n")
    (old / "final_report.json").write_text(json.dumps({"top_hypotheses": [{"hypothesis_id": "AB12"}],
                                                       "rationale": "done", "suggested_validation": "flow"}))
    _w(old / "transcript.jsonl", claude_call("t1", "Bash", {"command": "ls"}, "x"))
    out = tmp_path / "new"
    convert_v1(old, out)
    led = Ledger(out)
    create, update = led.entries("hypothesis")
    assert create["hypothesis"] == "old claim" and create["confidence"] == 0.5 and create["rationale"] == ""
    assert create["attributes"] == {"gene": "SIGLEC1", "cell_type": "Mono"}
    assert create["evidence"][0]["finding"] == "lfc 3" and create["evidence"][0]["link"] == "none"
    assert update["rationale"] == "DE found" and update["v1_rewrites"] == {"claim": "new claim"}
    assert update["evidence"][0]["finding"] == "paper"
    assert led.entries("note")[0]["note_kind"] == "debug"
    assert led.conclusion()["ranked_hypothesis_ids"] == ["H1"]
    assert len(led.entries("tool_call")) == 1
    assert "creation entries have no rationale" in led.run_info("start")["conversion_limits"]
    assert not (old / "ledger.jsonl").exists()     # the old run is left alone
    h = Workspace(out).read_ledger()["hypotheses"]["H1"]
    assert {e["evidence_id"] for e in h["evidence"]} == {"E1", "E2"}   # dropped item kept cited


def test_link_uses_words_of_match_and_rounded_numbers():
    calls = [
        {"tool_call_id": "T1", "tool": "Bash", "cli_tool": "Bash",
         "arguments": {"command": "cat > /tmp/deg.py << EOF\nDATASET_ID = \"218acb0f-9f2f\"\nEOF"},
         "output": "IFI44L 2.871843 4.7e-14\nDNASE1L3 -0.116963 1.366398e-01"},
        {"tool_call_id": "T2", "tool": "Bash", "cli_tool": "Bash",
         "arguments": {"command": "cat > /tmp/pseudobulk.py"}, "output": "IFI44L 1.0 0.5"},
        {"tool_call_id": "T3", "tool": "EWAS_x", "cli_tool": "mcp__tooluniverse__execute_tool",
         "via": "execute_tool", "arguments": {"tool_name": "EWAS_x", "arguments": {"gene_symbol": "IFI44L"}},
         "output": "rank 1"},
    ]
    # agent names a library, not the tool; words of match + rounded numbers find the call
    ids, link, _, signals = link_source({"tool": "cellxgene_census.get_anndata", "match": "dataset_id=218acb0f"},
                                        calls, "DNASE1L3 log2FC=-0.117, p=0.137")
    assert (ids, link) == (["T1"], "exact") and signals == ["match", "numbers"]
    # 'key=value' written by the agent vs JSON arguments
    assert link_source({"tool": "EWAS_x", "match": "gene_symbol=IFI44L"}, calls)[:2] == (["T3"], "exact")
    # numbers alone, among all calls
    assert link_source({}, calls, "log2FC 2.87, p=4.7e-14")[:2] == (["T1"], "exact")
    assert link_source({}, calls, "log2FC 2.87, p = 4.7×10⁻¹⁴")[:2] == (["T1"], "exact")
    # a single number that appears nowhere links nothing
    assert link_source({}, calls, "fold change 9.99")[:2] == ([], "none")
    # with no other signal, one matching number is not enough (shake-down run 2, E19)
    assert link_source({"citation": "PMID 1"}, calls, "log2FC=2.87 in a paper")[:2] == ([], "citation")
    # but it narrows candidates once the tool matched
    assert link_source({"tool": "Bash"}, calls, "log2FC=2.87")[:2] == (["T1"], "exact")


def test_evidence_entry_records_link_signals(run):
    _w(run / "transcript.jsonl", claude_call("t1", "Bash", {"command": "python deg.py"}, "lfc 2.1043 p 0.0312"))
    call(server._tool_write_hypothesis, "h", {"gene": "X", "cell_type": "B"}, 0.5,
         [ev("lfc 2.10, p 0.031", tool="Bash", match="deg.py")], "r")
    item = Ledger(run).entries("hypothesis")[0]["evidence"][0]
    assert item["link"] == "exact" and item["link_signals"] == ["tool", "match", "numbers"]
