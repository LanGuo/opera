import json
import pytest
from pathlib import Path
from hypothesis_ledger.workspace import Workspace
from reviewer.log_replay import (analyze_run, RunMetrics, _parse_transcript,
                                  _bare_tool_name, _confidence_sequence,
                                  _martingale_score, _directional_flip_count)


@pytest.fixture
def populated_run(tmp_path):
    ws = Workspace(str(tmp_path / "run_001"))

    def add_hypotheses(data):
        data["hypotheses"] = {
            "H001": {
                "cell_type": "pDC", "candidate_gene": "SIGLEC1",
                "claim": "elevated", "status": "terminal",
                "evidence": [{"source": "DEG"}, {"source": "GWAS"}],
                "confidence": 0.85, "open_questions": [],
                "update_history": [{"rationale": "GWAS hit raised confidence"}],
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            "H002": {
                "cell_type": "NK", "candidate_gene": "KLRB1",
                "claim": "possible", "status": "refuted",
                "evidence": [{"source": "DEG"}],
                "confidence": 0.2, "open_questions": [],
                "update_history": [],
                "created_at": "2026-01-01T00:01:00+00:00",
            },
        }
        data["tool_call_count"] = 42
    ws.update_ledger(add_hypotheses)

    ws.log_run({"tool": "write_hypothesis", "result": {}})
    ws.log_run({"tool": "write_hypothesis", "result": {}})
    ws.log_run({"tool": "read_hypotheses", "result": {}})
    ws.write_final_report({"top_hypotheses": []})

    # Simulate a transcript.jsonl in the real -p/--verbose format:
    # hypothesis-ledger tools use the mcp__ prefix; ToolUniverse tools are bare names.
    transcript_lines = [
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__hypothesis-ledger__write_hypothesis"},
            {"type": "tool_use", "name": "mcp__tooluniverse__BiomarkerDiscoveryWorkflow"},
        ]}}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__tooluniverse__HypothesisGenerator"},
            {"type": "tool_use", "name": "mcp__tooluniverse__BiomarkerDiscoveryWorkflow"},
        ]}}),
        json.dumps({"type": "content_block_start", "content_block": {"type": "text"}}),  # ignored
    ]
    (tmp_path / "run_001" / "transcript.jsonl").write_text("\n".join(transcript_lines))

    return str(tmp_path / "run_001")


def test_analyze_run_returns_metrics(populated_run):
    metrics = analyze_run(populated_run)
    assert isinstance(metrics, RunMetrics)


def test_belief_revision_count(populated_run):
    metrics = analyze_run(populated_run)
    assert metrics.belief_revision_count == 1


def test_refutation_rate(populated_run):
    metrics = analyze_run(populated_run)
    assert metrics.refutation_rate == pytest.approx(0.5)


def test_modality_coverage(populated_run):
    metrics = analyze_run(populated_run)
    assert metrics.modality_coverage["H001"] == {"DEG", "GWAS"}
    assert metrics.modality_coverage["H002"] == {"DEG"}


def test_tool_diversity_merges_ledger_and_transcript(populated_run):
    metrics = analyze_run(populated_run)
    # From ledger run_log.jsonl (authoritative, bare names)
    assert metrics.tool_diversity["write_hypothesis"] == 2
    assert metrics.tool_diversity["read_hypotheses"] == 1
    # From transcript.jsonl (ToolUniverse calls not in ledger, keep prefixed name)
    assert metrics.tool_diversity["mcp__tooluniverse__BiomarkerDiscoveryWorkflow"] == 2
    assert metrics.tool_diversity["mcp__tooluniverse__HypothesisGenerator"] == 1
    # mcp__hypothesis-ledger__write_hypothesis must NOT appear (deduped against ledger)
    assert "mcp__hypothesis-ledger__write_hypothesis" not in metrics.tool_diversity


def test_parse_transcript_ignores_non_tool_blocks(tmp_path):
    transcript = "\n".join([
        json.dumps({"type": "content_block_start", "content_block": {"type": "tool_use", "name": "find_tools"}}),
        json.dumps({"type": "content_block_start", "content_block": {"type": "text"}}),
        json.dumps({"type": "message_start"}),
    ])
    path = tmp_path / "transcript.jsonl"
    path.write_text(transcript)
    counts = _parse_transcript(path)
    assert counts == {"find_tools": 1}


def test_completed_flag(populated_run):
    metrics = analyze_run(populated_run)
    assert metrics.completed is True


def test_incomplete_run(tmp_path):
    Workspace(str(tmp_path / "run_incomplete"))
    metrics = analyze_run(str(tmp_path / "run_incomplete"))
    assert metrics.completed is False
    assert metrics.hypothesis_count == 0


def test_ledger_count_is_authoritative_over_transcript(populated_run):
    # transcript has mcp__hypothesis-ledger__write_hypothesis (1 call); ledger has write_hypothesis (2)
    # bare-name dedup: ledger's count wins → 2; prefixed name absent from result
    metrics = analyze_run(populated_run)
    assert metrics.tool_diversity["write_hypothesis"] == 2
    assert "mcp__hypothesis-ledger__write_hypothesis" not in metrics.tool_diversity


def test_parse_transcript_non_streaming_format(tmp_path):
    # The -p / --verbose format wraps tool_use blocks inside assistant messages
    transcript = "\n".join([
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__tooluniverse__execute_tool"},
            {"type": "tool_use", "name": "mcp__tooluniverse__find_tools"},
            {"type": "text", "text": "some text"},
        ]}}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__tooluniverse__execute_tool"},
        ]}}),
        json.dumps({"type": "system", "subtype": "hook_started"}),  # ignored
    ])
    path = tmp_path / "transcript.jsonl"
    path.write_text(transcript)
    counts = _parse_transcript(path)
    assert counts["mcp__tooluniverse__execute_tool"] == 2
    assert counts["mcp__tooluniverse__find_tools"] == 1
    assert "text" not in counts


def test_parse_transcript_gemini_format(tmp_path):
    # Gemini CLI emits {"type": "tool_use", "tool_name": ..., "tool_id": ...}
    transcript = "\n".join([
        json.dumps({"type": "tool_use", "tool_name": "execute_tool", "tool_id": "abc1"}),
        json.dumps({"type": "tool_use", "tool_name": "execute_tool", "tool_id": "abc2"}),
        json.dumps({"type": "tool_use", "tool_name": "find_tools", "tool_id": "abc3"}),
        json.dumps({"type": "message_start"}),  # ignored
    ])
    path = tmp_path / "transcript.jsonl"
    path.write_text(transcript)
    counts = _parse_transcript(path)
    assert counts == {"execute_tool": 2, "find_tools": 1}


def test_bare_tool_name_strips_mcp_prefix():
    assert _bare_tool_name("mcp__hypothesis-ledger__write_hypothesis") == "write_hypothesis"
    assert _bare_tool_name("mcp__tooluniverse__find_tools") == "find_tools"
    assert _bare_tool_name("Bash") == "Bash"
    assert _bare_tool_name("mcp__claude_ai_bioRxiv__search_preprints") == "search_preprints"


def test_modality_coverage_accepts_type_key(tmp_path):
    # Agent may store evidence with "type" instead of "source"
    ws = Workspace(str(tmp_path / "run_type_key"))
    def add(data):
        data["hypotheses"] = {
            "H001": {
                "cell_type": "monocyte", "candidate_gene": "SIGLEC1",
                "claim": "elevated", "status": "active",
                "evidence": [{"type": "scrnaseq_deg"}, {"type": "gwas"}],
                "confidence": 0.7, "open_questions": [],
                "update_history": [], "created_at": "2026-01-01T00:00:00+00:00",
            },
        }
    ws.update_ledger(add)
    metrics = analyze_run(str(tmp_path / "run_type_key"))
    assert metrics.modality_coverage["H001"] == {"scrnaseq_deg", "gwas"}


def test_confidence_sequence_initial_only():
    h = {"confidence": 0.6, "update_history": []}
    assert _confidence_sequence(h) == [0.6]

def test_confidence_sequence_with_updates():
    h = {
        "confidence": 0.4,
        "update_history": [
            {"changes": {"from": {"confidence": 0.5}, "to": {"confidence": 0.7}}},
            {"changes": {"from": {"confidence": 0.7}, "to": {"confidence": 0.4}}},
        ],
    }
    assert _confidence_sequence(h) == [0.5, 0.7, 0.4]

def test_confidence_sequence_skips_updates_without_confidence():
    h = {
        "confidence": 0.3,
        "update_history": [
            {"changes": {"from": {"status": "active"}, "to": {"status": "weakened"}}},
            {"changes": {"from": {"confidence": 0.5}, "to": {"confidence": 0.3}}},
        ],
    }
    assert _confidence_sequence(h) == [0.5, 0.3]

def test_confidence_sequence_starts_from_first_update_not_current_value():
    # update_hypothesis overwrites "confidence" in place, so the ledger holds the latest value
    h = {
        "confidence": 0.4,
        "update_history": [
            {"changes": {"from": {"confidence": 0.65}, "to": {"confidence": 0.5}}},
            {"changes": {"from": {"status": "active"}, "to": {"status": "weakened"}}},
            {"changes": {"from": {"confidence": 0.5}, "to": {"confidence": 0.4}}},
        ],
    }
    assert _confidence_sequence(h) == [0.65, 0.5, 0.4]

def test_confidence_sequence_status_only_updates_keep_current_value():
    h = {
        "confidence": 0.6,
        "update_history": [{"changes": {"from": {"status": "active"}, "to": {"status": "terminal"}}}],
    }
    assert _confidence_sequence(h) == [0.6]

def test_martingale_score_none_when_all_starting_values_equal():
    # Every b_t is 0.6: r is undefined, not 0
    r, n = _martingale_score([[0.6, 0.8], [0.6, 0.8], [0.6, 0.7]])
    assert r is None and n == 3

def test_martingale_score_none_when_no_confidence_changes():
    r, n = _martingale_score([[0.5, 0.5], [0.7, 0.7], [0.3, 0.3]])
    assert r is None and n == 3

def test_martingale_score_none_for_insufficient_data():
    r, n = _martingale_score([[0.5, 0.6]])  # only 1 pair, need >= 3
    assert r is None and n == 1

def test_martingale_score_entrenchment():
    # High b_t predicts large positive delta: entrenchment → positive r
    sequences = [[0.1, 0.3], [0.2, 0.5], [0.3, 0.6]]
    r, n = _martingale_score(sequences)
    assert r is not None and r > 0 and n == 3

def test_martingale_score_near_zero_for_random_updates():
    # Pairs constructed so covariance is ~0: no relationship between b_t and delta
    # b_t values: 0.3, 0.7, 0.2, 0.8 (mean 0.5); deltas: +0.2, +0.2, -0.2, -0.2 (mean 0)
    sequences = [[0.3, 0.5], [0.7, 0.9], [0.2, 0.0], [0.8, 0.6]]
    r, n = _martingale_score(sequences)
    assert r is not None and abs(r) < 0.01 and n == 4

def test_directional_flip_count_zero():
    assert _directional_flip_count([[0.3, 0.4, 0.45]]) == 0  # never crossed 0.5

def test_directional_flip_count_one():
    assert _directional_flip_count([[0.4, 0.6]]) == 1  # crossed 0.5 once

def test_directional_flip_count_multiple():
    # 0.4→0.6 (cross up), 0.6→0.3 (cross down), 0.3→0.8 (cross up)
    assert _directional_flip_count([[0.4, 0.6, 0.3, 0.8]]) == 3

def test_directional_flip_count_across_hypotheses():
    seqs = [[0.4, 0.6], [0.7, 0.3]]  # one flip each
    assert _directional_flip_count(seqs) == 2

def test_analyze_run_includes_new_metrics(populated_run):
    metrics = analyze_run(populated_run)
    assert hasattr(metrics, "martingale_score")
    assert hasattr(metrics, "directional_flip_count")
    assert isinstance(metrics.directional_flip_count, int)


# --- turns and cost across resumed sessions ---

import json as _json

from reviewer.log_replay import _parse_cost, _parse_turn_count


def _transcript(tmp_path, results):
    p = tmp_path / "transcript.jsonl"
    lines = []
    for r in results:
        lines.append(_json.dumps({"type": "system", "subtype": "init", "session_id": r["session"]}))
        lines.append(_json.dumps({"type": "result", "session_id": r["session"],
                                  "num_turns": r["turns"], "total_cost_usd": r["cost"]}))
    p.write_text("\n".join(lines) + "\n")
    return p


def test_turns_single_session(tmp_path):
    p = _transcript(tmp_path, [{"session": "s1", "turns": 31, "cost": 1.53}])
    assert _parse_turn_count(p) == 31
    assert _parse_cost(p) == pytest.approx(1.53)


def test_turns_summed_and_cost_last_for_resumed_session(tmp_path):
    # Claude Code reports num_turns per invocation but cost cumulatively within a session.
    p = _transcript(tmp_path, [{"session": "s1", "turns": 31, "cost": 0.98},
                               {"session": "s1", "turns": 31, "cost": 2.29},
                               {"session": "s1", "turns": 17, "cost": 3.43}])
    assert _parse_turn_count(p) == 79
    assert _parse_cost(p) == pytest.approx(3.43)


def test_cost_summed_across_different_sessions(tmp_path):
    p = _transcript(tmp_path, [{"session": "s1", "turns": 10, "cost": 1.0},
                               {"session": "s2", "turns": 5, "cost": 0.5}])
    assert _parse_turn_count(p) == 15
    assert _parse_cost(p) == pytest.approx(1.5)


# --- Gemini tool names, ToolUniverse layer, Gemini turns and prices ---

from reviewer.log_replay import _parse_cost_details, _parse_universe_calls


def _write_events(tmp_path, events):
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(_json.dumps(e) for e in events) + "\n")
    return p


def test_bare_tool_name_strips_gemini_prefix_for_opera_servers():
    assert _bare_tool_name("mcp_hypothesis-ledger_write_hypothesis") == "write_hypothesis"
    assert _bare_tool_name("mcp_tooluniverse_execute_tool") == "execute_tool"
    assert _bare_tool_name("mcp_exa_web_search_exa") == "mcp_exa_web_search_exa"  # unknown server


def test_gemini_calls_to_unregistered_tools_are_not_counted(tmp_path):
    p = _write_events(tmp_path, [
        {"type": "tool_use", "tool_name": "mcp_hypothesis_ledger_write_hypothesis", "tool_id": "t1"},
        {"type": "tool_result", "tool_id": "t1", "status": "error",
         "error": {"type": "tool_not_registered", "message": "not found"}},
        {"type": "tool_use", "tool_name": "mcp_hypothesis-ledger_write_hypothesis", "tool_id": "t2"},
        {"type": "tool_result", "tool_id": "t2", "status": "success"},
        {"type": "tool_use", "tool_name": "run_shell_command", "tool_id": "t3"},
        {"type": "tool_result", "tool_id": "t3", "status": "error", "error": {"type": "shell_error"}},
    ])
    assert _parse_transcript(p) == {"mcp_hypothesis-ledger_write_hypothesis": 1, "run_shell_command": 1}


def test_gemini_ledger_calls_merge_with_run_log(tmp_path):
    ws = Workspace(str(tmp_path / "run"))
    ws.log_run({"tool": "write_hypothesis", "result": {}})
    _write_events(tmp_path / "run", [
        {"type": "tool_use", "tool_name": "mcp_hypothesis-ledger_write_hypothesis", "tool_id": "t1"},
    ])
    m = analyze_run(str(tmp_path / "run"))
    assert m.tool_diversity == {"write_hypothesis": 1}


def test_universe_layer_counts_biomedical_tools_only(tmp_path):
    p = _write_events(tmp_path, [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__tooluniverse__find_tools", "input": {"query": "x"}},
            {"type": "tool_use", "name": "mcp__tooluniverse__execute_tool",
             "input": {"tool_name": "PubMed_search_articles", "arguments": {}}},
            {"type": "tool_use", "name": "mcp__tooluniverse__execute_tool",
             "input": {"tool_name": "PubMed_search_articles", "arguments": {}}},
            {"type": "tool_use", "name": "mcp__tooluniverse__BiomarkerDiscoveryWorkflow", "input": {}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_use", "name": "WebSearch", "input": {"query": "y"}},
            {"type": "tool_use", "name": "mcp__hypothesis-ledger__write_hypothesis", "input": {}},
        ]}},
        {"type": "tool_use", "tool_name": "mcp_tooluniverse_execute_tool", "tool_id": "g1",
         "parameters": {"tool_name": "DisGeNET_get_gda", "arguments": {}}},
    ])
    assert _parse_universe_calls(p) == {
        "PubMed_search_articles": 2, "BiomarkerDiscoveryWorkflow": 1, "DisGeNET_get_gda": 1,
    }


def test_gemini_turns_counted_from_responses(tmp_path):
    p = _write_events(tmp_path, [
        {"type": "init"},
        {"type": "message", "role": "user", "content": "Begin the task."},
        {"type": "tool_use", "tool_name": "a", "tool_id": "1"},
        {"type": "tool_use", "tool_name": "b", "tool_id": "2"},   # same response, parallel calls
        {"type": "tool_result", "tool_id": "1"}, {"type": "tool_result", "tool_id": "2"},
        {"type": "message", "role": "assistant", "content": "done"},
        {"type": "result", "stats": {}},
    ])
    assert _parse_turn_count(p) == 2


def test_gemini_cost_prices_known_models_and_lists_unpriced(tmp_path):
    p = _write_events(tmp_path, [{"type": "result", "stats": {"models": {
        "gemini-2.5-pro": {"input": 1_000_000, "output_tokens": 100_000},
        "gemini-3-flash-preview": {"input": 1_000_000, "output_tokens": 1_000_000},
        "gemini-9-mystery": {"input": 5, "output_tokens": 5},
    }}}])
    cost, unpriced = _parse_cost_details(p)
    assert cost == pytest.approx(1.25 + 1.0 + 0.5 + 3.0)
    assert unpriced == ["gemini-9-mystery"]
