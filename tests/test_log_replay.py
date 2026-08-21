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
        "confidence": 0.5,
        "update_history": [
            {"changes": {"from": {"confidence": 0.5}, "to": {"confidence": 0.7}}},
            {"changes": {"from": {"confidence": 0.7}, "to": {"confidence": 0.4}}},
        ],
    }
    assert _confidence_sequence(h) == [0.5, 0.7, 0.4]

def test_confidence_sequence_skips_updates_without_confidence():
    h = {
        "confidence": 0.5,
        "update_history": [
            {"changes": {"from": {"status": "active"}, "to": {"status": "weakened"}}},
            {"changes": {"from": {"confidence": 0.5}, "to": {"confidence": 0.3}}},
        ],
    }
    assert _confidence_sequence(h) == [0.5, 0.3]

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
