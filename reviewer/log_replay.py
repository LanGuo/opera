# eval/log_replay.py
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from hypothesis_ledger.workspace import Workspace

# Known hypothesis-ledger tool names (bare, without MCP prefix).
_LEDGER_TOOLS: frozenset[str] = frozenset({
    "write_hypothesis",
    "update_hypothesis",
    "read_hypotheses",
    "log_analysis_step",
    "declare_done",
    "request_human_input",
})
_LEDGER_TOOL_COUNT = len(_LEDGER_TOOLS)   # 6 available ops
_TOOLUNIVERSE_TOOL_COUNT = 600            # approximate; from Gao et al. 2025
# OPERA's MCP servers. Gemini CLI names their tools 'mcp_<server>_<tool>'; with single underscores
# the split is ambiguous, so the known server names are matched.
_MCP_SERVERS = ("hypothesis-ledger", "tooluniverse")
# ToolUniverse gateway tools: discovery, and execute_tool, which runs the tool named in its
# 'tool_name' argument. None of them is one of the ~600 biomedical tools.
_TOOLUNIVERSE_META_TOOLS = frozenset({"execute_tool", "find_tools", "grep_tools", "list_tools", "get_tool_info"})


def _normalized_entropy(counts: dict[str, int], n_available: int) -> float | None:
    """Shannon entropy H normalized by log2(n_available).

    Returns a value in [0, 1] where 1 means calls were spread perfectly uniformly
    across all available tools, and 0 means only one tool was ever called.
    Returns None if there are no calls or only one distinct tool.
    """
    total = sum(counts.values())
    if total == 0 or len(counts) < 2 or n_available < 2:
        return None
    h = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    return h / math.log2(n_available)


@dataclass
class RunMetrics:
    belief_revision_count: int
    refutation_rate: float
    modality_coverage: dict  # hypothesis_id -> set of source strings
    tool_diversity: dict     # tool_name -> call count (ledger + transcript merged)
    hypothesis_count: int
    tool_call_count: int
    completed: bool
    martingale_score: float | None
    martingale_n: int                          # number of (b_t, Δb_t) pairs used
    directional_flip_count: int
    cost_usd: float | None
    turn_count: int | None                     # number of model invocations (agent turns used)
    revisions_per_hypothesis: float | None     # belief_revision_count / hypothesis_count
    tool_diversity_entropy: float | None       # kept for backward compat (merged, unnormalized)
    tool_diversity_entropy_ledger: float | None    # H / log2(6), ledger ops only
    tool_diversity_entropy_universe: float | None  # H / log2(600), over universe_tool_counts
    # biomedical tool -> calls: execute_tool's target, or a ToolUniverse tool called directly
    universe_tool_counts: dict = field(default_factory=dict)
    cost_unpriced_models: list = field(default_factory=list)  # Gemini models left out of cost_usd


def _split_mcp_name(name: str) -> tuple[str | None, str]:
    """Return (server, bare tool name) for an MCP tool name, or (None, name) for other tools.

    Claude Code uses 'mcp__<server>__<tool>'; Gemini CLI uses 'mcp_<server>_<tool>' and is only
    recognized for OPERA's servers.
    """
    if name.startswith("mcp__"):
        rest = name[5:]          # drop 'mcp__'
        sep = rest.find("__")
        if sep >= 0:
            return rest[:sep], rest[sep + 2:]
    elif name.startswith("mcp_"):
        for server in _MCP_SERVERS:
            prefix = f"mcp_{server}_"
            if name.startswith(prefix):
                return server, name[len(prefix):]
    return None, name


def _bare_tool_name(name: str) -> str:
    """Strip the MCP server prefix from a tool name.

    Transcripts use prefixed names (e.g. 'mcp__hypothesis-ledger__write_hypothesis' or Gemini's
    'mcp_hypothesis-ledger_write_hypothesis') while the ledger run_log stores bare names
    ('write_hypothesis').  Stripping the prefix lets the skip-if-exists merge detect that they
    refer to the same tool.
    """
    return _split_mcp_name(name)[1]


def _universe_tool(name: str, args) -> str | None:
    """The ToolUniverse biomedical tool a call used, or None if it used none."""
    server, bare = _split_mcp_name(name)
    if bare == "execute_tool":
        target = args.get("tool_name") if isinstance(args, dict) else None
        return target if isinstance(target, str) and target else "unknown"
    if server == "tooluniverse" and bare not in _TOOLUNIVERSE_META_TOOLS:
        return bare
    return None


def _confidence_sequence(h: dict) -> list[float]:
    """Extract chronological confidence values from a hypothesis entry."""
    init = h.get("confidence")
    seq = [init] if init is not None else []
    for update in h.get("update_history", []):
        new_conf = update.get("changes", {}).get("to", {}).get("confidence")
        if new_conf is not None:
            seq.append(float(new_conf))
    return seq


def _martingale_score(sequences: list[list[float]]) -> tuple[float | None, int]:
    """Pearson r between b_t and (b_{t+1} - b_t) pooled across all hypothesis sequences.

    Returns (r, n) where n is the number of (b_t, Δb_t) pairs used.
    Returns (None, n) if n < 3 — too few pairs for a reliable estimate.

    Near-zero r: evidence-driven updates.
    Positive r: belief entrenchment (high confidence tends to keep rising).
    """
    xs, ys = [], []
    for seq in sequences:
        for i in range(len(seq) - 1):
            xs.append(seq[i])
            ys.append(seq[i + 1] - seq[i])
    n = len(xs)
    if n < 3:
        return None, n
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0, n
    return cov / (sx * sy), n


def _directional_flip_count(sequences: list[list[float]], threshold: float = 0.5) -> int:
    """Count confidence crossings of threshold across all sequences."""
    flips = 0
    for seq in sequences:
        for i in range(len(seq) - 1):
            if (seq[i] - threshold) * (seq[i + 1] - threshold) < 0:
                flips += 1
    return flips


_GEMINI_PRICING = {
    # model-substring -> (input $/1M, output $/1M)  — prompts <=200K context window
    # Standard paid tier, ai.google.dev/gemini-api/docs/pricing, checked 2026-11-01.
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-3-flash-preview": (0.50, 3.00),
}


def _parse_cost(transcript_path: Path) -> float | None:
    """Extract run cost from the 'result' events in a transcript; see _parse_cost_details."""
    return _parse_cost_details(transcript_path)[0]


def _parse_cost_details(transcript_path: Path) -> tuple[float | None, list[str]]:
    """Return (run cost, Gemini models with no price) from the 'result' events in a transcript.

    Claude transcripts include total_cost_usd. A resumed session writes one result event per
    invocation, each with the session's cumulative cost, so the last value per session is used and
    sessions are summed. Gemini transcripts include token counts per model in stats; cost is
    estimated with _GEMINI_PRICING (first matching model substring wins) from the first result.
    Models without a price are left out of the estimate and listed. Cost is None if no cost data
    is found.
    """
    unpriced: list[str] = []
    if not transcript_path.exists():
        return None, unpriced
    claude_costs: dict = {}
    gemini_cost: float | None = None
    for line in transcript_path.read_text(encoding="utf-8").strip().splitlines():
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "result":
            continue
        if "total_cost_usd" in event:
            claude_costs[event.get("session_id")] = float(event["total_cost_usd"])
            continue
        if gemini_cost is not None:
            continue
        models = event.get("stats", {}).get("models", {})
        if models:
            total = 0.0
            for model_name, model_stats in models.items():
                price = next(
                    (v for k, v in _GEMINI_PRICING.items() if k in model_name), None
                )
                if price is None:
                    unpriced.append(model_name)
                    continue
                inp = model_stats.get("input", 0) or 0       # non-cached input tokens
                out = model_stats.get("output_tokens", 0) or 0
                total += inp / 1e6 * price[0] + out / 1e6 * price[1]
            gemini_cost = total if total > 0 else None
    if claude_costs:
        return sum(claude_costs.values()), unpriced
    return gemini_cost, unpriced


def gemini_response_count(events: list[dict]) -> int:
    """Count model responses in a Gemini CLI stream-json transcript.

    Gemini reports no turn count. A response is a run of model output (text and/or parallel tool
    calls) ending when tool results come back; model.maxSessionTurns counts these.
    """
    responses, awaiting = 0, True
    for event in events:
        t = event.get("type")
        if t == "tool_use" or (t == "message" and event.get("role") == "assistant"):
            responses += awaiting
            awaiting = False
        elif t == "tool_result":
            awaiting = True
    return responses


def _parse_turn_count(transcript_path: Path) -> int | None:
    """Return the number of agent turns recorded in a transcript.

    Priority:
    1. Claude Code result events: ``num_turns`` is the authoritative count written by the
       harness at the end of each invocation. A resumed session writes one result event per
       invocation, each counting only that invocation's turns, so they are summed.
    2. Claude Code streaming: count ``message_start`` events (one per API call).
    3. Gemini CLI: count model responses (gemini_response_count).
    Returns None if the transcript does not exist or no turn signal is found.
    """
    if not transcript_path.exists():
        return None
    message_starts = 0
    events: list[dict] = []
    result_turns: list[int] = []
    for line in transcript_path.read_text(encoding="utf-8").strip().splitlines():
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(event)
        t = event.get("type")
        if t == "result" and "num_turns" in event:
            result_turns.append(int(event["num_turns"]))
        elif t == "message_start":
            message_starts += 1
    if result_turns:
        return sum(result_turns)
    count = message_starts or gemini_response_count(events)
    return count if count > 0 else None


def _tool_calls(transcript_path: Path) -> list[tuple[str, dict]]:
    """Extract (tool name, arguments) for each tool call in a harness transcript.

    Supports:
    - Claude Code: {"type": "content_block_start", "content_block": {"type": "tool_use", "name": ...}}
    - Claude Code (-p): {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": ...}]}}
    - Gemini CLI: {"type": "tool_use", "tool_name": ..., "tool_id": ..., "parameters": ...}

    Gemini calls to a tool name that doesn't exist (error type 'tool_not_registered') never ran
    and are left out.
    """
    calls: list[tuple[str, dict]] = []
    if not transcript_path.exists():
        return calls
    gemini_ids: list = []
    unregistered: set = set()
    for line in transcript_path.read_text(encoding="utf-8").strip().splitlines():
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Gemini CLI format
        if event.get("type") == "tool_use" and "tool_name" in event:
            calls.append((event["tool_name"], event.get("parameters") or {}))
            gemini_ids.append(event.get("tool_id"))
            continue
        if event.get("type") == "tool_result" and (event.get("error") or {}).get("type") == "tool_not_registered":
            unregistered.add(event.get("tool_id"))
            continue

        # Claude Code Streaming format
        if event.get("type") == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                calls.append((block.get("name", "unknown"), block.get("input") or {}))
                gemini_ids.append(None)
        # Claude Code Non-streaming / -p format
        elif event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    calls.append((block.get("name", "unknown"), block.get("input") or {}))
                    gemini_ids.append(None)
    return [c for c, tid in zip(calls, gemini_ids) if tid is None or tid not in unregistered]


def _parse_transcript(transcript_path: Path) -> dict[str, int]:
    """Count tool calls by tool name in a harness transcript (see _tool_calls)."""
    counts: dict[str, int] = {}
    for name, _ in _tool_calls(transcript_path):
        counts[name] = counts.get(name, 0) + 1
    return counts


def _parse_universe_calls(transcript_path: Path) -> dict[str, int]:
    """Count calls per ToolUniverse biomedical tool, resolving execute_tool to its target."""
    counts: dict[str, int] = {}
    for name, args in _tool_calls(transcript_path):
        tool = _universe_tool(name, args)
        if tool is not None:
            counts[tool] = counts.get(tool, 0) + 1
    return counts


def analyze_run(run_dir: str) -> RunMetrics:
    ws = Workspace(run_dir)
    data = ws.read_ledger()
    hypotheses = data.get("hypotheses", {})

    belief_revision_count = sum(
        len(h.get("update_history", [])) for h in hypotheses.values()
    )

    refuted_count = sum(
        1 for h in hypotheses.values()
        if h.get("status") in ("weakened", "refuted")
    )
    refutation_rate = refuted_count / len(hypotheses) if hypotheses else 0.0

    modality_coverage = {
        h_id: {(e.get("source") or e.get("type") or "unknown") if isinstance(e, dict) else str(e) for e in h.get("evidence", [])}
        for h_id, h in hypotheses.items()
    }

    # Ledger run_log: authoritative counts for hypothesis-ledger tool calls
    tool_diversity: dict[str, int] = {}
    if ws.run_log_path.exists():
        for line in ws.run_log_path.read_text(encoding="utf-8").strip().splitlines():
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            tool = entry.get("tool", "unknown")
            tool_diversity[tool] = tool_diversity.get(tool, 0) + 1

    # Transcript covers ALL MCP calls. Use it only for tools not already in the ledger.
    # Transcript names use the 'mcp__<server>__<tool>' prefix; ledger uses bare names.
    # Strip the prefix before the skip-if-exists check so they match correctly.
    transcript_path = ws.run_dir / "transcript.jsonl"
    for tool, count in _parse_transcript(transcript_path).items():
        if _bare_tool_name(tool) not in tool_diversity:
            tool_diversity[tool] = count

    sequences = [_confidence_sequence(h) for h in hypotheses.values()]
    martingale_score_val, martingale_n = _martingale_score(sequences)
    flip_count = _directional_flip_count(sequences)

    n_hypotheses = len(hypotheses)
    revisions_per_hypothesis = (
        belief_revision_count / n_hypotheses if n_hypotheses > 0 else None
    )

    # Entropy per tool layer: the ledger's six operations, and the ToolUniverse biomedical tools
    # actually run (built-in harness tools and ToolUniverse discovery calls are in neither).
    ledger_counts = {k: v for k, v in tool_diversity.items() if k in _LEDGER_TOOLS}
    universe_counts = _parse_universe_calls(transcript_path)
    cost_usd, unpriced = _parse_cost_details(transcript_path)

    # Merged unnormalized entropy (kept for backward compat).
    total_calls = sum(tool_diversity.values())
    if total_calls > 0 and len(tool_diversity) > 1:
        tool_diversity_entropy: float | None = -sum(
            (c / total_calls) * math.log2(c / total_calls)
            for c in tool_diversity.values()
            if c > 0
        )
    else:
        tool_diversity_entropy = None

    return RunMetrics(
        belief_revision_count=belief_revision_count,
        refutation_rate=refutation_rate,
        modality_coverage=modality_coverage,
        tool_diversity=tool_diversity,
        hypothesis_count=n_hypotheses,
        tool_call_count=data.get("tool_call_count", 0),
        completed=ws.final_report_path.exists(),
        martingale_score=martingale_score_val,
        martingale_n=martingale_n,
        directional_flip_count=flip_count,
        cost_usd=cost_usd,
        turn_count=_parse_turn_count(transcript_path),
        revisions_per_hypothesis=revisions_per_hypothesis,
        tool_diversity_entropy=tool_diversity_entropy,
        tool_diversity_entropy_ledger=_normalized_entropy(ledger_counts, _LEDGER_TOOL_COUNT),
        tool_diversity_entropy_universe=_normalized_entropy(universe_counts, _TOOLUNIVERSE_TOOL_COUNT),
        universe_tool_counts=universe_counts,
        cost_unpriced_models=unpriced,
    )


def print_report(metrics: RunMetrics) -> None:
    print("=== Run Analysis ===")
    print(f"Completed:             {metrics.completed}")
    print(f"Hypotheses explored:   {metrics.hypothesis_count}")
    print(f"Total tool calls:      {metrics.tool_call_count}")
    print(f"Belief revisions:      {metrics.belief_revision_count}")
    rph = metrics.revisions_per_hypothesis
    print(f"Revisions/hypothesis:  {rph:.2f}" if rph is not None else "Revisions/hypothesis:  N/A")
    print(f"Refutation rate:       {metrics.refutation_rate:.1%}")
    ms = metrics.martingale_score
    ms_str = f"{ms:.3f} (n={metrics.martingale_n} pairs)" if ms is not None else f"N/A (n={metrics.martingale_n} pairs, need ≥3)"
    print(f"Martingale score:      {ms_str}")
    print(f"Directional flips:     {metrics.directional_flip_count}")
    tc = metrics.turn_count
    print(f"Turns used:            {tc if tc is not None else 'N/A'}")
    cost_str = f"${metrics.cost_usd:.4f}" if metrics.cost_usd is not None else "N/A"
    if metrics.cost_unpriced_models:
        cost_str += f"  (excludes unpriced: {', '.join(metrics.cost_unpriced_models)})"
    print(f"Estimated cost:        {cost_str}")

    def _fmt_entropy(e: float | None) -> str:
        return f"{e:.3f}" if e is not None else "N/A"

    print(f"Tool diversity H (ledger):    {_fmt_entropy(metrics.tool_diversity_entropy_ledger)}  (normalized, 6 ops available)")
    print(f"Tool diversity H (universe):  {_fmt_entropy(metrics.tool_diversity_entropy_universe)}  (normalized, ~600 tools available)")
    print("\nModality coverage per hypothesis:")
    for h_id, sources in metrics.modality_coverage.items():
        print(f"  {h_id}: {sorted(sources)}")
    print("\nToolUniverse tools run:")
    for tool, count in sorted(metrics.universe_tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")
    print("\nTool usage (all tools):")
    for tool, count in sorted(metrics.tool_diversity.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m reviewer.log_replay <run_dir>")
        sys.exit(1)
    print_report(analyze_run(sys.argv[1]))
