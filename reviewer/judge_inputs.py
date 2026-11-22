"""Judge inputs, selected and formatted from ledger.jsonl fields only (judging-v2 §4.1).

Nothing here infers, summarises or rewords: each renderer prints named fields of named entry kinds
in ledger order, with one fixed template for every agent. The only change to any value is the
per-output budget. A cited tool output longer than `max_output_chars` is shown as an excerpt: its
start, then the passages where the evidence item's own numbers and identifiers occur (matched as
the ledger's source linking matches them, D-60), each marked with its character positions and the
gaps marked as not shown. With no match past the start, the output is cut at the budget and the
cut is marked. Which pass reads which field:

- evidence pass (E): hypothesis text and attributes; every evidence item the hypothesis ever cited
  (withdrawn ones marked) with the cited calls' arguments, output and error; latest open
  questions; the hypothesis's validation plan and declared artifacts when a criterion asks. Never a
  rationale, confidence or status.
- reasoning pass (R): the hypothesis's entries in order, each with its rationale, confidence,
  status and the evidence cited in that entry; derived_from; interpretation notes linked to it;
  its validation plan.
- run-level pass: every hypothesis's text, attributes, final status and confidence; counts; notes;
  human input; uncited tool calls; visible agent messages; the conclusion; run_info; artifacts.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path

import re

from hypothesis_ledger.ledger import _IDENTIFIER, _NUMBER, _OUTPUT_NUMBER, Ledger, _parse_number

MAX_OUTPUT_CHARS = 2000
UNCITED_OUTPUT_CHARS = 200
NO_SOURCE = "no recorded source"
RUN_PLAN_KEY = "run"   # conclusion.validation key of runs converted from v1
RUN_PLAN_NOTE = "(one plan for the whole run, not written for this hypothesis) "


EXCERPT_HEAD = 400      # characters of an output's start always shown
EXCERPT_BEFORE = 150    # characters shown before a matched value (a label usually precedes it)
EXCERPT_AFTER = 450     # and after it
_SYMBOL = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{2,9}(?![A-Za-z0-9])")   # gene-like: MAN2A1, SSB
_COMMON_CAPS = {"SLE", "GWAS", "PBMC", "PBMCS", "DNA", "RNA", "AUC", "IFN", "ISG", "ISGS", "DEG",
                "DEGS", "LOG", "THE", "AND", "NOT", "FDR", "CI", "OR"}


def _targets(finding: str, quantities) -> tuple[list[tuple[float, float]], list[str]]:
    """The item's numbers (value, half a unit in the last written place) and identifiers."""
    texts = _NUMBER.findall(finding or "")
    for q in quantities or []:
        if isinstance(q, dict) and isinstance(q.get("value"), (int, float)) and not isinstance(q["value"], bool):
            texts.append(repr(float(q["value"])) if isinstance(q["value"], float) else str(q["value"]))
    nums = []
    for t in texts:
        v, tol = _parse_number(t)
        if tol > 0 and (v, tol) not in nums:
            nums.append((v, tol))
    symbols = {w for w in _SYMBOL.findall(finding or "") if w not in _COMMON_CAPS}
    return nums, sorted({i for i in _IDENTIFIER.findall(finding or "")} | symbols)


def _matches(x: float, v: float, tol: float) -> bool:
    eps = tol * 1e-9
    for y in (x, x * 100):                       # a fraction the agent wrote as a percentage
        if abs(y - v) <= tol + eps or ((y >= 0) == (v >= 0) and 0 <= abs(y) - abs(v) < 2 * tol + eps):
            return True
    return False


def excerpt(text: str | None, limit: int, finding: str = "", quantities=None) -> str:
    """`text` within `limit` characters: whole if it fits; else its start plus the passages that
    contain the evidence item's numbers and identifiers, chosen greedily by how many not-yet-shown
    values each adds; else (nothing matched past the start) cut at the limit."""
    text = "" if text is None else str(text)
    if not limit or len(text) <= limit:
        return text
    nums, idents = _targets(finding, quantities)
    hits: list[tuple[int, int]] = []               # (position, target index)
    if nums:
        for m in _OUTPUT_NUMBER.finditer(text):
            try:
                x = float(m.group())
            except ValueError:
                continue
            hits += [(m.start(), k) for k, (v, tol) in enumerate(nums) if _matches(x, v, tol)]
    for j, ident in enumerate(idents):
        hits += [(m.start(), len(nums) + j) for m in
                 re.finditer(r"(?<![A-Za-z0-9])" + re.escape(ident) + r"(?![A-Za-z0-9])", text, re.I)]
    head = min(EXCERPT_HEAD, limit)
    covered = {k for p, k in hits if p < head}
    spans: list[tuple[int, int]] = []
    budget = limit - head
    while budget > 0:
        best = None
        for p, _ in hits:
            if p < head or any(a <= p < b for a, b in spans):
                continue
            a, b = max(head, p - EXCERPT_BEFORE), min(len(text), p + EXCERPT_AFTER)
            gain = len({k for q, k in hits if a <= q < b} - covered)
            if gain and (best is None or gain > best[0]):
                best = (gain, a, b)
        if best is None:
            break
        _, a, b = best
        b = min(b, a + budget)
        spans.append((a, b))
        covered |= {k for q, k in hits if a <= q < b}
        budget -= b - a
    if not spans:
        return cut(text, limit)
    spans.sort()
    # Budget left over: shared out to extend each passage forward (values follow their labels).
    if budget > 0:
        extra = budget // len(spans)
        spans = [(a, min(len(text), b + extra)) for a, b in spans]
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    parts = [f"[excerpt of {len(text):,} characters: the start and the passages containing this item's "
             "values; the rest is not shown]", text[:head]]
    last = head
    for a, b in merged:
        if a > last:
            parts.append(f"[... characters {last:,}-{a:,} not shown ...]")
        parts.append(text[max(a, last):b])
        last = b
    if last < len(text):
        parts.append(f"[... characters {last:,}-{len(text):,} not shown ...]")
    return "\n".join(parts)


def cut(text: str | None, limit: int) -> str:
    text = "" if text is None else str(text)
    if limit and len(text) > limit:
        return f"{text[:limit]}\n[output cut at {limit:,} of {len(text):,} characters]"
    return text


@dataclass
class EvidenceView:
    evidence_id: str
    finding: str
    source: dict
    relation: str
    link: str
    tool_call_ids: list[str]
    quantities: dict | None
    extra: dict | None
    recorded_in: str                 # entry_id that recorded it
    withdrawn_in: str | None = None  # entry_id of this hypothesis that withdrew it


@dataclass
class EntryView:
    entry_id: str
    action: str
    rationale: str
    confidence: float | None
    status: str
    cited: list[tuple[str, str]]      # (evidence_id, relation as cited in this entry)
    withdrawn: list[str]
    open_questions: list[str]


@dataclass
class HypothesisView:
    hypothesis_id: str
    hypothesis: str
    attributes: dict
    derived_from: str | None
    entries: list[EntryView]
    evidence: dict[str, EvidenceView]          # every item ever cited, in first-cited order
    notes: list[dict]                          # interpretation notes linked to it

    @property
    def final(self) -> EntryView:
        return self.entries[-1]


@dataclass
class RunView:
    run_dir: Path
    run_info: dict
    run_end: dict
    hypotheses: dict[str, HypothesisView]
    tool_calls: dict[str, dict]
    conclusion: dict | None
    notes: list[dict]
    human_inputs: list[dict]
    agent_messages: list[dict]                 # visible only
    artifacts: list[dict]
    cited_call_ids: set[str] = field(default_factory=set)

    def validation_for(self, hid: str) -> str:
        plans = (self.conclusion or {}).get("validation") or {}
        plan = str(plans.get(hid, "") or "").strip()
        if not plan and set(plans) == {RUN_PLAN_KEY} and str(plans[RUN_PLAN_KEY] or "").strip():
            # Runs converted from v1 have one plan for the whole run (final_report.suggested_validation).
            plan = RUN_PLAN_NOTE + str(plans[RUN_PLAN_KEY]).strip()
        return plan


def build_run_view(run_dir) -> RunView:
    led = Ledger(run_dir)
    if not led.exists():
        raise FileNotFoundError(f"no ledger.jsonl in {run_dir}")
    items = led.evidence_items()
    calls = led.tool_calls()
    notes = led.entries("note")
    hyps: dict[str, HypothesisView] = {}
    cited_calls: set[str] = set()
    for hid, entries in led.hypotheses().items():
        first = entries[0]
        evidence: dict[str, EvidenceView] = {}
        views = []
        for e in entries:
            cited = []
            for item in e.get("evidence") or []:
                eid = item.get("evidence_id") or item.get("ref")
                if not eid or eid not in items:
                    continue
                src = items[eid]
                cited.append((eid, item.get("relation") or src.get("relation", "")))
                if eid not in evidence:
                    evidence[eid] = EvidenceView(
                        eid, src.get("finding", ""), src.get("source") or {}, src.get("relation", ""),
                        src.get("link", "none"), list(src.get("tool_call_ids") or []),
                        src.get("quantities"), src.get("extra"), src["recorded_in"])
                    cited_calls.update(evidence[eid].tool_call_ids)
            for eid in e.get("evidence_withdrawn") or []:
                if eid in evidence and evidence[eid].withdrawn_in is None:
                    evidence[eid].withdrawn_in = e["entry_id"]
            views.append(EntryView(e["entry_id"], e.get("action", ""), e.get("rationale", ""),
                                   e.get("confidence"), e.get("status", ""), cited,
                                   list(e.get("evidence_withdrawn") or []),
                                   list(e.get("open_questions") or [])))
        linked = [n for n in notes
                  if n.get("note_kind") == "interpretation" and hid in (n.get("hypothesis_ids") or [])]
        hyps[hid] = HypothesisView(hid, first.get("hypothesis", ""), first.get("attributes") or {},
                                   first.get("derived_from"), views, evidence, linked)
    return RunView(
        run_dir=Path(run_dir), run_info=led.run_info("start") or {}, run_end=led.run_info("end") or {},
        hypotheses=hyps, tool_calls=calls, conclusion=led.conclusion(), notes=notes,
        human_inputs=led.entries("human_input"),
        agent_messages=[m for m in led.entries("agent_message") if m.get("visibility") == "visible"],
        artifacts=led.entries("artifact"), cited_call_ids=cited_calls)


# --- rendering -----------------------------------------------------------------------------------

def _source_text(ev: EvidenceView) -> str:
    s = ev.source or {}
    if s.get("citation"):
        return f"citation: {s['citation']}"
    if s.get("tool") or s.get("match"):
        return f"tool: {s.get('tool', '')}; match: {s.get('match', '')}"
    if s.get("unrecorded_reason"):      # e.g. runs converted from v1, whose tools took no source
        return f"not recorded ({s['unrecorded_reason']})"
    return NO_SOURCE


def _render_calls(run: RunView, ev: EvidenceView, max_chars: int) -> list[str]:
    out = []
    for tid in ev.tool_call_ids:
        c = run.tool_calls.get(tid)
        if c is None:
            continue
        out.append(f"    Tool call {tid}: {c.get('tool')}"
                   + (f" (via {c['via']})" if c.get("via") else ""))
        out.append(f"    Arguments: {cut(json.dumps(c.get('arguments'), ensure_ascii=False), max_chars)}")
        if c.get("error"):
            out.append(f"    Error: {c['error']}")
        out.append("    Output:\n" + "\n".join("      " + line for line in
                                              excerpt(c.get("output"), max_chars, ev.finding,
                                                      ev.quantities).splitlines() or [""]))
    return out


def _render_item(run: RunView, ev: EvidenceView, max_chars: int, relation: str | None = None,
                 with_calls: bool = True) -> list[str]:
    head = f"[{ev.evidence_id}] ({relation or ev.relation})"
    if ev.withdrawn_in:
        head += f" WITHDRAWN in {ev.withdrawn_in}"
    lines = [head, f"    Finding: {ev.finding}", f"    Source: {_source_text(ev)}; link: {ev.link}"]
    if ev.quantities:
        lines.append(f"    Quantities: {json.dumps(ev.quantities, ensure_ascii=False)}")
    if ev.link in ("exact", "probable") and with_calls:
        lines += _render_calls(run, ev, max_chars)
    elif ev.link in ("none",) and not ev.source.get("citation") and not ev.source.get("unrecorded_reason"):
        lines.append(f"    ({NO_SOURCE})")
    return lines


def _header(h: HypothesisView) -> list[str]:
    attrs = "; ".join(f"{k}: {v}" for k, v in h.attributes.items())
    return [f"Hypothesis {h.hypothesis_id}: {h.hypothesis}", f"Attributes: {attrs or '(none)'}"]


def render_evidence_input(run: RunView, hid: str, *, max_output_chars: int = MAX_OUTPUT_CHARS,
                          include_validation: bool = False, artifact_globs=()) -> str:
    """Pass E input for one hypothesis: findings and the tool outputs behind them, no interpretation."""
    h = run.hypotheses[hid]
    lines = _header(h) + ["", f"Evidence ({len(h.evidence)} items, every item this hypothesis cited):"]
    for ev in h.evidence.values():
        lines += _render_item(run, ev, max_output_chars)
    oq = h.final.open_questions
    lines += ["", "Open questions (latest): " + ("; ".join(oq) if oq else "none recorded")]
    if include_validation:
        lines += ["", "Validation experiment proposed for this hypothesis: "
                  + (run.validation_for(hid) or "none")]
    lines += _render_artifacts(run, artifact_globs, max_output_chars)
    return "\n".join(lines)


def render_reasoning_input(run: RunView, hid: str, *, max_output_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Pass R input for one hypothesis: each entry's rationale beside the evidence it cited."""
    h = run.hypotheses[hid]
    lines = _header(h)
    if h.derived_from:
        lines.append(f"Derived from: {h.derived_from}")
    for n, e in enumerate(h.entries, 1):
        lines += ["", f"Entry {e.entry_id} ({n} of {len(h.entries)}, {e.action}): "
                  f"confidence {e.confidence}, status {e.status}",
                  f"  Rationale: {e.rationale or '(none recorded)'}"]
        if e.withdrawn:
            lines.append(f"  Withdrew evidence: {', '.join(e.withdrawn)}")
        lines.append(f"  Evidence cited in this entry ({len(e.cited)}):")
        for eid, relation in e.cited:
            lines += ["  " + line for line in _render_item(run, h.evidence[eid], max_output_chars, relation)]
        if e.open_questions:
            lines.append(f"  Open questions: {'; '.join(e.open_questions)}")
    lines += ["", "Interpretation notes linked to this hypothesis:"]
    lines += [f"  Note {n['entry_id']}: {n.get('text', '')}" for n in h.notes] or ["  none recorded"]
    lines += ["", "Validation experiment proposed for this hypothesis: " + (run.validation_for(hid) or "none")]
    return "\n".join(lines)


def _render_artifacts(run: RunView, globs, max_chars: int) -> list[str]:
    lines = []
    for glob in globs:
        found = [a for a in run.artifacts if fnmatch.fnmatch(a["path"], glob)]
        lines += ["", f"Artifacts matching {glob!r}: {len(found)}"]
        for a in found:
            body = a.get("content")
            lines.append(f"  {a['path']} ({a.get('bytes')} bytes)"
                         + ("" if body is not None else " [binary or too large; not inlined]"))
            if body is not None:
                lines += ["    " + line for line in cut(body, max_chars).splitlines()]
    return lines


def run_counts(run: RunView) -> dict:
    final = [h.final for h in run.hypotheses.values()]
    return {
        "hypotheses": len(run.hypotheses),
        "updates": sum(len(h.entries) - 1 for h in run.hypotheses.values()),
        "by_final_status": {s: sum(1 for e in final if e.status == s)
                            for s in sorted({e.status for e in final})},
        "evidence_items": sum(len(h.evidence) for h in run.hypotheses.values()),
        "contradicting_items": sum(1 for h in run.hypotheses.values() for ev in h.evidence.values()
                                   if ev.relation == "contradicts"),
        "tool_calls": len(run.tool_calls),
        "failed_tool_calls": sum(1 for c in run.tool_calls.values() if c.get("error")),
        "concluded": run.conclusion is not None,
    }


def render_run_input(run: RunView, inputs=(), *, artifact_globs=(),
                     max_output_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Run-level input: the whole run at the level of hypotheses, plus what each criterion declares."""
    inputs = set(inputs)
    info, end = run.run_info, run.run_end
    lines = [f"Run: agent model {info.get('model_requested')} via {info.get('harness')} "
             f"{info.get('cli_version')}; turn cap {info.get('turn_cap')}; "
             f"model responses {((end.get('outcome') or {}).get('model_responses'))}; "
             f"stopped at cap: {((end.get('outcome') or {}).get('cap_hit'))}",
             f"Counts: {json.dumps(run_counts(run))}", "", "Hypotheses (final state):"]
    for h in run.hypotheses.values():
        f = h.final
        lines.append(f"  {h.hypothesis_id}: {h.hypothesis} | "
                     + "; ".join(f"{k}: {v}" for k, v in h.attributes.items())
                     + f" | final status {f.status}, confidence {f.confidence}, "
                     f"{len(h.entries) - 1} updates, {len(h.evidence)} evidence items"
                     + (f", derived from {h.derived_from}" if h.derived_from else ""))
        if "confidence_trajectory" in inputs:
            lines.append("    trajectory: " + " -> ".join(f"{e.confidence} ({e.status})" for e in h.entries))
        if "validation_plan" in inputs:
            lines.append(f"    validation: {run.validation_for(h.hypothesis_id) or 'none'}")
    if "evidence" in inputs:
        lines += ["", "Evidence findings (all hypotheses):"]
        for h in run.hypotheses.values():
            for ev in h.evidence.values():
                lines.append(f"  [{ev.evidence_id}] {h.hypothesis_id} ({ev.relation}, link {ev.link})"
                             + (" WITHDRAWN" if ev.withdrawn_in else "") + f": {ev.finding}")
    c = run.conclusion
    lines += ["", "Conclusion (declare_done): " + ("none — the run ended without declare_done" if c is None else "")]
    if c is not None:
        lines += [f"  Ranked: {', '.join(c.get('ranked_hypothesis_ids') or [])}",
                  f"  Rationale: {c.get('rationale', '')}",
                  f"  Confidence summary: {c.get('confidence_summary', '')}"]
    if "notes" in inputs:
        lines += ["", "Notes:"] + ([f"  {n['entry_id']} [{n.get('note_kind')}]"
                                     + (f" ({', '.join(n.get('hypothesis_ids') or [])})" if n.get('hypothesis_ids') else "")
                                     + f": {n.get('text', '')}" for n in run.notes] or ["  none recorded"])
    if "human_input" in inputs:
        lines += ["", "Human input:"] + ([f"  Q: {h.get('question')} | context: {h.get('context')} | "
                                          f"answer: {h.get('answer')}" for h in run.human_inputs]
                                         or ["  none recorded"])
    if "tool_calls" in inputs:
        uncited = [c for tid, c in run.tool_calls.items() if tid not in run.cited_call_ids]
        lines += ["", f"Tool calls not cited by any evidence ({len(uncited)} of {len(run.tool_calls)}):"]
        for c in uncited:
            lines.append(f"  {c['tool_call_id']} {c.get('tool')}"
                         + (f" ERROR {c['error']}" if c.get("error") else "")
                         + ": " + cut(c.get("output"), UNCITED_OUTPUT_CHARS).replace("\n", " "))
    if "agent_messages" in inputs:
        lines += ["", "Agent messages (visible text):"] + (
            [f"  {m['entry_id']}: {m.get('text', '')}" for m in run.agent_messages] or ["  none recorded"])
    lines += _render_artifacts(run, artifact_globs, max_output_chars)
    return "\n".join(lines)
