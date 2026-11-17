"""MCP server: the agent's tools for writing to the run ledger (`ledger.jsonl`).

The tools are the same for every task. What a hypothesis is made of (attributes) and any rules
about hypotheses (constraints) come from the task, through the `run_info` entry the harness writes
at the start of the run (or `task_spec.yaml` in the run directory when there is none).
"""
from __future__ import annotations

import json
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .ledger import (LIVE_STATUSES, NOTE_KINDS, RELATIONS, STATUSES, Ledger, LedgerError,
                     link_source)

mcp = FastMCP("hypothesis-ledger")
_ledger: Optional[Ledger] = None


def _get_ledger() -> Ledger:
    global _ledger
    if _ledger is None:
        _ledger = Ledger(os.environ.get("EVAL_RUN_DIR", "/workspace/runs/run_001"))
    return _ledger


def _reset_for_tests(run_dir) -> Ledger:
    global _ledger
    _ledger = Ledger(run_dir)
    return _ledger


def _error(e: Exception) -> str:
    return json.dumps({"status": "rejected", "error": str(e)})


# --- validation helpers ---

def _confidence(x) -> float:
    try:
        c = float(x)
    except (TypeError, ValueError):
        raise LedgerError(f"confidence must be a number between 0 and 1, got {x!r}")
    if not 0 <= c <= 1:
        raise LedgerError(f"confidence must be between 0 and 1, got {c}")
    return c


def _text(name: str, x) -> str:
    if not isinstance(x, str) or not x.strip():
        raise LedgerError(f"{name} is required and must be non-empty text")
    return x.strip()


def _str_list(name: str, x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if not isinstance(x, list) or not all(isinstance(i, str) for i in x):
        raise LedgerError(f"{name} must be a list of strings")
    return list(x)


def _attributes(given, rules: dict) -> dict:
    declared = {a["name"]: a for a in rules.get("hypothesis_attributes", [])}
    given = given or {}
    if not isinstance(given, dict):
        raise LedgerError("attributes must be an object, e.g. {\"name\": \"value\"}")
    if not declared:
        if given:
            raise LedgerError("this task declares no hypothesis attributes; pass attributes={} "
                              "and put everything in the hypothesis text")
        return {}
    unknown = sorted(set(given) - set(declared))
    if unknown:
        raise LedgerError(f"unknown attribute(s) {unknown}; this task's attributes are "
                          f"{sorted(declared)}")
    missing = [n for n, a in declared.items() if a.get("required", True) and not str(given.get(n, "")).strip()]
    if missing:
        raise LedgerError(f"missing required attribute(s) {missing}; this task's attributes: " +
                          "; ".join(f"{n}: {a.get('description', '')}".rstrip(": ") for n, a in declared.items()))
    return {k: v for k, v in given.items()}


def _check_constraints(attributes: dict, rules: dict, ledger: Ledger) -> None:
    for c in rules.get("hypothesis_constraints", []):
        keys = c.get("unique")
        if not keys:
            continue
        among = c.get("among") or list(LIVE_STATUSES)
        mine = tuple(str(attributes.get(k, "")).strip().lower() for k in keys)
        for hid, entries in ledger.hypotheses().items():
            latest = entries[-1]
            theirs = tuple(str(latest.get("attributes", {}).get(k, "")).strip().lower() for k in keys)
            if theirs == mine and latest.get("status") in among:
                shown = ", ".join(f"{k}={attributes.get(k)}" for k in keys)
                raise LedgerError(
                    f"{hid} is already a {latest['status']} hypothesis with {shown}; this task allows "
                    f"one at a time. Update {hid} instead, or first mark it 'abandoned' with "
                    f"update_hypothesis and then write the new one with derived_from=\"{hid}\".")


def _evidence(items, hypothesis_id: str, ledger: Ledger, ids, prior: set[str]) -> tuple[list, list]:
    """Validate and number evidence items, linking new ones to recorded tool calls."""
    if isinstance(items, (dict, str)):
        items = [items]
    if not isinstance(items, list) or not items:
        raise LedgerError("evidence is required: give at least one item — a new finding "
                          "{finding, source, relation} or a reference to earlier evidence "
                          "{ref: \"E3\", relation}")
    known = ledger.evidence_items()
    calls = list(ledger.tool_calls().values())
    out, warnings = [], []
    for n, item in enumerate(items, 1):
        if isinstance(item, str):
            item = {"finding": item}
        if not isinstance(item, dict):
            raise LedgerError(f"evidence item {n} must be an object")
        relation = item.get("relation")
        if relation not in RELATIONS:
            raise LedgerError(f"evidence item {n}: relation must be one of {list(RELATIONS)} "
                              f"(relative to this hypothesis), got {relation!r}")
        if "ref" in item:
            ref = item["ref"]
            if ref not in known:
                raise LedgerError(f"evidence item {n}: no evidence {ref!r} in the ledger")
            out.append({"ref": ref, "relation": relation})
            continue
        finding = item.get("finding")
        if not isinstance(finding, str) or not finding.strip():
            raise LedgerError(f"evidence item {n}: 'finding' (a factual statement of the result) is required")
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        tool_call_ids, link, candidates = link_source(source, calls)
        new = {"evidence_id": ids.next("E"), "finding": finding.strip(), "source": source,
               "relation": relation, "tool_call_ids": tool_call_ids, "link": link}
        if link == "probable":
            new["link_candidates"] = candidates
        q = item.get("quantities")
        if q:
            new["quantities"] = q if isinstance(q, list) else [q]
        extra = {k: v for k, v in item.items()
                 if k not in ("finding", "source", "relation", "quantities")}
        if extra:
            new["extra"] = extra
        if link == "none":
            warnings.append(f"{new['evidence_id']}: no recorded source — no tool call matched "
                            f"source={json.dumps(source)}; it will be shown to reviewers as "
                            "'no recorded source'")
        out.append(new)
    return out, warnings


# --- tools ---

def _tool_write_hypothesis(hypothesis, attributes, confidence, evidence, rationale,
                           open_questions=None, derived_from=None) -> str:
    led = _get_ledger()
    try:
        led.sync_stream()
        rules = led.task_rules()
        text = _text("hypothesis", hypothesis)
        attrs = _attributes(attributes, rules)
        conf = _confidence(confidence)
        why = _text("rationale", rationale)
        questions = _str_list("open_questions", open_questions)
        if derived_from is not None and derived_from not in led.hypotheses():
            raise LedgerError(f"derived_from: no hypothesis {derived_from!r}")
        warnings: list[str] = []

        def build(ids):
            _check_constraints(attrs, rules, led)
            items, w = _evidence(evidence, "", led, ids, set())
            warnings.extend(w)
            return [{"kind": "hypothesis", "action": "create", "hypothesis_id": ids.next("H"),
                     "hypothesis": text, "attributes": attrs, "derived_from": derived_from,
                     "confidence": conf, "status": "active", "evidence": items,
                     "evidence_withdrawn": [], "rationale": why, "open_questions": questions}]

        [entry] = led.transact(build, "agent")
    except LedgerError as e:
        return _error(e)
    return json.dumps({"status": "recorded", "hypothesis_id": entry["hypothesis_id"],
                       "entry_id": entry["entry_id"], "evidence": _evidence_summary(entry),
                       "warnings": warnings})


def _evidence_summary(entry: dict) -> list[dict]:
    return [{k: i[k] for k in ("evidence_id", "link", "tool_call_ids") if k in i}
            for i in entry["evidence"] if "evidence_id" in i]


def _cited(entries: list[dict]) -> set[str]:
    ids = set()
    for e in entries:
        for i in e.get("evidence", []):
            ids.add(i.get("evidence_id") or i.get("ref"))
    return ids


def _tool_update_hypothesis(hypothesis_id, confidence, status, evidence, rationale,
                            evidence_withdrawn=None, open_questions=None) -> str:
    led = _get_ledger()
    try:
        led.sync_stream()
        entries = led.hypotheses().get(hypothesis_id)
        if not entries:
            raise LedgerError(f"no hypothesis {hypothesis_id!r}; existing: {sorted(led.hypotheses())}")
        conf = _confidence(confidence)
        if status not in STATUSES:
            raise LedgerError(f"status must be one of {list(STATUSES)}, got {status!r}")
        why = _text("rationale", rationale)
        withdrawn = _str_list("evidence_withdrawn", evidence_withdrawn)
        cited = _cited(entries)
        bad = [w for w in withdrawn if w not in cited]
        if bad:
            raise LedgerError(f"evidence_withdrawn: {bad} were never cited for {hypothesis_id}")
        first, latest = entries[0], entries[-1]
        questions = latest.get("open_questions", []) if open_questions is None else \
            _str_list("open_questions", open_questions)
        warnings: list[str] = []

        def build(ids):
            if status in LIVE_STATUSES and latest.get("status") not in LIVE_STATUSES:
                _check_constraints(first.get("attributes", {}), led.task_rules(), led)
            items, w = _evidence(evidence, hypothesis_id, led, ids, cited)
            warnings.extend(w)
            return [{"kind": "hypothesis", "action": "update", "hypothesis_id": hypothesis_id,
                     "hypothesis": first["hypothesis"], "attributes": first.get("attributes", {}),
                     "derived_from": first.get("derived_from"), "confidence": conf, "status": status,
                     "evidence": items, "evidence_withdrawn": withdrawn, "rationale": why,
                     "open_questions": questions}]

        [entry] = led.transact(build, "agent")
    except LedgerError as e:
        return _error(e)
    return json.dumps({"status": "recorded", "hypothesis_id": hypothesis_id,
                       "entry_id": entry["entry_id"], "evidence": _evidence_summary(entry),
                       "warnings": warnings})


def _tool_read_hypotheses(status_filter: str = "live") -> str:
    led = _get_ledger()
    out = []
    for hid, entries in led.hypotheses().items():
        latest = entries[-1]
        if status_filter == "live" and latest["status"] not in LIVE_STATUSES:
            continue
        if status_filter not in ("live", "all") and latest["status"] != status_filter:
            continue
        withdrawn = {w for e in entries for w in e.get("evidence_withdrawn", [])}
        out.append({"hypothesis_id": hid, "hypothesis": latest["hypothesis"],
                    "attributes": latest.get("attributes", {}), "derived_from": latest.get("derived_from"),
                    "confidence": latest["confidence"], "status": latest["status"],
                    "evidence_ids": sorted(_cited(entries) - withdrawn, key=lambda s: int(s[1:])),
                    "evidence_withdrawn": sorted(withdrawn), "entries": len(entries),
                    "open_questions": latest.get("open_questions", [])})
    return json.dumps({"hypotheses": out, "total": len(led.hypotheses())}, indent=2)


def _tool_recent_tool_calls(n: int = 10) -> str:
    led = _get_ledger()
    led.sync_stream()
    calls = list(led.tool_calls().values())[-max(1, min(int(n), 50)):]
    return json.dumps([{"tool_call_id": c["tool_call_id"], "tool": c["tool"],
                        "arguments": json.dumps(c.get("arguments"), ensure_ascii=False)[:200],
                        "error": c.get("error"), "output": (c.get("output") or "")[:200]}
                       for c in calls], indent=2)


def _tool_log_note(kind, text, hypothesis_ids=None) -> str:
    led = _get_ledger()
    try:
        if kind not in NOTE_KINDS:
            raise LedgerError(f"kind must be one of {list(NOTE_KINDS)}, got {kind!r}")
        body = _text("text", text)
        hids = _str_list("hypothesis_ids", hypothesis_ids)
        unknown = [h for h in hids if h not in led.hypotheses()]
        if unknown:
            raise LedgerError(f"hypothesis_ids: no hypothesis {unknown}")
        [entry] = led.append([{"kind": "note", "note_kind": kind, "text": body,
                               "hypothesis_ids": hids}], "agent")
    except LedgerError as e:
        return _error(e)
    return json.dumps({"status": "recorded", "entry_id": entry["entry_id"]})


def _tool_request_human_input(question, context) -> str:
    led = _get_ledger()
    human_enabled = os.environ.get("ALLOW_HUMAN_INPUT", "false").lower() == "true"
    answer = None
    if human_enabled:
        print(f"\n[HUMAN INPUT]\nContext: {context}\nQuestion: {question}\nAnswer: ", flush=True)
        answer = input()
    [entry] = led.append([{"kind": "human_input", "question": question, "context": context,
                           "mode": "human" if human_enabled else "headless", "answer": answer}], "agent")
    if not human_enabled:
        return json.dumps({"status": "headless_mode", "entry_id": entry["entry_id"],
                           "message": "No human is available in this run; your question is recorded."})
    return json.dumps({"status": "received", "answer": answer})


def _tool_declare_done(ranked_hypothesis_ids, rationale, confidence_summary, validation) -> str:
    led = _get_ledger()
    try:
        ranked = _str_list("ranked_hypothesis_ids", ranked_hypothesis_ids)
        if not ranked:
            raise LedgerError("ranked_hypothesis_ids is required: your final hypotheses, best first")
        known = led.hypotheses()
        unknown = [h for h in ranked if h not in known]
        if unknown:
            raise LedgerError(f"ranked_hypothesis_ids: no hypothesis {unknown}")
        why = _text("rationale", rationale)
        warnings = []
        if isinstance(validation, str):
            validation = {"run": validation}
            warnings.append("validation was one text for the run; give one plan per hypothesis "
                            "({\"H1\": \"...\"}) so each is judged on its own plan")
        if not isinstance(validation, dict):
            raise LedgerError("validation must map hypothesis ids to a proposed validation experiment")
        missing = [h for h in ranked if h not in validation]
        if missing and "run" not in validation:
            warnings.append(f"no validation plan for {missing}")
        [entry] = led.append([{"kind": "conclusion", "ranked_hypothesis_ids": ranked, "rationale": why,
                               "confidence_summary": confidence_summary, "validation": validation}],
                             "agent")
    except LedgerError as e:
        return _error(e)
    return json.dumps({"status": "completed", "entry_id": entry["entry_id"], "warnings": warnings})


# --- FastMCP registrations ---

@mcp.tool()
def write_hypothesis(hypothesis: str, attributes: dict, confidence: float, evidence: list[dict],
                     rationale: str, open_questions: list[str] = [], derived_from: str = "") -> str:
    """Record a new hypothesis. Its text and attributes can never be changed afterwards: to change
    them, write a new hypothesis with derived_from set to the old id.

    hypothesis: what you propose is true, in one or two sentences.
    attributes: the fields this task asks for (see your task instructions); {} if none.
    confidence: 0-1.
    evidence: at least one item. A new item is {"finding": factual statement of what a result shows
      (numbers, not interpretation), "source": {"tool": name of the tool you called, "match": short
      text from that call's arguments} or {"citation": "PMID ..."}, "relation": "supports" |
      "contradicts" | "context", "quantities": optional [{"name","value","unit"}]}. An earlier item
      is {"ref": "E3", "relation": ...}.
    rationale: why your confidence is what it is, given this evidence.
    """
    return _tool_write_hypothesis(hypothesis, attributes, confidence, evidence, rationale,
                                  open_questions, derived_from or None)


@mcp.tool()
def update_hypothesis(hypothesis_id: str, confidence: float, status: str, evidence: list[dict],
                      rationale: str, evidence_withdrawn: list[str] = [],
                      open_questions: list[str] = []) -> str:
    """Record a change in belief about a hypothesis: new evidence, confidence, or status.

    status: active | supported | weakened | refuted | abandoned.
    evidence: at least one item — the new evidence that prompted this update, or references
      ({"ref": "E3", "relation": ...}) to the earlier evidence this decision rests on. Same item format
      as write_hypothesis.
    rationale: why confidence and status are what they are now, given this evidence.
    evidence_withdrawn: ids of earlier evidence you no longer rely on (say why in the rationale).
    open_questions: the current list of open questions; leave empty to keep the previous list.
    The hypothesis text and attributes can't be changed here; write a new hypothesis instead.
    """
    return _tool_update_hypothesis(hypothesis_id, confidence, status, evidence, rationale,
                                   evidence_withdrawn, open_questions or None)


@mcp.tool()
def read_hypotheses(status_filter: str = "live") -> str:
    """Current state of each hypothesis: 'live' (active, supported, weakened), 'all', or one status."""
    return _tool_read_hypotheses(status_filter)


@mcp.tool()
def recent_tool_calls(n: int = 10) -> str:
    """The last n tool calls recorded in the ledger, with their ids (T1, T2, ...). Optional: cite
    an id in an evidence source as {"tool_call_ids": ["T12"]}."""
    return _tool_recent_tool_calls(n)


@mcp.tool()
def log_note(kind: str, text: str, hypothesis_ids: list[str] = []) -> str:
    """Record a note in the ledger. kind: interpretation (what a result means, linked to the
    hypotheses it bears on) | plan | debug | other."""
    return _tool_log_note(kind, text, hypothesis_ids)


@mcp.tool()
def request_human_input(question: str, context: str) -> str:
    """Ask a human expert. In headless runs the question is recorded and no answer comes back."""
    return _tool_request_human_input(question, context)


@mcp.tool()
def declare_done(ranked_hypothesis_ids: list[str], rationale: str, confidence_summary: str,
                 validation: dict) -> str:
    """Finish the task. ranked_hypothesis_ids: your final hypotheses, best first. validation: one
    proposed validation experiment per ranked hypothesis, {"H1": "...", "H2": "..."}."""
    return _tool_declare_done(ranked_hypothesis_ids, rationale, confidence_summary, validation)


def run():
    mcp.run()


if __name__ == "__main__":
    run()
