"""Read access to a run directory for the reviewer.

A run recorded with the current ledger has one file, `ledger.jsonl` (see hypothesis_ledger.ledger).
Runs recorded before it have `ledger.json`, `step_log.jsonl`, `run_log.jsonl` and
`final_report.json`.

Interim (until the judge reads ledger.jsonl directly): for a ledger.jsonl run, `read_ledger`,
`read_final_report` and `read_step_log` present the ledger in the old files' shape, so the current
judge, metrics and review page work on new runs. The view only selects and renames fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .ledger import LEDGER_FILE, LIVE_STATUSES, Ledger


class Workspace:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.ledger_path = self.run_dir / "ledger.json"           # old runs
        self.run_log_path = self.run_dir / "run_log.jsonl"        # old runs
        self.step_log_path = self.run_dir / "step_log.jsonl"      # old runs
        self.final_report_path = self.run_dir / "final_report.json"  # old runs
        self.ledger = Ledger(self.run_dir)

    @property
    def is_ledger_run(self) -> bool:
        """True for runs recorded in ledger.jsonl."""
        return (self.run_dir / LEDGER_FILE).exists()

    def is_run(self) -> bool:
        return self.is_ledger_run or self.ledger_path.exists()

    def read_ledger(self) -> dict:
        if self.is_ledger_run:
            return view_hypotheses(self.ledger)
        if not self.ledger_path.exists():
            return {"hypotheses": {}, "tool_call_count": 0}
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def has_final_report(self) -> bool:
        if self.is_ledger_run:
            return self.ledger.conclusion() is not None
        return self.final_report_path.exists()

    def read_final_report(self) -> Optional[dict]:
        if self.is_ledger_run:
            return view_final_report(self.ledger)
        if not self.final_report_path.exists():
            return None
        return json.loads(self.final_report_path.read_text(encoding="utf-8"))

    def read_step_log(self) -> list[dict]:
        if self.is_ledger_run:
            return [{"timestamp": n["time"], "step_type": n["note_kind"], "tool_used": "",
                     "inputs_summary": "", "outputs_summary": "", "interpretation": n["text"],
                     "next_action": "", "hypothesis_ids": n.get("hypothesis_ids", [])}
                    for n in self.ledger.entries("note")]
        return _jsonl(self.step_log_path)

    def read_run_log(self) -> list[dict]:
        if self.is_ledger_run:
            return [{"timestamp": e["time"], "tool": _ledger_tool_of(e)}
                    for e in self.ledger.entries() if e.get("written_by") == "agent"]
        return _jsonl(self.run_log_path)

    @property
    def task_spec_path(self) -> Path:
        return self.run_dir / "task_spec.yaml"

    def read_task_spec(self) -> Optional[dict]:
        if not self.task_spec_path.exists():
            return None
        import yaml
        return yaml.safe_load(self.task_spec_path.read_text(encoding="utf-8"))


_LEDGER_TOOL_OF = {"note": "log_note", "human_input": "request_human_input",
                   "conclusion": "declare_done"}


def _ledger_tool_of(entry: dict) -> str:
    """The ledger tool that wrote an agent entry. (Read-only calls write no entry.)"""
    if entry["kind"] == "hypothesis":
        return "write_hypothesis" if entry.get("action") == "create" else "update_hypothesis"
    return _LEDGER_TOOL_OF.get(entry["kind"], entry["kind"])


def _jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


# --- interim view of ledger.jsonl in the old files' shape --------------------------------------

def _source_label(source: dict) -> str:
    if not isinstance(source, dict):
        return ""
    return str(source.get("tool") or source.get("citation") or "")


def _item_view(item: dict) -> dict:
    out = {"evidence_id": item["evidence_id"], "finding": item["finding"],
           "source": _source_label(item.get("source")), "relation": item.get("relation")}
    if item.get("source", {}).get("citation"):
        out["citation"] = item["source"]["citation"]
    if item.get("quantities"):
        out["quantities"] = item["quantities"]
    out["link"] = item.get("link")
    return out


def _key_attribute(attributes: dict, hypothesis_id: str) -> str:
    for k in ("gene", "candidate_gene"):
        if attributes.get(k):
            return str(attributes[k])
    for v in attributes.values():
        if v:
            return str(v)
    return hypothesis_id


def view_hypotheses(ledger: Ledger) -> dict:
    items = ledger.evidence_items()
    conclusion = ledger.conclusion()
    ranked = set(conclusion["ranked_hypothesis_ids"]) if conclusion else set()
    out = {}
    for hid, entries in ledger.hypotheses().items():
        first, latest = entries[0], entries[-1]
        attrs = first.get("attributes", {})
        active: list[str] = []
        history = []
        prev = None
        for e in entries:
            for i in e.get("evidence", []):
                eid = i.get("evidence_id") or i.get("ref")
                if eid not in active:
                    active.append(eid)
            for w in e.get("evidence_withdrawn", []):
                if w in active:
                    active.remove(w)
            state = {"confidence": e["confidence"], "status": e["status"],
                     "evidence": [_item_view(items[x]) for x in active if x in items]}
            if prev is not None:
                history.append({"timestamp": e["time"], "rationale": e["rationale"],
                                "changes": {"from": prev, "to": state}, "entry_id": e["entry_id"]})
            prev = state
        status = latest["status"]
        if hid in ranked and status in LIVE_STATUSES:
            status = "terminal"
        out[hid] = {
            "hypothesis_id": hid, **{k: v for k, v in attrs.items()},
            "candidate_gene": _key_attribute(attrs, hid), "cell_type": str(attrs.get("cell_type", "")),
            "attributes": attrs, "claim": first["hypothesis"], "derived_from": first.get("derived_from"),
            "evidence": prev["evidence"] if prev else [], "confidence": latest["confidence"],
            "open_questions": latest.get("open_questions", []), "status": status,
            "update_history": history, "created_at": first["time"],
            "creation_rationale": first["rationale"],
        }
    n_calls = len(ledger.entries("tool_call")) + sum(
        1 for e in ledger.entries() if e.get("written_by") == "agent")
    return {"hypotheses": out, "tool_call_count": n_calls}


def view_final_report(ledger: Ledger) -> Optional[dict]:
    c = ledger.conclusion()
    if c is None:
        return None
    hyps = view_hypotheses(ledger)
    validation = c.get("validation") or {}
    return {
        "completed_at": c["time"],
        "top_hypotheses": [hyps["hypotheses"][h] for h in c["ranked_hypothesis_ids"]
                           if h in hyps["hypotheses"]],
        "rationale": c["rationale"],
        "confidence_summary": c.get("confidence_summary"),
        "suggested_validation": "\n".join(f"{k}: {v}" for k, v in validation.items()),
        "validation": validation,
        "total_tool_calls": hyps["tool_call_count"],
        "total_hypotheses_explored": len(hyps["hypotheses"]),
    }
