"""Convert a run recorded before ledger.jsonl into a ledger.jsonl, for scoring old runs only.

  python -m hypothesis_ledger.legacy <old_run_dir> <out_dir>

The old run directory is not modified; out_dir gets ledger.jsonl and a copy of task_spec.yaml.
What the old files can't supply is stated in the run_info start entry (`conversion_limits`):

- creation entries have no rationale (the old write_hypothesis took none);
- evidence items aren't linked to tool calls (`link: none`), since the old tools recorded no source;
- old updates rewrote the hypothesis text and attributes in place; the creation values are kept and
  each rewrite is listed in the update entry's `v1_rewrites`;
- evidence an old update dropped from the list is kept cited (most drops were accidental, see D-29);
- entries are grouped by kind (tool calls, hypotheses, notes, …), not interleaved in time, because
  Claude Code transcripts carry no timestamps.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .ledger import STATUSES, IdCounter, Ledger, StreamParser, task_rules_from_spec

CONVERSION_LIMITS = [
    "creation entries have no rationale",
    "evidence items are not linked to tool calls",
    "in-place rewrites of hypothesis text or attributes are listed in v1_rewrites",
    "evidence dropped by old updates is kept cited",
    "entries are grouped by kind, not in time order",
]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _initial_state(h: dict) -> dict:
    """The hypothesis as first written: the 'from' side of the first update that changed a field."""
    state = {k: h.get(k) for k in ("claim", "cell_type", "candidate_gene", "evidence", "confidence",
                                   "status", "open_questions")}
    seen = set()
    for u in h.get("update_history", []):
        for k, v in ((u.get("changes") or {}).get("from") or {}).items():
            if k in state and k not in seen:
                state[k] = v
                seen.add(k)
    return state


def _norm(item) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False) if not isinstance(item, str) else item


def _status(v) -> tuple[str, str | None]:
    if v in STATUSES:
        return v, None
    return "active", v


def convert_v1(old_run_dir, out_dir) -> Path:
    from reviewer.evidence import evidence_parts

    old, out = Path(old_run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ledger.jsonl").exists():
        raise FileExistsError(f"{out / 'ledger.jsonl'} exists")
    spec_path = old / "task_spec.yaml"
    spec = {}
    if spec_path.exists():
        import yaml
        shutil.copy2(spec_path, out / "task_spec.yaml")
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    rules = task_rules_from_spec(spec)
    if not rules["hypothesis_attributes"]:
        rules["hypothesis_attributes"] = [{"name": "gene"}, {"name": "cell_type"}]

    led = Ledger(out)
    manifest = _read_json(old / "run_manifest.json") or {}
    led.append([{"kind": "run_info", "phase": "start", **manifest, "converted_from": str(old),
                 "conversion_limits": CONVERSION_LIMITS, "task_rules": rules}], "harness")

    led._stream = StreamParser(old / "transcript.jsonl")
    led.sync_stream(final=True)

    ledger_v1 = _read_json(old / "ledger.json") or {"hypotheses": {}}
    hyps = sorted(ledger_v1.get("hypotheses", {}).values(), key=lambda h: h.get("created_at", ""))
    id_map: dict[str, str] = {}

    def item_of(raw, ids) -> dict:
        kind, text = evidence_parts(raw)
        return {"evidence_id": ids.next("E"), "finding": text or _norm(raw),
                "source": {"v1_kind": kind} if kind else {}, "relation": "supports",
                "tool_call_ids": [], "link": "none", "extra": {"v1_item": raw}}

    for h in hyps:
        def build(ids, h=h):
            entries = []
            first = _initial_state(h)
            hid = ids.next("H")
            id_map[h["hypothesis_id"]] = hid
            attrs = {"gene": first.get("candidate_gene") or "", "cell_type": first.get("cell_type") or ""}
            cited: dict[str, str] = {}
            items = []
            for raw in first.get("evidence") or []:
                it = item_of(raw, ids)
                cited[_norm(raw)] = it["evidence_id"]
                items.append(it)
            status, v1_status = _status(first.get("status"))
            entries.append({"kind": "hypothesis", "action": "create", "hypothesis_id": hid,
                            "hypothesis": first.get("claim") or "", "attributes": attrs,
                            "derived_from": None, "confidence": first.get("confidence"),
                            "status": status, "evidence": items, "evidence_withdrawn": [],
                            "rationale": "", "open_questions": first.get("open_questions") or [],
                            "time_v1": h.get("created_at"), "v1_status": v1_status,
                            "v1_hypothesis_id": h["hypothesis_id"]})
            conf, stat, questions = first.get("confidence"), status, first.get("open_questions") or []
            for u in h.get("update_history", []):
                to = (u.get("changes") or {}).get("to") or {}
                new_items, refs = [], []
                for raw in to.get("evidence") or []:
                    key = _norm(raw)
                    if key in cited:
                        refs.append({"ref": cited[key], "relation": "supports"})
                    else:
                        it = item_of(raw, ids)
                        cited[key] = it["evidence_id"]
                        new_items.append(it)
                conf = to.get("confidence", conf)
                stat, v1_status = _status(to.get("status", stat)) if "status" in to else (stat, None)
                questions = to.get("open_questions", questions)
                rewrites = {k: to[k] for k in ("claim", "cell_type", "candidate_gene") if k in to}
                entries.append({"kind": "hypothesis", "action": "update", "hypothesis_id": hid,
                                "hypothesis": first.get("claim") or "", "attributes": attrs,
                                "derived_from": None, "confidence": conf, "status": stat,
                                "evidence": new_items + refs, "evidence_withdrawn": [],
                                "rationale": u.get("rationale") or "", "open_questions": questions,
                                "time_v1": u.get("timestamp"), "v1_status": v1_status,
                                "v1_rewrites": rewrites or None})
            return entries

        led.transact(build, "agent")

    notes = []
    for st in _read_jsonl(old / "step_log.jsonl"):
        step = str(st.get("step_type", "")).lower()
        kind = "debug" if any(w in step for w in ("debug", "troubleshoot", "fail", "error")) else "other"
        text = "\n".join(f"{k}: {st[k]}" for k in ("tool_used", "inputs_summary", "outputs_summary",
                                                    "interpretation", "next_action") if st.get(k))
        notes.append({"kind": "note", "note_kind": kind, "text": text, "hypothesis_ids": [],
                      "time_v1": st.get("timestamp"), "v1_step": st})
    for r in _read_jsonl(old / "run_log.jsonl"):
        if r.get("tool") == "request_human_input":
            inp = r.get("inputs") or {}
            notes.append({"kind": "human_input", "question": inp.get("question"),
                          "context": inp.get("context"), "mode": "headless", "answer": None,
                          "time_v1": r.get("timestamp")})
    if notes:
        led.append(notes, "agent")

    report = _read_json(old / "final_report.json")
    if report:
        ranked = [id_map.get(t.get("hypothesis_id"), t.get("hypothesis_id"))
                  for t in report.get("top_hypotheses", [])]
        led.append([{"kind": "conclusion", "ranked_hypothesis_ids": ranked,
                     "rationale": report.get("rationale") or "",
                     "confidence_summary": report.get("confidence_summary"),
                     "validation": {"run": report.get("suggested_validation") or ""},
                     "time_v1": report.get("completed_at")}], "agent")

    from reviewer.log_replay import _parse_cost_details
    cost, unpriced = _parse_cost_details(old / "transcript.jsonl")
    led.append([{"kind": "run_info", "phase": "end", "outcome": manifest.get("outcome") or {},
                 "cost_usd": cost, "cost_unpriced_models": unpriced,
                 "concluded": report is not None}], "harness")
    return out / "ledger.jsonl"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(convert_v1(sys.argv[1], sys.argv[2]))
