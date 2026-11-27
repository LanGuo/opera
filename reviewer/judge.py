"""The ledger judge: pass E, pass R, the run-level pass and reasoning scores.

Every input comes from ledger.jsonl through reviewer.judge_inputs; nothing is read from any other
file of the run except task_spec.yaml (the rubric and goal). Passes are independent: pass R and the
reasoning-criterion scores never see pass-E output, and pass E never sees the agent's rationales,
confidence or status.

Results go to `judge_<timestamp>.json` next to the run, never into ledger.jsonl. Each call's exact
prompt and raw response are saved with what was parsed from it.

  uv run bioagent-judge <run_dir>            # every configured backend (Claude Haiku, Gemini Flash)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from reviewer.judge_inputs import (MAX_OUTPUT_CHARS, RunView, build_run_view, render_evidence_input,
                                   render_reasoning_input, render_run_input, run_counts)
from reviewer.llm import LLMClient, extract_json_object
from reviewer.rubric import Criterion, load_criteria, missing_inputs

JUDGE_VERSION = "2.1"   # 2.1: long tool outputs shown as finding-matched excerpts (D-60)
BACKENDS = {"anthropic": ("claude-haiku-4-5", "ANTHROPIC_API_KEY"),
            "google": ("gemini-2.5-flash", "GOOGLE_API_KEY")}
# Gemini 2.5 Flash counts its thinking tokens against the output limit: at 8,192 a long reasoning
# audit was cut off mid-JSON (shake-down pair 6/7), so it gets more room.
JUDGE_MAX_TOKENS = {"anthropic": 8192, "google": 32768}
LABELS = ("supported", "overstated", "unsupported", "contradicted", "not_checkable")
CONTESTED_GAP = 2

ROLE = ("You are judging a completed run of an AI research agent. Use only the information below: "
        "do not look anything up or use outside sources of evidence about this run. Do not reward "
        "length, confident wording or presentation; judge the scientific content.")

EVIDENCE_WEIGHING = (
    "How to weigh the evidence: judge what each item shows, not how many items there are or how "
    "long they are. Items that restate the same finding, or come from the same source or analysis, "
    "count as one line of evidence. A short item with a quantified result can outweigh several long "
    "descriptive ones. Check each finding against the tool output shown with it: a finding the "
    "output does not show is not evidence. Items with no recorded source rest on the agent's word.")


_STR = {"type": "string"}
_IDS = {"type": "array", "items": _STR}
SCORE_SCHEMA = {"type": "object", "required": ["criteria"], "properties": {"criteria": {
    "type": "array", "items": {"type": "object", "required": ["criterion_id", "score", "rationale"],
                               "properties": {"criterion_id": _STR, "score": _STR, "rationale": _STR,
                                              "evidence_ids": _IDS}}}}}
REASONING_SCHEMA = {"type": "object", "required": ["assertions"], "properties": {
    "assertions": {"type": "array", "items": {"type": "object", "required": ["entry_id", "quote", "label"],
                   "properties": {"entry_id": _STR, "quote": _STR, "label": {"type": "string", "enum": list(LABELS)},
                                  "evidence_ids": _IDS, "reason": _STR}}},
    "entry_directions": {"type": "array", "items": {"type": "object", "properties": {
        "entry_id": _STR, "evidence_direction": {"type": "string", "enum": ["up", "down", "neutral"]}}}},
    "contradictions": {"type": "array", "items": {"type": "object", "properties": {
        "evidence_id": _STR, "acknowledged": {"type": "boolean"}, "entry_id": _STR, "quote": _STR}}}}}
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def parse_response(raw: str, wrap_list: str | None = None) -> dict:
    """The response's JSON object. A bare list is wrapped as {wrap_list: [...]} (Haiku sometimes
    answers a scoring prompt with just the array)."""
    text = _FENCE.sub("", raw or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = extract_json_object(text)
    if isinstance(obj, list) and wrap_list:
        obj = {wrap_list: obj}
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


def _criteria_block(criteria: list[Criterion]) -> str:
    return "\n".join(f"- criterion_id: {c.id}\n  label: {c.label}\n  scale: {c.scale}\n"
                     f"  guidance: {c.guidance}" for c in criteria)


def _score_format(with_evidence_ids: bool) -> str:
    ev = ', "evidence_ids": ["E3", "E7"]' if with_evidence_ids else ""
    note = (" `evidence_ids` lists the evidence items (by id) your score relies on."
            if with_evidence_ids else "")
    return ("Return ONLY a JSON object, no markdown fence:\n"
            '{"criteria": [{"criterion_id": "<id>", "score": "<a value on that criterion\'s scale, '
            'e.g. 4 or yes/no/partial>", "rationale": "<2-3 sentences>"' + ev + "}]}\n"
            "One entry per criterion listed." + note)


def numeric(score) -> float | None:
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(score))
    return float(m.group(1)) if m else None


def _norm(text: str) -> str:
    return " ".join(str(text).replace("’", "'").replace("“", '"').replace("”", '"').split()).lower()


class Judge:
    def __init__(self, run_dir, client: LLMClient, *, max_output_chars: int = MAX_OUTPUT_CHARS,
                 spec: dict | None = None):
        self.run_dir = Path(run_dir)
        self.client = client
        notes: list[str] = []
        if spec is None:
            spec, notes = _read_spec(self.run_dir)
        self.spec = spec
        self.criteria, warnings = load_criteria(self.spec)
        self.rubric_warnings = notes + warnings
        judge_inputs = (self.spec.get("judge_inputs") or {})
        self.max_output_chars = int(judge_inputs.get("max_output_chars", max_output_chars))
        self.run: RunView = build_run_view(self.run_dir)
        self.goal = " ".join(str(self.spec.get("goal", "")).split())
        self.calls: list[dict] = []

    # --- LLM calls -------------------------------------------------------------------------------

    def _call(self, kind: str, prompt: str, hypothesis_id: str | None = None) -> tuple[dict | None, dict]:
        """One judge call; a response that isn't valid JSON is asked for once more (both kept)."""
        schema = REASONING_SCHEMA if kind == "reasoning" else SCORE_SCHEMA
        rec = {"call": len(self.calls) + 1, "pass": kind, "hypothesis_id": hypothesis_id,
               "prompt": prompt, "prompt_chars": len(prompt)}
        parsed, failed = None, []
        for attempt in range(2):
            try:
                # The retry goes without the schema: Gemini's schema decoding sometimes degenerates
                # (a string of endless newlines), and plain decoding is a different path.
                raw = self.client.chat([{"role": "user", "content": prompt}],
                                       schema=schema if attempt == 0 else None)
            except Exception as exc:          # recorded, never fatal for the other calls
                rec["error"] = f"{type(exc).__name__}: {exc}"
                break
            rec["response"] = raw
            rec["model_reported"] = getattr(self.client, "last_model", None)
            if getattr(self.client, "last_finish_reason", None):
                rec["finish_reason"] = self.client.last_finish_reason
            try:
                parsed = parse_response(raw, None if kind == "reasoning" else "criteria")
                rec.pop("error", None)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                failed.append(raw)
        if failed:
            rec["unparsed_responses"] = failed
        self.calls.append(rec)
        return parsed, rec

    def _scores(self, parsed: dict | None, criteria: list[Criterion], call: dict,
                known_evidence: set[str] | None) -> dict:
        out = {}
        by_id = {c.get("criterion_id"): c for c in (parsed or {}).get("criteria", []) if isinstance(c, dict)}
        for c in criteria:
            got = by_id.get(c.id)
            if got is None:
                out[c.id] = {"status": "no_score", "call": call["call"]}
                continue
            res = {"status": "scored", "score": str(got.get("score", "")).strip(),
                   "numeric": numeric(got.get("score")), "rationale": str(got.get("rationale", "")).strip(),
                   "call": call["call"]}
            if known_evidence is not None:
                cited = [str(e) for e in got.get("evidence_ids") or []]
                res["evidence_ids"] = [e for e in cited if e in known_evidence]
                res["unknown_evidence_ids"] = [e for e in cited if e not in known_evidence]
            out[c.id] = res
        return out

    def _split_assessable(self, criteria: list[Criterion], hid: str | None):
        ok, skipped = [], {}
        for c in criteria:
            miss = missing_inputs(c, self.run, hid)
            if miss:
                skipped[c.id] = {"status": "not_assessable", "missing": miss}
            else:
                ok.append(c)
        return ok, skipped

    # --- pass E ----------------------------------------------------------------------------------

    def evidence_pass(self) -> dict:
        per = [c for c in self.criteria if c.judge_pass == "evidence" and c.scope == "per_hypothesis"]
        overall = [c for c in self.criteria if c.judge_pass == "evidence" and c.scope == "overall"]
        out = {"per_hypothesis": {}, "overall": {}}
        for hid in self.run.hypotheses:
            ok, results = self._split_assessable(per, hid)
            if ok:
                text = render_evidence_input(
                    self.run, hid, max_output_chars=self.max_output_chars,
                    include_validation=any("validation_plan" in c.inputs for c in ok),
                    artifact_globs=[g for c in ok for g in c.artifact_globs])
                prompt = "\n\n".join([f"Task goal: {self.goal}", ROLE,
                                      "Score this hypothesis on the evidence recorded for it.", text,
                                      EVIDENCE_WEIGHING, _score_format(True), _criteria_block(ok)])
                parsed, call = self._call("evidence", prompt, hid)
                results.update(self._scores(parsed, ok, call, set(self.run.hypotheses[hid].evidence)))
            out["per_hypothesis"][hid] = results
        if overall:
            ok, results = self._split_assessable(overall, None)
            if ok:
                texts = [render_evidence_input(self.run, hid, max_output_chars=self.max_output_chars,
                                               include_validation=any("validation_plan" in c.inputs for c in ok))
                         for hid in self.run.hypotheses]
                prompt = "\n\n".join([f"Task goal: {self.goal}", ROLE,
                                      "Score the run's evidence as a whole.", *texts,
                                      EVIDENCE_WEIGHING, _score_format(True), _criteria_block(ok)])
                parsed, call = self._call("evidence", prompt)
                known = {e for h in self.run.hypotheses.values() for e in h.evidence}
                results.update(self._scores(parsed, ok, call, known))
            out["overall"] = results
        return out

    # --- pass R ----------------------------------------------------------------------------------

    _R_INSTRUCTIONS = (
        "Audit the reasoning: does what the agent SAID follow from what it RECORDED? Do not judge how "
        "strong the evidence is in general; judge whether each statement is backed by the evidence "
        "cited alongside it (the finding and the tool output shown with it).\n\n"
        "An assertion is one statement of fact or inference in the agent's own reasoning: the text "
        "after 'Rationale:' in an entry, an interpretation Note, or the Conclusion rationale. Evidence "
        "'Finding:' lines and tool outputs are what the reasoning is checked AGAINST; never quote them "
        "as assertions. Split each reasoning text into its assertions. Quote each exactly as written "
        "(a contiguous excerpt, verbatim) and give the entry id of the Rationale, Note or Conclusion it "
        "comes from (e.g. L00079). Use only the five labels below. Label each:\n"
        "- supported: the cited evidence shows it;\n"
        "- overstated: the evidence points the same way but the statement claims more (stronger, more "
        "specific, more certain) than it shows;\n"
        "- unsupported: nothing cited shows it;\n"
        "- contradicted: cited evidence shows the opposite;\n"
        "- not_checkable: a plan, a question, or a statement about the agent's own intent.\n"
        "For every entry after the first, say which way the evidence cited IN THAT ENTRY moves the "
        "hypothesis: up, down or neutral.\n"
        "For every evidence item with relation 'contradicts', say whether a rationale acknowledges "
        "it (true/false) and quote where.")

    _R_FORMAT = (
        "Return ONLY a JSON object, no markdown fence:\n"
        '{"assertions": [{"entry_id": "L00079", "quote": "<verbatim>", "label": "supported", '
        '"evidence_ids": ["E9"], "reason": "<one sentence>"}],\n'
        ' "entry_directions": [{"entry_id": "L00090", "evidence_direction": "down"}],\n'
        ' "contradictions": [{"evidence_id": "E20", "acknowledged": true, "entry_id": "L00090", '
        '"quote": "<verbatim or empty>"}]}')

    def _sources(self, hid: str) -> dict[str, str]:
        h = self.run.hypotheses[hid]
        texts = {e.entry_id: e.rationale for e in h.entries if e.rationale}
        texts.update({n["entry_id"]: n.get("text", "") for n in h.notes})
        return texts

    def reasoning_pass(self) -> dict:
        out = {"per_hypothesis": {}, "run": None}
        for hid, h in self.run.hypotheses.items():
            sources = self._sources(hid)
            if not any(t.strip() for t in sources.values()):
                out["per_hypothesis"][hid] = {"status": "no_reasoning_recorded",
                                              "metrics": self._reasoning_metrics(hid, [], {}, {})}
                continue
            text = render_reasoning_input(self.run, hid, max_output_chars=self.max_output_chars)
            prompt = "\n\n".join([f"Task goal: {self.goal}", ROLE, self._R_INSTRUCTIONS, text, self._R_FORMAT])
            parsed, call = self._call("reasoning", prompt, hid)
            out["per_hypothesis"][hid] = self._reasoning_result(hid, parsed, call, sources)
        c = self.run.conclusion
        if c and str(c.get("rationale", "")).strip():
            sources = {c["entry_id"]: c.get("rationale", "")}
            text = render_run_input(self.run, ["evidence", "confidence_trajectory"],
                                    max_output_chars=self.max_output_chars)
            prompt = "\n\n".join([f"Task goal: {self.goal}", ROLE, self._R_INSTRUCTIONS,
                                  f"Audit only the Conclusion rationale (entry {c['entry_id']}) against "
                                  "the evidence findings and hypothesis states below.", text, self._R_FORMAT])
            parsed, call = self._call("reasoning", prompt)
            out["run"] = self._reasoning_result(None, parsed, call, sources)
        else:
            out["run"] = {"status": "no_conclusion_rationale"}
        return out

    def _reasoning_result(self, hid, parsed, call, sources: dict[str, str]) -> dict:
        valid, rejected = [], []
        for a in (parsed or {}).get("assertions", []) or []:
            if not isinstance(a, dict):
                continue
            eid, quote, label = str(a.get("entry_id", "")), str(a.get("quote", "")), a.get("label")
            reason = None
            if label not in LABELS:
                reason = f"unknown label {label!r}"
            elif eid not in sources:
                reason = f"entry {eid} is not a rationale, note or conclusion of this scope"
            elif not quote.strip() or _norm(quote) not in _norm(sources[eid]):
                reason = "quote not found verbatim in that entry"
            item = {"entry_id": eid, "quote": quote, "label": label,
                    "evidence_ids": [str(e) for e in a.get("evidence_ids") or []],
                    "reason": str(a.get("reason", ""))}
            (rejected if reason else valid).append(dict(item, rejected=reason) if reason else item)
        directions = {str(d.get("entry_id")): d.get("evidence_direction")
                      for d in (parsed or {}).get("entry_directions", []) or [] if isinstance(d, dict)}
        acks = {str(d.get("evidence_id")): d for d in (parsed or {}).get("contradictions", []) or []
                if isinstance(d, dict)}
        if parsed is None:
            status = "call_failed"
        elif not valid:
            status = "no_checkable_assertions"
        else:
            status = "ok"
        res = {"status": status, "call": call["call"], "assertions": valid, "rejected_assertions": rejected}
        res["metrics"] = (self._reasoning_metrics(hid, valid, directions, acks) if hid
                          else _label_rates(valid))
        return res

    def _reasoning_metrics(self, hid, assertions, directions, acks) -> dict:
        """Computed in code from the labels and the ledger, never by the LLM."""
        m = _label_rates(assertions)
        h = self.run.hypotheses[hid]
        changes = []
        for prev, e in zip(h.entries, h.entries[1:]):
            if e.confidence is None or prev.confidence is None:
                continue
            delta = round(e.confidence - prev.confidence, 6)
            moved = "up" if delta > 0 else "down" if delta < 0 else "neutral"
            judged = directions.get(e.entry_id)
            changes.append({"entry_id": e.entry_id, "confidence_change": delta, "moved": moved,
                            "evidence_direction": judged,
                            "matches": None if judged not in ("up", "down", "neutral") else judged == moved})
        judged = [c for c in changes if c["matches"] is not None]
        m["confidence_changes"] = changes
        m["calibration"] = (sum(c["matches"] for c in judged) / len(judged)) if judged else None
        contra = []
        for n, e in enumerate(h.entries):
            for eid, relation in e.cited:
                if relation != "contradicts" or any(c["evidence_id"] == eid for c in contra):
                    continue
                later = h.entries[n:]
                # Confidence before the item was cited (the entry it was cited in, if a create).
                before = h.entries[n - 1].confidence if n else e.confidence
                dropped = before is not None and any(
                    x.confidence is not None and x.confidence < before for x in later)
                status_change = any(x.status in ("weakened", "refuted", "abandoned") for x in later)
                ack = bool((acks.get(eid) or {}).get("acknowledged"))
                contra.append({"evidence_id": eid, "cited_in": e.entry_id, "acknowledged": ack,
                               "followed_by_lower_confidence": dropped,
                               "followed_by_status_change": status_change,
                               "handled": ack or dropped or status_change})
        m["contradicting_items"] = contra
        m["disconfirmation_handled"] = (sum(c["handled"] for c in contra) / len(contra)) if contra else None
        return m

    # --- reasoning-criterion scores (from pass-R metrics, never pass-E output) -------------------

    def reasoning_scores(self, reasoning: dict) -> dict:
        crits = [c for c in self.criteria if c.judge_pass == "reasoning"]
        out = {"per_hypothesis": {}, "overall": {}}
        per = [c for c in crits if c.scope == "per_hypothesis"]
        overall = [c for c in crits if c.scope == "overall"]
        for hid in self.run.hypotheses if per else []:
            ok, results = self._split_assessable(per, hid)
            if ok:
                prompt = "\n\n".join([f"Task goal: {self.goal}", ROLE,
                                      "Score the agent's reasoning about this hypothesis from the "
                                      "reasoning audit below (assertion labels and measures computed "
                                      "from them) and the recorded entries.",
                                      _reasoning_report(hid, reasoning["per_hypothesis"][hid]),
                                      render_reasoning_input(self.run, hid, max_output_chars=self.max_output_chars),
                                      _score_format(False), _criteria_block(ok)])
                parsed, call = self._call("reasoning_score", prompt, hid)
                results.update(self._scores(parsed, ok, call, None))
            out["per_hypothesis"][hid] = results
        if overall:
            ok, results = self._split_assessable(overall, None)
            if ok:
                reports = [_reasoning_report(hid, r) for hid, r in reasoning["per_hypothesis"].items()]
                if reasoning.get("run") and reasoning["run"].get("assertions") is not None:
                    reports.append(_reasoning_report("Conclusion", reasoning["run"]))
                inputs = sorted({i for c in ok for i in c.inputs})
                prompt = "\n\n".join([f"Task goal: {self.goal}", ROLE,
                                      "Score the agent's reasoning over the whole run from the "
                                      "reasoning audit below (assertion labels and measures computed "
                                      "from them) and the run record.", *reports,
                                      render_run_input(self.run, inputs, max_output_chars=self.max_output_chars),
                                      _score_format(False), _criteria_block(ok)])
                parsed, call = self._call("reasoning_score", prompt)
                results.update(self._scores(parsed, ok, call, None))
            out["overall"] = results
        return out

    # --- run-level pass --------------------------------------------------------------------------

    def run_pass(self) -> dict:
        crits = [c for c in self.criteria if c.judge_pass == "run"]
        ok, results = self._split_assessable(crits, None)
        if ok:
            inputs = sorted({i for c in ok for i in c.inputs})
            prompt = "\n\n".join([f"Task goal: {self.goal}", ROLE, "Score the run as a whole.",
                                  render_run_input(self.run, inputs,
                                                   artifact_globs=[g for c in ok for g in c.artifact_globs],
                                                   max_output_chars=self.max_output_chars),
                                  _score_format(False), _criteria_block(ok)])
            parsed, call = self._call("run", prompt)
            results.update(self._scores(parsed, ok, call, None))
        return results

    def judge(self) -> dict:
        evidence = self.evidence_pass()
        reasoning = self.reasoning_pass()
        return {"model": getattr(self.client, "model", None), "backend": getattr(self.client, "backend", None),
                "evidence": evidence, "reasoning": reasoning,
                "reasoning_scores": self.reasoning_scores(reasoning),
                "run": self.run_pass(), "calls": self.calls}


def _label_rates(assertions: list[dict]) -> dict:
    n = len(assertions)
    checkable = [a for a in assertions if a["label"] != "not_checkable"]
    rates = {f"{lab}_rate": (sum(a["label"] == lab for a in checkable) / len(checkable)) if checkable else None
             for lab in LABELS if lab != "not_checkable"}
    return {"n_assertions": n, "n_checkable": len(checkable),
            "n_not_checkable": n - len(checkable), **rates}


def _reasoning_report(name: str, r: dict) -> str:
    m = r.get("metrics") or {}
    lines = [f"Reasoning audit — {name}: status {r.get('status')}",
             "  " + ", ".join(f"{k} {v if not isinstance(v, float) else round(v, 2)}"
                              for k, v in m.items() if not isinstance(v, list))]
    for a in r.get("assertions") or []:
        lines.append(f"  [{a['label']}] ({a['entry_id']}) \"{a['quote']}\" — {a.get('reason', '')}")
    for c in m.get("confidence_changes") or []:
        lines.append(f"  confidence change {c['entry_id']}: {c['confidence_change']:+} (moved {c['moved']}; "
                     f"evidence in that entry points {c['evidence_direction']})")
    for c in m.get("contradicting_items") or []:
        lines.append(f"  contradicting {c['evidence_id']}: acknowledged {c['acknowledged']}, lower confidence "
                     f"after {c['followed_by_lower_confidence']}, status change {c['followed_by_status_change']}")
    return "\n".join(lines)


TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


def _read_spec(run_dir: Path, tasks_dir: Path = TASKS_DIR) -> tuple[dict, list[str]]:
    """The run's copy of task_spec.yaml (the rubric it was run under), plus notes.

    `pass` and `inputs` configure only the judge (the agent never sees them), so when the task's
    current spec has the same criteria (ids and guidance) its declarations are used, and any
    difference from the run's copy is noted. If the criteria differ, the run's copy is used as is.
    """
    p = run_dir / "task_spec.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no task_spec.yaml in {run_dir}")
    spec = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rubric = spec.get("rubric") or []
    current = tasks_dir / str(spec.get("task_id", "")) / "task_spec.yaml"
    if not rubric or not current.exists():
        return spec, []
    cur = {c.get("id"): c for c in (yaml.safe_load(current.read_text(encoding="utf-8")) or {}).get("rubric") or []}
    same = all(c.get("id") in cur and " ".join(str(c.get("guidance", "")).split())
               == " ".join(str(cur[c["id"]].get("guidance", "")).split()) for c in rubric)
    where = current.relative_to(tasks_dir.parent) if current.is_relative_to(tasks_dir.parent) else current
    if not same:
        return spec, [f"the run's rubric differs from {where}; its own pass/inputs (or defaults) are used"]
    keys = ("pass", "inputs")
    changed = [c["id"] for c in rubric if any(c.get(k) != cur[c["id"]].get(k) for k in keys)]
    if not changed:
        return spec, []
    merged = dict(spec, rubric=[dict({k: v for k, v in c.items() if k not in keys},
                                     **{k: cur[c["id"]][k] for k in keys if k in cur[c["id"]]})
                                for c in rubric])
    return merged, [f"pass/inputs for {', '.join(changed)} taken from {where} (they configure only "
                    "the judge; the criteria are otherwise identical to the run's copy)"]


def contested(results: dict[str, dict]) -> list[dict]:
    """Criteria where two backends' numeric scores differ by CONTESTED_GAP or more, per pass."""
    flags = []
    if len(results) < 2:
        return flags
    keys = list(results)

    def walk(section_name: str, getter):
        rows = {}
        for k in keys:
            for loc, cid, res in getter(results[k]):
                if res.get("numeric") is not None:
                    rows.setdefault((loc, cid), {})[k] = res["numeric"]
        for (loc, cid), by in rows.items():
            if len(by) >= 2 and max(by.values()) - min(by.values()) >= CONTESTED_GAP:
                flags.append({"pass": section_name, "hypothesis_id": loc, "criterion_id": cid, "scores": by})

    def per_and_overall(section):
        def get(r):
            sec = r.get(section) or {}
            for hid, crits in (sec.get("per_hypothesis") or {}).items():
                for cid, res in crits.items():
                    yield hid, cid, res
            for cid, res in (sec.get("overall") or {}).items():
                yield None, cid, res
        return get

    walk("evidence", per_and_overall("evidence"))
    walk("reasoning", per_and_overall("reasoning_scores"))
    walk("run", lambda r: ((None, cid, res) for cid, res in (r.get("run") or {}).items()))
    return flags


def configured_backends(environ=os.environ) -> dict[str, LLMClient]:
    out = {}
    for name, (model, key) in BACKENDS.items():
        if environ.get(key):
            out[name] = LLMClient(backend=name, model=model, max_tokens=JUDGE_MAX_TOKENS[name])
    return out


def judge_run(run_dir, clients: dict[str, LLMClient] | None = None, *, write: bool = True) -> dict:
    run_dir = Path(run_dir)
    clients = configured_backends() if clients is None else clients
    if not clients:
        raise RuntimeError("no judge backend configured (set ANTHROPIC_API_KEY and/or GOOGLE_API_KEY)")
    judges = {k: Judge(run_dir, c) for k, c in clients.items()}
    first = next(iter(judges.values()))
    with ThreadPoolExecutor(max_workers=len(judges)) as pool:
        futures = {k: pool.submit(j.judge) for k, j in judges.items()}
        results = {}
        for k, f in futures.items():
            try:
                results[k] = f.result()
            except Exception as exc:
                results[k] = {"error": f"{type(exc).__name__}: {exc}", "calls": judges[k].calls}
    out = {"judge_version": JUDGE_VERSION, "created": datetime.now(timezone.utc).isoformat(),
           "run_dir": str(run_dir), "rubric_warnings": first.rubric_warnings,
           "max_output_chars": first.max_output_chars, "counts": run_counts(first.run),
           "criteria": [{"id": c.id, "scope": c.scope, "pass": c.judge_pass, "inputs": c.inputs, "optional": c.optional,
                         "artifacts": c.artifact_globs, "defaulted": c.defaulted} for c in first.criteria],
           "backends": results, "contested": contested(results)}
    if write:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = run_dir / f"judge_{stamp}.json"
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        out["path"] = str(path)
    return out


def summary(out: dict) -> str:
    lines = [f"judge v{out['judge_version']} — {out['run_dir']}"]
    lines += [f"  warning: {w}" for w in out["rubric_warnings"]]
    for k, r in out["backends"].items():
        if "error" in r:
            lines.append(f"[{k}] FAILED: {r['error']}")
            continue
        lines.append(f"[{k}] {r['model']} — {len(r['calls'])} calls, "
                     f"{sum(1 for c in r['calls'] if 'error' in c)} failed")
        for hid, crits in r["evidence"]["per_hypothesis"].items():
            lines.append(f"  E {hid}: " + ", ".join(f"{cid} {v.get('score', v['status'])}" for cid, v in crits.items()))
        for hid, rr in r["reasoning"]["per_hypothesis"].items():
            m = rr.get("metrics") or {}
            lines.append(f"  R {hid}: {rr['status']}, {m.get('n_assertions')} assertions, supported "
                         f"{_fmt(m.get('supported_rate'))}, overstated {_fmt(m.get('overstated_rate'))}, "
                         f"calibration {_fmt(m.get('calibration'))}, disconfirmation {_fmt(m.get('disconfirmation_handled'))}")
        for scope in ("per_hypothesis", "overall"):
            sec = r["reasoning_scores"][scope]
            items = sec.items() if scope == "overall" else [(f"{h}:{c}", v) for h, cs in sec.items() for c, v in cs.items()]
            for cid, v in items:
                lines.append(f"  reasoning score {cid}: {v.get('score', v['status'])}")
        for cid, v in r["run"].items():
            lines.append(f"  run {cid}: {v.get('score', v['status'])}" + (f" ({'; '.join(v['missing'])})" if v.get("missing") else ""))
    lines += [f"  CONTESTED {f['pass']} {f['hypothesis_id'] or 'run'} {f['criterion_id']}: {f['scores']}"
              for f in out["contested"]]
    if out.get("path"):
        lines.append(f"written: {out['path']}")
    return "\n".join(lines)


def _fmt(x):
    return "n/a" if x is None else f"{x:.2f}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Judge a ledger.jsonl run (judging v2).")
    p.add_argument("run_dir")
    p.add_argument("--backend", action="append", choices=sorted(BACKENDS),
                   help="limit to these backends (default: every one with an API key set)")
    a = p.parse_args(argv)
    clients = configured_backends()
    if a.backend:
        clients = {k: v for k, v in clients.items() if k in a.backend}
    try:
        out = judge_run(a.run_dir, clients)
    except (RuntimeError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    print(summary(out))


if __name__ == "__main__":
    main()
