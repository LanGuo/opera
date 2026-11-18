"""Task review step before any run (judging-v2 §4.2): does the agent prompt produce what the rubric reads?

For every rubric criterion: the pass that judges it, the ledger inputs it reads, which agent tool or
harness step produces each, and whether the delivered agent prompt (agent_task.md plus the generated
ledger section) asks the agent for it. Mismatches are flagged; with --confirm the author's
confirmation is appended to task_spec.yaml.

  uv run bioagent-check-task tasks/<task_id> [--confirm]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evaluator.harness import delivered_task_text
from reviewer.rubric import INPUTS, RubricError, load_criteria

# What the delivered prompt must mention for an input to be produced at all. Inputs not listed are
# produced for every run (the harness records them, or the ledger tools require them).
_ASKS = {
    "validation_plan": (r"validation", "declare_done must be asked for a validation experiment per hypothesis"),
    "human_input": (r"request_human_input|clarif|ask .*question|questions? (to|for) the (user|human)",
                    "nothing in the prompt asks the agent to ask questions (request_human_input)"),
    "notes": (r"log_note", "the prompt never mentions log_note"),
    "conclusion": (r"declare_done", "the prompt never mentions declare_done"),
}
_ARTIFACT_HINTS = {".png": r"\.png|plot|figure|heatmap|visuali[sz]", ".py": r"\.py\b|script",
                   ".csv": r"\.csv|table", ".pdf": r"\.pdf|report"}
CONFIRM_KEY = "judge_inputs_confirmed"


def review(task_dir) -> dict:
    task_dir = Path(task_dir)
    spec = yaml.safe_load((task_dir / "task_spec.yaml").read_text(encoding="utf-8")) or {}
    criteria, warnings = load_criteria(spec)
    prompt = delivered_task_text(task_dir)
    rows, problems = [], []
    for c in criteria:
        row = {"id": c.id, "scope": c.scope, "pass": c.judge_pass, "inputs": [], "defaulted": c.defaulted}
        for name in c.inputs:
            source, producer = INPUTS[name]
            ok, why = True, ""
            if name in _ASKS:
                pattern, why = _ASKS[name]
                ok = re.search(pattern, prompt, re.IGNORECASE) is not None
            row["inputs"].append({"input": name, "ledger": source, "produced_by": producer,
                                  "prompt_asks": ok, "problem": "" if ok else why})
            if not ok:
                problems.append(f"{c.id}: needs {name} — {why}")
        for glob in c.artifact_globs:
            ext = Path(glob).suffix.lower()
            hint = _ARTIFACT_HINTS.get(ext, re.escape(ext) if ext else r"\bfile")
            ok = re.search(hint, prompt, re.IGNORECASE) is not None
            if ok and "/" in glob:
                ok = glob.split("/")[0] in prompt
            why = "" if ok else f"the prompt never says to save {glob}"
            row["inputs"].append({"input": f"artifacts: {glob}", "ledger": INPUTS["artifacts"][0],
                                  "produced_by": "the agent writing that file (recorded by the harness)",
                                  "prompt_asks": ok, "problem": why})
            if not ok:
                problems.append(f"{c.id}: needs artifacts {glob!r} — {why}")
        if c.scope == "per_hypothesis" and "validation_plan" in c.inputs and not re.search(
                r"validation experiment per (ranked )?hypothesis|per[- ]hypothesis", prompt, re.IGNORECASE):
            problems.append(f"{c.id}: per-hypothesis criterion needs a validation plan per hypothesis; "
                            "the prompt doesn't ask declare_done for one per hypothesis")
        rows.append(row)
    return {"task_dir": str(task_dir), "criteria": rows, "warnings": warnings, "problems": problems,
            "criteria_hash": criteria_hash(spec), "confirmed": spec.get(CONFIRM_KEY)}


def criteria_hash(spec: dict) -> str:
    rubric = [{k: c.get(k) for k in ("id", "scope", "pass", "inputs", "guidance")} for c in spec.get("rubric") or []]
    return hashlib.sha256(json.dumps(rubric, sort_keys=True).encode()).hexdigest()[:16]


def render(r: dict) -> str:
    lines = [f"Task: {r['task_dir']}"]
    for c in r["criteria"]:
        lines.append(f"\n{c['id']} ({c['scope']}) — judged by the {c['pass']} pass"
                     + (" [pass/inputs defaulted]" if c["defaulted"] else ""))
        for i in c["inputs"]:
            mark = "ok " if i["prompt_asks"] else "!! "
            lines.append(f"  {mark}{i['input']}: {i['ledger']} ← {i['produced_by']}"
                         + (f"\n       {i['problem']}" if i["problem"] else ""))
    lines += [""] + [f"warning: {w}" for w in r["warnings"]]
    lines += [f"PROBLEM: {p}" for p in r["problems"]] or ["No mismatches found."]
    conf = r["confirmed"]
    lines.append("Confirmed: " + (f"{conf.get('date')} (rubric {conf.get('criteria_hash')}"
                                   + ("" if conf.get("criteria_hash") == r["criteria_hash"] else ", RUBRIC CHANGED SINCE")
                                   + ")" if conf else "not yet"))
    return "\n".join(lines)


def confirm(task_dir, r: dict) -> None:
    """Append the confirmation to task_spec.yaml (appended, so the file's comments and layout stay)."""
    path = Path(task_dir) / "task_spec.yaml"
    text = path.read_text(encoding="utf-8")
    text = re.sub(rf"\n{CONFIRM_KEY}:.*?(?=\n\S|\Z)", "", text, flags=re.DOTALL).rstrip("\n")
    block = {CONFIRM_KEY: {"date": datetime.now(timezone.utc).date().isoformat(),
                           "criteria_hash": r["criteria_hash"], "problems_acknowledged": r["problems"]}}
    path.write_text(text + "\n" + yaml.safe_dump(block, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task_dir")
    p.add_argument("--confirm", action="store_true", help="record that the author reviewed this")
    a = p.parse_args(argv)
    try:
        r = review(a.task_dir)
    except (RubricError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    print(render(r))
    if a.confirm:
        confirm(a.task_dir, r)
        print(f"\nConfirmation written to {Path(a.task_dir) / 'task_spec.yaml'}")


if __name__ == "__main__":
    main()
