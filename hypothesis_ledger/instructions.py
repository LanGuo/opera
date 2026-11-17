"""The ledger section of the agent prompt, generated from the task spec.

The harness appends it to the task's agent_task.md, so every task states the same ledger rules and
its own hypothesis attributes and constraints the same way. Task authors describe the science in
agent_task.md and leave the ledger tools to this section.
"""
from __future__ import annotations

from .ledger import LIVE_STATUSES, task_rules_from_spec

_BASE = """\
## Hypothesis ledger

The hypothesis-ledger tools record your work. Reviewers read only this ledger, together with the
tool calls the harness records automatically (every tool you call and its full output).

- **`write_hypothesis`** when you form a hypothesis: the hypothesis in words, its attributes,
  confidence (0-1), at least one evidence item, and a rationale for your confidence.
- **`update_hypothesis`** whenever evidence changes your confidence or status (`active`,
  `supported`, `weakened`, `refuted`, `abandoned`): at least one evidence item and a rationale.
  If evidence contradicts a hypothesis, record it with `"relation": "contradicts"` and lower its
  confidence or change its status. Do not drop it.
- **A hypothesis's text and attributes never change.** To reword or re-scope one, write a new
  hypothesis with `derived_from` set to the old id, and mark the old one `abandoned`.
- **Evidence items** state facts: `finding` is what the result shows, with numbers (not why it
  matters; that goes in `rationale`). `source` names the tool call it came from:
  `{"tool": "<tool name as you called it>", "match": "<short text from that call's arguments>"}`,
  or `{"citation": "PMID ..."}` for literature. Reuse earlier evidence with `{"ref": "E3"}`.
  Every item has `"relation"`: `supports`, `contradicts` or `context`.
- **`log_note`** for interpretations (link them to hypotheses), plans and debugging.
- **`declare_done`** to finish: your final hypotheses ranked best first, a rationale, a confidence
  summary, and one validation experiment per ranked hypothesis.
"""


def ledger_instructions(spec: dict | None) -> str:
    rules = task_rules_from_spec(spec or {})
    lines = [_BASE]
    attrs = rules["hypothesis_attributes"]
    if attrs:
        lines.append("**This task's hypothesis attributes** (pass them in `attributes`):\n")
        for a in attrs:
            req = "required" if a.get("required", True) else "optional"
            desc = f": {a['description']}" if a.get("description") else ""
            lines.append(f"- `{a['name']}` ({req}){desc}")
        lines.append("")
    else:
        lines.append("This task declares no hypothesis attributes: pass `attributes={}`.\n")
    for c in rules["hypothesis_constraints"]:
        keys = c.get("unique") or []
        if keys:
            among = c.get("among") or list(LIVE_STATUSES)
            names = " + ".join(f"`{k}`" for k in keys)
            lines.append(f"**Rule:** at most one hypothesis per {names} among hypotheses that are "
                         f"{', '.join(among)}. To change such a hypothesis (its wording or another "
                         "attribute), first mark it `abandoned`, then write the new one with "
                         "`derived_from`.\n")
    return "\n".join(lines).rstrip() + "\n"
