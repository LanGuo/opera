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
  `{"tool": "<the tool you called>", "match": "<short text from that call's arguments>"}`.
  The tool is what you invoked (e.g. `Bash`, `run_shell_command`, or a ToolUniverse tool name),
  not a library used inside your code; for code you ran, `match` can be the script's file name.
  Use `{"citation": "PMID ..."}` for literature. Reuse earlier evidence with `{"ref": "E3"}`.
  Every item has `"relation"`: `supports`, `contradicts` or `context`.
- **`log_note`** for interpretations (link them to hypotheses), plans and debugging.
- **`declare_done`** to finish: your final hypotheses ranked best first, a rationale, a confidence
  summary, and one validation experiment per ranked hypothesis.
"""

_HUMAN_INPUT = """\
- **`request_human_input`** to ask the human expert a question (with the context it needs).
  The question is recorded in the ledger whether or not anyone answers; if the reply says no
  human is available, carry on with your own judgement.
"""


def _scores_human_input(spec: dict) -> bool:
    """True if any rubric criterion reads the recorded human_input entries."""
    for c in spec.get("rubric") or []:
        for i in (c.get("inputs") if isinstance(c, dict) else None) or []:
            if isinstance(i, str) and i.rstrip("?") == "human_input":
                return True
    return False


def ledger_instructions(spec: dict | None, turn_budget: tuple[int, int] | None = None) -> str:
    """The ledger section for a task. turn_budget (cap, warn_at) adds the run's turn cap."""
    rules = task_rules_from_spec(spec or {})
    lines = [_BASE + (_HUMAN_INPUT if _scores_human_input(spec or {}) else "")]
    if turn_budget:
        cap, warn_at = turn_budget
        lines.append(f"**Turn budget:** this run stops after {cap} turns; each of your responses "
                     f"uses one. From {warn_at} turns before the end, tool results tell you how "
                     "many are left. Leave yourself a turn to call `declare_done`.\n")
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
