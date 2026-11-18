"""Rubric criteria with the judge pass that scores them and the ledger inputs they need.

Each criterion in a task spec's `rubric:` may declare

    pass: evidence | reasoning | run
    inputs: [evidence, tool_outputs, ...]          # names below; artifacts take a glob:
    inputs: [conclusion, {artifacts: "results/*.png"}]
    inputs: [evidence, "conclusion?"]              # '?' (quoted in YAML): read if present, not required

Criteria without them get defaults (per-hypothesis → evidence pass, `reasoning_quality` →
reasoning pass, other overall criteria → run pass) and a warning. Every input name maps to ledger
entry kinds and fields (judging-v2 §4.2); the judge reads nothing else.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

PASSES = ("evidence", "reasoning", "run")
SCOPES = ("per_hypothesis", "overall")

# input name -> (ledger source, what produces it)
INPUTS = {
    "evidence": ("hypothesis.evidence[] and evidence_withdrawn",
                 "write_hypothesis / update_hypothesis"),
    "tool_outputs": ("tool_call.output, arguments, error of the calls evidence cites",
                     "harness (every tool call the agent made)"),
    "rationales": ("hypothesis.rationale of every entry", "write_hypothesis / update_hypothesis"),
    "confidence_trajectory": ("hypothesis.confidence and status of every entry",
                              "write_hypothesis / update_hypothesis"),
    "open_questions": ("hypothesis.open_questions (latest)", "write_hypothesis / update_hypothesis"),
    "validation_plan": ("conclusion.validation[hypothesis_id]", "declare_done"),
    "conclusion": ("conclusion.ranked_hypothesis_ids, rationale, confidence_summary",
                   "declare_done"),
    "notes": ("note.note_kind, text, hypothesis_ids", "log_note"),
    "human_input": ("human_input.question, context, answer", "request_human_input"),
    "tool_calls": ("every tool_call entry (tool, error, start of output)", "harness"),
    "agent_messages": ("agent_message entries with visibility 'visible'", "harness"),
    "artifacts": ("artifact entries whose path matches the glob", "harness (workspace files)"),
}

# What each pass may read (judging-v2 §4.1). Pass E never reads the agent's interpretation.
PASS_INPUTS = {
    "evidence": {"evidence", "tool_outputs", "open_questions", "validation_plan", "artifacts"},
    "reasoning": {"evidence", "tool_outputs", "rationales", "confidence_trajectory",
                  "open_questions", "validation_plan", "notes", "conclusion"},
    "run": {"evidence", "confidence_trajectory", "validation_plan", "conclusion", "notes",
            "human_input", "tool_calls", "agent_messages", "artifacts"},
}

DEFAULTS = {
    "evidence": ["evidence", "tool_outputs"],
    "reasoning": ["rationales", "confidence_trajectory", "evidence", "conclusion?"],
    "run": ["evidence", "conclusion?", "notes", "tool_calls"],
}


class RubricError(ValueError):
    pass


@dataclass
class Criterion:
    id: str
    label: str
    scope: str
    scale: str
    guidance: str
    judge_pass: str
    inputs: list[str]
    artifact_globs: list[str] = field(default_factory=list)
    defaulted: bool = False
    optional: list[str] = field(default_factory=list)    # inputs marked 'name?'


    def as_prompt_item(self) -> dict:
        return {"id": self.id, "label": self.label, "scale": self.scale, "guidance": self.guidance}


def _parse_inputs(raw, cid: str) -> tuple[list[str], list[str], list[str]]:
    names, globs, optional = [], [], []
    for item in raw or []:
        if isinstance(item, dict) and set(item) == {"artifacts"}:
            names.append("artifacts")
            globs.append(str(item["artifacts"]))
        elif isinstance(item, str) and item.rstrip("?") in INPUTS and item.rstrip("?") != "artifacts":
            names.append(item.rstrip("?"))
            if item.endswith("?"):
                optional.append(item.rstrip("?"))
        else:
            raise RubricError(f"criterion {cid!r}: unknown input {item!r}; use one of "
                              f"{sorted(INPUTS)} (artifacts as {{artifacts: '<glob>'}})")
    return list(dict.fromkeys(names)), globs, optional


def load_criteria(spec: dict) -> tuple[list[Criterion], list[str]]:
    """The spec's rubric as Criterion objects, plus warnings for anything defaulted."""
    criteria, warnings = [], []
    for i, item in enumerate((spec or {}).get("rubric") or []):
        cid = item.get("id") or f"criterion_{i}"
        scope = item.get("scope", "per_hypothesis")
        if scope not in SCOPES:
            raise RubricError(f"criterion {cid!r}: scope must be one of {SCOPES}, got {scope!r}")
        judge_pass = item.get("pass")
        defaulted = judge_pass is None or "inputs" not in item
        if judge_pass is None:
            judge_pass = ("evidence" if scope == "per_hypothesis"
                          else "reasoning" if cid == "reasoning_quality" else "run")
        if judge_pass not in PASSES:
            raise RubricError(f"criterion {cid!r}: pass must be one of {PASSES}, got {judge_pass!r}")
        if judge_pass == "run" and scope == "per_hypothesis":
            raise RubricError(f"criterion {cid!r}: the run pass scores overall criteria only")
        if "inputs" in item:
            inputs, globs, optional = _parse_inputs(item["inputs"], cid)
        else:
            inputs, globs, optional = _parse_inputs(DEFAULTS[judge_pass], cid)
        bad = set(inputs) - PASS_INPUTS[judge_pass]
        if bad:
            raise RubricError(f"criterion {cid!r}: the {judge_pass} pass doesn't read {sorted(bad)}")
        if defaulted:
            warnings.append(f"criterion {cid!r} declares no "
                            f"{'pass' if item.get('pass') is None else 'inputs'}; using pass "
                            f"{judge_pass!r} with inputs {inputs}")
        criteria.append(Criterion(cid, item.get("label", cid), scope, str(item.get("scale", "1-5")),
                                  " ".join(str(item.get("guidance", "")).split()), judge_pass,
                                  inputs, globs, defaulted, optional))
    return criteria, warnings


def missing_inputs(c: Criterion, run, hypothesis_id: str | None = None) -> list[str]:
    """Inputs the criterion needs that this run doesn't have (→ not_assessable).

    `run` is a judge_inputs.RunView. Evidence and rationales always exist in v2 ledgers (both
    tools require them); notes, tool calls and messages may be empty, which is itself information
    and is shown as "none recorded", not treated as missing.
    """
    missing = []
    conclusion = run.conclusion
    for name in c.inputs:
        if name in c.optional:
            continue
        if name == "conclusion" and conclusion is None:
            missing.append("conclusion (the agent never called declare_done)")
        elif name == "validation_plan":
            plans = (conclusion or {}).get("validation") or {}
            if hypothesis_id is not None and not str(plans.get(hypothesis_id, "")).strip():
                missing.append(f"validation_plan for {hypothesis_id}"
                               + ("" if conclusion else " (no declare_done)"))
            elif hypothesis_id is None and not plans:
                missing.append("validation_plan (no declare_done)" if conclusion is None
                               else "validation_plan (none given)")
        elif name == "human_input" and not run.human_inputs:
            missing.append("human_input (none recorded)")
    for glob in c.artifact_globs:
        if not any(fnmatch.fnmatch(a["path"], glob) for a in run.artifacts):
            missing.append(f"artifacts matching {glob!r}")
    return missing
