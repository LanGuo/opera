"""Instruction wording for the LLM judge and the falsification observer.

Only instruction prose lives here. The rubric, run data, section labels, JSON keys, and the
observer's e-value method are fixed and are assembled by score.py and falsification.py.

"original" is the production wording, byte-for-byte, including a defect: the judge's pass-1
output instruction shows the model doubled braces, because the string is never passed through
.format(). P0 fixes only that; P1 and P2 are rewordings used to measure how much judge and
observer outputs depend on wording (see reviewer.variability).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PromptVariant:
    judge_role: str                 # role line and bias rules, pass 1
    judge_pass1_format: str         # output-format instruction, pass 1
    judge_pass2_hyp_lead: str       # line before the pass-1 scores, per-hypothesis pass 2
    judge_pass2_hyp_task: str       # instruction after the scores, per-hypothesis pass 2
    judge_pass2_overall_lead: str
    judge_pass2_overall_task: str
    judge_pass2_format: str         # output-format instruction, pass 2
    observer_lead: str              # observer text before the output shape


_PASS1_SHAPE = (
    '{"criteria": [{"criterion_id": <str>, "score": <str, e.g. "4/5" or "yes/no/partial">, '
    '"rationale": <str, 2-3 sentences>}, ...]}'
)
_PASS2_SHAPE = '{"agent_notes": [{"criterion_id": <str>, "agent_note": <str, 1-2 sentences>}, ...]}'

OBSERVER_SHAPE = (
    '{"observer_confidence": <float 0-1>, '
    '"evidence_assessments": [{"source": <str>, "e_value": <float>, "reasoning": <str>}], '
    '"falsification_criteria": [<str>, <str>, <str>]}'
)
OBSERVER_METHOD = (
    "where e_value > 1 means evidence supporting the hypothesis, "
    "e_value < 1 means evidence against, e_value = 1 means neutral. "
    "Calibrated scale: p<0.001→e=10, p<0.01→e=5, p<0.05→e=2, "
    "p>0.1→e=0.5, supporting text (no p-value)→e=1.5, neutral→e=1.0, contradicting→e=0.5. "
    "Cap individual e_values at 10. "
    "Set observer_confidence = min(0.99, E_product / (1 + E_product)) "
    "where E_product is the product of all e_values (capped at 100)."
)


def observer_system(v: PromptVariant) -> str:
    return v.observer_lead + OBSERVER_SHAPE + " " + OBSERVER_METHOD


_ORIGINAL = PromptVariant(
    judge_role=(
        "You are scoring a completed AI agent evaluation run. "
        "Score based solely on the information provided below — do not query external databases, "
        "verify claims via search, or use any tools. Do not reward length, confidence of language, "
        "or presentation style — only scientific content and rigor."
    ),
    judge_pass1_format=(
        "\nReturn ONLY a JSON object with no markdown code fence, matching this shape:\n"
        + _PASS1_SHAPE.replace("{", "{{").replace("}", "}}")
        + "\nInclude exactly one entry per criterion below, using the exact criterion_id given for each.\n"
    ),
    judge_pass2_hyp_lead="You just scored this hypothesis based on observed evidence:",
    judge_pass2_hyp_task=(
        "Now review the agent's own rationale for the same hypothesis and note where your "
        "independent assessment agrees or diverges, for each criterion above."
    ),
    judge_pass2_overall_lead="You just scored the overall run quality based on observed metrics:",
    judge_pass2_overall_task=(
        "Now review the agent's own termination rationale and confidence summary, and note "
        "where your independent assessment agrees or diverges, for each criterion above."
    ),
    judge_pass2_format=(
        "\nReturn ONLY a JSON object with no markdown code fence, matching this shape:\n"
        + _PASS2_SHAPE
        + "\nInclude one entry per criterion_id listed above — do not restate scores or rationale."
    ),
    observer_lead=(
        "You are an independent scientific reviewer assessing evidence for a biological hypothesis. "
        "You will NOT be told the researcher's confidence estimate — form your own. "
        "For each evidence item, extract any quantitative value (p-value, fold-change, odds ratio) "
        "if explicitly stated in the text. "
        "Return ONLY a JSON object with no markdown wrapper: "
    ),
)

VARIANTS: dict[str, PromptVariant] = {
    "original": _ORIGINAL,
    "P0": replace(
        _ORIGINAL,
        judge_pass1_format=(
            "\nReturn ONLY a JSON object with no markdown code fence, matching this shape:\n"
            + _PASS1_SHAPE
            + "\nInclude exactly one entry per criterion below, using the exact criterion_id given for each.\n"
        ),
    ),
    "P1": PromptVariant(
        judge_role=(
            "You are evaluating a finished research run produced by an AI agent. "
            "Base every score only on the material given below. Do not consult external databases, "
            "check claims with a search, or use any tools. Give no credit for length, assertive "
            "wording, or presentation; judge only the scientific content and its rigor."
        ),
        judge_pass1_format=(
            "\nRespond with a single JSON object and nothing else (no markdown code fence), in this form:\n"
            + _PASS1_SHAPE
            + "\nGive exactly one entry for each criterion listed below, and copy each criterion_id "
            "exactly as written.\n"
        ),
        judge_pass2_hyp_lead="These are the scores you gave this hypothesis from the observed evidence:",
        judge_pass2_hyp_task=(
            "Next, read the agent's own rationale for this hypothesis and, for each criterion above, "
            "note where your independent assessment agrees with it or departs from it."
        ),
        judge_pass2_overall_lead=(
            "These are the scores you gave the run's overall quality from the observed metrics:"
        ),
        judge_pass2_overall_task=(
            "Next, read the agent's own termination rationale and confidence summary and, for each "
            "criterion above, note where your independent assessment agrees with them or departs from them."
        ),
        judge_pass2_format=(
            "\nRespond with a single JSON object and nothing else (no markdown code fence), in this form:\n"
            + _PASS2_SHAPE
            + "\nGive one entry for each criterion_id listed above, and do not repeat the scores or rationales."
        ),
        observer_lead=(
            "You are an independent scientist reviewing the evidence for a biological hypothesis. "
            "The researcher's own confidence estimate is withheld from you; reach your own. "
            "For each evidence item, pull out any quantitative value (p-value, fold-change, odds ratio) "
            "that the text states explicitly. "
            "Respond with a single JSON object and nothing else (no markdown wrapper): "
        ),
    ),
    "P2": PromptVariant(
        judge_role=(
            "Task: score a completed run by an AI research agent.\n"
            "Rules:\n"
            "1. Use only the information in this message.\n"
            "2. Do not search, query databases, verify claims externally, or use tools.\n"
            "3. Score scientific content and rigor only. Length, confident wording, and presentation "
            "earn no credit."
        ),
        judge_pass1_format=(
            "\nOutput format: one JSON object, with no markdown code fence and no other text.\n"
            "Shape: " + _PASS1_SHAPE
            + "\nList every criterion below exactly once, no more and no fewer, and copy each "
            "criterion_id exactly.\n"
        ),
        judge_pass2_hyp_lead="Your evidence-based scores for this hypothesis:",
        judge_pass2_hyp_task=(
            "Task: compare these with the agent's own rationale below. For each criterion above, "
            "note where your independent assessment agrees or diverges."
        ),
        judge_pass2_overall_lead="Your metric-based scores for overall run quality:",
        judge_pass2_overall_task=(
            "Task: compare these with the agent's termination rationale and confidence summary below. "
            "For each criterion above, note where your independent assessment agrees or diverges."
        ),
        judge_pass2_format=(
            "\nOutput format: one JSON object, with no markdown code fence and no other text.\n"
            "Shape: " + _PASS2_SHAPE
            + "\nOne entry per criterion_id listed above. Do not restate scores or rationales."
        ),
        observer_lead=(
            "Role: independent reviewer of the evidence for a biological hypothesis.\n"
            "The researcher's confidence estimate is not shown to you. Produce your own.\n"
            "For each evidence item, extract any explicitly stated quantitative value "
            "(p-value, fold-change, odds ratio).\n"
            "Output: one JSON object, with no markdown wrapper and no other text: "
        ),
    ),
}


def get_variant(name: str) -> PromptVariant:
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"unknown prompt variant {name!r}; choose from {sorted(VARIANTS)}") from None
