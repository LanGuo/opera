import json
import re
from dataclasses import dataclass

from reviewer.llm import LLMClient, get_llm_client

_OBSERVER_SYSTEM = (
    "You are an independent scientific reviewer assessing evidence for a biological hypothesis. "
    "You will NOT be told the researcher's confidence estimate — form your own. "
    "For each evidence item, extract any quantitative value (p-value, fold-change, odds ratio) "
    "if explicitly stated in the text. "
    "Return ONLY a JSON object with no markdown wrapper: "
    '{"observer_confidence": <float 0-1>, '
    '"evidence_assessments": [{"source": <str>, "e_value": <float>, "reasoning": <str>}], '
    '"falsification_criteria": [<str>, <str>, <str>]} '
    "where e_value > 1 means evidence supporting the hypothesis, "
    "e_value < 1 means evidence against, e_value = 1 means neutral. "
    "Calibrated scale: p<0.001→e=10, p<0.01→e=5, p<0.05→e=2, "
    "p>0.1→e=0.5, supporting text (no p-value)→e=1.5, neutral→e=1.0, contradicting→e=0.5. "
    "Cap individual e_values at 10. "
    "Set observer_confidence = min(0.99, E_product / (1 + E_product)) "
    "where E_product is the product of all e_values (capped at 100)."
)

_P_PATTERN = re.compile(r"p\s*[<=]+\s*([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", re.IGNORECASE)

# Threshold beyond which observer_confidence and e_value_confidence are considered
# divergent enough to flag for downstream review. Not a correctness oracle — e_value_conf
# is a rougher regex-based heuristic, not assumed to be more "correct" than the LLM's read.
_DIVERGENCE_THRESHOLD = 0.2


@dataclass
class EvidenceAssessment:
    source: str
    e_value: float
    reasoning: str


@dataclass
class FalsificationResult:
    hypothesis_id: str
    agent_confidence: float | None
    observer_confidence: float
    e_value_product: float
    e_value_confidence: float
    confidence_delta: float
    falsification_criteria: list[str]
    evidence_assessments: list[EvidenceAssessment]
    deterministic_delta: float = 0.0
    confidence_flagged: bool = False


def _evidence_text(item) -> str:
    """Return a plain-text representation of an evidence item (str or dict)."""
    if isinstance(item, str):
        return item
    return item.get("description", "") + " " + item.get("source", "")


def _extract_e_values_from_text(evidence: list) -> float:
    """Compute e-value product from evidence items using regex-extracted p-values."""
    E = 1.0
    for item in evidence:
        text = _evidence_text(item)
        match = _P_PATTERN.search(text)
        if match:
            p = float(match.group(1))
            if p < 0.001:
                e = 10.0
            elif p < 0.01:
                e = 5.0
            elif p < 0.05:
                e = 2.0
            elif p < 0.1:
                e = 1.0
            else:
                e = 0.5
        elif any(w in text.lower() for w in ("support", "confirm", "significant", "enriched")):
            e = 1.5
        elif any(w in text.lower() for w in ("contradict", "refute", "not significant", "fail")):
            e = 0.5
        else:
            e = 1.0
        E *= min(e, 10.0)
        E = min(E, 100.0)
    return E


def assess_hypothesis(hypothesis: dict, client: LLMClient) -> FalsificationResult:
    """Run observer LLM on one hypothesis's evidence. Returns statistically grounded confidence."""
    h_id = hypothesis.get("hypothesis_id", "unknown")
    agent_conf = hypothesis.get("confidence")
    gene = hypothesis.get("candidate_gene", "")
    cell_type = hypothesis.get("cell_type", "")
    claim = hypothesis.get("claim", "")
    evidence = hypothesis.get("evidence", [])

    user = f"Hypothesis: {gene} in {cell_type} — {claim}\n\nEvidence items:\n"
    for i, e in enumerate(evidence):
        text = _evidence_text(e)
        src = e.get("source", "?") if isinstance(e, dict) else "?"
        user += f"{i + 1}. [{src}] {text}\n"

    raw_text = client.chat([
        {"role": "system", "content": _OBSERVER_SYSTEM},
        {"role": "user", "content": user},
    ])
    e_product = _extract_e_values_from_text(evidence)
    e_value_conf = min(0.99, e_product / (1 + e_product))
    try:
        # Strip markdown code fences if the model wrapped the JSON
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        import sys
        print(f"  [falsification] WARNING: observer returned unparseable JSON for {h_id}: {exc}", file=sys.stderr)
        print(f"  [falsification] Raw response (first 500 chars): {raw_text[:500]!r}", file=sys.stderr)
        fallback_observer_conf = 0.5
        fallback_det_delta = round(fallback_observer_conf - e_value_conf, 3)
        fallback_flagged = abs(fallback_det_delta) > _DIVERGENCE_THRESHOLD
        if fallback_flagged:
            print(
                f"  [falsification] WARNING: observer/e-value confidence diverge for {h_id} "
                f"(observer={fallback_observer_conf:.2f}, e_value={e_value_conf:.2f}, "
                f"delta={fallback_det_delta:+.2f})",
                file=sys.stderr,
            )
        return FalsificationResult(
            hypothesis_id=h_id,
            agent_confidence=agent_conf,
            observer_confidence=fallback_observer_conf,
            e_value_product=round(e_product, 3),
            e_value_confidence=round(e_value_conf, 3),
            confidence_delta=0.0,
            deterministic_delta=fallback_det_delta,
            confidence_flagged=fallback_flagged,
            falsification_criteria=[],
            evidence_assessments=[],
        )

    assessments = [
        EvidenceAssessment(
            source=a.get("source", ""),
            e_value=float(a.get("e_value", 1.0)),
            reasoning=a.get("reasoning", ""),
        )
        for a in parsed.get("evidence_assessments", [])
    ]

    observer_conf = float(parsed.get("observer_confidence", 0.5))
    delta = round(observer_conf - (agent_conf if agent_conf is not None else 0.5), 3)

    # Sanity-check the LLM's self-reported observer_conf against the independently
    # (deterministically) computed e_value_conf. Divergence doesn't mean observer_conf is
    # wrong — e_value_conf is a rougher regex heuristic — but it flags cases where the LLM's
    # arithmetic may be inconsistent with the evidence it claims to have used.
    det_delta = round(observer_conf - e_value_conf, 3)
    flagged = abs(det_delta) > _DIVERGENCE_THRESHOLD
    if flagged:
        import sys
        print(
            f"  [falsification] WARNING: observer/e-value confidence diverge for {h_id} "
            f"(observer={observer_conf:.2f}, e_value={e_value_conf:.2f}, delta={det_delta:+.2f})",
            file=sys.stderr,
        )

    return FalsificationResult(
        hypothesis_id=h_id,
        agent_confidence=agent_conf,
        observer_confidence=round(observer_conf, 3),
        e_value_product=round(e_product, 3),
        e_value_confidence=round(e_value_conf, 3),
        confidence_delta=delta,
        deterministic_delta=det_delta,
        confidence_flagged=flagged,
        falsification_criteria=parsed.get("falsification_criteria", []),
        evidence_assessments=assessments,
    )


def assess_terminal_hypotheses(run_dir: str, client: LLMClient | None = None) -> list[FalsificationResult]:
    """Assess all hypotheses with evidence at end of run.

    Includes terminal, active, and weakened hypotheses — any hypothesis the
    agent accumulated evidence for. Refuted hypotheses are excluded as the
    agent already rejected them. The observer's validity is independent of
    whether the agent formally called declare_done on a hypothesis.
    """
    from hypothesis_ledger.workspace import Workspace
    c = client or get_llm_client()
    ws = Workspace(run_dir)
    data = ws.read_ledger()
    to_assess = [
        h for h in data.get("hypotheses", {}).values()
        if h.get("status") != "refuted" and h.get("evidence")
    ]
    return [assess_hypothesis(h, c) for h in to_assess]
