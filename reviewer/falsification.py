import json
import re
from dataclasses import dataclass

from reviewer.llm import LLMClient, extract_json_object, get_llm_client

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
# Diagnostic-performance percentages (sensitivity/specificity/accuracy/AUC) — common in
# clinical-biomarker evidence but invisible to _P_PATTERN since no p-value is stated.
_PERCENT_PATTERN = re.compile(
    r"(sensitivity|specificity|accuracy|auc)\D{0,15}([0-9]{1,3}(?:\.[0-9]+)?)\s*%", re.IGNORECASE
)
# Named effect sizes (odds/hazard ratio, fold-change) given as a bare number, no p-value.
_EFFECT_SIZE_PATTERN = re.compile(
    r"\b(odds ratio|hazard ratio|fold-change|fold change|\bOR\b|\bHR\b)\D{0,10}([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
# Database interaction/confidence scores (e.g. STRING "combined score 0.83").
_INTERACTION_SCORE_PATTERN = re.compile(
    r"(combined score|interaction score|confidence score)\D{0,5}(0?\.[0-9]+)", re.IGNORECASE
)
# Caveat language that undercuts a hypothesis without using the original narrow
# "contradict/refute/not significant/fail" wording — explicit specificity caveats are
# common in this evidence (e.g. "CAUTION: ... is not SLE-specific").
_CAUTION_KEYWORDS = (
    "caution", "not specific", "non-specific", "nonspecific", "cross-react",
    "cross reactiv", "not sle-specific", "no known natural substrate", "orphan",
)
_SUPPORTING_KEYWORDS = (
    "support", "confirm", "significant", "enriched", "validated", "clonally expanded",
    "associated with", "driver",
)
# Negated forms of the supporting keywords above, caught explicitly rather than relying
# on keyword polarity alone — see _keyword_pattern's docstring for why a naive substring
# check gets these backwards (e.g. "insignificant" contains "significant").
_NEGATED_KEYWORDS = ("insignificant", "unsupported", "invalidated", "unconfirmed", "inconclusive")
_CONTRADICTING_KEYWORDS = (
    "contradict", "refute", "not significant", "fail"
) + _NEGATED_KEYWORDS + _CAUTION_KEYWORDS


def _keyword_pattern(keywords: tuple) -> re.Pattern:
    """Compile keywords/phrases into one case-insensitive, start-anchored pattern.

    Anchoring only the start of each keyword (\\b before, nothing after) means a
    suffix still matches freely — "support" matches "supports"/"supported" — but a
    negation prefix glued directly onto the word does NOT, since there is no word
    boundary between it and the root (e.g. "insignificant" does not match
    "significant"; "unsupported" does not match "support"; "invalidated" does not
    match "validated"). A naive `keyword in text` substring check gets exactly these
    cases backwards, silently flipping a negative finding into a supporting one.
    """
    alternation = "|".join(re.escape(k) for k in keywords)
    return re.compile(r"\b(?:" + alternation + r")", re.IGNORECASE)


_SUPPORTING_PATTERN = _keyword_pattern(_SUPPORTING_KEYWORDS)
_CONTRADICTING_PATTERN = _keyword_pattern(_CONTRADICTING_KEYWORDS)

# Threshold beyond which observer_confidence and e_value_confidence are considered
# divergent enough to flag for downstream review. Not a correctness oracle — e_value_conf
# is a rougher regex-based heuristic, not assumed to be more "correct" than the LLM's read.
_DIVERGENCE_THRESHOLD = 0.2

# Tolerance for comparing the LLM's self-reported observer_confidence against what its
# OWN stated per-item e-values arithmetically multiply out to (E = prod(e_i), C = E/(1+E)).
# Unlike _DIVERGENCE_THRESHOLD above (which compares against an independently-derived,
# necessarily imperfect regex read of the raw text), this is pure math over numbers the
# LLM itself already committed to — there is no legitimate "different but reasonable"
# reading here. A gap beyond rounding noise means the LLM's final confidence number is
# arithmetically inconsistent with the evidence weights it itself assigned.
_AGGREGATION_TOLERANCE = 0.05


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
    llm_e_value_product: float = 1.0
    llm_recomputed_confidence: float = 0.5
    aggregation_delta: float = 0.0
    aggregation_inconsistent: bool = False


def _evidence_text(item) -> str:
    """Return a plain-text representation of an evidence item (str or dict)."""
    if isinstance(item, str):
        return item
    return item.get("description", "") + " " + item.get("source", "")


def _extract_e_values_from_text(evidence: list) -> float:
    """Compute e-value product from evidence items using regex-extracted quantitative signal.

    Checks, in order: explicit p-value -> diagnostic-performance percentage
    (sensitivity/specificity/accuracy/AUC) -> named effect size (OR/HR/fold-change) ->
    database interaction/confidence score -> caution/caveat keywords (undercut the
    hypothesis even without the original narrow "contradict" wording) -> generic
    supporting keywords -> neutral default. This remains a heuristic text scan, not a
    ground truth — it exists to sanity-check the LLM's self-reported confidence, not to
    replace it.
    """
    E = 1.0
    for item in evidence:
        text = _evidence_text(item)
        text_lower = text.lower()
        p_match = _P_PATTERN.search(text)
        pct_match = _PERCENT_PATTERN.search(text)
        effect_match = _EFFECT_SIZE_PATTERN.search(text)
        score_match = _INTERACTION_SCORE_PATTERN.search(text)
        if p_match:
            p = float(p_match.group(1))
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
        elif pct_match:
            pct = float(pct_match.group(2))
            e = 3.0 if pct >= 90 else 2.0 if pct >= 70 else 1.5
        elif effect_match:
            e = 2.0
        elif score_match:
            e = 1.5
        elif _CONTRADICTING_PATTERN.search(text_lower):
            e = 0.5
        elif _SUPPORTING_PATTERN.search(text_lower):
            e = 1.5
        else:
            e = 1.0
        E *= min(e, 10.0)
        E = min(E, 100.0)
    return E


def _recompute_confidence_from_assessments(assessments: list) -> tuple[float, float]:
    """Deterministically recompute E and C from the LLM's own stated per-item e-values.

    This checks whether the LLM's self-reported observer_confidence is
    arithmetically consistent with the e-values it assigned to the evidence —
    a pure math check (prod -> ratio), independent of whether those e-values
    are themselves well-chosen. It answers a different question than
    e_value_confidence (which re-derives e-values from scratch via regex):
    "did the model correctly aggregate the numbers it already committed to?"
    """
    E = 1.0
    for a in assessments:
        E *= min(max(a.e_value, 0.0), 10.0)
    E = min(E, 100.0)
    C = min(0.99, E / (1 + E))
    return E, C


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
        parsed = extract_json_object(raw_text)
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

    # Separately, verify the LLM's self-reported observer_conf is arithmetically
    # consistent with the per-item e-values it itself assigned (not a re-derivation
    # from text — a pure recomputation of E = prod(e_i), C = E/(1+E) from numbers
    # the model already committed to in evidence_assessments). Skipped (not flagged)
    # when the model returned no per-item breakdown to check against — that's a
    # missing-data case, not an arithmetic inconsistency.
    if assessments:
        llm_e_product, llm_recomputed_conf = _recompute_confidence_from_assessments(assessments)
        agg_delta = round(observer_conf - llm_recomputed_conf, 3)
        agg_inconsistent = abs(agg_delta) > _AGGREGATION_TOLERANCE
    else:
        llm_e_product, llm_recomputed_conf = 1.0, 0.5
        agg_delta = 0.0
        agg_inconsistent = False
    if agg_inconsistent:
        import sys
        print(
            f"  [falsification] WARNING: observer's reported confidence is arithmetically "
            f"inconsistent with its own stated e-values for {h_id} "
            f"(reported={observer_conf:.2f}, recomputed-from-own-e-values={llm_recomputed_conf:.2f}, "
            f"delta={agg_delta:+.2f})",
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
        llm_e_value_product=round(llm_e_product, 3),
        llm_recomputed_confidence=round(llm_recomputed_conf, 3),
        aggregation_delta=agg_delta,
        aggregation_inconsistent=agg_inconsistent,
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
