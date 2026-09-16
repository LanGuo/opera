import json
import os
import re
import sys
from dataclasses import dataclass, field

from hypothesis_ledger.workspace import Workspace
from reviewer.llm import LLMClient, get_llm_client


@dataclass
class CriterionScore:
    criterion_id: str   # matches rubric item id (e.g. "novelty")
    rationale: str      # 2-3 sentence evidence-grounded justification
    score: str          # "4/5", "yes", "partial", etc — matches rubric scale
    agent_note: str = ""  # pass-2 agreement/divergence note; empty string if no pass 2


@dataclass
class HypothesisScores:
    hypothesis_id: str
    candidate_gene: str
    per_hypothesis_raw: str  # raw LLM text covering all per_hypothesis rubric items
    criteria: list[CriterionScore] = field(default_factory=list)  # structured per-criterion scores


@dataclass
class RunScores:
    hypothesis_scores: list[HypothesisScores]
    overall_raw: str   # raw LLM text covering all overall rubric items
    run_dir: str
    overall_criteria: list[CriterionScore] = field(default_factory=list)  # structured overall scores


# Fallback prompts used when task_spec.yaml is absent from the run directory.
_DESIGN_PROMPT = """\
You are an expert in experimental biology and clinical research design.
Evaluate the proposed validation experiment for the following SLE biomarker hypothesis.

Hypothesis: {hypothesis}

Proposed validation experiment:
{design_description}

Score the experimental design on these criteria (1–5 each):
1. Feasibility — realistic cell numbers, reagents, and assays?
2. Specificity — does the experiment directly test the hypothesis?
3. Statistical power — appropriate cohort size and controls?
4. Translatability — potential for clinical use?

Respond with a concise evaluation (≤200 words) ending with:
Overall score: X/20
"""

_NOVELTY_PROMPT = """\
You are a scientific reviewer assessing the novelty and significance of a biomarker hypothesis.

Hypothesis: {hypothesis}
Cell type: {cell_type}
Evidence summary: {evidence_str}

Evaluate:
1. Is this gene/cell-type combination a well-known SLE association or genuinely novel?
2. Does the multi-modal evidence (DEG, GWAS, protein, literature) strengthen the claim?
3. What is the potential clinical impact if validated?

Respond concisely (≤200 words) ending with:
Novelty score: X/5
Significance score: X/5
"""

_BIAS_PREAMBLE = (
    "Score based solely on the information provided below — do not query external databases, "
    "verify claims via search, or use any tools. Do not reward length, confidence of language, "
    "or presentation style — only scientific content and rigor."
)

# Shared JSON-output contract for both pass-1 scoring calls (per-hypothesis and overall).
# Free-text output ("Score: X/5" lines) proved unreliable to parse back out: judge models
# inconsistently wrapped it in markdown, restructured pass-2 responses entirely, or used
# numbered sub-lists inside a criterion's own rationale that collided with the block
# splitter. JSON has no such ambiguity — the model's prose lives inside string values,
# never in a position that could be mistaken for a structural boundary.
_JSON_PASS1_INSTRUCTION = (
    '\nReturn ONLY a JSON object with no markdown code fence, matching this shape:\n'
    '{{"criteria": [{{"criterion_id": <str>, "score": <str, e.g. "4/5" or "yes/no/partial">, '
    '"rationale": <str, 2-3 sentences>}}, ...]}}\n'
    'Include exactly one entry per criterion below, using the exact criterion_id given for each.\n'
)

_JSON_PASS2_INSTRUCTION = (
    '\nReturn ONLY a JSON object with no markdown code fence, matching this shape:\n'
    '{"agent_notes": [{"criterion_id": <str>, "agent_note": <str, 1-2 sentences>}, ...]}\n'
    "Include one entry per criterion_id listed above — do not restate scores or rationale."
)


def _evidence_text(item, default_type: str = "unknown") -> tuple[str, str]:
    """Return (type, description) for an evidence item that may be a plain string
    (Gemini's format) or a dict (Claude's format)."""
    if isinstance(item, str):
        return default_type, item
    return item.get("type", default_type), item.get("description", "")


def _format_evidence_trail(update_history: list) -> str:
    """Render update_history as an evidence timeline, excluding confidence numbers and rationale."""
    if not update_history:
        return ""
    lines = ["Evidence accumulation timeline:"]
    for i, update in enumerate(update_history, 1):
        new_evidence = update.get("changes", {}).get("to", {}).get("evidence", [])
        old_evidence = update.get("changes", {}).get("from", {}).get("evidence", [])
        added = [e for e in new_evidence if e not in old_evidence]
        for ev in added:
            ev_type, description = _evidence_text(ev)
            lines.append(f"  Step {i} — {ev_type}: {description}")
    return "\n".join(lines)


def _format_open_questions(open_questions: list | None) -> str:
    if not open_questions:
        return ""
    items = "\n".join(f"  - {q}" for q in open_questions)
    return f"Unresolved questions flagged by the agent:\n{items}"


def _format_refuted_hypotheses(all_hypotheses: dict) -> str:
    refuted = [h for h in all_hypotheses.values() if h.get("status") == "refuted"]
    if not refuted:
        return ""
    lines = ["Refuted hypotheses (agent discarded these during the run):"]
    for h in refuted:
        gene = h.get("candidate_gene", "unknown")
        cell_type = h.get("cell_type", "unknown")
        evidence = h.get("evidence", [])
        descriptions = [_evidence_text(e)[1] for e in evidence]
        ev_summary = "; ".join(d[:120] for d in descriptions if d)
        lines.append(f"  - {cell_type}/{gene}: evidence found — {ev_summary or 'none'}")
    return "\n".join(lines)


def _format_criteria_listing(items: list) -> str:
    """Render rubric items with explicit criterion_id, so the model echoes back an id
    we can match on rather than relying on positional/count alignment."""
    return "\n\n".join(
        f"- criterion_id: {item.get('id', f'criterion_{i}')}\n"
        f"  label: {item.get('label', item.get('id', 'unknown'))}\n"
        f"  guidance: {item.get('guidance', '')}"
        for i, item in enumerate(items)
    )


def _build_per_hypothesis_prompt_pass1(
    goal: str, per_items: list, hypothesis_str: str, evidence_str: str,
    validation: str, evidence_trail: str, open_questions_str: str,
) -> str:
    parts = [
        f"Task goal: {goal}\n",
        f"You are scoring a completed AI agent evaluation run. {_BIAS_PREAMBLE}\n",
        f"Hypothesis: {hypothesis_str}",
        f"Final evidence:\n{evidence_str}",
    ]
    if evidence_trail:
        parts.append(evidence_trail)
    if open_questions_str:
        parts.append(open_questions_str)
    if validation:
        parts.append(f"Proposed validation: {validation}")
    parts.append(_JSON_PASS1_INSTRUCTION)
    parts.append(_format_criteria_listing(per_items))
    return "\n\n".join(parts)


def _build_per_hypothesis_prompt_pass2(
    criteria: list["CriterionScore"], agent_rationale: str,
) -> str:
    scores_summary = "\n".join(f"- {c.criterion_id}: {c.score} — {c.rationale}" for c in criteria)
    return (
        "You just scored this hypothesis based on observed evidence:\n"
        f"{scores_summary}\n\n"
        "Now review the agent's own rationale for the same hypothesis and note where your "
        "independent assessment agrees or diverges, for each criterion above.\n\n"
        f"Agent's rationale for the final belief update:\n{agent_rationale}"
        f"{_JSON_PASS2_INSTRUCTION}"
    )


def _build_overall_prompt_pass1(
    goal: str, overall_items: list, metrics: dict, refuted_str: str,
) -> str:
    parts = [
        f"Task goal: {goal}\n",
        f"You are scoring a completed AI agent evaluation run. {_BIAS_PREAMBLE}\n",
        "Run summary:",
        f"  Hypotheses formed: {metrics['hypothesis_count']}",
        f"  Belief revisions: {metrics['belief_revision_count']}",
        f"  Refutation rate: {metrics['refutation_rate']:.0%}",
    ]
    if refuted_str:
        parts.append(f"\n{refuted_str}")
    parts.append(_JSON_PASS1_INSTRUCTION)
    parts.append(_format_criteria_listing(overall_items))
    return "\n\n".join(parts)


def _build_overall_prompt_pass2(
    criteria: list["CriterionScore"], termination_rationale: str, confidence_summary: str,
) -> str:
    scores_summary = "\n".join(f"- {c.criterion_id}: {c.score} — {c.rationale}" for c in criteria)
    return (
        "You just scored the overall run quality based on observed metrics:\n"
        f"{scores_summary}\n\n"
        "Now review the agent's own termination rationale and confidence summary, and note "
        "where your independent assessment agrees or diverges, for each criterion above.\n\n"
        f"Agent's termination rationale:\n{termination_rationale}\n\n"
        f"Agent's confidence summary:\n{confidence_summary}"
        f"{_JSON_PASS2_INSTRUCTION}"
    )


def _strip_json_fence(raw: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())


def _parse_json_criteria(raw: str, rubric_items: list) -> list[CriterionScore]:
    """Parse pass-1's JSON criteria response, keeping only entries whose criterion_id
    matches a real rubric item (guards against a hallucinated or malformed id rather
    than silently accepting it). Returns [] on any parse failure.
    """
    if not raw or not rubric_items:
        return []
    valid_ids = {item.get("id", f"criterion_{i}") for i, item in enumerate(rubric_items)}
    try:
        parsed = json.loads(_strip_json_fence(raw))
        result = []
        for c in parsed.get("criteria", []):
            cid = c.get("criterion_id", "")
            if cid not in valid_ids:
                continue
            result.append(CriterionScore(
                criterion_id=cid,
                rationale=str(c.get("rationale", "")).strip(),
                score=str(c.get("score", "")).strip(),
            ))
        return result
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return []


def _parse_json_agent_notes(raw: str) -> dict[str, str]:
    """Parse pass-2's JSON agent-notes response into {criterion_id: note}. Returns {}
    on any parse failure — pass-1's scores/rationale are never affected by this,
    since agent_note is applied as an annotation on top of pass-1's own CriterionScore
    objects rather than by re-deriving anything from pass-2's text.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(_strip_json_fence(raw))
        return {
            n["criterion_id"]: str(n.get("agent_note", "")).strip()
            for n in parsed.get("agent_notes", [])
            if n.get("criterion_id")
        }
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError, KeyError):
        return {}


def _score_criteria_two_pass(
    client: LLMClient, prompt1: str, rubric_items: list,
    build_prompt2: "callable | None", raw_label: str,
) -> tuple[str, list[CriterionScore]]:
    """Run pass-1 scoring, then optionally pass-2 (agent-note-only) annotation.

    build_prompt2, if given, is a zero-arg callable returning the pass-2 prompt string
    (built from the already-parsed pass-1 criteria) — deferred so pass-2 is skipped
    entirely when pass-1 yields nothing to annotate.
    Returns (raw_text_for_display, criteria). Pass-2 can only ADD agent_note to
    criteria pass-1 already produced; it can never replace score or rationale.
    """
    pass1 = client.chat([{"role": "user", "content": prompt1}])
    criteria = _parse_json_criteria(pass1, rubric_items)
    raw_display = pass1

    if build_prompt2 is not None and criteria:
        prompt2 = build_prompt2(criteria)
        pass2 = client.chat([{"role": "user", "content": prompt2}])
        notes = _parse_json_agent_notes(pass2)
        if notes:
            for c in criteria:
                if c.criterion_id in notes:
                    c.agent_note = notes[c.criterion_id]
        raw_display = f"{pass1}\n\n--- {raw_label} agreement notes ---\n\n{pass2}"

    return raw_display, criteria


def score_run(run_dir: str, client: LLMClient | None = None) -> RunScores:
    ws = Workspace(run_dir)
    if not ws.final_report_path.exists():
        raise FileNotFoundError(f"No final_report.json in {run_dir}. Run must be completed.")

    report = json.loads(ws.final_report_path.read_text(encoding="utf-8"))
    task_spec = ws.read_task_spec()

    if client is None:
        client = get_llm_client()

    ledger = ws.read_ledger()
    all_hypotheses = ledger.get("hypotheses", {})

    top_hypotheses = report.get("top_hypotheses") or []
    if not top_hypotheses:
        top_hypotheses = [
            h for h in all_hypotheses.values()
            if h.get("status") in ("terminal", "active")
        ]

    validation = report.get("suggested_validation", "")
    termination_rationale = report.get("rationale", "")
    confidence_summary = report.get("confidence_summary", "")
    refuted_str = _format_refuted_hypotheses(all_hypotheses)
    scores = []

    if task_spec:
        per_items = [r for r in task_spec.get("rubric", []) if r.get("scope") == "per_hypothesis"]
        overall_items = [r for r in task_spec.get("rubric", []) if r.get("scope") == "overall"]
        goal = task_spec.get("goal", "")

        for h in top_hypotheses:
            gene = h.get("candidate_gene", "unknown")
            cell_type = h.get("cell_type", "unknown")
            claim = h.get("claim", "")
            evidence_str = json.dumps(h.get("evidence", []))
            hypothesis_str = f"{cell_type} / {gene}: {claim}"
            update_history = h.get("update_history", [])
            open_questions = h.get("open_questions")

            evidence_trail = _format_evidence_trail(update_history)
            open_questions_str = _format_open_questions(open_questions)

            # Last update's rationale for pass 2 (agent interpretation, kept out of pass 1)
            agent_rationale = ""
            if update_history:
                agent_rationale = update_history[-1].get("rationale", "")

            per_hyp_raw = ""
            criteria: list[CriterionScore] = []
            if per_items:
                prompt1 = _build_per_hypothesis_prompt_pass1(
                    goal, per_items, hypothesis_str, evidence_str,
                    validation, evidence_trail, open_questions_str,
                )
                build_prompt2 = (
                    (lambda crit: _build_per_hypothesis_prompt_pass2(crit, agent_rationale))
                    if agent_rationale else None
                )
                per_hyp_raw, criteria = _score_criteria_two_pass(
                    client, prompt1, per_items, build_prompt2, "per-hypothesis",
                )
            scores.append(HypothesisScores(
                hypothesis_id=h.get("hypothesis_id", "unknown"),
                candidate_gene=gene,
                per_hypothesis_raw=per_hyp_raw,
                criteria=criteria,
            ))

        overall_raw = ""
        overall_criteria: list[CriterionScore] = []
        if overall_items:
            n_hyps = len(top_hypotheses)
            metrics_summary = {
                "hypothesis_count": n_hyps,
                "belief_revision_count": sum(
                    len(h.get("update_history", [])) for h in top_hypotheses
                ),
                "refutation_rate": sum(
                    1 for h in top_hypotheses if h.get("status") == "refuted"
                ) / max(n_hyps, 1),
            }
            prompt1 = _build_overall_prompt_pass1(goal, overall_items, metrics_summary, refuted_str)
            build_prompt2 = (
                (lambda crit: _build_overall_prompt_pass2(crit, termination_rationale, confidence_summary))
                if (termination_rationale or confidence_summary) else None
            )
            overall_raw, overall_criteria = _score_criteria_two_pass(
                client, prompt1, overall_items, build_prompt2, "overall",
            )

    else:
        # Fallback: no task_spec.yaml in run dir — use hardcoded SLE prompts
        for h in top_hypotheses:
            gene = h.get("candidate_gene", "unknown")
            cell_type = h.get("cell_type", "unknown")
            claim = h.get("claim", "")
            evidence_str = json.dumps(h.get("evidence", []))
            hypothesis_str = f"{cell_type} / {gene}: {claim}"

            design_raw = client.chat([{"role": "user", "content": _DESIGN_PROMPT.format(
                hypothesis=hypothesis_str, design_description=validation,
            )}])
            novelty_raw = client.chat([{"role": "user", "content": _NOVELTY_PROMPT.format(
                hypothesis=hypothesis_str, cell_type=cell_type, evidence_str=evidence_str,
            )}])
            scores.append(HypothesisScores(
                hypothesis_id=h.get("hypothesis_id", "unknown"),
                candidate_gene=gene,
                per_hypothesis_raw=design_raw + "\n\n" + novelty_raw,
                criteria=[],
            ))
        overall_raw = ""
        overall_criteria = []

    return RunScores(
        hypothesis_scores=scores,
        overall_raw=overall_raw,
        run_dir=run_dir,
        overall_criteria=overall_criteria,
    )


def score_run_all_backends(run_dir: str) -> dict[str, RunScores]:
    """Score a run with all configured LLM backends.

    Detects backends from environment:
      - ANTHROPIC_API_KEY  → key "anthropic",   model claude-haiku-4-5
      - GOOGLE_API_KEY     → key "google",       model gemini-2.5-flash
      - OLLAMA_MODEL=x     → key "ollama:x"

    Returns a dict mapping backend key → RunScores. Individual backend failures
    are logged to stderr and skipped. Returns {} if no backends are configured.
    """
    backends: list[tuple[str, LLMClient]] = []

    # JSON output (criterion_id/score/rationale repeated per entry, quoted and
    # escaped) runs noticeably longer than the old free-text format did for the
    # same content -- bump past the client's 4096 default so a detailed multi-
    # criterion response doesn't get truncated mid-JSON.
    JUDGE_MAX_TOKENS = 8192

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            backends.append(("anthropic", LLMClient(
                backend="anthropic", model="claude-haiku-4-5", max_tokens=JUDGE_MAX_TOKENS,
            )))
        except Exception as exc:
            print(f"[score] skipping anthropic backend: {exc}", file=sys.stderr)

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            backends.append(("google", LLMClient(
                backend="google", model="gemini-2.5-flash", max_tokens=JUDGE_MAX_TOKENS,
            )))
        except Exception as exc:
            print(f"[score] skipping google backend: {exc}", file=sys.stderr)

    ollama_model = os.environ.get("OLLAMA_MODEL")
    if ollama_model:
        try:
            backends.append((f"ollama:{ollama_model}", LLMClient(
                backend="ollama", model=ollama_model, max_tokens=JUDGE_MAX_TOKENS,
            )))
        except Exception as exc:
            print(f"[score] skipping ollama backend: {exc}", file=sys.stderr)

    if not backends:
        return {}

    results: dict[str, RunScores] = {}
    for key, client in backends:
        try:
            results[key] = score_run(run_dir, client=client)
        except Exception as exc:
            print(f"[score] backend {key!r} failed: {exc}", file=sys.stderr)

    return results


def print_scores(scores: RunScores) -> None:
    print(f"=== Automated Scores for {scores.run_dir} ===")
    for hs in scores.hypothesis_scores:
        print(f"\n--- {hs.hypothesis_id}: {hs.candidate_gene} ---")
        print(f"[Per-Hypothesis Evaluation]\n{hs.per_hypothesis_raw}")
    if scores.overall_raw:
        print(f"\n[Overall Run Evaluation]\n{scores.overall_raw}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m reviewer.score <run_dir>")
        sys.exit(1)
    print_scores(score_run(sys.argv[1]))
