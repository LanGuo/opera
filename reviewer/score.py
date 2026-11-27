import json
import os
import re
import sys
from dataclasses import dataclass, field

from hypothesis_ledger.workspace import Workspace
from reviewer.evidence import (
    DROPPED, SUPERSEDED, all_evidence, evidence_for_judging, evidence_parts, format_evidence_list,
)
from reviewer.llm import LLMClient, extract_json_object, get_llm_client
from reviewer.prompt_variants import PromptVariant, get_variant


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
    status: str = ""  # ledger status at end of run; "weakened"/"refuted" ones aren't final candidates
    final_candidate: bool = True  # False for weakened/refuted hypotheses scored for completeness
    # Size of what the judge was shown (see _hypothesis_input_stats), recorded so score-vs-length
    # can be checked after the fact.
    input_stats: dict = field(default_factory=dict)


@dataclass
class RunScores:
    hypothesis_scores: list[HypothesisScores]
    overall_raw: str   # raw LLM text covering all overall rubric items
    run_dir: str
    overall_criteria: list[CriterionScore] = field(default_factory=list)  # structured overall scores
    completed: bool = True  # False: no final_report.json (agent never called declare_done)
    overall_input_stats: dict = field(default_factory=dict)  # size of the overall-pass inputs


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

# Instruction wording (role, bias rules, JSON output contract) lives in reviewer.prompt_variants.
# The judge answers in JSON because free text ("Score: X/5" lines) proved unreliable to parse:
# models wrapped it in markdown, restructured pass-2 responses, or used numbered sub-lists that
# collided with the block splitter. In JSON the model's prose stays inside string values.


# Fixed across prompt variants (like the observer's e-value method): how to weigh evidence, so the
# score doesn't follow item count or text length. Saved runs showed pass-1 scores rising with the
# number of evidence items and, across agents, with characters per item.
_EVIDENCE_WEIGHING = (
    "How to weigh the evidence: judge what each item shows, not how many items there are or how "
    "long they are. Items that restate the same finding, or come from the same source or analysis, "
    "count as one line of evidence. A short item with a quantified result can outweigh several long "
    "descriptive ones. Do not raise a score because the record is longer or lower it because it is "
    "terse; score the strength and independence of what is actually shown."
)


def _evidence_text(item, default_type: str = "unknown") -> tuple[str, str]:
    """Return (type, text) for an evidence item: a plain string (Gemini's format) or a dict
    with any of the key sets agents use (see reviewer.evidence.evidence_parts)."""
    return evidence_parts(item, default_type)


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
    """Refuted and weakened hypotheses with a short evidence summary (weakened listed separately:
    the agent lowered them below its final candidates without rejecting them outright)."""
    sections = []
    for status, heading in (
        ("refuted", "Refuted hypotheses (agent discarded these during the run):"),
        ("weakened", "Weakened hypotheses (agent downgraded these during the run):"),
    ):
        hs = [h for h in all_hypotheses.values() if h.get("status") == status]
        if not hs:
            continue
        lines = [heading]
        for h in hs:
            gene = h.get("candidate_gene", "unknown")
            cell_type = h.get("cell_type", "unknown")
            descriptions = [_evidence_text(e)[1] for e, _ in evidence_for_judging(h)]
            ev_summary = "; ".join(d[:120] for d in descriptions if d)
            lines.append(f"  - {cell_type}/{gene}: evidence found — {ev_summary or 'none'}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


_SCORED_STATUSES = ("terminal", "active")
_DISCARDED_STATUSES = ("weakened", "refuted")


def _hypotheses_to_score(report: dict, all_hypotheses: dict) -> list[dict]:
    """The agent's final candidates, then every weakened or refuted hypothesis.

    Final candidates are the report's top_hypotheses, or the active/terminal ledger entries when
    the run has no report (or an empty list). Weakened/refuted ones were previously never scored,
    so the judge never saw how the agent handled disconfirmation; they are appended, taking their
    status from the ledger.
    """
    top = list(report.get("top_hypotheses") or [])
    if not top:
        top = [h for h in all_hypotheses.values() if h.get("status") in _SCORED_STATUSES]
    seen = {h.get("hypothesis_id") for h in top}
    discarded = [
        h for h in all_hypotheses.values()
        if h.get("status") in _DISCARDED_STATUSES and h.get("hypothesis_id") not in seen
    ]
    return top + discarded


def _run_metrics(all_hypotheses: dict, scored: list[dict]) -> dict:
    """Run-level counts over every hypothesis the agent formed.

    Previously computed over the final candidates only, which by construction contain no refuted
    hypotheses, so the judge was shown a 0% refutation rate for every run.
    """
    pool = list(all_hypotheses.values()) or scored
    n = len(pool)
    by_status = {s: sum(1 for h in pool if h.get("status") == s) for s in _DISCARDED_STATUSES}
    return {
        "hypothesis_count": n,
        "final_count": sum(1 for h in pool if h.get("status") in _SCORED_STATUSES),
        "weakened_count": by_status["weakened"],
        "refuted_count": by_status["refuted"],
        "belief_revision_count": sum(len(h.get("update_history") or []) for h in pool),
        "refutation_rate": by_status["refuted"] / max(n, 1),
        "weakened_rate": by_status["weakened"] / max(n, 1),
    }


def _hypothesis_input_stats(h: dict, evidence_str: str, trail: str, rationale: str) -> dict:
    """How much the judge was shown for one hypothesis. Item counts follow what the prompt lists
    (final + dropped items; superseded rewordings are not shown but counted separately)."""
    states = [s for _, s in all_evidence(h)]
    return {
        "evidence_items": sum(1 for s in states if s != SUPERSEDED),
        "dropped_items": states.count(DROPPED),
        "superseded_items": states.count(SUPERSEDED),
        "evidence_chars": len(evidence_str),
        "trail_lines": max(len(trail.splitlines()) - 1, 0) if trail else 0,
        "rationale_chars": len(rationale or ""),
    }


def _hypothesis_label(h: dict) -> str:
    return f"{h.get('cell_type', 'unknown')} / {h.get('candidate_gene', 'unknown')}"


def _format_hypothesis_table(scored: list[dict], final_ids: set) -> str:
    """Every hypothesis the judge scores, for the overall pass: label, status, claim, how much
    evidence and how many revisions. No confidence numbers (pass 1 excludes agent confidence)."""
    if not scored:
        return ""
    lines = ["Hypotheses at the end of the run:"]
    for h in scored:
        role = "final candidate" if h.get("hypothesis_id") in final_ids else h.get("status", "unknown")
        n_ev = len(evidence_for_judging(h))
        n_up = len(h.get("update_history") or [])
        lines.append(
            f"  - {_hypothesis_label(h)} [{role}]: {h.get('claim', '')} "
            f"(evidence items: {n_ev}; revisions: {n_up})"
        )
    return "\n".join(lines)


def _format_criteria_listing(items: list) -> str:
    """Render rubric items with explicit criterion_id, so the model echoes back an id
    we can match on rather than relying on positional/count alignment.

    Includes each item's own `scale` explicitly (e.g. "1-5" vs. "yes/no/partial").
    Without this, the model has to infer the right format per criterion from the
    pass-1 instruction's generic example alone, which shows both formats without
    tying either to a specific criterion — observed in practice to produce a
    "partial" answer for a 1-5-scale criterion.
    """
    return "\n\n".join(
        f"- criterion_id: {item.get('id', f'criterion_{i}')}\n"
        f"  label: {item.get('label', item.get('id', 'unknown'))}\n"
        f"  scale: {item.get('scale', '1-5')}\n"
        f"  guidance: {item.get('guidance', '')}"
        for i, item in enumerate(items)
    )


def _build_per_hypothesis_prompt_pass1(
    goal: str, per_items: list, hypothesis_str: str, evidence_str: str,
    validation: str, evidence_trail: str, open_questions_str: str, v: PromptVariant,
) -> str:
    parts = [
        f"Task goal: {goal}\n",
        f"{v.judge_role}\n",
        f"Hypothesis: {hypothesis_str}",
        f"Evidence recorded by the agent:\n{evidence_str}",
    ]
    if evidence_trail:
        parts.append(evidence_trail)
    if open_questions_str:
        parts.append(open_questions_str)
    if validation:
        parts.append(f"Proposed validation: {validation}")
    parts.append(_EVIDENCE_WEIGHING)
    parts.append(v.judge_pass1_format)
    parts.append(_format_criteria_listing(per_items))
    return "\n\n".join(parts)


def _build_per_hypothesis_prompt_pass2(
    criteria: list["CriterionScore"], agent_rationale: str, v: PromptVariant,
) -> str:
    scores_summary = "\n".join(f"- {c.criterion_id}: {c.score} — {c.rationale}" for c in criteria)
    return (
        f"{v.judge_pass2_hyp_lead}\n"
        f"{scores_summary}\n\n"
        f"{v.judge_pass2_hyp_task}\n\n"
        f"Agent's rationale for the final belief update:\n{agent_rationale}"
        f"{v.judge_pass2_format}"
    )


def _build_overall_prompt_pass1(
    goal: str, overall_items: list, metrics: dict, refuted_str: str, v: PromptVariant,
    hypothesis_table: str = "",
) -> str:
    n = metrics["hypothesis_count"]
    parts = [
        f"Task goal: {goal}\n",
        f"{v.judge_role}\n",
        "Run summary:",
        f"  Hypotheses formed: {n} (final candidates: {metrics.get('final_count', n)}, "
        f"weakened: {metrics.get('weakened_count', 0)}, refuted: {metrics.get('refuted_count', 0)})",
        f"  Belief revisions (all hypotheses): {metrics['belief_revision_count']}",
        f"  Refutation rate: {metrics['refutation_rate']:.0%}",
        f"  Weakened rate: {metrics.get('weakened_rate', 0.0):.0%}",
    ]
    if hypothesis_table:
        parts.append(f"\n{hypothesis_table}")
    if refuted_str:
        parts.append(f"\n{refuted_str}")
    parts.append(_EVIDENCE_WEIGHING)
    parts.append(v.judge_pass1_format)
    parts.append(_format_criteria_listing(overall_items))
    return "\n\n".join(parts)


def _build_overall_prompt_pass2(
    criteria: list["CriterionScore"], termination_rationale: str, confidence_summary: str,
    v: PromptVariant,
) -> str:
    scores_summary = "\n".join(f"- {c.criterion_id}: {c.score} — {c.rationale}" for c in criteria)
    return (
        f"{v.judge_pass2_overall_lead}\n"
        f"{scores_summary}\n\n"
        f"{v.judge_pass2_overall_task}\n\n"
        f"Agent's termination rationale:\n{termination_rationale}\n\n"
        f"Agent's confidence summary:\n{confidence_summary}"
        f"{v.judge_pass2_format}"
    )


def _parse_json_criteria(raw: str, rubric_items: list) -> list[CriterionScore]:
    """Parse pass-1's JSON criteria response, keeping only entries whose criterion_id
    matches a real rubric item (guards against a hallucinated or malformed id rather
    than silently accepting it). Returns [] on any parse failure.
    """
    if not raw or not rubric_items:
        return []
    valid_ids = {item.get("id", f"criterion_{i}") for i, item in enumerate(rubric_items)}
    try:
        parsed = extract_json_object(raw)
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
        parsed = extract_json_object(raw)
        return {
            n["criterion_id"]: str(n.get("agent_note", "")).strip()
            for n in parsed.get("agent_notes", [])
            if n.get("criterion_id")
        }
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError, KeyError):
        return {}


def _score_criteria_two_pass(
    client: LLMClient, prompt1: str, rubric_items: list,
    build_prompt2: "callable | None", raw_label: str, stats: dict | None = None,
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
    if stats is not None:
        stats["pass1_prompt_chars"] = len(prompt1)
        stats["pass2_prompt_chars"] = 0  # 0 = pass 2 skipped

    if build_prompt2 is not None and criteria:
        prompt2 = build_prompt2(criteria)
        if stats is not None:
            stats["pass2_prompt_chars"] = len(prompt2)
        pass2 = client.chat([{"role": "user", "content": prompt2}])
        notes = _parse_json_agent_notes(pass2)
        if notes:
            for c in criteria:
                if c.criterion_id in notes:
                    c.agent_note = notes[c.criterion_id]
        raw_display = f"{pass1}\n\n--- {raw_label} agreement notes ---\n\n{pass2}"

    return raw_display, criteria


def score_run(
    run_dir: str, client: LLMClient | None = None, prompt_variant: str = "original",
) -> RunScores:
    """Score a run. prompt_variant selects instruction wording (reviewer.prompt_variants).

    A run that ended without declare_done (e.g. at its turn cap) has no final report: its
    active and terminal ledger hypotheses are scored with no termination rationale, confidence
    summary, or suggested validation, and the result is marked completed=False.
    """
    v = get_variant(prompt_variant)
    ws = Workspace(run_dir)
    completed = ws.has_final_report()
    report = ws.read_final_report() or {}
    task_spec = ws.read_task_spec()

    if client is None:
        client = get_llm_client()

    ledger = ws.read_ledger()
    all_hypotheses = ledger.get("hypotheses", {})

    scored_hypotheses = _hypotheses_to_score(report, all_hypotheses)
    top_hypotheses = [h for h in scored_hypotheses if h.get("status") not in _DISCARDED_STATUSES]
    final_ids = {h.get("hypothesis_id") for h in top_hypotheses}

    validation = report.get("suggested_validation", "")
    termination_rationale = report.get("rationale", "")
    confidence_summary = report.get("confidence_summary", "")
    refuted_str = _format_refuted_hypotheses(all_hypotheses)
    scores = []

    if task_spec:
        per_items = [r for r in task_spec.get("rubric", []) if r.get("scope") == "per_hypothesis"]
        overall_items = [r for r in task_spec.get("rubric", []) if r.get("scope") == "overall"]
        goal = task_spec.get("goal", "")

        for h in scored_hypotheses:
            gene = h.get("candidate_gene", "unknown")
            cell_type = h.get("cell_type", "unknown")
            claim = h.get("claim", "")
            status = h.get("status", "")
            is_final = h.get("hypothesis_id") in final_ids
            evidence_str = format_evidence_list(h)
            hypothesis_str = f"{cell_type} / {gene}: {claim}"
            if not is_final:
                hypothesis_str += (
                    f"\nStatus at end of run: {status} (the agent did not keep this as a final candidate)"
                )
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
            input_stats = _hypothesis_input_stats(h, evidence_str, evidence_trail, agent_rationale)
            if per_items:
                prompt1 = _build_per_hypothesis_prompt_pass1(
                    goal, per_items, hypothesis_str, evidence_str,
                    validation, evidence_trail, open_questions_str, v,
                )
                build_prompt2 = (
                    (lambda crit: _build_per_hypothesis_prompt_pass2(crit, agent_rationale, v))
                    if agent_rationale else None
                )
                per_hyp_raw, criteria = _score_criteria_two_pass(
                    client, prompt1, per_items, build_prompt2, "per-hypothesis", input_stats,
                )
            scores.append(HypothesisScores(
                hypothesis_id=h.get("hypothesis_id", "unknown"),
                candidate_gene=gene,
                per_hypothesis_raw=per_hyp_raw,
                criteria=criteria,
                status=status,
                final_candidate=is_final,
                input_stats=input_stats,
            ))

        overall_raw = ""
        overall_criteria: list[CriterionScore] = []
        overall_input_stats: dict = {}
        if overall_items:
            metrics_summary = _run_metrics(all_hypotheses, scored_hypotheses)
            hypothesis_table = _format_hypothesis_table(scored_hypotheses, final_ids)
            prompt1 = _build_overall_prompt_pass1(
                goal, overall_items, metrics_summary, refuted_str, v,
                hypothesis_table=hypothesis_table,
            )
            overall_input_stats = {
                "hypotheses_listed": len(scored_hypotheses),
                "hypothesis_table_chars": len(hypothesis_table),
                "rationale_chars": len(termination_rationale or "") + len(confidence_summary or ""),
            }
            build_prompt2 = (
                (lambda crit: _build_overall_prompt_pass2(crit, termination_rationale, confidence_summary, v))
                if (termination_rationale or confidence_summary) else None
            )
            overall_raw, overall_criteria = _score_criteria_two_pass(
                client, prompt1, overall_items, build_prompt2, "overall", overall_input_stats,
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
        overall_input_stats = {}

    return RunScores(
        hypothesis_scores=scores,
        overall_raw=overall_raw,
        run_dir=run_dir,
        overall_criteria=overall_criteria,
        completed=completed,
        overall_input_stats=overall_input_stats,
    )


# Shared with reviewer.variability so repeat experiments use the production judge settings.
JUDGE_BACKEND_MODELS = {"anthropic": "claude-haiku-4-5", "google": "gemini-2.5-flash"}

# JSON output (criterion_id/score/rationale repeated per entry, quoted and
# escaped) runs noticeably longer than the old free-text format did for the
# same content -- bump past the client's 4096 default so a detailed multi-
# criterion response doesn't get truncated mid-JSON.
JUDGE_MAX_TOKENS = 8192


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

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            backends.append(("anthropic", LLMClient(
                backend="anthropic", model=JUDGE_BACKEND_MODELS["anthropic"],
                max_tokens=JUDGE_MAX_TOKENS,
            )))
        except Exception as exc:
            print(f"[score] skipping anthropic backend: {exc}", file=sys.stderr)

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            backends.append(("google", LLMClient(
                backend="google", model=JUDGE_BACKEND_MODELS["google"],
                max_tokens=JUDGE_MAX_TOKENS,
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
