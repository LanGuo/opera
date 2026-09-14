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
    agent_note: str     # pass-2 agreement/divergence note; empty string if no pass 2


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
            ev_type = ev.get("type", "unknown")
            description = ev.get("description", "")
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
        ev_summary = "; ".join(
            e.get("description", "")[:120] for e in evidence if e.get("description")
        )
        lines.append(f"  - {cell_type}/{gene}: evidence found — {ev_summary or 'none'}")
    return "\n".join(lines)


def _build_per_hypothesis_prompt_pass1(
    goal: str, per_items: list, hypothesis_str: str, evidence_str: str,
    validation: str, evidence_trail: str, open_questions_str: str,
) -> str:
    criteria = "\n\n".join(
        f"{i + 1}. {item.get('label', item.get('id', 'unknown'))}\n   {item.get('guidance', '')}"
        for i, item in enumerate(per_items)
    )
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
    parts.append(
        '\nFor each criterion below write 2–3 sentences then end with '
        '"Score: X/5" or "Score: yes/no/partial".\n'
    )
    parts.append(criteria)
    return "\n\n".join(parts)


def _build_per_hypothesis_prompt_pass2(
    pass1_response: str, agent_rationale: str,
) -> str:
    return (
        "You just scored this hypothesis based on observed evidence. "
        "Now review the agent's own rationale for the same hypothesis and note where your "
        "independent assessment agrees or diverges. Keep your scores — only add a brief "
        '"Agent self-assessment note:" paragraph at the end.\n\n'
        f"Your prior scoring:\n{pass1_response}\n\n"
        f"Agent's rationale for the final belief update:\n{agent_rationale}"
    )


def _build_overall_prompt_pass1(
    goal: str, overall_items: list, metrics: dict, refuted_str: str,
) -> str:
    criteria = "\n\n".join(
        f"{i + 1}. {item.get('label', item.get('id', 'unknown'))}\n   {item.get('guidance', '')}"
        for i, item in enumerate(overall_items)
    )
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
    parts.append(
        '\nFor each criterion below write 2–3 sentences then end with '
        '"Score: X/5" or "Score: yes/no/partial".\n'
    )
    parts.append(criteria)
    return "\n\n".join(parts)


def _build_overall_prompt_pass2(
    pass1_response: str, termination_rationale: str, confidence_summary: str,
) -> str:
    return (
        "You just scored the overall run quality based on observed metrics. "
        "Now review the agent's own termination rationale and confidence summary, and note "
        "where your independent assessment agrees or diverges. Keep your scores — only add a brief "
        '"Agent self-assessment note:" paragraph at the end.\n\n'
        f"Your prior scoring:\n{pass1_response}\n\n"
        f"Agent's termination rationale:\n{termination_rationale}\n\n"
        f"Agent's confidence summary:\n{confidence_summary}"
    )


def _parse_criterion_scores(raw: str, rubric_items: list) -> list[CriterionScore]:
    """Parse per-criterion scores from LLM free-text output.

    Splits raw text on double-newlines or numbered headings to find per-criterion
    blocks, then extracts Score lines and agent notes. Returns empty list only if
    raw is empty or no blocks were found at all; partial results use score="".
    """
    if not raw or not rubric_items:
        return []

    # Markdown formatting the model may wrap around structural markers (bold, headers).
    # Tolerated on both sides of "Score:" / "Agent self-assessment note:" / numbered
    # headings, since judge models inconsistently emit "**Score: 4/5**", "## 1. Novelty",
    # "**Score:** 4/5", etc. instead of the plain text the prompt asked for.
    # Deliberately excludes \n -- matching across newlines here would bridge the blank
    # line between blocks and corrupt the block-split boundaries.
    _MD = r'[#*]{0,4}[ \t]{0,3}'

    try:
        # Split on numbered headings (e.g. "1. " or "2. ", optionally markdown-wrapped)
        # or double newlines. Try numbered heading split first.
        blocks = re.split(r'\n(?=' + _MD + r'\d+\.\s)', raw.strip())
        if len(blocks) < 2:
            # Fall back to double-newline split
            blocks = [b.strip() for b in re.split(r'\n\n+', raw.strip()) if b.strip()]
        else:
            # Fix 1: drop preamble block if block 0 doesn't look like a criterion block
            # (i.e. doesn't start with a digit and has no Score: line)
            if blocks and not re.match(r'^' + _MD + r'\d+\.', blocks[0].strip()) and not re.search(
                r'(?:^|\n)' + _MD + r'Score:', blocks[0], re.IGNORECASE | re.MULTILINE
            ):
                blocks = blocks[1:]

        if not blocks:
            return []

        # Match blocks to rubric items by position
        result = []
        for i, item in enumerate(rubric_items):
            if i >= len(blocks):
                break
            block = blocks[i].strip()

            # Extract agent note if present
            agent_note = ""
            note_match = re.search(
                _MD + r'Agent self-assessment note:' + _MD + r'(.+?)(?:\n|$)',
                block, re.IGNORECASE | re.DOTALL,
            )
            if note_match:
                agent_note = note_match.group(1).strip(' *#')
                block = block[:note_match.start()].strip()

            # Fix 3: anchor Score regex to line start to avoid matching "Novelty Score:" mid-rationale
            score_match = re.search(
                r'(?:^|\n)' + _MD + r'Score:' + _MD + r'([^\n]+)', block, re.IGNORECASE | re.MULTILINE
            )
            # Fix 2: emit partial result instead of returning [] on missing Score line
            if not score_match:
                result.append(CriterionScore(
                    criterion_id=item.get("id", f"criterion_{i}"),
                    rationale=block.strip(),
                    score="",
                    agent_note=agent_note,
                ))
                continue

            score_val = score_match.group(1).strip().strip(' *#')
            rationale = block[:score_match.start()].strip()
            # Remove leading numbered heading (optionally markdown-wrapped) from rationale
            rationale = re.sub(r'^' + _MD + r'\d+\.\s+[^\n]*\n', '', rationale).strip()

            result.append(CriterionScore(
                criterion_id=item.get("id", f"criterion_{i}"),
                rationale=rationale,
                score=score_val,
                agent_note=agent_note,
            ))

        return result
    except Exception:
        return []


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
            if per_items:
                prompt1 = _build_per_hypothesis_prompt_pass1(
                    goal, per_items, hypothesis_str, evidence_str,
                    validation, evidence_trail, open_questions_str,
                )
                pass1 = client.chat([{"role": "user", "content": prompt1}])

                if agent_rationale:
                    prompt2 = _build_per_hypothesis_prompt_pass2(pass1, agent_rationale)
                    per_hyp_raw = client.chat([
                        {"role": "user", "content": prompt1},
                        {"role": "assistant", "content": pass1},
                        {"role": "user", "content": prompt2},
                    ])
                else:
                    per_hyp_raw = pass1

            criteria = _parse_criterion_scores(per_hyp_raw, per_items)
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
            pass1 = client.chat([{"role": "user", "content": prompt1}])

            if termination_rationale or confidence_summary:
                prompt2 = _build_overall_prompt_pass2(pass1, termination_rationale, confidence_summary)
                overall_raw = client.chat([
                    {"role": "user", "content": prompt1},
                    {"role": "assistant", "content": pass1},
                    {"role": "user", "content": prompt2},
                ])
            else:
                overall_raw = pass1
            overall_criteria = _parse_criterion_scores(overall_raw, overall_items)

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

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            backends.append(("anthropic", LLMClient(backend="anthropic", model="claude-haiku-4-5")))
        except Exception as exc:
            print(f"[score] skipping anthropic backend: {exc}", file=sys.stderr)

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            backends.append(("google", LLMClient(backend="google", model="gemini-2.5-flash")))
        except Exception as exc:
            print(f"[score] skipping google backend: {exc}", file=sys.stderr)

    ollama_model = os.environ.get("OLLAMA_MODEL")
    if ollama_model:
        try:
            backends.append((f"ollama:{ollama_model}", LLMClient(backend="ollama", model=ollama_model)))
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
