import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .workspace import Workspace

mcp = FastMCP("hypothesis-ledger")
_workspace: Optional[Workspace] = None


def _get_ws() -> Workspace:
    global _workspace
    if _workspace is None:
        run_dir = os.environ.get("EVAL_RUN_DIR", "/workspace/runs/run_001")
        _workspace = Workspace(run_dir)
    return _workspace


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Tool implementations (also exposed as _tool_* for direct unit testing) ---

def _tool_write_hypothesis(
    cell_type: str,
    candidate_gene: str,
    claim: str,
    evidence: list,
    confidence: float,
    open_questions: list,
) -> str:
    ws = _get_ws()
    h_id = uuid.uuid4().hex[:6].upper()
    entry = {
        "hypothesis_id": h_id,
        "cell_type": cell_type,
        "candidate_gene": candidate_gene,
        "claim": claim,
        "evidence": evidence,
        "confidence": confidence,
        "open_questions": open_questions,
        "status": "active",
        "update_history": [],
        "created_at": _now(),
    }
    count = ws.increment_tool_call_count()

    def _add(data):
        data["hypotheses"][h_id] = entry

    ws.update_ledger(_add)
    ws.log_run({"timestamp": _now(), "tool": "write_hypothesis",
                "inputs": {"cell_type": cell_type, "candidate_gene": candidate_gene},
                "result": {"hypothesis_id": h_id}, "tool_call_count": count})
    return json.dumps({"hypothesis_id": h_id, "status": "recorded"})


def _tool_read_hypotheses(status_filter: str = "active") -> str:
    ws = _get_ws()
    data = ws.read_ledger()
    hyps = data["hypotheses"]
    if status_filter != "all":
        hyps = {k: v for k, v in hyps.items() if v["status"] == status_filter}
    count = ws.increment_tool_call_count()
    ws.log_run({"timestamp": _now(), "tool": "read_hypotheses",
                "status_filter": status_filter, "tool_call_count": count})
    return json.dumps({
        "hypotheses": hyps,
        "total_count": len(data["hypotheses"]),
        "filtered_count": len(hyps),
        "tool_call_count": count,
    }, indent=2)


def _tool_update_hypothesis(hypothesis_id: str, updates: dict, rationale: str) -> str:
    ws = _get_ws()

    def _update(data):
        if hypothesis_id not in data["hypotheses"]:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        h = data["hypotheses"][hypothesis_id]
        old_values = {k: h.get(k) for k in updates}
        h.update(updates)
        h["update_history"].append({
            "timestamp": _now(),
            "rationale": rationale,
            "changes": {"from": old_values, "to": updates},
        })

    ws.update_ledger(_update)
    count = ws.increment_tool_call_count()
    ws.log_run({"timestamp": _now(), "tool": "update_hypothesis",
                "inputs": {"hypothesis_id": hypothesis_id, "updates": updates},
                "result": {"status": "updated"}, "tool_call_count": count})
    return json.dumps({"hypothesis_id": hypothesis_id, "status": "updated"})


def _tool_log_analysis_step(
    step_type: str,
    tool_used: str,
    inputs_summary: str,
    outputs_summary: str,
    interpretation: str,
    next_action: str,
) -> str:
    ws = _get_ws()
    count = ws.increment_tool_call_count()
    entry = {
        "timestamp": _now(), "step_type": step_type, "tool_used": tool_used,
        "inputs_summary": inputs_summary, "outputs_summary": outputs_summary,
        "interpretation": interpretation, "next_action": next_action,
        "tool_call_count": count,
    }
    ws.log_step(entry)
    ws.log_run({"timestamp": _now(), "tool": "log_analysis_step",
                "inputs": {"step_type": step_type, "tool_used": tool_used},
                "result": {"logged": True}, "tool_call_count": count})
    return json.dumps({"status": "logged", "step_count": count})


def _tool_declare_done(
    top_hypotheses: list,
    rationale: str,
    confidence_summary: str,
    suggested_validation: str,
) -> str:
    ws = _get_ws()
    data = ws.read_ledger()

    # If the agent omitted hypothesis IDs, derive them deterministically: all
    # non-refuted hypotheses sorted by confidence descending.
    resolved_ids = list(top_hypotheses) if top_hypotheses else []
    if not resolved_ids:
        resolved_ids = sorted(
            [h_id for h_id, h in data["hypotheses"].items()
             if h.get("status") not in ("refuted",)],
            key=lambda h_id: data["hypotheses"][h_id].get("confidence", 0),
            reverse=True,
        )

    def _mark_terminal(d):
        for h_id in resolved_ids:
            if h_id in d["hypotheses"]:
                d["hypotheses"][h_id]["status"] = "terminal"

    ws.update_ledger(_mark_terminal)
    data = ws.read_ledger()
    terminal = [data["hypotheses"][h] for h in resolved_ids if h in data["hypotheses"]]
    report = {
        "completed_at": _now(),
        "top_hypotheses": terminal,
        "rationale": rationale,
        "confidence_summary": confidence_summary,
        "suggested_validation": suggested_validation,
        "total_tool_calls": data.get("tool_call_count", 0),
        "total_hypotheses_explored": len(data["hypotheses"]),
    }
    ws.write_final_report(report)
    count = ws.increment_tool_call_count()
    ws.log_run({"timestamp": _now(), "tool": "declare_done",
                "inputs": {"top_hypotheses": resolved_ids},
                "result": {"report_written": True}, "tool_call_count": count})
    return json.dumps({
        "status": "completed",
        "report_path": str(ws.final_report_path),
        "hypothesis_count": len(terminal),
    })


def _tool_request_human_input(question: str, context: str) -> str:
    ws = _get_ws()
    human_enabled = os.environ.get("ALLOW_HUMAN_INPUT", "false").lower() == "true"
    ws.log_run({"timestamp": _now(), "tool": "request_human_input",
                "inputs": {"question": question, "context": context},
                "result": {"mode": "human" if human_enabled else "headless"}})
    if not human_enabled:
        return json.dumps({
            "status": "headless_mode",
            "message": "Human input requested but running headlessly. Question logged.",
            "question": question,
        })
    print(f"\n[HUMAN INPUT]\nContext: {context}\nQuestion: {question}\nAnswer: ", flush=True)
    return json.dumps({"status": "received", "answer": input()})


# --- FastMCP tool registrations ---

@mcp.tool()
def write_hypothesis(
    cell_type: str,
    candidate_gene: str,
    claim: str,
    evidence: list,
    confidence: float,
    open_questions: list,
) -> str:
    """Record a new candidate hypothesis with structured evidence in the ledger."""
    return _tool_write_hypothesis(cell_type, candidate_gene, claim, evidence, confidence, open_questions)


@mcp.tool()
def read_hypotheses(status_filter: str = "active") -> str:
    """Return ledger entries filtered by status ('active', 'weakened', 'refuted', 'terminal', 'all')."""
    return _tool_read_hypotheses(status_filter)


@mcp.tool()
def update_hypothesis(hypothesis_id: str, updates: dict, rationale: str) -> str:
    """Revise confidence, status, or evidence for a hypothesis. Rationale is required and logged."""
    return _tool_update_hypothesis(hypothesis_id, updates, rationale)


@mcp.tool()
def log_analysis_step(
    step_type: str,
    tool_used: str,
    inputs_summary: str,
    outputs_summary: str,
    interpretation: str,
    next_action: str,
) -> str:
    """Record a structured analysis step. Call after every non-trivial tool result."""
    return _tool_log_analysis_step(step_type, tool_used, inputs_summary,
                                   outputs_summary, interpretation, next_action)


@mcp.tool()
def declare_done(
    top_hypotheses: list,
    rationale: str,
    confidence_summary: str,
    suggested_validation: str,
) -> str:
    """Signal task complete. Writes final_report.json and marks top hypotheses as terminal."""
    return _tool_declare_done(top_hypotheses, rationale, confidence_summary, suggested_validation)


@mcp.tool()
def request_human_input(question: str, context: str) -> str:
    """Request human expert judgment. In headless mode (default), logs the question and returns immediately."""
    return _tool_request_human_input(question, context)


def run():
    mcp.run()


if __name__ == "__main__":
    run()
