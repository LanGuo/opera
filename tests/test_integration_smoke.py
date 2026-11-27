"""Integration smoke test: the hypothesis-ledger server over the MCP stdio protocol.

Starts `python -m hypothesis_ledger.server` as a subprocess, calls every writing tool through an
MCP client, and checks what lands in ledger.jsonl.
"""
import asyncio
import json
import os
import sys

import pytest

from hypothesis_ledger.ledger import Ledger

pytestmark = pytest.mark.integration


async def _session_calls(run_dir: str):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "hypothesis_ledger.server"],
                                   env={**os.environ, "EVAL_RUN_DIR": run_dir, "ALLOW_HUMAN_INPUT": "false"})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}

            async def call(name, **args):
                r = await session.call_tool(name, args)
                return json.loads(r.content[0].text)

            out = {"tools": names}
            out["write"] = await call(
                "write_hypothesis", hypothesis="SIGLEC1 is raised on SLE monocytes",
                attributes={"gene": "SIGLEC1", "cell_type": "monocytes"}, confidence=0.6,
                evidence=[{"finding": "lfc 2.1, padj 1e-8", "source": {"tool": "Bash", "match": "deg"},
                           "relation": "supports"}],
                rationale="Large effect in the starting cohort")
            out["update"] = await call(
                "update_hypothesis", hypothesis_id="H1", confidence=0.75, status="supported",
                evidence=[{"ref": "E1", "relation": "supports"},
                          {"finding": "IFN signature GWAS locus", "source": {"citation": "PMID 1"},
                           "relation": "supports"}],
                rationale="Second modality")
            out["note"] = await call("log_note", kind="interpretation", text="IFN-driven", hypothesis_ids=["H1"])
            out["human"] = await call("request_human_input", question="Valid?", context="ctx")
            out["read"] = await call("read_hypotheses", status_filter="all")
            out["done"] = await call("declare_done", ranked_hypothesis_ids=["H1"], rationale="converged",
                                     confidence_summary="0.75", validation={"H1": "flow cytometry"})
            return out


def test_ledger_server_over_stdio(tmp_path):
    Ledger(tmp_path).append([{"kind": "run_info", "phase": "start", "task_rules": {
        "hypothesis_attributes": [{"name": "gene"}, {"name": "cell_type"}],
        "hypothesis_constraints": [{"unique": ["gene"]}]}}], "harness")
    out = asyncio.run(_session_calls(str(tmp_path)))
    assert "log_analysis_step" not in out["tools"] and "log_note" in out["tools"]
    assert out["write"]["status"] == "recorded" and out["write"]["hypothesis_id"] == "H1"
    assert out["update"]["status"] == "recorded"
    assert out["note"]["status"] == "recorded" and out["human"]["status"] == "headless_mode"
    assert out["read"]["hypotheses"][0]["status"] == "supported"
    assert out["done"]["status"] == "completed" and out["done"]["warnings"] == []
    kinds = [e["kind"] for e in Ledger(tmp_path).entries()]
    assert kinds == ["run_info", "hypothesis", "hypothesis", "note", "human_input", "conclusion"]
    for f in ("ledger.json", "step_log.jsonl", "run_log.jsonl", "final_report.json"):
        assert not (tmp_path / f).exists()
