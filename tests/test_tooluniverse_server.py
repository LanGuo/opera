"""The pinned ToolUniverse server, checked over MCP as the agents use it.

Needs `uvx` and network access to install tooluniverse; the MCP test also calls live
biomedical APIs through BiomarkerDiscoveryWorkflow.
"""
import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from evaluator.harness import _mcp_servers

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "evaluator" / "tooluniverse_server.py"


def test_launcher_refuses_another_tooluniverse_version():
    r = subprocess.run(["uvx", "--from", "tooluniverse==1.5.8", "python", "-P", str(LAUNCHER)],
                       capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "OPERA pins" in r.stderr


async def _call_workflow_then_execute(server):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=server["command"], args=server["args"],
                                   env={**os.environ, **server["env"]})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        await session.call_tool("execute_tool", {
            "tool_name": "BiomarkerDiscoveryWorkflow",
            "arguments": {"disease_condition": "systemic lupus erythematosus", "sample_type": "blood"}})
        after = await session.call_tool("execute_tool", {
            "tool_name": "UniProt_get_function_by_accession", "arguments": {"accession": "P05231"}})
        return "".join(getattr(c, "text", "") for c in after.content)


def test_execute_tool_still_works_after_the_workflow_over_mcp(tmp_path):
    """The 1.5.8 failure as agents saw it (fixed upstream in 1.5.9): after the workflow,
    execute_tool itself was gone."""
    server = _mcp_servers(REPO, tmp_path, {}, trust=False)["tooluniverse"]
    text = asyncio.run(asyncio.wait_for(_call_workflow_then_execute(server), timeout=900))
    assert "not found even after loading tools" not in text
    assert "interleukin" in text.lower() or "IL6" in text or "cytokine" in text.lower(), text[:500]


async def _census_versions(server):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=server["command"], args=server["args"],
                                   env={**os.environ, **server["env"]})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        r = await session.call_tool("execute_tool", {
            "tool_name": "CELLxGENE_get_census_versions", "arguments": {}})
        return "".join(getattr(c, "text", "") for c in r.content)


def test_census_tools_run_in_the_tooluniverse_environment(tmp_path):
    server = _mcp_servers(REPO, tmp_path, {}, trust=False)["tooluniverse"]
    text = asyncio.run(asyncio.wait_for(_census_versions(server), timeout=900))
    assert "2025-11-08" in text, text[:800]
