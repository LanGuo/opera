"""The ToolUniverse wrapper, checked in the pinned ToolUniverse environment.

Needs `uvx` and network access to install tooluniverse; the MCP test also calls live
biomedical APIs through BiomarkerDiscoveryWorkflow.
"""
import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

from evaluator.harness import _mcp_servers
from evaluator.tooluniverse_server import TOOLUNIVERSE_VERSION

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "evaluator" / "tooluniverse_server.py"
LONG_NAME = "HPA_get_comprehensive_gene_details_by_ensembl_id"

PATCH_CHECK = r"""
import contextlib, importlib.util, io, json, sys
spec = importlib.util.spec_from_file_location("wrapper", sys.argv[1])
wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
wrapper.patch_compose_tool()
from tooluniverse import ToolUniverse
from tooluniverse.compose_tool import ComposeTool
with contextlib.redirect_stdout(io.StringIO()):
    tu = ToolUniverse(enable_name_shortening=True)
    tu.load_tools(categories=["hpa", "compose", "pubmed"])
    before = set(tu.all_tool_dict)
    wf = ComposeTool(tu.all_tool_dict["BiomarkerDiscoveryWorkflow"], tooluniverse=tu)
    shortened = wf._load_missing_dependencies({sys.argv[2]})
    after_shortened = set(tu.all_tool_dict)
    unloaded = wf._load_missing_dependencies({"UniProt_get_entry_by_accession"})
print(json.dumps({
    "long_name_registered": sys.argv[2] in before,
    "shortened": [sorted(s) for s in shortened],
    "registry_unchanged": after_shortened == before,
    "unloaded": [sorted(s) for s in unloaded],
    "kept_after_load": before <= set(tu.all_tool_dict),
    "uniprot_added": "UniProt_get_entry_by_accession" in tu.all_tool_dict,
    "load_tools_restored": "load_tools" not in vars(tu),
}))
"""


def test_patch_finds_shortened_dependencies_and_keeps_the_registry():
    r = subprocess.run(
        ["uvx", "--from", f"tooluniverse=={TOOLUNIVERSE_VERSION}", "python", "-P", "-c",
         PATCH_CHECK, str(WRAPPER), LONG_NAME],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {
        "long_name_registered": False,  # registered in shortened form: the case being fixed
        "shortened": [[LONG_NAME], []],
        "registry_unchanged": True,
        "unloaded": [["UniProt_get_entry_by_accession"], []],
        "kept_after_load": True,
        "uniprot_added": True,
        "load_tools_restored": True,
    }


def test_wrapper_refuses_another_tooluniverse_version():
    r = subprocess.run(["uvx", "--from", "tooluniverse==1.5.7", "python", "-P", str(WRAPPER)],
                       capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    assert r.returncode != 0 and "checked against" in r.stderr


async def _call_workflow_then_execute(server):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=server["command"], args=server["args"],
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        await session.call_tool("execute_tool", {
            "tool_name": "BiomarkerDiscoveryWorkflow",
            "arguments": {"disease_condition": "systemic lupus erythematosus", "sample_type": "blood"}})
        after = await session.call_tool("execute_tool", {
            "tool_name": "UniProt_get_function_by_accession", "arguments": {"accession": "P05231"}})
        return "".join(getattr(c, "text", "") for c in after.content)


def test_execute_tool_still_works_after_the_workflow_over_mcp(tmp_path):
    """The failure as agents saw it: after the workflow, execute_tool itself was gone."""
    server = _mcp_servers(REPO, tmp_path, {}, trust=False)["tooluniverse"]
    text = asyncio.run(asyncio.wait_for(_call_workflow_then_execute(server), timeout=900))
    assert "not found even after loading tools" not in text
    assert "interleukin" in text.lower() or "IL6" in text or "cytokine" in text.lower(), text[:500]
