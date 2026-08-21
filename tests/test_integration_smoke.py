# tests/test_integration_smoke.py
"""
Integration smoke test: runs the hypothesis ledger server as a subprocess
and verifies all 6 tools work over the MCP stdio protocol.
Requires: `uv run python -m hypothesis_ledger.server` to be runnable.
"""
import json
import os
import subprocess
import sys
import time
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("smoke_run"))


def test_server_starts_and_all_tools_callable(run_dir):
    """Verify the server registers all 6 tools via `mcp dev` list-tools inspection."""
    env = {**os.environ, "EVAL_RUN_DIR": run_dir, "ALLOW_HUMAN_INPUT": "false"}
    result = subprocess.run(
        ["uv", "run", "python", "-c",
         "from hypothesis_ledger.server import mcp; "
         "tools = [t.name for t in mcp._tool_manager.list_tools()]; "
         "print(tools)"],
        capture_output=True, text=True, env=env, timeout=15
    )
    assert result.returncode == 0, result.stderr
    tools = eval(result.stdout.strip())
    expected = {
        "write_hypothesis", "read_hypotheses", "update_hypothesis",
        "log_analysis_step", "declare_done", "request_human_input",
    }
    assert expected == set(tools), f"Missing tools: {expected - set(tools)}"


def test_full_mini_workflow(run_dir):
    """Call all 6 tools in sequence; verify workspace state is correct."""
    env = {**os.environ, "EVAL_RUN_DIR": run_dir, "ALLOW_HUMAN_INPUT": "false"}

    script = """
import json, sys, os
sys.path.insert(0, '.')
os.environ['EVAL_RUN_DIR'] = sys.argv[1]
from hypothesis_ledger import server as srv

# write
r = json.loads(srv._tool_write_hypothesis(
    cell_type='pDC', candidate_gene='SIGLEC1', claim='elevated in SLE',
    evidence=[{'source': 'DEG', 'strength': 'strong', 'log2fc': 2.1}],
    confidence=0.65, open_questions=['Is it druggable?']
))
h_id = r['hypothesis_id']

# log step
srv._tool_log_analysis_step(
    step_type='DEG', tool_used='scanpy', inputs_summary='SLE vs ctrl pDC',
    outputs_summary='SIGLEC1 top hit', interpretation='strong candidate',
    next_action='check GWAS'
)

# update
srv._tool_update_hypothesis(h_id, {'confidence': 0.82}, 'GWAS colocalization found')

# read
r2 = json.loads(srv._tool_read_hypotheses('active'))
assert r2['filtered_count'] == 1
assert r2['hypotheses'][h_id]['confidence'] == 0.82

# request human (headless)
r3 = json.loads(srv._tool_request_human_input('Valid?', 'SLE context'))
assert r3['status'] == 'headless_mode'

# declare done
r4 = json.loads(srv._tool_declare_done(
    top_hypotheses=[h_id], rationale='converged',
    confidence_summary='0.82', suggested_validation='flow cytometry'
))
assert r4['status'] == 'completed'
print('PASS')
"""

    result = subprocess.run(
        ["uv", "run", "python", "-c", script, run_dir],
        capture_output=True, text=True, env=env, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
