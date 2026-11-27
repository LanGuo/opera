"""A synthetic ledger.jsonl run for judge tests (no agent-run data).

Two variants: one that ends with declare_done (validation plans for two of its three hypotheses)
and one that never concludes. Between them they cover every input the judge reads: tool calls
with a long output (cut at 2,000 characters), a failed call, a citation, a ref, a withdrawn item,
contradicting evidence followed by a lower confidence and a status change, an interpretation note,
derived_from, a visible and a thinking message, an artifact.
"""
import json
from pathlib import Path

from hypothesis_ledger.ledger import Ledger

TASK = Path(__file__).resolve().parent.parent / "tasks" / "sle_biomarker_discovery" / "task_spec.yaml"


def _call(n, tool, args, output, error=None, via="execute_tool"):
    return {"kind": "tool_call", "tool_call_id": f"T{n}", "cli_call_id": f"c{n}", "tool": tool,
            "via": via, "server": "tooluniverse" if via else None, "arguments": args,
            "output": output, "error": error, "started": None, "finished": None}


def _ev(eid, finding, relation, link, calls=(), citation=None, quantities=None):
    source = {"citation": citation} if citation else {"tool": "execute_tool", "match": finding[:20]}
    item = {"evidence_id": eid, "finding": finding, "source": source, "relation": relation,
            "link": link, "tool_call_ids": list(calls), "link_signals": []}
    if quantities:
        item["quantities"] = quantities
    return item


def _hyp(hid, action, text, attrs, conf, status, evidence, rationale, withdrawn=(), derived=None):
    return {"kind": "hypothesis", "hypothesis_id": hid, "action": action, "hypothesis": text,
            "attributes": attrs, "derived_from": derived, "confidence": conf, "status": status,
            "evidence": evidence, "evidence_withdrawn": list(withdrawn), "rationale": rationale,
            "open_questions": ["Is it cell-type specific?"]}


def write_synthetic_run(run_dir, concluded: bool = True) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    led = Ledger(run_dir)
    table = "\n".join(f"GENE{i}\tlog2FC {i / 10:.3f}\tp {i}e-5" for i in range(400))   # > 2,000 chars
    led.append([{"kind": "run_info", "phase": "start", "profile": "matched", "harness": "claude",
                 "cli_version": "test", "model_requested": "agent-model", "turn_cap": "--max-turns 30",
                 "task_text": "Find SLE biomarkers.", "task_rules": {}}], "harness")
    led.append([
        {"kind": "agent_message", "visibility": "thinking", "text": "private thoughts"},
        {"kind": "agent_message", "visibility": "visible", "text": "Starting the analysis."},
        _call(1, "DEG_table", {"tool_name": "DEG_table", "arguments": {"cell": "pDC"}},
              "IFI44L pDC log2FC 2.47 padj 3.1e-12\n" + table),
        _call(2, "GWAS_x", {"tool_name": "GWAS_x", "arguments": {"gene": "STAT4"}},
              "rs7574865 STAT4 p 1e-40 OR 1.57"),
        _call(3, "Bash", {"command": "python broken.py"}, "Traceback ... ModuleNotFoundError",
              error="tool reported an error", via=None),
        _call(4, "DEG_table", {"tool_name": "DEG_table", "arguments": {"cell": "B"}},
              "ZBP1 B cell log2FC 0.05 padj 0.81"),
        _call(5, "Notes_tool", {"tool_name": "Notes_tool"}, "an uncited call"),
    ], "harness")
    led.append([
        _hyp("H1", "create", "IFI44L is raised in SLE pDCs.", {"gene": "IFI44L", "cell_type": "pDC"}, 0.6,
             "active", [_ev("E1", "IFI44L log2FC 2.47 in pDC (padj 3.1e-12)", "supports", "exact", ["T1"],
                            quantities=[{"name": "padj", "value": 3.1e-12}])],
             "The DEG table shows a large IFI44L increase in pDCs. That makes it a strong pDC marker."),
        _hyp("H2", "create", "STAT4 drives SLE T-cell activation.", {"gene": "STAT4", "cell_type": "CD4 T"},
             0.5, "active", [_ev("E2", "rs7574865 at STAT4, p 1e-40, OR 1.57", "supports", "exact", ["T2"]),
                             _ev("E3", "STAT4 risk allele raises IL-12 signalling", "supports", "citation",
                                 citation="PMID 1")],
             "STAT4 is a replicated GWAS locus. Genetics alone implies it is causal in T cells."),
        _hyp("H3", "create", "ZBP1 is raised in SLE B cells.", {"gene": "ZBP1", "cell_type": "B cell"}, 0.55,
             "active", [_ev("E4", "ZBP1 reported in SLE plasmablasts", "supports", "citation", citation="PMID 2")],
             "Literature links ZBP1 to plasmablasts."),
        {"kind": "note", "note_kind": "interpretation", "text": "H1 fits an interferon signature.",
         "hypothesis_ids": ["H1"]},
        {"kind": "note", "note_kind": "plan", "text": "Check B cells next.", "hypothesis_ids": []},
        _hyp("H1", "update", "IFI44L is raised in SLE pDCs.", {"gene": "IFI44L", "cell_type": "pDC"}, 0.75,
             "supported", [{"ref": "E1", "relation": "supports"}],
             "Re-checked the table: the pDC effect holds, so confidence rises."),
        _hyp("H3", "update", "ZBP1 is raised in SLE B cells.", {"gene": "ZBP1", "cell_type": "B cell"}, 0.3,
             "weakened", [_ev("E5", "ZBP1 B cell log2FC 0.05, padj 0.81", "contradicts", "exact", ["T4"])],
             "Our own DEG shows no ZBP1 change in B cells, which contradicts the literature claim.",
             withdrawn=["E4"]),
        _hyp("H4", "create", "ZBP1 is raised in SLE plasmablasts only.",
             {"gene": "ZBP1-pb", "cell_type": "plasmablast"}, 0.4, "active",
             [{"ref": "E4", "relation": "supports"}], "Narrowing H3 to plasmablasts.", derived="H3"),
    ], "agent")
    if concluded:
        led.append([{"kind": "conclusion", "ranked_hypothesis_ids": ["H1", "H2"],
                     "rationale": "H1 has the strongest quantified support. H2 rests on genetics only.",
                     "confidence_summary": "H1 0.75, H2 0.5",
                     "validation": {"H1": "Sort pDCs from 20 SLE and 20 controls; qPCR for IFI44L.",
                                    "H2": "Phospho-flow for pSTAT4 by genotype."}}], "agent")
    led.append([{"kind": "artifact", "path": "deg.py", "bytes": 20, "content": "print('deg')\n"},
                {"kind": "run_info", "phase": "end", "outcome": {"model_responses": 20, "cap_hit": not concluded}}],
               "harness")
    (run_dir / "task_spec.yaml").write_text(TASK.read_text(encoding="utf-8"), encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    import sys
    write_synthetic_run(sys.argv[1], concluded=len(sys.argv) < 3 or sys.argv[2] != "unconcluded")
    print(json.dumps({"written": sys.argv[1]}))
