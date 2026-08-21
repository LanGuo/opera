# BioEval: Protein Synergy Discovery

## Task
You are a computational biologist and data scientist. Your goal:

Investigate the **synergistic (non-additive) effects** of double mutations in a 
protein fitness landscape. Determine if synergy is primarily driven by:
1. **Proximity to the active site** (functional constraint).
2. **Physical distance** between the two mutation sites (structural constraint).

**Starting data:** You must discover a relevant dataset (e.g., Protein GB1 fitness 
landscape, or similar deep mutational scanning data) using available tools.

## Required Behavior (The "Dos")

1. **The Interview:** Before performing any analysis, call `request_human_input` 
   with at least two probing questions about the data schema or biological 
   context to "load the context into your head." Even in headless mode, this 
   logs your intent.

2. **The Journal:** Maintain a `journal.md` in your workspace. Every time you 
   perform EDA or "stare at the data," record your observations, hypotheses, 
   and changing beliefs. Use `log_analysis_step` to summarize these entries.

3. **Three Variations:** When visualizing synergy or distance correlations, 
   propose and implement at least **three distinct approaches** (e.g., 3D 
   structure mapping, distance-vs-synergy scatter plots, or heatmaps of pairs).

4. **Hypothesis Ledger:** Call `write_hypothesis` whenever you form a new 
   candidate rule for synergy. Call `update_hypothesis` with a rationale 
   whenever new evidence (structural or functional) changes your confidence.

5. **Reproducibility:** Your final analysis must include a standalone Python 
   script (e.g., `discovery_pipeline.py`) that can be executed via `bash` 
   to reproduce your primary findings.

## Constraints (The "Don'ts")

- **No Monoliths:** Do not generate more than 50 lines of code without a 
  verification step (e.g., checking data shapes, nulls, or plotting a subset).
- **Scale Awareness:** If plotting raw activity [0,1], do NOT use a divergent 
  color scale. Use divergent scales ONLY for "delta," "synergy," or "relative" 
  properties.
- **No Cherry-Picking:** If evidence contradicts your proximity hypothesis, 
  you must update your ledger with `status: "weakened"` or `"refuted"`.

## Recommended Tools (via ToolUniverse)

- **`find_tools("protein fitness landscape")`** — Discover tools for data access.
- **`StructureAnalysisTool`** — Calculate distances between residues in a PDB.
- **`LiteratureSearchTool`** — Find the active site residues for your chosen protein.

## Success Criteria
- A `journal.md` documenting the evolution of your understanding.
- A final report via `declare_done` naming the top drivers of synergy.
- A `uv`-compatible script that reproduces your primary discovery plot.
- Updated confidence scores in the ledger reflecting both structural and functional data.
