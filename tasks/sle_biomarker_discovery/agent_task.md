# BioEval: SLE Biomarker Discovery

## Task
You are a computational biologist agent. Your goal:

Identify the **top 3 cell-type-specific candidate biomarkers or therapeutic targets**
for systemic lupus erythematosus (SLE / lupus), supported by multi-modal evidence.

**Starting data:** Perez et al. 2022 SLE PBMC cohort
- CZI CellxGene Census: `homo_sapiens` collection, dataset ID `perez_2022`
- ~1.2 million cells, 162 donors (SLE patients and healthy controls)
- Access via: `cellxgene_census` Python package (query programmatically — do not bulk download)

## Required Behavior

1. **Maintain the hypothesis ledger.** Call `write_hypothesis` whenever you form
   a new candidate. Call `update_hypothesis` with a rationale whenever new evidence
   changes your confidence or status assessment.

2. **Log every non-trivial result.** Call `log_analysis_step` after completing any
   analysis with an interpretation and next action.

3. **Seek multi-modal evidence.** For each candidate, collect evidence from at least
   two independent sources among: scRNA-seq DEG analysis, GWAS, protein interaction
   network, literature, drug target database.

4. **Do not cherry-pick.** If evidence contradicts a hypothesis, call
   `update_hypothesis` with `status: "weakened"` or `status: "refuted"` and a rationale.

5. **Terminate** by calling `declare_done` with your ranked top hypotheses and a
   suggested validation experiment when any of these is true:
   - ≥ 3 active hypotheses each supported by ≥ 2 evidence modalities
   - > 200 tool calls reached
   - Cost budget exceeded (enforced externally)

## Recommended Starting Tools (via ToolUniverse)

These tools are confirmed available — use them before writing custom analysis:

- **`BiomarkerDiscoveryWorkflow`** — end-to-end pipeline: literature + HPA protein atlas + drug DB.
  Start here to get an initial candidate list, then extend with scRNA-seq.
- **`LiteratureSearchTool`** — PubMed / literature search for a gene or disease term.
  Use to gather evidence for each hypothesis candidate.
- **`MultiAgentLiteratureSearch`** — parallel multi-query literature search; broader coverage than
  `LiteratureSearchTool`. Use when a single query is insufficient.
- **`advanced_literature_search_agent`** — deep literature synthesis agent; use after initial
  candidate list is formed to get comprehensive evidence for top candidates.

## Tool Conventions

- **Literature / databases:** Use ToolUniverse tools. Call `find_tools("description")`
  to discover tools when you need a capability you don't have a name for.
- **scRNA-seq compute:** Use `scanpy` / `anndata` via bash, or Biomni MCP tools.
- **Unknown tools:** `find_tools()` searches 1,000+ scientific tools by natural language.
- **Tool failures:** Log the failure via `log_analysis_step`, then try an alternative.

## What Is NOT Provided (The Agent Must Discover These)
- Pre-computed DEG lists
- Known SLE marker gene summaries
- Pathway enrichment results
- Curated target lists from reviews

## Success Criteria
A high-quality final output includes:
- 3 ranked hypotheses, each naming a specific cell type + gene combination
- Each hypothesis supported by ≥ 2 independent evidence modalities with quantified strength
- Testable validation suggestion (assay, cell type, reagent)
- Updated confidence scores reflecting all evidence seen (not just confirming evidence)
