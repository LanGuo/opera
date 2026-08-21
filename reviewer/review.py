# eval/review.py
import base64
import csv
from datetime import datetime
import json
import os
import re
import sys
import html as _html_module
from pathlib import Path

from reviewer.log_replay import analyze_run
from hypothesis_ledger.workspace import Workspace


DEFAULT_RUBRIC = [
    {"id": "novelty", "label": "Scientific Novelty", "scope": "per_hypothesis",
     "scale": "1-5", "guidance": "Is this hypothesis novel given existing literature?"},
    {"id": "evidence_quality", "label": "Evidence Quality", "scope": "per_hypothesis",
     "scale": "1-5", "guidance": "Quality and independence of evidence supporting this hypothesis."},
    {"id": "overall_quality", "label": "Overall Scientific Quality", "scope": "overall",
     "scale": "1-5", "guidance": "Overall scientific rigor and quality of the run."},
    {"id": "wet_lab_worthy", "label": "Worth Pursuing in Wet Lab", "scope": "overall",
     "scale": "yes/no/partial", "guidance": "Would the top hypotheses be worth pursuing in wet lab validation?"},
]


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def _render_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    size = path.stat().st_size
    name = path.name

    if suffix in (".png", ".jpg", ".jpeg"):
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode()
        return {"name": name, "render_type": "image",
                "content": f"data:{mime};base64,{b64}", "row_count": 0, "size_bytes": size}

    if suffix == ".svg":
        b64 = base64.b64encode(path.read_bytes()).decode()
        return {"name": name, "render_type": "image",
                "content": "data:image/svg+xml;base64," + b64, "row_count": 0, "size_bytes": size}

    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        rows: list = []
        total = 0
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.reader(f, delimiter=delim):
                    total += 1
                    if len(rows) < 21:
                        rows.append(row)
        except Exception:
            return {"name": name, "render_type": "binary", "content": "", "row_count": 0, "size_bytes": size}
        return {"name": name, "render_type": "table",
                "content": json.dumps(rows), "row_count": total, "size_bytes": size}

    if suffix in (".json", ".txt", ".log", ".md"):
        limit = 200 if suffix == ".json" else 100
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            truncated = "\n".join(lines[:limit])
            if len(lines) > limit:
                truncated += f"\n... ({len(lines) - limit} more lines)"
        except Exception:
            truncated = ""
        return {"name": name, "render_type": "text",
                "content": truncated, "row_count": 0, "size_bytes": size}

    return {"name": name, "render_type": "binary", "content": "", "row_count": 0, "size_bytes": size}


def _collect_supporting_files(run_dir: Path) -> list:
    files = []
    for subdir in ("data", "results"):
        d = run_dir / subdir
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file():
                files.append({**_render_file(f), "subdir": subdir})
    return files


def build_review_data(run_dir: str, llm_scores=None) -> dict:
    ws = Workspace(run_dir)
    metrics = analyze_run(run_dir)
    ledger = ws.read_ledger()

    steps = []
    if ws.step_log_path.exists():
        for line in ws.step_log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    hypotheses = sorted(
        ledger.get("hypotheses", {}).values(),
        key=lambda h: h.get("created_at", ""),
    )

    supporting_files = _collect_supporting_files(ws.run_dir)
    task_spec = ws.read_task_spec()
    rubric = (task_spec or {}).get("rubric") or DEFAULT_RUBRIC

    locked: dict = {
        "rationale": None, "confidence_summary": None,
        "suggested_validation": None, "llm_scores": None,
        "overall_raw_by_backend": None,
    }
    total_tool_calls = None
    if ws.final_report_path.exists():
        report = json.loads(ws.final_report_path.read_text(encoding="utf-8"))
        locked["rationale"] = report.get("rationale")
        locked["confidence_summary"] = report.get("confidence_summary")
        locked["suggested_validation"] = report.get("suggested_validation")
        total_tool_calls = report.get("total_tool_calls")

    if llm_scores is not None:
        if hasattr(llm_scores, "hypothesis_scores"):
            scores_dict = {"default": llm_scores}
        else:
            scores_dict = llm_scores

        hyp_map: dict = {}
        overall_by_backend: dict = {}
        for backend_key, run_scores in scores_dict.items():
            for hs in run_scores.hypothesis_scores:
                entry = hyp_map.setdefault(hs.hypothesis_id, {
                    "hypothesis_id": hs.hypothesis_id,
                    "gene": hs.candidate_gene,
                    "scores_by_backend": {},
                })
                entry["scores_by_backend"][backend_key] = {
                    "per_hypothesis_raw": hs.per_hypothesis_raw,
                    "criteria": [
                        {"criterion_id": c.criterion_id, "rationale": c.rationale,
                         "score": c.score, "agent_note": c.agent_note}
                        for c in (hs.criteria or [])
                    ],
                }
            if run_scores.overall_raw:
                overall_by_backend[backend_key] = run_scores.overall_raw

        locked["llm_scores"] = list(hyp_map.values())
        locked["overall_raw_by_backend"] = overall_by_backend or None

    # Parse model and date from run directory name (format: model_YYYYMMDD_HHMMSS)
    run_id_str = Path(run_dir).name
    _m = re.match(r'^(.+)_(\d{8})_\d{6}$', run_id_str)
    if _m:
        run_model = _m.group(1)
        run_date = _m.group(2)[:4] + "-" + _m.group(2)[4:6] + "-" + _m.group(2)[6:8]
    else:
        run_model = "unknown"
        run_date = "unknown"

    return {
        "run_id": Path(run_dir).name,
        "run_dir": str(run_dir),
        "run_model": run_model,
        "run_date": run_date,
        "total_tool_calls": total_tool_calls,
        "metrics": {
            "hypothesis_count": metrics.hypothesis_count,
            "belief_revision_count": metrics.belief_revision_count,
            "refutation_rate": metrics.refutation_rate,
            "modality_coverage": {k: sorted(v) for k, v in metrics.modality_coverage.items()},
            "tool_diversity_top5": sorted(
                metrics.tool_diversity.items(), key=lambda x: -x[1]
            )[:5],
            "tool_diversity_entropy_ledger": metrics.tool_diversity_entropy_ledger,
            "tool_diversity_entropy_universe": metrics.tool_diversity_entropy_universe,
            "martingale_score": metrics.martingale_score,
            "martingale_n": metrics.martingale_n,
            "directional_flip_count": metrics.directional_flip_count,
            "revisions_per_hypothesis": metrics.revisions_per_hypothesis,
            "turn_count": metrics.turn_count,
            "completed": metrics.completed,
        },
        "steps": steps,
        "supporting_files": supporting_files,
        "hypotheses": list(hypotheses),
        "locked": locked,
        "task_spec": task_spec,
        "rubric": rubric,
    }


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _esc(s) -> str:
    return _html_module.escape(str(s) if s is not None else "")


def _radio_group(name: str, options: list) -> str:
    parts = [
        f'<label><input type="radio" name="{_esc(name)}" value="{_esc(str(v))}"> {_esc(str(lb))}</label>'
        for v, lb in options
    ]
    return '<div class="radio-group">' + "".join(parts) + "</div>"


def _safe_json(obj) -> str:
    return json.dumps(obj).replace("</", "<\\/")


def _render_task_context(data: dict) -> str:
    task_spec = data.get("task_spec")
    if not task_spec:
        return ""

    goal = task_spec.get("goal", "")
    tc = task_spec.get("termination_criteria") or {}
    if not isinstance(tc, dict):
        tc = {}
    rubric = data.get("rubric", [])

    tc_items = "".join(
        f"<li>{_esc(k)}: <strong>{_esc(str(v))}</strong></li>"
        for k, v in tc.items()
    )

    rubric_items = "".join(
        f'<li><strong>{_esc(r["label"])}</strong>'
        f' <span style="color:#888">({_esc(r["scale"])})</span></li>'
        for r in rubric
    )

    return (
        '<div class="task-context">'
        '<div class="task-context-title">Task</div>'
        f'<p class="task-goal">{_esc(goal)}</p>'
        '<div class="task-context-cols">'
        '<div>'
        '<div class="task-context-subtitle">Termination Criteria</div>'
        f'<ul class="tc-list">{tc_items}</ul>'
        '</div>'
        '<div>'
        '<div class="task-context-subtitle">Rubric Dimensions</div>'
        f'<ul class="tc-list">{rubric_items}</ul>'
        '</div>'
        '</div>'
        '</div>'
    )


def _css() -> str:
    return """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto;
       padding: 20px 24px 60px; color: #1a1a1a; line-height: 1.5; }
h1 { font-size: 1.4rem; margin-bottom: 4px; }
h2 { font-size: 1.05rem; margin: 28px 0 12px; border-bottom: 2px solid #e0e0e0;
     padding-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em; color: #333; }
h3 { font-size: 0.95rem; margin: 16px 0 8px; color: #555; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 12px; }
.metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
           gap: 8px; margin: 12px 0; }
.metric-card { background: #f5f5f5; border-radius: 6px; padding: 10px 14px; }
.metric-label { font-size: 0.7rem; color: #888; text-transform: uppercase;
                letter-spacing: 0.06em; margin-bottom: 2px; }
.metric-value { font-size: 1.3rem; font-weight: 700; }
details { border: 1px solid #e0e0e0; border-radius: 6px; margin-bottom: 8px; }
summary { padding: 10px 14px; cursor: pointer; font-weight: 500;
          user-select: none; list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "▶ "; font-size: 0.7em; opacity: 0.6; }
details[open] > summary::before { content: "▼ "; }
.details-body { padding: 12px 14px; border-top: 1px solid #e0e0e0; }
.step-field { margin-bottom: 10px; }
.step-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
              color: #999; letter-spacing: 0.06em; }
.step-value { margin-top: 2px; font-size: 0.9rem; white-space: pre-wrap; }
.rubric-section { background: #f8f9ff; border-left: 3px solid #4a90e2;
                  padding: 16px 20px; margin: 20px 0; border-radius: 0 6px 6px 0; }
.rubric-title { font-weight: 700; font-size: 0.8rem; text-transform: uppercase;
                letter-spacing: 0.07em; color: #4a90e2; margin-bottom: 14px; }
.rubric-item { margin-bottom: 14px; }
.rubric-label { font-size: 0.9rem; margin-bottom: 6px; }
.radio-group { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 4px; }
.radio-group label { display: flex; align-items: center; gap: 5px;
                     font-size: 0.85rem; cursor: pointer; }
textarea { width: 100%; padding: 8px 10px; border: 1px solid #ccc; border-radius: 4px;
           font-family: inherit; font-size: 0.9rem; resize: vertical; min-height: 60px;
           margin-top: 4px; }
input[type="text"] { width: 100%; padding: 8px 10px; border: 1px solid #ccc;
                     border-radius: 4px; font-family: inherit; font-size: 0.9rem;
                     margin-top: 4px; }
.hypothesis-block { border: 1px solid #e0e0e0; border-radius: 8px;
                    padding: 16px; margin-bottom: 16px; }
.hypothesis-header { display: flex; align-items: center; gap: 12px;
                     margin-bottom: 12px; flex-wrap: wrap; }
.h-gene { font-size: 1.1rem; font-weight: 700; }
.h-celltype { color: #666; font-size: 0.9rem; }
.h-status { padding: 2px 9px; border-radius: 12px; font-size: 0.73rem; font-weight: 700; }
.status-active { background: #e8f5e9; color: #2e7d32; }
.status-terminal { background: #e3f2fd; color: #1565c0; }
.status-weakened { background: #fff8e1; color: #e65100; }
.status-refuted { background: #fce4ec; color: #c62828; }
.evidence-item { margin-bottom: 8px; padding: 8px 10px; background: #fafafa;
                 border-radius: 4px; }
.evidence-type { font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
                 color: #999; margin-bottom: 2px; }
.update-item { margin-bottom: 8px; padding: 8px 10px; background: #fafafa;
               border-radius: 4px; border-left: 2px solid #4a90e2; }
.update-meta { font-size: 0.75rem; color: #999; margin-bottom: 3px; }
.file-item { margin-bottom: 16px; }
.file-name { font-weight: 600; font-size: 0.88rem; margin-bottom: 6px; }
.file-size { color: #999; font-size: 0.8rem; }
table.preview { border-collapse: collapse; width: 100%; font-size: 0.78rem;
                overflow-x: auto; display: block; }
table.preview th { background: #f0f0f0; padding: 4px 8px;
                   border: 1px solid #ddd; white-space: nowrap; }
table.preview td { padding: 3px 8px; border: 1px solid #ddd; }
pre.preview { background: #f5f5f5; padding: 10px 12px; border-radius: 4px;
              overflow-x: auto; font-size: 0.78rem; max-height: 280px;
              overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
section { margin-bottom: 36px; }
.submit-btn { background: #4a90e2; color: white; border: none; padding: 12px 36px;
              font-size: 1rem; border-radius: 6px; cursor: pointer; margin-top: 20px;
              font-weight: 600; }
.submit-btn:hover { background: #357abd; }
.locked-wrapper { margin-top: 40px; border-top: 2px dashed #ccc; padding-top: 24px; }
.task-context { background: #f0f4ff; border: 1px solid #c5d5f5; border-radius: 8px;
                padding: 16px 20px; margin: 16px 0 24px; }
.task-context-title { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
                      letter-spacing: 0.08em; color: #4a6fa5; margin-bottom: 8px; }
.task-context-subtitle { font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
                         letter-spacing: 0.06em; color: #666; margin-bottom: 6px;
                         margin-top: 12px; }
.task-goal { font-size: 0.92rem; margin-bottom: 12px; color: #222; }
.task-context-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.tc-list { list-style: none; padding: 0; font-size: 0.82rem; }
.tc-list li { margin-bottom: 4px; padding-left: 12px; position: relative; }
.tc-list li::before { content: "•"; color: #4a6fa5; position: absolute; left: 0; }
.rubric-guidance { font-size: 0.8rem; color: #666; margin: 2px 0 6px; font-style: italic; }
.criterion-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }
.criterion-table th, .criterion-table td { padding: 0.35rem 0.5rem; border: 1px solid #dde; text-align: left; vertical-align: top; }
.criterion-table th { background: #f0f2f5; font-weight: 600; }
.criterion-table .score-cell { font-weight: bold; white-space: nowrap; }
.score-contested { background: #fff3cd; }
"""


def _render_header(data: dict) -> str:
    m = data["metrics"]
    tool_str = ", ".join(f"{_esc(t)}&times;{c}" for t, c in m["tool_diversity_top5"])
    coverage_rows = "".join(
        f"<tr><td><code>{_esc(hid)}</code></td><td>{_esc(', '.join(srcs))}</td></tr>"
        for hid, srcs in m["modality_coverage"].items()
    )
    ms = m.get("martingale_score")
    martingale_n = m.get("martingale_n", 0)
    if ms is not None:
        ms_str = f"{ms:.3f} (n={martingale_n})"
    else:
        ms_str = f"N/A (n={martingale_n})"
    rph = m.get("revisions_per_hypothesis")
    rph_str = f"{rph:.1f}" if rph is not None else "N/A"
    h_ledger = m.get("tool_diversity_entropy_ledger")
    h_universe = m.get("tool_diversity_entropy_universe")
    h_ledger_str = f"{h_ledger:.2f}" if h_ledger is not None else "N/A"
    h_universe_str = f"{h_universe:.2f}" if h_universe is not None else "N/A"
    flip_count = m.get("directional_flip_count", 0)
    return (
        "<header>"
        "<h1>BioAgent-Eval Human Review</h1>"
        f'<div class="meta">Run: <strong>{_esc(data["run_id"])}</strong></div>'
        f'<div class="meta">Model: <strong>{_esc(data.get("run_model", "unknown"))}</strong>'
        f' &middot; Date: {_esc(data.get("run_date", "unknown"))}'
        f' &middot; Turns: {_esc(str(m.get("turn_count") or "N/A"))}'
        f' &middot; Tool calls: {_esc(data.get("total_tool_calls") or "N/A")}</div>'
        '<div class="metrics">'
        f'<div class="metric-card"><div class="metric-label">Hypotheses</div>'
        f'<div class="metric-value">{m["hypothesis_count"]}</div></div>'
        f'<div class="metric-card"><div class="metric-label">Revisions / Hyp.</div>'
        f'<div class="metric-value" title="Mean belief revision events per hypothesis">{_esc(rph_str)}</div></div>'
        f'<div class="metric-card"><div class="metric-label">Refutation Rate</div>'
        f'<div class="metric-value">{m["refutation_rate"]:.0%}</div></div>'
        f'<div class="metric-card"><div class="metric-label">Martingale Score</div>'
        f'<div class="metric-value" title="Pearson r(b_t, Δb_t) pooled across hypotheses; positive = entrenchment">{_esc(ms_str)}</div></div>'
        f'<div class="metric-card"><div class="metric-label">H: Ledger ops</div>'
        f'<div class="metric-value" title="Normalized Shannon entropy over hypothesis-ledger op usage (6 ops available)">{_esc(h_ledger_str)}</div></div>'
        f'<div class="metric-card"><div class="metric-label">H: Bio tools</div>'
        f'<div class="metric-value" title="Normalized Shannon entropy over ToolUniverse usage (~600 tools available)">{_esc(h_universe_str)}</div></div>'
        f'<div class="metric-card"><div class="metric-label">Completed</div>'
        f'<div class="metric-value">{"&#10003;" if m["completed"] else "&#10007;"}</div></div>'
        "</div>"
        f'<div class="meta">Top tools: {tool_str}</div>'
        "<details>"
        "<summary>Modality coverage per hypothesis</summary>"
        '<div class="details-body">'
        '<table class="preview"><tr><th>Hypothesis</th><th>Evidence modalities</th></tr>'
        f"{coverage_rows}</table></div></details>"
        "</header>"
    )


def _render_supporting_data(files: list) -> str:
    if not files:
        return ""

    def render_one(f: dict) -> str:
        if f["render_type"] == "image":
            img = f'<img src="{f["content"]}" style="max-width:100%;border-radius:4px;" alt="{_esc(f["name"])}">'
            return f'<div class="file-item"><div class="file-name">{_esc(f["name"])}</div>{img}</div>'
        if f["render_type"] == "table":
            rows = json.loads(f["content"])
            if not rows:
                return f'<div class="file-item"><div class="file-name">{_esc(f["name"])}</div><em>Empty</em></div>'
            header = "".join(f"<th>{_esc(str(c))}</th>" for c in rows[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{_esc(str(c))}</td>" for c in row) + "</tr>"
                for row in rows[1:]
            )
            total = f["row_count"]
            shown = len(rows) - 1
            note = (f" <small>({shown} of {total - 1} data rows shown)</small>"
                    if total > len(rows) else "")
            tbl = f'<table class="preview"><tr>{header}</tr>{body}</table>{note}'
            return f'<div class="file-item"><div class="file-name">{_esc(f["name"])}</div>{tbl}</div>'
        if f["render_type"] == "text":
            return (f'<div class="file-item"><div class="file-name">{_esc(f["name"])}</div>'
                    f'<pre class="preview">{_esc(f["content"])}</pre></div>')
        size_kb = f["size_bytes"] / 1024
        return (f'<div class="file-item"><div class="file-name">{_esc(f["name"])}</div>'
                f'<span class="file-size">Binary &mdash; {size_kb:.1f} KB</span></div>')

    by_subdir: dict = {}
    for f in files:
        by_subdir.setdefault(f["subdir"], []).append(f)

    inner = ""
    for subdir, subfiles in by_subdir.items():
        items = "".join(render_one(f) for f in subfiles)
        n = len(subfiles)
        inner += (
            "<details>"
            f'<summary>&#128193; {_esc(subdir)}/ ({n} file{"s" if n != 1 else ""})</summary>'
            f'<div class="details-body">{items}</div>'
            "</details>"
        )
    return (
        "<details><summary>Supporting Data</summary>"
        f'<div class="details-body">{inner}</div></details>'
    )


def _render_section1(data: dict) -> str:
    steps_html = ""
    for i, step in enumerate(data["steps"], 1):
        steps_html += (
            "<details>"
            f'<summary>Step {i} &middot; {_esc(step.get("step_type",""))} &middot; {_esc(step.get("tool_used",""))}</summary>'
            '<div class="details-body">'
            f'<div class="step-field"><div class="step-label">Inputs</div>'
            f'<div class="step-value">{_esc(step.get("inputs_summary",""))}</div></div>'
            f'<div class="step-field"><div class="step-label">Outputs</div>'
            f'<div class="step-value">{_esc(step.get("outputs_summary",""))}</div></div>'
            f'<div class="step-field"><div class="step-label">Interpretation</div>'
            f'<div class="step-value">{_esc(step.get("interpretation",""))}</div></div>'
            f'<div class="step-field"><div class="step-label">Next action</div>'
            f'<div class="step-value">{_esc(step.get("next_action",""))}</div></div>'
            "</div></details>"
        )

    p = []
    p.append("<section><h2>Section 1: Reasoning Process</h2>")
    p.append(steps_html)
    p.append(_render_supporting_data(data["supporting_files"]))
    p.append("</section>")
    return "".join(p)


def _render_section2(data: dict) -> str:
    YNP = [("yes", "Yes"), ("no", "No"), ("partial", "Partial")]
    per_items = [r for r in data.get("rubric", []) if r.get("scope") == "per_hypothesis"]
    hyp_blocks = []

    for h in data["hypotheses"]:
        h_id = h.get("hypothesis_id", "unknown")
        safe = h_id.replace("-", "_").replace(" ", "_")
        gene = h.get("candidate_gene", h.get("gene", "unknown"))
        cell_type = h.get("cell_type", "unknown")
        status = h.get("status", "active")
        status_class = {
            "active": "status-active", "terminal": "status-terminal",
            "weakened": "status-weakened", "refuted": "status-refuted",
        }.get(status, "status-active")

        def _ev_html(ev) -> str:
            if isinstance(ev, str):
                return f'<div class="evidence-item"><div>{_esc(ev)}</div></div>'
            src = _esc(ev.get("source") or ev.get("type") or "unknown")
            desc = _esc(ev.get("description", ""))
            return (f'<div class="evidence-item">'
                    f'<div class="evidence-type">{src}</div>'
                    f'<div>{desc}</div></div>')
        ev_items = "".join(_ev_html(ev) for ev in h.get("evidence", [])) or "<em>No evidence recorded.</em>"

        upd_items = ""
        for upd in h.get("update_history", []):
            ch = upd.get("changes", {})
            fc = ch.get("from", {}).get("confidence", "")
            tc = ch.get("to", {}).get("confidence", "")
            conf_str = f" ({_esc(str(fc))} &rarr; {_esc(str(tc))})" if fc and tc else ""
            upd_items += (
                '<div class="update-item">'
                f'<div class="update-meta">{_esc(upd.get("timestamp",""))}{conf_str}</div>'
                f'<div>{_esc(upd.get("rationale",""))}</div></div>'
            )
        if not upd_items:
            upd_items = "<em>No updates &mdash; confidence unchanged from initial assessment.</em>"

        if status in ("active", "terminal"):
            status_q = "Should this hypothesis have been refuted?"
        else:
            status_q = "Was this refutation/weakening warranted?"

        p = []
        p.append(f'<div class="hypothesis-block" id="h_{_esc(safe)}">')
        p.append('<div class="hypothesis-header">')
        p.append(f'<span class="h-gene">{_esc(gene)}</span>')
        p.append(f'<span class="h-celltype">{_esc(cell_type)}</span>')
        p.append(f'<span class="h-status {status_class}">{_esc(status)}</span>')
        p.append("</div>")
        p.append(f"<details><summary>Evidence ({len(h.get('evidence', []))} items)</summary>")
        p.append(f'<div class="details-body">{ev_items}</div></details>')
        p.append(f"<details><summary>Confidence trajectory ({len(h.get('update_history', []))} updates)</summary>")
        p.append(f'<div class="details-body">{upd_items}</div></details>')

        p.append('<div class="rubric-section">')
        if per_items:
            p.append(f'<div class="rubric-title">Evaluation &mdash; {_esc(gene)} ({_esc(h_id)})</div>')
            for item in per_items:
                rid = item["id"]
                label = item["label"]
                guidance = item.get("guidance", "")
                scale = item.get("scale", "1-5")
                field_name = f"hyp_{safe}_{rid}"
                if scale == "1-5":
                    opts = [(1,"1"),(2,"2"),(3,"3"),(4,"4"),(5,"5")]
                else:
                    opts = [("yes","Yes"),("no","No"),("partial","Partial")]
                p.append(f'<div class="rubric-item"><div class="rubric-label">{_esc(label)}</div>')
                if guidance:
                    p.append(f'<div class="rubric-guidance">{_esc(guidance)}</div>')
                p.append(_radio_group(field_name, opts))
                p.append("</div>")
        # Status question and notes always shown regardless of rubric
        p.append(f'<div class="rubric-item"><div class="rubric-label">Status decision: {_esc(status_q)}</div>')
        p.append(_radio_group(f"hyp_{safe}_status", YNP))
        p.append("</div>")
        p.append(f'<div class="rubric-item"><div class="rubric-label">Notes</div>')
        p.append(f'<textarea id="hyp_{_esc(safe)}_notes" rows="2" placeholder="Optional notes..."></textarea></div>')
        p.append("</div>")

        p.append("</div>")
        hyp_blocks.append("".join(p))

    return "<section><h2>Section 2: Hypothesis Review</h2>" + "".join(hyp_blocks) + "</section>"


def _render_section3(data: dict) -> str:
    overall_items = [r for r in data.get("rubric", []) if r.get("scope") == "overall"]

    full_log = "".join(
        f'<div class="step-field">'
        f'<div class="step-label">Step {i}: {_esc(s.get("step_type",""))} &mdash; {_esc(s.get("tool_used",""))}</div>'
        f'<div class="step-value">{_esc(s.get("interpretation",""))}</div></div>'
        for i, s in enumerate(data["steps"], 1)
    ) or "<em>No steps recorded.</em>"

    p = []
    p.append("<section><h2>Section 3: Overall Assessment</h2>")
    p.append(f"<details><summary>Full step log ({len(data['steps'])} steps)</summary>")
    p.append(f'<div class="details-body">{full_log}</div></details>')

    if overall_items:
        p.append('<div class="rubric-section">')
        p.append('<div class="rubric-title">Overall Evaluation</div>')
        for item in overall_items:
            rid = item["id"]
            label = item["label"]
            guidance = item.get("guidance", "")
            scale = item.get("scale", "1-5")
            field_name = f"ovr_{rid}"
            if scale == "1-5":
                opts = [(1,"1"),(2,"2"),(3,"3"),(4,"4"),(5,"5")]
            else:
                opts = [("yes","Yes"),("no","No"),("partial","Partial")]
            p.append(f'<div class="rubric-item"><div class="rubric-label">{_esc(label)}</div>')
            if guidance:
                p.append(f'<div class="rubric-guidance">{_esc(guidance)}</div>')
            p.append(_radio_group(field_name, opts))
            p.append("</div>")
        p.append("</div>")

    p.append('<div class="rubric-section">')
    p.append('<div class="rubric-title">Reviewer Info</div>')
    p.append('<div class="rubric-item"><div class="rubric-label">Reviewer name</div>')
    p.append('<input type="text" id="rd_reviewer" placeholder="Your name"></div>')
    p.append('<div class="rubric-item"><div class="rubric-label">Overall notes</div>')
    p.append('<textarea id="rd_notes" rows="4" placeholder="Overall assessment..."></textarea></div>')
    p.append("</div>")
    p.append('<button class="submit-btn" onclick="submitReview()">Submit &amp; Download review.json</button>')
    p.append("</section>")
    return "".join(p)


def _render_locked(data: dict) -> str:
    locked = data.get("locked", {})

    llm_scores = locked.get("llm_scores")
    overall_raw_by_backend = locked.get("overall_raw_by_backend")

    if llm_scores is None and overall_raw_by_backend is None:
        scores_html = (
            "<p><em>LLM scores not available for this run "
            "(set ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OLLAMA_MODEL when generating review.html).</em></p>"
        )
    else:
        scores_html = ""

        if llm_scores:
            scores_html += "<h3>Per-Hypothesis Evaluation</h3>"
            for hs in llm_scores:
                backends = hs.get("scores_by_backend", {})
                n_backends = len(backends)
                header = (
                    '<div class="hypothesis-block">'
                    '<div class="hypothesis-header">'
                    f'<span class="h-gene">{_esc(hs["gene"])}</span>'
                    f' <span style="color:#999">({_esc(hs["hypothesis_id"])})</span>'
                    "</div>"
                )
                if n_backends > 1:
                    header += (
                        f'<p style="font-size:0.85rem;color:#666;margin-bottom:8px;">'
                        f'{n_backends} backends scored &mdash; expand each to compare</p>'
                    )
                # Collect per-criterion scores across backends for conflict detection
                # criterion_id -> {backend_key -> numeric_score}
                criterion_backend_nums: dict = {}
                for backend_key, backend_scores in backends.items():
                    for c in backend_scores.get("criteria") or []:
                        cid = c["criterion_id"]
                        m = __import__("re").match(r"(\d+)", str(c.get("score", "")))
                        if m:
                            criterion_backend_nums.setdefault(cid, {})[backend_key] = int(m.group(1))

                # Determine contested criteria (differ by >=2 across backends)
                contested_criteria: set = set()
                for cid, bmap in criterion_backend_nums.items():
                    nums = list(bmap.values())
                    if len(nums) >= 2 and (max(nums) - min(nums)) >= 2:
                        contested_criteria.add(cid)

                per_hyp_html = ""
                for backend_key, backend_scores in backends.items():
                    criteria = backend_scores.get("criteria") or []
                    if criteria:
                        rows_html = ""
                        for c in criteria:
                            cid = c["criterion_id"]
                            contested_class = "score-contested" if cid in contested_criteria else ""
                            rows_html += (
                                "<tr>"
                                f"<td>{_esc(cid)}</td>"
                                f'<td class="score-cell {contested_class}">{_esc(c.get("score", ""))}</td>'
                                f"<td>{_esc(c.get('rationale', ''))}</td>"
                                f'<td class="agent-note">{_esc(c.get("agent_note", ""))}</td>'
                                "</tr>"
                            )
                        body_html = (
                            '<table class="criterion-table">'
                            "<thead><tr><th>Criterion</th><th>Score</th><th>Rationale</th><th>Agent note</th></tr></thead>"
                            f"<tbody>{rows_html}</tbody>"
                            "</table>"
                        )
                    else:
                        body_html = (
                            f'<pre class="preview">'
                            f'{_esc(backend_scores.get("per_hypothesis_raw", ""))}'
                            f"</pre>"
                        )
                    per_hyp_html += (
                        "<details>"
                        f"<summary>{_esc(backend_key)}</summary>"
                        f'<div class="details-body">{body_html}</div></details>'
                    )
                scores_html += header + per_hyp_html + "</div>"

        if overall_raw_by_backend:
            scores_html += "<h3>Overall Run Evaluation</h3>"
            for backend_key, overall_raw in overall_raw_by_backend.items():
                scores_html += (
                    "<details>"
                    f"<summary>{_esc(backend_key)}</summary>"
                    f'<div class="details-body"><pre class="preview">'
                    f"{_esc(overall_raw)}"
                    f"</pre></div></details>"
                )

    rationale = _esc(locked.get("rationale") or "")
    conf = _esc(locked.get("confidence_summary") or "")
    validation = _esc(locked.get("suggested_validation") or "")

    content = (
        "<h3>LLM-as-Judge Scores</h3>" + scores_html
        + "<h3>Agent Rationale</h3>"
        + (f"<p>{rationale}</p>" if rationale else "<p><em>Not available.</em></p>")
        + "<h3>Confidence Summary</h3>"
        + (f"<p>{conf}</p>" if conf else "<p><em>Not available.</em></p>")
        + "<h3>Suggested Validation Experiment</h3>"
        + (f"<p>{validation}</p>" if validation else "<p><em>Not available.</em></p>")
    )

    return (
        '<div class="locked-wrapper">'
        "<h2>&#128274; Agent Assessment "
        '<small style="font-weight:normal;font-size:0.78rem;text-transform:none;">'
        "(revealed after submitting your scores)</small></h2>"
        '<div style="position:relative;">'
        '<div id="locked-section" style="filter:blur(5px);pointer-events:none;user-select:none;">'
        + content
        + "</div>"
        '<div id="locked-overlay" style="position:absolute;top:0;left:0;right:0;bottom:0;'
        "display:flex;align-items:center;justify-content:center;"
        'background:rgba(255,255,255,0.45);">'
        '<div style="text-align:center;padding:20px;background:white;border-radius:8px;'
        'box-shadow:0 2px 12px rgba(0,0,0,0.15);">'
        '<div style="font-size:2rem;">&#128274;</div>'
        "<p>Submit your scores above to reveal agent assessment.</p>"
        "</div></div></div></div>"
    )


def _js(data: dict) -> str:
    h_ids = [h.get("hypothesis_id", "unknown") for h in data["hypotheses"]]
    safe_ids = [hid.replace("-", "_").replace(" ", "_") for hid in h_ids]
    h_meta = {
        h.get("hypothesis_id", "unknown"): {
            "gene": h.get("candidate_gene", h.get("gene", "unknown")),
            "cell_type": h.get("cell_type", "unknown"),
            "status": h.get("status", "active"),
        }
        for h in data["hypotheses"]
    }
    rubric_js = [
        {"id": r["id"], "label": r["label"], "scope": r.get("scope", "overall"),
         "scale": r.get("scale", "1-5")}
        for r in data.get("rubric", [])
    ]
    constants = (
        "const RUN_ID=" + _safe_json(data["run_id"]) + ";\n"
        "const RUN_DIR=" + _safe_json(data["run_dir"]) + ";\n"
        "const HYPOTHESIS_IDS=" + _safe_json(h_ids) + ";\n"
        "const HYPOTHESIS_SAFE_IDS=" + _safe_json(safe_ids) + ";\n"
        "const HYPOTHESIS_META=" + _safe_json(h_meta) + ";\n"
        "const RUBRIC=" + _safe_json(rubric_js) + ";\n"
    )
    fn = r"""
function submitReview() {
    const getR = (n) => { const e = document.querySelector('input[name="'+n+'"]:checked'); if (!e) return null; const v=e.value; return isNaN(v)?v:parseInt(v,10); };
    const getT = (id) => { const e = document.getElementById(id); return e ? e.value.trim() : ''; };
    const errs = [];

    const per_hypothesis = {};
    const per_items = RUBRIC.filter(r => r.scope === 'per_hypothesis');
    const ovr_items = RUBRIC.filter(r => r.scope === 'overall');

    for (let i = 0; i < HYPOTHESIS_IDS.length; i++) {
        const hid = HYPOTHESIS_IDS[i], safe = HYPOTHESIS_SAFE_IDS[i];
        per_hypothesis[hid] = {};
        for (const item of per_items) {
            const val = getR('hyp_' + safe + '_' + item.id);
            per_hypothesis[hid][item.id] = val;
            if (!val) errs.push(hid + ': ' + item.label + ' not scored');
        }
        const statusVal = getR('hyp_' + safe + '_status');
        per_hypothesis[hid].status_decision = statusVal;
        if (!statusVal) errs.push(hid + ': status decision not answered');
        per_hypothesis[hid].notes = getT('hyp_' + safe + '_notes');
    }

    const overall = {};
    for (const item of ovr_items) {
        const val = getR('ovr_' + item.id);
        overall[item.id] = val;
        if (!val) errs.push('Overall: ' + item.label + ' not scored');
    }

    const reviewer = getT('rd_reviewer');
    if (!reviewer) errs.push('Reviewer name required');
    overall.notes = getT('rd_notes');

    if (errs.length > 0) { alert('Please complete all required fields:\n\n' + errs.join('\n')); return; }

    const review = {
        run_dir: RUN_DIR, run_id: RUN_ID, reviewer,
        reviewed_at: new Date().toISOString(),
        per_hypothesis, overall
    };
    const blob = new Blob([JSON.stringify(review, null, 2)], {type: 'application/json'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = RUN_ID + '_review.json';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    const s = document.getElementById('locked-section');
    if (s) { s.style.filter='none'; s.style.pointerEvents='auto'; s.style.userSelect='auto'; }
    const ov = document.getElementById('locked-overlay'); if (ov) ov.style.display = 'none';
}
"""
    return constants + fn


def _render_falsification_section(results: list) -> str:
    if not results:
        return ""
    from reviewer.falsification import _DIVERGENCE_THRESHOLD
    rows = ""
    for r in results:
        delta_color = "#c0392b" if r.confidence_delta < -0.1 else (
            "#27ae60" if r.confidence_delta > 0.1 else "#888"
        )
        agent_conf_str = f"{r.agent_confidence:.2f}" if r.agent_confidence is not None else "N/A"
        flag_str = (
            f"<span style='color:#c0392b;font-weight:bold' "
            f"title='observer vs. e-value confidence diverge by {r.deterministic_delta:+.2f}'>&#9888; flagged</span>"
            if getattr(r, "confidence_flagged", False) else ""
        )
        rows += (
            f"<tr><td>{r.hypothesis_id}</td>"
            f"<td>{agent_conf_str}</td>"
            f"<td>{r.observer_confidence:.2f}</td>"
            f"<td>{r.e_value_confidence:.2f}</td>"
            f"<td style='color:{delta_color}'>{r.confidence_delta:+.2f}</td>"
            f"<td>{flag_str}</td>"
            f"<td>{'<br>'.join(r.falsification_criteria[:3])}</td></tr>"
        )
    return (
        "<h3>Falsification Assessment (Independent Observer)</h3>"
        "<p style='font-size:0.85rem;color:#666'>"
        "Observer confidence is assessed by a separate LLM with no access to the agent's "
        "stated confidence. E-value confidence is derived from quantitative tool outputs "
        "(p-values, effect sizes). A negative delta means the agent was overconfident. "
        "A hypothesis is flagged when observer and e-value confidence diverge by more than "
        f"{_DIVERGENCE_THRESHOLD:.2f} — this is a sanity-check flag, not a correctness verdict.</p>"
        "<table border='1' style='border-collapse:collapse;width:100%;font-size:0.88rem'>"
        "<tr><th>Hypothesis</th><th>Agent conf.</th><th>Observer conf.</th>"
        "<th>E-value conf.</th><th>Delta</th><th>Consistency</th><th>Falsification criteria</th></tr>"
        f"{rows}</table>"
    )


def render_html(data: dict, falsification_html: str = "") -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>BioAgent-Eval Review &mdash; {_esc(data["run_id"])}</title>\n'
        f"<style>{_css()}</style>\n"
        "</head>\n<body>\n"
        + _render_header(data)
        + _render_task_context(data)
        + _render_section1(data)
        + _render_section2(data)
        + falsification_html
        + _render_section3(data)
        + _render_locked(data)
        + f"<script>{_js(data)}</script>\n"
        "</body>\n</html>"
    )


def generate_review_html(run_dir, metrics, scores=None, falsification_results=None) -> str:
    """Build a complete review HTML string from a RunMetrics object and optional scores/falsification.

    This is a convenience wrapper around build_review_data + render_html that accepts
    pre-computed metrics (e.g. from tests or programmatic callers) instead of re-reading
    disk artifacts.

    ``scores`` may be:
      - None
      - a single RunScores object (legacy / single-backend)
      - a dict[str, RunScores] (multi-backend, keys are backend names)
    """
    # Build a minimal data dict from the provided metrics object
    tool_diversity_top5 = sorted(
        metrics.tool_diversity.items(), key=lambda x: -x[1]
    )[:5]
    modality_coverage = {k: sorted(v) for k, v in metrics.modality_coverage.items()}

    # Normalise scores into the locked structure via build_review_data's logic.
    # We create a temporary minimal data dict and let build_review_data fill
    # locked["llm_scores"] by passing scores as llm_scores.
    # To avoid re-reading disk we build the data dict manually, then inject llm_scores.
    llm_scores_structured = None
    if scores is not None:
        if hasattr(scores, "hypothesis_scores"):
            # Legacy single RunScores — wrap so build_review_data normalises it
            llm_scores_structured = scores
        else:
            llm_scores_structured = scores  # dict[str, RunScores]

    # Build locked["llm_scores"] using the same normalisation in build_review_data
    locked_llm_scores = None
    locked_overall_raw_by_backend = None
    if llm_scores_structured is not None:
        if hasattr(llm_scores_structured, "hypothesis_scores"):
            scores_dict = {"default": llm_scores_structured}
        else:
            scores_dict = llm_scores_structured

        hyp_map: dict = {}
        overall_by_backend: dict = {}
        for backend_key, run_scores in scores_dict.items():
            for hs in run_scores.hypothesis_scores:
                entry = hyp_map.setdefault(hs.hypothesis_id, {
                    "hypothesis_id": hs.hypothesis_id,
                    "gene": hs.candidate_gene,
                    "scores_by_backend": {},
                })
                entry["scores_by_backend"][backend_key] = {
                    "per_hypothesis_raw": hs.per_hypothesis_raw,
                    "criteria": [
                        {"criterion_id": c.criterion_id, "rationale": c.rationale,
                         "score": c.score, "agent_note": c.agent_note}
                        for c in (hs.criteria or [])
                    ],
                }
            if run_scores.overall_raw:
                overall_by_backend[backend_key] = run_scores.overall_raw
        locked_llm_scores = list(hyp_map.values())
        locked_overall_raw_by_backend = overall_by_backend or None

    data = {
        "run_id": Path(run_dir).name,
        "run_dir": str(run_dir),
        "run_model": "unknown",
        "run_date": "unknown",
        "total_tool_calls": metrics.tool_call_count,
        "metrics": {
            "hypothesis_count": metrics.hypothesis_count,
            "belief_revision_count": metrics.belief_revision_count,
            "refutation_rate": metrics.refutation_rate,
            "modality_coverage": modality_coverage,
            "tool_diversity_top5": tool_diversity_top5,
            "tool_diversity_entropy_ledger": metrics.tool_diversity_entropy_ledger,
            "tool_diversity_entropy_universe": metrics.tool_diversity_entropy_universe,
            "martingale_score": metrics.martingale_score,
            "martingale_n": metrics.martingale_n,
            "directional_flip_count": metrics.directional_flip_count,
            "revisions_per_hypothesis": metrics.revisions_per_hypothesis,
            "turn_count": metrics.turn_count,
            "completed": metrics.completed,
        },
        "steps": [],
        "supporting_files": [],
        "hypotheses": [],
        "locked": {
            "rationale": None,
            "confidence_summary": None,
            "suggested_validation": None,
            "llm_scores": locked_llm_scores,
            "overall_raw_by_backend": locked_overall_raw_by_backend,
        },
        "task_spec": None,
        "rubric": DEFAULT_RUBRIC,
    }
    falsification_html = _render_falsification_section(falsification_results or [])
    return render_html(data, falsification_html=falsification_html)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m reviewer.review <run_dir>", file=sys.stderr)
        sys.exit(1)

    run_dir = sys.argv[1]
    if not Path(run_dir).is_dir():
        print(f"Error: run_dir does not exist or is not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Computing offline metrics for {run_dir} ...")
    try:
        from reviewer.log_replay import analyze_run, print_report
        metrics = analyze_run(run_dir)
        print_report(metrics)
    except Exception as exc:
        print(f"  ERROR: offline metrics failed: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("\n[2/4] Running LLM-as-judge scoring (all configured backends) ...")
    llm_scores = None
    try:
        from reviewer.score import score_run_all_backends
        import os
        backends = []
        if os.environ.get("ANTHROPIC_API_KEY"): backends.append("anthropic")
        if os.environ.get("GOOGLE_API_KEY"): backends.append("google")
        if os.environ.get("OLLAMA_MODEL"): backends.append("ollama")
        if not backends:
            print("  SKIP: no API keys found (ANTHROPIC_API_KEY / GOOGLE_API_KEY / OLLAMA_MODEL).")
        else:
            print(f"  Backends detected: {', '.join(backends)}")
            result = score_run_all_backends(run_dir)
            llm_scores = result or None
            n = sum(len(rs.hypothesis_scores) for rs in (result.values() if isinstance(result, dict) else [result])) if llm_scores else 0
            print(f"  Done: scored across {len(backends)} backend(s).")
    except Exception as exc:
        print(f"  ERROR: LLM scoring failed — locked section will show N/A.\n  Reason: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()

    print("\n[3/4] Running falsification observer ...")
    falsification_results = []
    try:
        from reviewer.falsification import assess_terminal_hypotheses
        import os
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OLLAMA_MODEL")):
            print("  SKIP: no API keys found.")
        else:
            falsification_results = assess_terminal_hypotheses(run_dir)
            print(f"  Done: assessed {len(falsification_results)} hypothesis/es.")
            for r in falsification_results:
                print(f"    {r.hypothesis_id}: agent={r.agent_confidence:.2f}  observer={r.observer_confidence:.2f}  e_value_conf={r.e_value_confidence:.2f}")
    except Exception as exc:
        print(f"  ERROR: falsification observer failed — section will be absent.\n  Reason: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()

    print("\n[4/4] Assembling review HTML ...")
    data = build_review_data(run_dir, llm_scores=llm_scores)
    falsification_html = _render_falsification_section(falsification_results)
    html = render_html(data, falsification_html=falsification_html)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = Path(run_dir) / f"review_{timestamp}.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nDone. Written: {out}")
    print("Open it in a browser to begin review.")


if __name__ == "__main__":
    main()
