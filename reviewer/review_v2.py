"""Review-page sections for ledger.jsonl runs, built from the judge's own inputs and saved prompts.

`render_judge_inputs` prints, for every hypothesis, exactly the pass-E and pass-R inputs the judge
renders (reviewer.judge_inputs), plus the run-level input, so a human reviewer reads what the LLM
judges read. `render_judge_results` shows a judge_<timestamp>.json: every pass, criterion and
backend, the contested flags, and each call's saved prompt and response.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from reviewer.judge_inputs import (build_run_view, render_evidence_input, render_reasoning_input,
                                   render_run_input)
from reviewer.rubric import INPUTS


def _e(s) -> str:
    return html.escape("" if s is None else str(s))


def latest_judge_file(run_dir) -> Path | None:
    found = sorted(Path(run_dir).glob("judge_*.json"))
    return found[-1] if found else None


def render_judge_inputs(run_dir, which: str = "evidence", max_output_chars: int | None = None) -> str:
    """which='evidence': the pass-E inputs (shown before the reviewer scores, as they contain no
    agent interpretation). which='reasoning': the pass-R and run-level inputs (agent rationales,
    confidence, conclusion), shown with the locked section."""
    run = build_run_view(run_dir)
    kw = {"max_output_chars": max_output_chars} if max_output_chars else {}
    if which == "evidence":
        parts = ["<section><h2>What the evidence judge reads</h2>",
                 "<p>Exactly the text the LLM judge's evidence pass receives for each hypothesis, built "
                 "from <code>ledger.jsonl</code> fields only: findings and the tool outputs behind them, "
                 "never the agent's rationale, confidence or status.</p>"]
        for hid in run.hypotheses:
            parts.append(f"<details><summary>{_e(hid)} — evidence pass input</summary><pre class='judge-input'>"
                         f"{_e(render_evidence_input(run, hid, include_validation=True, **kw))}</pre></details>")
    else:
        parts = ["<h3>What the reasoning judge reads</h3>",
                 "<p>The reasoning pass input per hypothesis (each rationale beside the evidence cited "
                 "with it) and the run-level input. The reasoning pass never sees evidence-pass output.</p>"]
        for hid in run.hypotheses:
            parts.append(f"<details><summary>{_e(hid)} — reasoning pass input</summary><pre class='judge-input'>"
                         f"{_e(render_reasoning_input(run, hid, **kw))}</pre></details>")
        all_inputs = sorted(set(INPUTS) - {"artifacts", "tool_outputs", "rationales", "open_questions"})
        parts.append("<details><summary>Run-level input</summary><pre class='judge-input'>"
                     f"{_e(render_run_input(run, all_inputs, **kw))}</pre></details>")
    if which == "evidence":
        parts.append("</section>")
    return "".join(parts)


def _score_cell(v: dict) -> str:
    if v.get("status") != "scored":
        miss = "; ".join(v.get("missing") or [])
        return f"<em>{_e(v.get('status'))}</em>" + (f"<br><small>{_e(miss)}</small>" if miss else "")
    extra = ""
    if v.get("evidence_ids"):
        extra += f"<br><small>relies on {_e(', '.join(v['evidence_ids']))}</small>"
    if v.get("unknown_evidence_ids"):
        extra += f"<br><small>unknown ids {_e(', '.join(v['unknown_evidence_ids']))}</small>"
    return (f"<strong>{_e(v.get('score'))}</strong> — {_e(v.get('rationale'))}{extra}"
            f"<br><small>call {v.get('call')}</small>")


def _criteria_table(rows: list[tuple[str, str]], backends: dict, getter) -> str:
    out = ["<table class='judge-table'><tr><th>criterion</th>"
           + "".join(f"<th>{_e(k)} ({_e(r.get('model'))})</th>" for k, r in backends.items()) + "</tr>"]
    for label, key in rows:
        out.append(f"<tr><td>{_e(label)}</td>")
        for r in backends.values():
            v = getter(r, key)
            out.append(f"<td>{_score_cell(v) if v else '—'}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _fmt(x) -> str:
    return "n/a" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def render_judge_results(path) -> str:
    out = json.loads(Path(path).read_text(encoding="utf-8"))
    backends = {k: r for k, r in out["backends"].items() if "error" not in r}
    parts = [f"<h3>Judge results (v{out['judge_version']})</h3>",
             f"<p><code>{_e(Path(path).name)}</code>, {_e(out['created'])}; tool outputs cut at "
             f"{_e(out['max_output_chars'])} characters per call in prompts.</p>"]
    for k, r in out["backends"].items():
        if "error" in r:
            parts.append(f"<p><strong>{_e(k)} failed:</strong> {_e(r['error'])}</p>")
    parts += [f"<p><em>{_e(w)}</em></p>" for w in out.get("rubric_warnings") or []]
    if out.get("contested"):
        parts.append("<h3>Contested (backends differ by 2 or more points)</h3><ul>"
                     + "".join(f"<li>{_e(f['pass'])} · {_e(f['hypothesis_id'] or 'run')} · "
                               f"{_e(f['criterion_id'])}: {_e(f['scores'])}</li>" for f in out["contested"])
                     + "</ul>")
    if not backends:
        return "".join(parts)
    first = next(iter(backends.values()))
    hids = list(first["evidence"]["per_hypothesis"])
    for hid in hids:
        parts.append(f"<h3>{_e(hid)} — evidence pass</h3>")
        cids = list(first["evidence"]["per_hypothesis"][hid])
        parts.append(_criteria_table([(c, c) for c in cids], backends,
                                     lambda r, c, hid=hid: r["evidence"]["per_hypothesis"].get(hid, {}).get(c)))
        parts.append(f"<h3>{_e(hid)} — reasoning audit</h3>")
        for k, r in backends.items():
            rr = r["reasoning"]["per_hypothesis"].get(hid) or {}
            m = rr.get("metrics") or {}
            parts.append(f"<p><strong>{_e(k)}</strong>: {_e(rr.get('status'))}; "
                         f"{_e(m.get('n_assertions'))} assertions — supported {_fmt(m.get('supported_rate'))}, "
                         f"overstated {_fmt(m.get('overstated_rate'))}, unsupported {_fmt(m.get('unsupported_rate'))}, "
                         f"contradicted {_fmt(m.get('contradicted_rate'))}; calibration {_fmt(m.get('calibration'))}; "
                         f"disconfirmation handled {_fmt(m.get('disconfirmation_handled'))}</p>")
            rows = "".join(f"<tr><td>{_e(a['label'])}</td><td>{_e(a['entry_id'])}</td><td>{_e(a['quote'])}</td>"
                           f"<td>{_e(', '.join(a.get('evidence_ids') or []))}</td><td>{_e(a.get('reason'))}</td></tr>"
                           for a in rr.get("assertions") or [])
            if rows:
                parts.append("<details><summary>assertions</summary><table class='judge-table'>"
                             "<tr><th>label</th><th>entry</th><th>quote</th><th>evidence</th><th>why</th></tr>"
                             f"{rows}</table></details>")
            if rr.get("rejected_assertions"):
                parts.append(f"<p><small>{len(rr['rejected_assertions'])} assertion(s) rejected "
                             "(quote not verbatim, or wrong entry)</small></p>")
    overall_rows = ([(f"evidence · {c}", ("evidence", c)) for c in first["evidence"]["overall"]]
                    + [(f"reasoning · {c}", ("reasoning_scores", c)) for c in first["reasoning_scores"]["overall"]]
                    + [(f"run · {c}", ("run", c)) for c in first["run"]])
    per_reason = first["reasoning_scores"]["per_hypothesis"]
    overall_rows += [(f"reasoning · {h} · {c}", ("reasoning_per", (h, c))) for h, cs in per_reason.items() for c in cs]

    def get(r, key):
        sec, c = key
        if sec == "run":
            return r["run"].get(c)
        if sec == "reasoning_per":
            return r["reasoning_scores"]["per_hypothesis"].get(c[0], {}).get(c[1])
        return r[sec]["overall"].get(c)
    if overall_rows:
        parts.append("<h3>Run-level criteria</h3>" + _criteria_table(overall_rows, backends, get))
    for k, r in backends.items():
        parts.append(f"<details><summary>{_e(k)}: {len(r['calls'])} calls — prompts and responses</summary>")
        for c in r["calls"]:
            parts.append(f"<details><summary>call {c['call']} · {_e(c['pass'])} · {_e(c.get('hypothesis_id') or 'run')}"
                         f" · {c['prompt_chars']:,} characters" + (" · ERROR" if c.get("error") else "") + "</summary>"
                         f"<pre class='judge-input'>{_e(c['prompt'])}</pre><p>Response:</p>"
                         f"<pre class='judge-input'>{_e(c.get('response') or c.get('error'))}</pre></details>")
        parts.append("</details>")
    return "".join(parts)


CSS = (".judge-input{white-space:pre-wrap;font-size:12px;background:#f6f7f9;padding:8px;max-height:480px;"
       "overflow:auto}.judge-table{border-collapse:collapse;font-size:13px}.judge-table td,.judge-table th"
       "{border:1px solid #ddd;padding:4px 6px;vertical-align:top}")
