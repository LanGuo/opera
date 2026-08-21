# task_proposer/wizard.py
import json
import os
import re
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from reviewer.llm import get_llm_client

app = FastAPI()

REPO_ROOT = Path(__file__).parent.parent

EXTRACTION_SYSTEM = (
    "You are helping a domain scientist define an open-world evaluation task for an LLM agent. "
    "Extract structured fields from their free-text description. Return JSON only matching this schema:\n"
    "{\n"
    '  "task_id": "short_snake_case_slug",\n'
    '  "goal": "goal-oriented, non-prescriptive description",\n'
    '  "termination_criteria": {\n'
    '    "min_hypotheses_with_multimodal_evidence": 3,\n'
    '    "max_tool_calls": 200,\n'
    '    "custom": []\n'
    "  },\n"
    '  "evaluator": {"model": "claude-opus-4-7", "max_turns": 250},\n'
    '  "rubric": [\n'
    '    {"id": "snake_case_id", "label": "Human-readable label", '
    '"scope": "per_hypothesis", "scale": "1-5", '
    '"guidance": "1-2 sentence scoring guidance shown verbatim to both LLM and human reviewers"}\n'
    "  ]\n"
    "}\n"
    "For task_id, generate a short snake_case slug. "
    "For goal, preserve intent but rewrite to be goal-oriented with no how-to steps. "
    "For rubric, generate 4-8 items that reflect the task's success criteria: "
    "typically 3-4 per_hypothesis items (scientific quality of each hypothesis) and "
    "2-3 overall items (run-level quality). "
    "scope must be 'per_hypothesis' or 'overall'. scale must be '1-5' or 'yes/no/partial'. "
    "guidance must be specific to this task, not generic. "
    "Suggest sensible defaults for missing fields."
)


class ExtractRequest(BaseModel):
    free_text: str


def _call_llm(system: str, user: str) -> str:
    client = get_llm_client()
    return client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])


AGENT_TASK_SYSTEM = (
    "Generate a agent_task.md agent task prompt from the given task spec. "
    "Write in second person ('Your task is to...'). "
    "Be goal-oriented and intent-driven. Do NOT prescribe methodology. "
    "Include the goal, any dataset pointers mentioned, and the termination criteria "
    "rendered as a Markdown checklist. "
    "The 'Success Criteria' section in agent_task.md must be consistent with the rubric items "
    "in the task spec — the rubric items formalize the same evaluation criteria in prose. "
    "Return only the agent_task.md content, no JSON wrapper."
)

QUALITY_SYSTEM = (
    "Review this evaluation task spec and agent_task.md for quality. "
    'Return a JSON array of issue objects. Each: field (string), severity ("blocking"|"warning"), '
    "issue (string), suggestion (string). "
    "Flag: goal too prescriptive or too vague, termination criteria with no quantitative anchor, "
    "agent_task.md methodology prescriptions, missing termination criteria in agent_task.md. "
    "Return [] if no issues."
)


class GenerateRequest(BaseModel):
    task_id: str
    goal: str
    termination_criteria: dict
    evaluator: dict
    rubric_overrides: list | dict


class SaveRequest(BaseModel):
    task_id: str
    goal: str
    termination_criteria: dict
    evaluator: dict
    rubric_overrides: list | dict
    agent_task_md: str


def _build_yaml(req: SaveRequest) -> str:
    lines = [f"task_id: {req.task_id}", "goal: >"]
    for line in req.goal.strip().splitlines():
        lines.append(f"  {line}")
    lines.append("termination_criteria:")
    for k, v in req.termination_criteria.items():
        if isinstance(v, bool):
            lines.append(f"  {k}: {str(v).lower()}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("evaluator:")
    lines.append(f"  model: {req.evaluator.get('model', 'claude-opus-4-7')}")
    lines.append(f"  max_turns: {req.evaluator.get('max_turns', 250)}")
    rubric = req.rubric_overrides if isinstance(req.rubric_overrides, list) else []
    if rubric:
        lines.append("rubric:")
        for item in rubric:
            _id = str(item.get('id', 'unknown')).replace(' ', '_')
            lines.append(f"  - id: {_id}")
            _label = str(item.get('label', '')).replace("'", "''")
            lines.append(f"    label: '{_label}'")
            lines.append(f"    scope: {item.get('scope', 'per_hypothesis')}")
            lines.append(f"    scale: \"{item.get('scale', '1-5')}\"")
            guidance = item.get("guidance", "").replace("\n", " ").strip()
            lines.append(f"    guidance: >-")
            lines.append(f"      {guidance}")
    else:
        lines.append("rubric: []")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _css() -> str:
    return (
        "* { box-sizing: border-box; }"
        "body { font-family: system-ui, -apple-system, sans-serif; margin: 0;"
        " background: #f0f2f5; color: #1a1a2e; }"
        ".page-header { background: #1a1a2e; color: #e8e8f0; padding: 0.75rem 1.5rem;"
        " font-size: 0.9rem; letter-spacing: 0.05em; }"
        ".page-header span { opacity: 0.5; }"
        ".container { display: grid; grid-template-columns: 1fr 1fr;"
        " height: calc(100vh - 40px); overflow: hidden; }"
        ".panel { padding: 1.5rem; overflow-y: auto; }"
        ".left-panel { background: #fff; border-right: 1px solid #dde; }"
        ".right-panel { background: #fafafa; }"
        "label { display: block; font-size: 0.78rem; font-weight: 600; color: #555;"
        " text-transform: uppercase; letter-spacing: 0.04em;"
        " margin-top: 1.2rem; margin-bottom: 0.3rem; }"
        "input[type=text], input[type=number], textarea, select {"
        " width: 100%; padding: 0.45rem 0.6rem; border: 1px solid #ccd;"
        " border-radius: 4px; font-size: 0.9rem; font-family: inherit;"
        " background: #fff; color: #1a1a2e; }"
        "textarea { resize: vertical; }"
        "#free-text { height: 160px; }"
        "#goal { height: 120px; }"
        "#agent-task-preview { height: 140px; font-family: monospace; font-size: 0.8rem;"
        " background: #f5f7fa; color: #444; }"
        ".criteria-section { border: 1px solid #dde; border-radius: 6px;"
        " padding: 0.75rem; margin-top: 0.3rem; }"
        ".criteria-section label { margin-top: 0.6rem; text-transform: none;"
        " font-weight: normal; font-size: 0.85rem; color: #333; }"
        ".custom-criterion-row { display: flex; align-items: center; gap: 0.5rem;"
        " margin-top: 0.5rem; flex-wrap: wrap; }"
        ".custom-criterion-row .criterion-name { flex: 1; min-width: 120px; }"
        ".custom-criterion-row .criterion-value { width: 80px; flex: none; }"
        ".custom-criterion-row .remove-btn { background: none; border: none;"
        " color: #c44; cursor: pointer; font-size: 1.1rem; padding: 0 0.3rem; }"
        ".type-label { font-size: 0.8rem; white-space: nowrap; }"
        ".btn { display: inline-block; padding: 0.5rem 1rem; border: none;"
        " border-radius: 5px; cursor: pointer; font-size: 0.9rem; font-weight: 500; }"
        ".btn-primary { background: #2563eb; color: #fff; }"
        ".btn-primary:hover { background: #1d4ed8; }"
        ".btn-primary:disabled { background: #93c5fd; cursor: default; }"
        ".btn-secondary { background: #e5e7eb; color: #374151; }"
        ".btn-secondary:hover { background: #d1d5db; }"
        ".btn-add { background: none; border: 1px dashed #aab; color: #556;"
        " font-size: 0.82rem; padding: 0.3rem 0.6rem; border-radius: 4px;"
        " cursor: pointer; margin-top: 0.5rem; }"
        ".btn-add:hover { background: #f0f2f5; }"
        ".buttons { display: flex; gap: 0.75rem; margin-top: 1.5rem; }"
        "#phase1 { height: 100%; display: flex; flex-direction: column; gap: 1rem; }"
        "#input-summary { font-size: 0.82rem; color: #666; border: 1px solid #dde;"
        " border-radius: 4px; padding: 0.5rem 0.75rem; background: #f5f7fa; }"
        ".issue { margin-bottom: 0.75rem; padding: 0.75rem; border-radius: 5px;"
        " font-size: 0.88rem; }"
        ".issue-blocking { background: #fef2f2; border: 1px solid #fca5a5; }"
        ".issue-warning { background: #fffbeb; border: 1px solid #fcd34d; }"
        ".issue-icon { margin-right: 0.3rem; }"
        ".issue strong { font-weight: 600; }"
        ".suggestion { color: #555; margin: 0.4rem 0 0.5rem; font-style: italic; }"
        ".apply-btn { font-size: 0.78rem; padding: 0.2rem 0.5rem; border: 1px solid #aab;"
        " border-radius: 3px; cursor: pointer; background: #fff; }"
        ".apply-btn:hover { background: #f0f2f5; }"
        ".no-issues { color: #16a34a; font-weight: 500; }"
        ".error-msg { color: #dc2626; font-size: 0.88rem; }"
        ".issues-header { font-weight: 600; font-size: 0.88rem; color: #555;"
        " margin-bottom: 0.5rem; }"
        "fieldset:disabled input, fieldset:disabled textarea, fieldset:disabled select,"
        " fieldset:disabled button.btn-add {"
        " opacity: 0.45; cursor: not-allowed; pointer-events: none; }"
        "fieldset:disabled { opacity: 1; }"
    )


def _js() -> str:
    return r"""
function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function collectTerminationCriteria() {
    var criteria = {
        min_hypotheses_with_multimodal_evidence: parseInt(document.getElementById('min-hypotheses').value) || 3,
        max_tool_calls: parseInt(document.getElementById('max-tool-calls').value) || 200
    };
    document.querySelectorAll('.custom-criterion-row').forEach(function(row) {
        var name = row.querySelector('.criterion-name').value.trim();
        if (!name) return;
        var quantRadio = row.querySelector('input[value="quant"]');
        if (quantRadio && quantRadio.checked) {
            criteria[name] = parseFloat(row.querySelector('.criterion-value').value) || 0;
        } else {
            criteria[name] = true;
        }
    });
    return criteria;
}

function collectRubric() {
    var items = [];
    document.querySelectorAll('.rubric-item-row').forEach(function(row) {
        var id = row.querySelector('.rubric-id').value.trim().replace(/\s+/g, '_');
        var label = row.querySelector('.rubric-label-input').value.trim();
        if (!id || !label) return;
        items.push({
            id: id,
            label: label,
            scope: row.querySelector('.rubric-scope').value,
            scale: row.querySelector('.rubric-scale').value,
            guidance: row.querySelector('.rubric-guidance-input').value.trim()
        });
    });
    return items;
}

function collectSpec() {
    return {
        task_id: document.getElementById('task-id').value.trim(),
        goal: document.getElementById('goal').value.trim(),
        termination_criteria: collectTerminationCriteria(),
        evaluator: {
            model: document.getElementById('evaluator-model').value,
            max_turns: parseInt(document.getElementById('max-turns').value) || 250
        },
        rubric_overrides: collectRubric()
    };
}

async function extractTask() {
    var freeText = document.getElementById('free-text').value.trim();
    if (!freeText) return;
    var btn = document.querySelector('#phase1 .btn-primary');
    btn.textContent = 'Extracting...';
    btn.disabled = true;
    try {
        var resp = await fetch('/extract', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({free_text: freeText})
        });
        var data = await resp.json();
        if (data.error) {
            btn.textContent = 'Generate open-world eval task';
            btn.disabled = false;
            showError(data.error);
            return;
        }
        document.getElementById('task-id').value = data.task_id || '';
        document.getElementById('goal').value = data.goal || '';
        document.getElementById('min-hypotheses').value =
            (data.termination_criteria && data.termination_criteria.min_hypotheses_with_multimodal_evidence != null)
            ? data.termination_criteria.min_hypotheses_with_multimodal_evidence : 3;
        document.getElementById('max-tool-calls').value =
            (data.termination_criteria && data.termination_criteria.max_tool_calls != null)
            ? data.termination_criteria.max_tool_calls : 200;
        document.getElementById('max-turns').value =
            (data.evaluator && data.evaluator.max_turns) ? data.evaluator.max_turns : 250;
        if (data.evaluator && data.evaluator.model) {
            document.getElementById('evaluator-model').value = data.evaluator.model;
        }
        document.getElementById('custom-criteria').innerHTML = '';
        ((data.termination_criteria && data.termination_criteria.custom) || []).forEach(function(c) {
            addCriterionRow(c);
        });
        document.getElementById('rubric-items').innerHTML = '';
        ((data.rubric) || []).forEach(function(item) {
            addRubricItem(item);
        });
        document.getElementById('input-summary').textContent =
            freeText.length > 100 ? freeText.slice(0, 100) + '...' : freeText;
        document.getElementById('phase1').style.display = 'none';
        document.getElementById('phase2').style.display = 'block';
        enableRightPanel();
        setStep(2);
    } catch(e) {
        btn.textContent = 'Generate open-world eval task';
        btn.disabled = false;
        showError('Network error: ' + e.message);
    }
}

function addCriterion() {
    addCriterionRow({name: '', quantitative: true, value: null});
}

function addCriterionRow(c) {
    var container = document.getElementById('custom-criteria');
    var uid = 'ctype_' + Date.now() + '_' + Math.random().toString(36).slice(2);
    var isQuant = c.quantitative !== false;
    var row = document.createElement('div');
    row.className = 'custom-criterion-row';
    row.innerHTML =
        '<input class="criterion-name" type="text" placeholder="criterion name" value="' +
        escHtml(c.name || '') + '">' +
        '<label class="type-label"><input type="radio" name="' + uid + '" value="quant"' +
        (isQuant ? ' checked' : '') +
        ' onchange="toggleCriterionType(this, this.closest(\'.custom-criterion-row\'))"> quant</label>' +
        '<label class="type-label"><input type="radio" name="' + uid + '" value="qual"' +
        (!isQuant ? ' checked' : '') +
        ' onchange="toggleCriterionType(this, this.closest(\'.custom-criterion-row\'))"> qual</label>' +
        '<input class="criterion-value" type="number" step="any" value="' +
        (c.value != null ? c.value : '') +
        '" style="display:' + (isQuant ? 'inline-block' : 'none') + '">' +
        '<button class="remove-btn" onclick="this.closest(\'.custom-criterion-row\').remove()"' +
        ' title="Remove">\xd7</button>';
    container.appendChild(row);
}

function toggleCriterionType(radio, row) {
    var valInput = row.querySelector('.criterion-value');
    valInput.style.display = radio.value === 'quant' ? 'inline-block' : 'none';
}

function addRubricItem(item) {
    item = item || {id:'', label:'', scope:'per_hypothesis', scale:'1-5', guidance:''};
    var container = document.getElementById('rubric-items');
    var row = document.createElement('div');
    row.className = 'rubric-item-row';
    row.style.cssText = 'border:1px solid #dde;border-radius:4px;padding:0.5rem;margin-bottom:0.5rem;';
    row.innerHTML =
        '<div style="display:grid;grid-template-columns:1fr 2fr auto;gap:0.5rem;margin-bottom:0.4rem">' +
        '<input class="rubric-id" type="text" placeholder="id (snake_case)" value="' + escHtml(item.id) + '">' +
        '<input class="rubric-label-input" type="text" placeholder="label" value="' + escHtml(item.label) + '">' +
        '<button onclick="this.closest(\'.rubric-item-row\').remove()" style="background:none;border:none;color:#c44;cursor:pointer;font-size:1.1rem;">\xd7</button>' +
        '</div>' +
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:0.4rem">' +
        '<select class="rubric-scope"><option value="per_hypothesis"' + (item.scope==='per_hypothesis'?' selected':'') + '>per_hypothesis</option>' +
        '<option value="overall"' + (item.scope==='overall'?' selected':'') + '>overall</option></select>' +
        '<select class="rubric-scale"><option value="1-5"' + (item.scale==='1-5'?' selected':'') + '>1-5</option>' +
        '<option value="yes/no/partial"' + (item.scale==='yes/no/partial'?' selected':'') + '>yes/no/partial</option></select>' +
        '</div>' +
        '<textarea class="rubric-guidance-input" rows="2" placeholder="Scoring guidance (shown verbatim to LLM and human reviewer)">' + escHtml(item.guidance) + '</textarea>';
    container.appendChild(row);
}

async function generateAndCheck() {
    var spec = collectSpec();
    var btn = document.getElementById('generate-btn');
    btn.textContent = 'Checking...';
    btn.disabled = true;
    clearErrors();
    try {
        var resp = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(spec)
        });
        var data = await resp.json();
        btn.textContent = 'Generate & Check';
        btn.disabled = false;
        if (data.error) { showError(data.error); return; }
        document.getElementById('agent-task-preview').value = data.agent_task_md || '';
        renderIssues(data.issues || []);
        var hasBlocking = (data.issues || []).some(function(i) { return i.severity === 'blocking'; });
        document.getElementById('save-btn').disabled = hasBlocking;
        document.getElementById('issues-header-label').textContent =
            (data.issues && data.issues.length) ? 'Issues found:' : '';
        if (!hasBlocking) { setStep(3); }
    } catch(e) {
        btn.textContent = 'Generate & Check';
        btn.disabled = false;
        showError('Network error: ' + e.message);
    }
}

function renderIssues(issues) {
    var section = document.getElementById('issues-section');
    if (!issues.length) {
        section.innerHTML = '<p class="no-issues">✓ No issues — ready to save</p>';
        return;
    }
    section.innerHTML = issues.map(function(iss) {
        var icon = iss.severity === 'blocking' ? '🚫' : '⚠';
        return '<div class="issue issue-' + escHtml(iss.severity) + '">' +
            '<span class="issue-icon">' + icon + '</span>' +
            '<strong>' + escHtml(iss.field) + ':</strong> ' + escHtml(iss.issue) +
            '<div class="suggestion">→ ' + escHtml(iss.suggestion) + '</div>' +
            '<button class="apply-btn" onclick="applyFix(' +
            JSON.stringify(iss.field) + ',' + JSON.stringify(iss.suggestion) +
            ')">Apply fix</button></div>';
    }).join('');
}

function applyFix(field, suggestion) {
    var fieldMap = {goal: 'goal', task_id: 'task-id', rubric_overrides: 'rubric-overrides'};
    var elId = fieldMap[field];
    if (elId) { document.getElementById(elId).value = suggestion; }
}

async function saveTask() {
    var spec = collectSpec();
    var claudeMd = document.getElementById('agent-task-preview').value;
    var btn = document.getElementById('save-btn');
    btn.textContent = 'Saving...';
    btn.disabled = true;
    clearErrors();
    try {
        var resp = await fetch('/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(Object.assign({}, spec, {agent_task_md: claudeMd}))
        });
        var data = await resp.json();
        if (data.error) {
            btn.textContent = 'Save Task';
            btn.disabled = false;
            showError(data.error);
            return;
        }
        document.body.innerHTML =
            '<div style="text-align:center;padding:4rem;font-family:system-ui,sans-serif">' +
            '<h2 style="color:#16a34a">✓ Task saved!</h2>' +
            '<p>Written to <code>' + escHtml(data.path) + '</code></p>' +
            '<p style="color:#666">You can close this window. The server will shut down automatically.</p>' +
            '</div>';
    } catch(e) {
        btn.textContent = 'Save Task';
        btn.disabled = false;
        showError('Network error: ' + e.message);
    }
}

function showError(msg) {
    var banner = document.getElementById('error-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'error-banner';
        banner.style.cssText = 'background:#fef2f2;border:1px solid #fca5a5;color:#dc2626;padding:10px 14px;border-radius:6px;font-size:0.88rem;margin-top:10px;';
        var phase1 = document.getElementById('phase1');
        var phase2 = document.getElementById('phase2');
        var target = (phase2 && phase2.style.display !== 'none') ? phase2 : phase1;
        if (target) target.appendChild(banner);
    }
    banner.textContent = 'Error: ' + msg;
    banner.style.display = 'block';
}

function clearErrors() {
    var banner = document.getElementById('error-banner');
    if (banner) banner.style.display = 'none';
    var section = document.getElementById('issues-section');
    if (section && section.querySelector('.error-msg')) { section.innerHTML = ''; }
}

function disableRightPanel() {
    var form = document.getElementById('right-panel-form');
    if (form) form.disabled = true;
}

function enableRightPanel() {
    var form = document.getElementById('right-panel-form');
    if (form) {
        form.disabled = false;
        // Brief highlight to show fields were just populated
        form.style.transition = 'background 0.4s';
        form.style.background = '#f0fdf4';
        setTimeout(function() { form.style.background = ''; }, 800);
    }
}

function setStep(n) {
    var indicator = document.getElementById('step-indicator');
    var labels = ['', 'Step 1 of 3', 'Step 2 of 3', 'Step 3 of 3'];
    if (indicator) indicator.textContent = labels[n] || ('Step ' + n);

    var instructions = [
        '',
        'Describe your evaluation task in plain language — the scientific question, any known datasets, and what a good outcome looks like. The LLM will extract structured fields.',
        'Review and edit the extracted fields on the right. When ready, click <strong>Generate &amp; Check</strong> to produce the agent task prompt.',
        'Review the agent task prompt and any issues below. Click <strong>Save Task</strong> to write files to disk.'
    ];
    var inst = document.getElementById('step-instruction');
    if (inst && instructions[n]) inst.innerHTML = instructions[n];

    var genBtn = document.getElementById('generate-btn');
    if (genBtn) {
        genBtn.className = n === 2 ? 'btn btn-primary' : 'btn btn-secondary';
    }
}

function startOver() {
    document.getElementById('phase1').style.display = 'flex';
    document.getElementById('phase2').style.display = 'none';
    disableRightPanel();
    setStep(1);
    document.getElementById('agent-task-preview').value = '';
    document.getElementById('save-btn').disabled = true;
    document.getElementById('issues-section').innerHTML = '';
    document.getElementById('issues-header-label').textContent = '';
    document.getElementById('task-id').value = '';
    document.getElementById('goal').value = '';
    document.getElementById('custom-criteria').innerHTML = '';
    document.getElementById('rubric-items').innerHTML = '';
    document.getElementById('input-summary').textContent = '';
    clearErrors();
}
"""


def _html() -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>BioAgent-Eval / New Task</title>\n"
        f"<style>{_css()}</style>\n"
        "</head>\n<body>\n"
        '<div class="page-header">BioAgent-Eval <span>/</span> New Task'
        '<span id="step-indicator" style="float:right;font-size:0.8rem;opacity:0.7">Step 1 of 3</span>'
        '</div>\n'
        '<div class="container">\n'

        '<div class="panel left-panel">\n'
        '<div id="phase1">\n'
        '<div id="step-instruction" style="font-size:0.88rem;color:#444;line-height:1.5;'
        'padding:0.6rem 0.8rem;background:#f0f4ff;border-radius:5px;border-left:3px solid #2563eb">'
        "Describe your evaluation task in plain language — the scientific question, any known "
        "datasets, and what a good outcome looks like. The LLM will extract structured fields."
        "</div>\n"
        '<textarea id="free-text" placeholder="e.g. Identify the top cell-type-specific '
        'biomarkers for lupus using single-cell RNA-seq data from PBMCs..."></textarea>\n'
        '<button class="btn btn-primary" onclick="extractTask()">'
        "Generate open-world eval task</button>\n"
        "</div>\n"
        '<div id="phase2" style="display:none">\n'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">\n'
        '<div class="issues-header" style="margin:0">Extracted from your description:</div>\n'
        '<button onclick="startOver()" style="background:none;border:none;color:#6b7280;'
        'font-size:0.82rem;cursor:pointer;text-decoration:underline">← Start over</button>\n'
        '</div>\n'
        '<div id="input-summary"></div>\n'
        '<div class="issues-header" style="margin-top:1.2rem" id="issues-header-label"></div>\n'
        '<div id="issues-section"></div>\n'
        "</div>\n"
        "</div>\n"

        '<div class="panel right-panel">\n'
        '<fieldset id="right-panel-form" disabled style="border:none;padding:0;margin:0">\n'
        '<label>task_id</label>\n'
        '<input id="task-id" type="text" placeholder="short_snake_case_slug">\n'
        '<label>goal</label>\n'
        '<textarea id="goal" placeholder="What should the agent discover or produce?'
        ' Goal-oriented, not prescriptive."></textarea>\n'
        '<label>termination_criteria</label>\n'
        '<div class="criteria-section">\n'
        '<label>min hypotheses with multimodal evidence</label>\n'
        '<input id="min-hypotheses" type="number" value="3" min="1">\n'
        '<label>max tool calls</label>\n'
        '<input id="max-tool-calls" type="number" value="200" min="1">\n'
        '<div id="custom-criteria"></div>\n'
        '<button class="btn-add" onclick="addCriterion()">+ Add criterion</button>\n'
        "</div>\n"
        '<label>evaluator model</label>\n'
        '<select id="evaluator-model">\n'
        '<option value="claude-opus-4-7">claude-opus-4-7</option>\n'
        '<option value="claude-sonnet-4-6">claude-sonnet-4-6</option>\n'
        '<option value="claude-haiku-4-5">claude-haiku-4-5</option>\n'
        "</select>\n"
        '<label>max_turns</label>\n'
        '<input id="max-turns" type="number" value="250" min="1">\n'
        '<label>rubric items</label>\n'
        '<div id="rubric-items"></div>\n'
        '<button class="btn-add" onclick="addRubricItem()">+ Add rubric item</button>\n'
        '<label>agent_task.md preview</label>\n'
        '<textarea id="agent-task-preview" readonly '
        'placeholder="Appears after Generate &amp; Check..."></textarea>\n'
        '<div class="buttons">\n'
        '<button id="generate-btn" class="btn btn-secondary" onclick="generateAndCheck()">'
        "Generate &amp; Check</button>\n"
        '<button class="btn btn-primary" id="save-btn" onclick="saveTask()" disabled>'
        "Save Task</button>\n"
        "</div>\n"
        "</fieldset>\n"
        "</div>\n"

        "</div>\n"
        f"<script>{_js()}</script>\n"
        "</body>\n</html>"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def get_index():
    return _html()


@app.post("/extract")
def post_extract(req: ExtractRequest):
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OLLAMA_MODEL"):
        return {"error": "Set ANTHROPIC_API_KEY or OLLAMA_MODEL to enable LLM features"}
    try:
        raw = _call_llm(EXTRACTION_SYSTEM, req.free_text)
        if not raw.strip():
            return {"error": "LLM returned empty response — check your API key and model access"}
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
        return json.loads(text)
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/generate")
def post_generate(req: GenerateRequest):
    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OLLAMA_MODEL"):
        return {"error": "Set ANTHROPIC_API_KEY or OLLAMA_MODEL to enable LLM features"}
    try:
        spec_text = (
            f"task_id: {req.task_id}\n"
            f"goal: {req.goal}\n"
            f"termination_criteria: {json.dumps(req.termination_criteria)}"
        )
        claude_md = _call_llm(AGENT_TASK_SYSTEM, spec_text)
        quality_input = f"spec:\n{spec_text}\n\nagent_task.md:\n{claude_md}"
        issues_raw = _call_llm(QUALITY_SYSTEM, quality_input)
        issues = json.loads(issues_raw)
        return {"agent_task_md": claude_md, "issues": issues}
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/save")
def post_save(req: SaveRequest):
    task_dir = REPO_ROOT / "tasks" / req.task_id
    if task_dir.exists():
        return {"error": f"Task already exists: tasks/{req.task_id}/"}
    task_dir.mkdir(parents=True)
    (task_dir / "task_spec.yaml").write_text(_build_yaml(req), encoding="utf-8")
    (task_dir / "agent_task.md").write_text(req.agent_task_md, encoding="utf-8")

    def _shutdown():
        time.sleep(2)
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()
    return {"status": "ok", "path": f"tasks/{req.task_id}/"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    port = 7464
    url = f"http://localhost:{port}"

    def _open_browser():
        time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    print(f"Starting BioAgent-Eval Task Proposer at {url}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
