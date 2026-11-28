"""bioagent-check-task: rubric inputs against what the agent prompt asks for."""
import yaml

from reviewer.check_task import CONFIRM_KEY, confirm, render, review


def _task(tmp_path, prompt, rubric):
    (tmp_path / "agent_task.md").write_text(prompt)
    (tmp_path / "task_spec.yaml").write_text("# comment kept\n" + yaml.safe_dump({"task_id": "t", "rubric": rubric}))
    return tmp_path


def test_flags_inputs_the_prompt_never_asks_for(tmp_path):
    t = _task(tmp_path, "Analyse the data.", [
        {"id": "interview", "scope": "overall", "pass": "run", "inputs": ["human_input"]},
        {"id": "viz", "scope": "overall", "pass": "run", "inputs": [{"artifacts": "results/*.png"}]},
        {"id": "design", "scope": "per_hypothesis", "pass": "evidence", "inputs": ["validation_plan"]},
        {"id": "plain", "scope": "per_hypothesis"}])
    r = review(t)
    text = " ".join(r["problems"])
    assert "interview: needs human_input" in text and "viz: needs artifacts 'results/*.png'" in text
    assert not any(p.startswith("design") for p in r["problems"])   # ledger section asks per hypothesis
    assert r["warnings"] and "plain" in r["warnings"][0]
    assert "!! human_input" in render(r)


def test_prompt_that_asks_passes_and_confirm_appends(tmp_path):
    t = _task(tmp_path, "First call request_human_input with questions. Save plots as results/fig.png.", [
        {"id": "interview", "scope": "overall", "pass": "run", "inputs": ["human_input"]},
        {"id": "viz", "scope": "overall", "pass": "run", "inputs": [{"artifacts": "results/*.png"}]}])
    r = review(t)
    assert r["problems"] == [] and r["confirmed"] is None
    confirm(t, r)
    confirm(t, review(t))                      # re-confirming replaces, doesn't duplicate
    text = (t / "task_spec.yaml").read_text()
    assert text.startswith("# comment kept") and text.count(CONFIRM_KEY) == 1
    assert review(t)["confirmed"]["criteria_hash"] == r["criteria_hash"]


def test_human_input_ask_must_be_in_the_task_text_and_may_wrap(tmp_path):
    rubric = [{"id": "interview", "scope": "overall", "pass": "run", "inputs": ["human_input"]}]
    wrapped = _task(tmp_path, "Before any analysis, ask the human expert at\nleast two probing questions.", rubric)
    assert not review(wrapped)["problems"]
    # The generated ledger section explains request_human_input but is not an ask by itself.
    (tmp_path / "agent_task.md").write_text("Analyse the data.")
    assert any(p.startswith("interview: needs human_input") for p in review(tmp_path)["problems"])
