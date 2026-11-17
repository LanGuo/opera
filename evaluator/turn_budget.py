"""Turn-budget warning, run by the CLI as a hook after every tool call (matched profile).

Both CLIs stop the agent at the turn cap without warning, so an agent mid-analysis never calls
declare_done and the run ends with no ranked hypotheses or validation plans. This hook counts the
agent's model responses so far in the run's transcript.jsonl and, once `warn_at` or fewer remain,
adds a notice to the tool result the model reads next (Claude Code PostToolUse / Gemini CLI
AfterTool `additionalContext`). Each notice is also written to the ledger as a `harness_message`
entry, once per model response, so reviewers see exactly what the agent was told.

  python -m evaluator.turn_budget --harness claude --cap 30 --warn-at 5   (EVAL_RUN_DIR set)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from hypothesis_ledger.ledger import STREAM_FILE, Ledger
from reviewer.log_replay import gemini_response_count

HOOK_EVENTS = {"claude": "PostToolUse", "gemini": "AfterTool"}


def count_responses(transcript: Path, harness: str) -> int:
    """Main-agent model responses in the CLI's stream-json output so far (what the cap counts)."""
    recs = []
    try:
        lines = transcript.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 0
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # the CLI may be mid-write on the last line
        if isinstance(r, dict):
            recs.append(r)
    if harness == "gemini":
        return gemini_response_count(recs)
    return len({(r.get("message") or {}).get("id") for r in recs
                if r.get("type") == "assistant" and not r.get("parent_tool_use_id")
                and (r.get("message") or {}).get("id")})


def notice(left: int, cap: int) -> str:
    turns = "1 turn" if left == 1 else f"{left} turns"
    return (f"Turn budget: {turns} left of {cap}, counting your next response; each response uses "
            "one turn. The run stops at the cap. A run that stops before `declare_done` has no "
            "ranked hypotheses or validation experiments. Record any remaining evidence with "
            "`update_hypothesis`, then call `declare_done` before the turns run out.")


def check(run_dir: Path, harness: str, cap: int, warn_at: int) -> str | None:
    """The notice for this point in the run, or None. Writes it to the ledger once per response."""
    done = count_responses(run_dir / STREAM_FILE, harness)
    left = cap - done
    if warn_at <= 0 or left < 1 or left > warn_at:
        return None
    text = notice(left, cap)
    ledger = Ledger(run_dir)

    def build(_ids):
        told = any(e.get("turn") == done for e in ledger.entries("harness_message"))
        return [] if told else [{"kind": "harness_message", "purpose": "turn_budget", "turn": done,
                                 "turns_left": left, "turn_cap": cap, "text": text}]

    ledger.transact(build, "harness")
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--harness", choices=sorted(HOOK_EVENTS), required=True)
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--warn-at", type=int, required=True)
    a = p.parse_args(argv)
    sys.stdin.read()  # the hook payload; the count comes from our own transcript
    run_dir = os.environ.get("EVAL_RUN_DIR")
    try:
        text = check(Path(run_dir), a.harness, a.cap, a.warn_at) if run_dir else None
    except Exception as exc:  # never break the agent's tool call
        print(f"turn_budget: {exc!r}", file=sys.stderr)
        return 0
    if text:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": HOOK_EVENTS[a.harness],
                                                 "additionalContext": text}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
