"""Repeat reviewer scoring on fixed run ledgers to measure run-to-run variability.

The LLM judge and falsification observer give different outputs on identical input. This
module re-runs them many times on the same saved runs, records every call in a JSONL file,
and summarizes which results hold across repeats.

  python -m reviewer.variability run --repeats 10 --out OUT.jsonl RUN_DIR [RUN_DIR ...]
  python -m reviewer.variability run --repeats 10 --temperature 0 --out OUT.jsonl RUN_DIR ...
  python -m reviewer.variability summarize OUT.jsonl [--json SUMMARY.json]

`run` is resumable: units already recorded successfully are skipped, failed ones are retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from reviewer.falsification import assess_terminal_hypotheses
from reviewer.llm import DEFAULT_MODELS, LLMClient
from reviewer.prompt_variants import VARIANTS, get_variant
from reviewer.review import CONTESTED_SCORE_GAP
from reviewer.score import JUDGE_BACKEND_MODELS, JUDGE_MAX_TOKENS, score_run

# A direction, sign, or ranking counts as stable when it holds in at least this share of repeats.
STABLE_FRACTION = 0.9
# Below this many repeats, no stability verdict is given (reported as None).
MIN_REPEATS = 5

# In production the observer comes from get_llm_client(), which prefers Anthropic when its key is set.
OBSERVER_BACKEND = "anthropic"


class RecordingClient:
    """Wraps an LLM client and keeps each call's prompt hash, raw response, and reported model."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], schema: dict | None = None) -> str:
        prompt = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        start = time.monotonic()
        text = self.inner.chat(messages, schema)
        self.calls.append({
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response": text,
            "model_returned": getattr(self.inner, "last_model", None),
            "seconds": round(time.monotonic() - start, 2),
        })
        return text


def default_client_factory(component: str, backend: str, temperature: float | None):
    if component == "judge":
        return LLMClient(
            backend=backend, model=JUDGE_BACKEND_MODELS[backend],
            max_tokens=JUDGE_MAX_TOKENS, temperature=temperature,
        )
    return LLMClient(
        backend=OBSERVER_BACKEND, model=DEFAULT_MODELS[OBSERVER_BACKEND], temperature=temperature,
    )


def _code_version() -> dict:
    """Commit of the repo this code runs from, and whether tracked files differ from it."""
    repo = Path(__file__).resolve().parent.parent

    def git(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"code_commit": commit, "code_dirty": None if status is None else bool(status)}


def _arm_name(temperature: float | None, prompt_variant: str = "original") -> str:
    base = "default" if temperature is None else f"temp{temperature}"
    return base if prompt_variant == "original" else f"{base}-{prompt_variant}"


def _key(rec: dict) -> tuple:
    return (rec["run_id"], rec["component"], rec["backend"], rec["arm"], rec["repeat"])


def load_records(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _judge_result(scores) -> dict:
    return {
        "hypotheses": [
            {
                "hypothesis_id": hs.hypothesis_id,
                "candidate_gene": hs.candidate_gene,
                "criteria": [asdict(c) for c in hs.criteria],
            }
            for hs in scores.hypothesis_scores
        ],
        "overall_criteria": [asdict(c) for c in scores.overall_criteria],
    }


def _run_unit(
    run_dir, component, backend, repeat, *, temperature, arm, client_factory, prompt_variant,
) -> dict:
    rec = {
        "run_id": Path(run_dir).name, "run_dir": str(run_dir), "component": component,
        "backend": backend, "arm": arm, "temperature": temperature, "repeat": repeat,
        "prompt_variant": prompt_variant,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    client = None
    try:
        inner = client_factory(component, backend, temperature)
        rec["model_requested"] = getattr(inner, "model", None)
        client = RecordingClient(inner)
        if component == "judge":
            rec["result"] = _judge_result(
                score_run(str(run_dir), client=client, prompt_variant=prompt_variant)
            )
        else:
            rec["result"] = [
                asdict(r) for r in assess_terminal_hypotheses(str(run_dir), client, prompt_variant)
            ]
        rec["ok"] = True
    except Exception as exc:
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    calls = client.calls if client else []
    rec["calls"] = calls
    rec["models_returned"] = sorted({c["model_returned"] for c in calls if c["model_returned"]})
    return rec


def run_repeats(
    run_dirs: list[str],
    repeats: int,
    out_path: str,
    *,
    client_factory=default_client_factory,
    workers: int = 4,
    temperature: float | None = None,
    arm: str | None = None,
    components: tuple[str, ...] = ("judge", "observer"),
    backends: tuple[str, ...] = ("anthropic", "google"),
    prompt_variant: str = "original",
) -> int:
    """Run every missing (run, component, backend, repeat) unit and append one record each.

    Returns the number of units attempted in this call.
    """
    # The observer returns an empty result for a missing ledger rather than failing,
    # so a bad path would otherwise be recorded as a successful, empty unit.
    for run_dir in run_dirs:
        if not (Path(run_dir) / "ledger.json").is_file():
            raise FileNotFoundError(f"not a run directory (no ledger.json): {run_dir}")
    get_variant(prompt_variant)  # fail before any call on an unknown name
    arm = arm or _arm_name(temperature, prompt_variant)
    done = {_key(r) for r in load_records(out_path) if r.get("ok")}
    units = []
    for run_dir in run_dirs:
        for component in components:
            for backend in (backends if component == "judge" else (OBSERVER_BACKEND,)):
                for repeat in range(repeats):
                    if (Path(run_dir).name, component, backend, arm, repeat) not in done:
                        units.append((run_dir, component, backend, repeat))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    version = _code_version()

    def work(unit):
        rec = _run_unit(
            *unit, temperature=temperature, arm=arm, client_factory=client_factory,
            prompt_variant=prompt_variant,
        )
        rec.update(version)
        with lock, open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for fut in as_completed([pool.submit(work, u) for u in units]):
            rec = fut.result()
            status = "ok" if rec["ok"] else f"FAILED {rec['error']}"
            print(
                f"[variability] {rec['run_id']} {rec['component']}/{rec['backend']} "
                f"{rec['arm']} #{rec['repeat']}: {status}",
                file=sys.stderr,
            )
    return len(units)


# --- analysis ---

def _numeric(score) -> int | None:
    m = re.match(r"\s*(\d+)", str(score or ""))
    return int(m.group(1)) if m else None


def _ordering(totals: dict[str, float]) -> str:
    """'IFI44L>GZMH=SIGLEC1': descending by total, ties joined by '=' in name order."""
    groups: dict[float, list[str]] = defaultdict(list)
    for name, total in totals.items():
        groups[total].append(name)
    return ">".join("=".join(sorted(groups[t])) for t in sorted(groups, reverse=True))


def load_human_scores(run_dir: str) -> dict[str, dict[str, int]]:
    """Numeric per-hypothesis criterion scores from the run's latest human_review*.json."""
    files = sorted(Path(run_dir).glob("human_review*.json")) if Path(run_dir).is_dir() else []
    if not files:
        return {}
    per_h = json.loads(files[-1].read_text(encoding="utf-8")).get("per_hypothesis", {})
    return {
        hid: {k: v for k, v in crit.items() if isinstance(v, int) and not isinstance(v, bool)}
        for hid, crit in per_h.items()
    }


def _verdict(share: float, n: int) -> bool | None:
    return None if n < MIN_REPEATS else share >= STABLE_FRACTION


def _stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _per_hypothesis_numeric(rec: dict) -> dict[str, dict[str, int]]:
    out = {}
    for h in rec["result"]["hypotheses"]:
        nums = {c["criterion_id"]: _numeric(c["score"]) for c in h["criteria"]}
        out[h["hypothesis_id"]] = {k: v for k, v in nums.items() if v is not None}
    return out


def _all_numeric(rec: dict) -> list[int]:
    vals = [v for crit in _per_hypothesis_numeric(rec).values() for v in crit.values()]
    vals += [n for c in rec["result"]["overall_criteria"] if (n := _numeric(c["score"])) is not None]
    return vals


def _pairs(judge: list[dict]) -> dict[tuple, dict[int, dict[str, dict]]]:
    """(run_id, arm) -> repeat -> backend -> record, for pairing backends by repeat index."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in judge:
        out[(r["run_id"], r["arm"])][r["repeat"]][r["backend"]] = r
    return out


def _criterion_spread(judge: list[dict]) -> list[dict]:
    values: dict[tuple, list[int]] = defaultdict(list)
    for r in judge:
        for hid, crit in _per_hypothesis_numeric(r).items():
            for cid, v in crit.items():
                values[(r["run_id"], r["arm"], r["backend"], hid, cid)].append(v)
        for c in r["result"]["overall_criteria"]:
            if (v := _numeric(c["score"])) is not None:
                values[(r["run_id"], r["arm"], r["backend"], "(overall)", c["criterion_id"])].append(v)
    return [
        {"run_id": k[0], "arm": k[1], "backend": k[2], "hypothesis_id": k[3], "criterion_id": k[4],
         **_stats(v)}
        for k, v in sorted(values.items())
    ]


def _rankings(judge: list[dict]) -> list[dict]:
    orderings: dict[tuple, Counter] = defaultdict(Counter)
    for r in judge:
        totals = {
            (h["candidate_gene"] or h["hypothesis_id"]): sum(
                v for v in (_numeric(c["score"]) for c in h["criteria"]) if v is not None
            )
            for h in r["result"]["hypotheses"]
        }
        if len(totals) >= 2:
            orderings[(r["run_id"], r["arm"], r["backend"])][_ordering(totals)] += 1
    rows = []
    for k, counts in sorted(orderings.items()):
        n = sum(counts.values())
        top_share = counts.most_common(1)[0][1] / n
        rows.append({"run_id": k[0], "arm": k[1], "backend": k[2], "n": n,
                     "orderings": dict(counts.most_common()), "top_share": top_share,
                     "stable": _verdict(top_share, n)})
    return rows


def _leniency(judge: list[dict]) -> list[dict]:
    rows = []
    for (run_id, arm), by_repeat in sorted(_pairs(judge).items()):
        diffs = []
        for recs in by_repeat.values():
            if "anthropic" in recs and "google" in recs:
                a, g = _all_numeric(recs["anthropic"]), _all_numeric(recs["google"])
                if a and g:
                    diffs.append(statistics.fmean(g) - statistics.fmean(a))
        if diffs:
            higher = sum(d > 0 for d in diffs) / len(diffs)
            lower = sum(d < 0 for d in diffs) / len(diffs)
            rows.append({"run_id": run_id, "arm": arm, "n_pairs": len(diffs),
                         "frac_google_higher": higher, "mean_diff": statistics.fmean(diffs),
                         "stable": _verdict(max(higher, lower), len(diffs))})
    return rows


def _contested(judge: list[dict], genes: dict) -> list[dict]:
    flags: dict[tuple, list[bool]] = defaultdict(list)
    for (run_id, arm), by_repeat in _pairs(judge).items():
        for recs in by_repeat.values():
            if "anthropic" not in recs or "google" not in recs:
                continue
            a = _per_hypothesis_numeric(recs["anthropic"])
            g = _per_hypothesis_numeric(recs["google"])
            for hid in a.keys() & g.keys():
                for cid in a[hid].keys() & g[hid].keys():
                    flags[(run_id, arm, hid, cid)].append(abs(a[hid][cid] - g[hid][cid]) >= CONTESTED_SCORE_GAP)
    return [
        {"run_id": k[0], "arm": k[1], "hypothesis_id": k[2],
         "candidate_gene": genes.get((k[0], k[2])), "criterion_id": k[3],
         "n_pairs": len(v), "flag_rate": sum(v) / len(v)}
        for k, v in sorted(flags.items())
    ]


def _human_mad(judge: list[dict], human_scores: dict) -> tuple[list[dict], list[dict]]:
    mads: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in judge:
        human = human_scores.get(r["run_id"])
        if not human:
            continue
        judged = _per_hypothesis_numeric(r)
        diffs = [
            abs(judged[hid][cid] - human[hid][cid])
            for hid in judged.keys() & human.keys()
            for cid in judged[hid].keys() & human[hid].keys()
        ]
        if diffs:
            mads[(r["run_id"], r["arm"], r["backend"])][r["repeat"]] = statistics.fmean(diffs)
    per_backend = [
        {"run_id": k[0], "arm": k[1], "backend": k[2], **_stats(list(v.values()))}
        for k, v in sorted(mads.items())
    ]
    gaps = []
    for run_id, arm in sorted({(k[0], k[1]) for k in mads}):
        a, g = mads.get((run_id, arm, "anthropic"), {}), mads.get((run_id, arm, "google"), {})
        shared = a.keys() & g.keys()
        if shared:
            closer = sum(a[i] < g[i] for i in shared) / len(shared)
            gaps.append({"run_id": run_id, "arm": arm, "n_pairs": len(shared),
                         "frac_anthropic_closer": closer,
                         "stable": _verdict(max(closer, 1 - closer), len(shared))})
    return per_backend, gaps


def _observer(observer: list[dict], genes: dict) -> list[dict]:
    rows_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in observer:
        for res in r["result"]:
            if res.get("parse_failed"):
                continue
            rows_by_key[(r["run_id"], r["arm"], res["hypothesis_id"])].append(res)
    rows = []
    for k, results in sorted(rows_by_key.items()):
        obs = [x["observer_confidence"] for x in results]
        with_agent = [x for x in results if x.get("agent_confidence") is not None]
        above = sum(x["observer_confidence"] > x["agent_confidence"] for x in with_agent)
        below = sum(x["observer_confidence"] < x["agent_confidence"] for x in with_agent)
        n_agent = len(with_agent) or 1
        rows.append({
            "run_id": k[0], "arm": k[1], "hypothesis_id": k[2],
            "candidate_gene": genes.get((k[0], k[2])), **_stats(obs),
            "agent_confidence": with_agent[0]["agent_confidence"] if with_agent else None,
            "frac_observer_above_agent": above / n_agent,
            "frac_observer_below_agent": below / n_agent,
            "sign_stable": _verdict(max(above, below) / n_agent, len(with_agent)),
            "frac_conservative": sum(
                x["observer_confidence"] < x.get("llm_recomputed_confidence", x["observer_confidence"])
                for x in results
            ) / len(results),
        })
    return rows


def _health(records: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = defaultdict(
        lambda: {"attempts": 0, "failures": 0, "observer_parse_failures": 0, "models_returned": set()}
    )
    for r in records:
        g = groups[(r["component"], r["backend"], r["arm"])]
        g["attempts"] += 1
        g["failures"] += not r.get("ok")
        if r["component"] == "observer" and r.get("ok"):
            g["observer_parse_failures"] += sum(bool(x.get("parse_failed")) for x in r["result"])
        g["models_returned"].update(r.get("models_returned", []))
    return [
        {"component": k[0], "backend": k[1], "arm": k[2], "attempts": v["attempts"],
         "failures": v["failures"], "observer_parse_failures": v["observer_parse_failures"],
         "models_returned": sorted(v["models_returned"])}
        for k, v in sorted(groups.items())
    ]


def summarize(records: list[dict], human_scores: dict | None = None) -> dict:
    """Section-5 statistics from the variability plan, using the latest successful record per unit."""
    latest: dict[tuple, dict] = {}
    for r in records:
        if r.get("ok"):
            latest[_key(r)] = r
    judge = [r for r in latest.values() if r["component"] == "judge"]
    observer = [r for r in latest.values() if r["component"] == "observer"]
    human_mad, human_mad_gap = _human_mad(judge, human_scores or {})
    genes = {
        (r["run_id"], h["hypothesis_id"]): h["candidate_gene"]
        for r in judge for h in r["result"]["hypotheses"] if h.get("candidate_gene")
    }
    return {
        "health": _health(records),
        "criterion_spread": _criterion_spread(judge),
        "rankings": _rankings(judge),
        "leniency": _leniency(judge),
        "contested": _contested(judge, genes),
        "human_mad": human_mad,
        "human_mad_gap": human_mad_gap,
        "observer": _observer(observer, genes),
    }


def compare(records: list[dict], baseline_arm: str) -> dict:
    """Check each other arm's scores against the baseline arm's repeat-to-repeat range.

    A variant value outside [min, max] of the baseline's repeats on the same run, backend,
    hypothesis, and criterion is a change the baseline's own noise doesn't cover.
    """
    latest: dict[tuple, dict] = {}
    for r in records:
        if r.get("ok"):
            latest[_key(r)] = r

    judge_vals: dict[tuple, list[int]] = defaultdict(list)  # (arm, run, backend, hid, cid)
    for r in latest.values():
        if r["component"] != "judge":
            continue
        for hid, crit in _per_hypothesis_numeric(r).items():
            for cid, v in crit.items():
                judge_vals[(r["arm"], r["run_id"], r["backend"], hid, cid)].append(v)
        for cr in r["result"]["overall_criteria"]:
            if (v := _numeric(cr["score"])) is not None:
                judge_vals[(r["arm"], r["run_id"], r["backend"], "(overall)", cr["criterion_id"])].append(v)

    genes = {
        (r["run_id"], h["hypothesis_id"]): h["candidate_gene"]
        for r in latest.values() if r["component"] == "judge"
        for h in r["result"]["hypotheses"] if h.get("candidate_gene")
    }
    obs_vals: dict[tuple, list[float]] = defaultdict(list)  # (arm, run, hid)
    for r in latest.values():
        if r["component"] != "observer":
            continue
        for res in r["result"]:
            if not res.get("parse_failed"):
                obs_vals[(r["arm"], r["run_id"], res["hypothesis_id"])].append(res["observer_confidence"])

    def rows(values: dict, group_len: int) -> list[dict]:
        groups: dict[tuple, dict] = {}
        for key, vals in sorted(values.items()):
            arm, rest = key[0], key[1:]
            if arm == baseline_arm or (baseline_arm,) + rest not in values:
                continue
            base = values[(baseline_arm,) + rest]
            g = groups.setdefault((arm,) + rest[:group_len], {"n_criteria": 0, "outside": []})
            g["n_criteria"] += 1
            if min(vals) < min(base) or max(vals) > max(base):
                hid = rest[group_len] if len(rest) > group_len else None
                g["outside"].append({
                    "hypothesis_id": hid,
                    "candidate_gene": genes.get((rest[0], hid)),
                    "criterion_id": rest[group_len + 1] if len(rest) > group_len + 1 else None,
                    "baseline_range": [min(base), max(base)],
                    "variant_values": sorted(vals),
                })
        return [{"arm": k[0], "run_id": k[1], **({"backend": k[2]} if group_len == 2 else {}),
                 "n_criteria": g["n_criteria"], "n_outside": len(g["outside"]), "outside": g["outside"]}
                for k, g in groups.items()]

    return {"baseline_arm": baseline_arm, "judge": rows(judge_vals, 2), "observer": rows(obs_vals, 1)}


def format_compare(c: dict) -> str:
    lines = [f"Compared against the baseline arm {c['baseline_arm']!r}: a value counts as outside "
             f"when it falls beyond the baseline repeats' min-max range."]
    lines.append("\n=== Judge criterion scores ===")
    for r in c["judge"]:
        lines.append(f"{r['arm']:16} {r['run_id']} {r['backend']:9} "
                     f"{r['n_outside']}/{r['n_criteria']} criteria outside")
        for o in r["outside"]:
            lines.append(f"    {o['candidate_gene'] or o['hypothesis_id']} {o['criterion_id']}: "
                         f"baseline {o['baseline_range']}, "
                         f"variant {o['variant_values']}")
    lines.append("\n=== Observer confidence ===")
    for r in c["observer"]:
        lines.append(f"{r['arm']:16} {r['run_id']} {r['n_outside']}/{r['n_criteria']} hypotheses outside")
        for o in r["outside"]:
            lines.append(f"    {o['candidate_gene'] or o['hypothesis_id']}: baseline {o['baseline_range']}, "
                         f"variant {o['variant_values']}")
    return "\n".join(lines)


def _label(stable: bool | None) -> str:
    return {True: "STABLE", False: "unstable", None: f"(n<{MIN_REPEATS}, no verdict)"}[stable]


def format_summary(s: dict) -> str:
    lines: list[str] = []

    def section(title):
        lines.append(f"\n=== {title} ===")

    section("Health (attempts / failed units / observer parse failures / model versions returned)")
    for h in s["health"]:
        lines.append(f"{h['component']:8} {h['backend']:9} {h['arm']:10} "
                     f"{h['attempts']:4} / {h['failures']:<3} / {h['observer_parse_failures']:<3} "
                     f"{', '.join(h['models_returned'])}")

    section("Criterion score spread across repeats (per run, arm, backend)")
    spread: dict[tuple, list[dict]] = defaultdict(list)
    for row in s["criterion_spread"]:
        spread[(row["run_id"], row["arm"], row["backend"])].append(row)
    for k, rows in sorted(spread.items()):
        same = sum(r["min"] == r["max"] for r in rows)
        lines.append(f"{k[0]} {k[1]} {k[2]}: {len(rows)} criteria, {same} identical in every repeat, "
                     f"mean SD {statistics.fmean(r['sd'] for r in rows):.2f}, "
                     f"max range {max(r['max'] - r['min'] for r in rows)}")

    section(f"Hypothesis rankings by summed score (stable = same order in >= {STABLE_FRACTION:.0%})")
    for r in s["rankings"]:
        orders = ", ".join(f"{o} x{c}" for o, c in r["orderings"].items())
        lines.append(f"{r['run_id']} {r['arm']} {r['backend']}: {_label(r['stable'])}  {orders}")

    section("Leniency: Gemini mean score minus Claude mean score, paired by repeat")
    for r in s["leniency"]:
        lines.append(f"{r['run_id']} {r['arm']}: Gemini higher in {r['frac_google_higher']:.0%} of "
                     f"{r['n_pairs']} pairs, mean diff {r['mean_diff']:+.2f} "
                     f"{_label(r['stable'])}")

    section(f"Contested criteria (|Claude - Gemini| >= {CONTESTED_SCORE_GAP}) flag rate")
    for r in s["contested"]:
        if r["flag_rate"] > 0:
            lines.append(f"{r['run_id']} {r['arm']} {r['candidate_gene'] or r['hypothesis_id']} {r['criterion_id']}: "
                         f"{r['flag_rate']:.0%} of {r['n_pairs']}")

    if s["human_mad"]:
        section("Mean absolute difference from the human reviewer")
        for r in s["human_mad"]:
            lines.append(f"{r['run_id']} {r['arm']} {r['backend']}: {r['mean']:.2f} ± {r['sd']:.2f} "
                         f"(range {r['min']:.2f}-{r['max']:.2f}, n={r['n']})")
        for r in s["human_mad_gap"]:
            lines.append(f"{r['run_id']} {r['arm']}: Claude closer to human in "
                         f"{r['frac_anthropic_closer']:.0%} of {r['n_pairs']} pairs "
                         f"{_label(r['stable'])}")

    section("Falsification observer confidence vs. agent's final confidence")
    for r in s["observer"]:
        sign = ("observer above" if r["frac_observer_above_agent"] >= r["frac_observer_below_agent"]
                else "observer below")
        share = max(r["frac_observer_above_agent"], r["frac_observer_below_agent"])
        lines.append(f"{r['run_id']} {r['arm']} {r['candidate_gene'] or r['hypothesis_id']}: C {r['mean']:.2f} ± {r['sd']:.2f} "
                     f"vs agent {r['agent_confidence']}; {sign} in {share:.0%} "
                     f"{_label(r['sign_stable'])}; "
                     f"C below own e-value product in {r['frac_conservative']:.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run reviewer repeats on saved runs")
    run_p.add_argument("run_dirs", nargs="+")
    run_p.add_argument("--out", required=True)
    run_p.add_argument("--repeats", type=int, default=10)
    run_p.add_argument("--temperature", type=float, default=None,
                       help="sampling temperature; omit for the provider default")
    run_p.add_argument("--arm", default=None,
                       help="arm label (default: derived from temperature and prompt variant)")
    run_p.add_argument("--prompt-variant", default="original", choices=sorted(VARIANTS),
                       help="instruction wording (reviewer.prompt_variants); default is production")
    run_p.add_argument("--components", nargs="+", default=["judge", "observer"], choices=["judge", "observer"])
    run_p.add_argument("--backends", nargs="+", default=["anthropic", "google"],
                       choices=sorted(JUDGE_BACKEND_MODELS))
    run_p.add_argument("--workers", type=int, default=4)

    cmp_p = sub.add_parser("compare", help="check other arms against a baseline arm's range")
    cmp_p.add_argument("path")
    cmp_p.add_argument("--baseline", required=True, help="baseline arm, e.g. temp0.0")
    cmp_p.add_argument("--json", default=None, help="also write the comparison as JSON here")

    sum_p = sub.add_parser("summarize", help="summarize a variability JSONL file")
    sum_p.add_argument("path")
    sum_p.add_argument("--json", default=None, help="also write the full summary as JSON here")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        n = run_repeats(
            args.run_dirs, args.repeats, args.out, workers=args.workers,
            temperature=args.temperature, arm=args.arm,
            components=tuple(args.components), backends=tuple(args.backends),
            prompt_variant=args.prompt_variant,
        )
        print(f"[variability] attempted {n} units -> {args.out}", file=sys.stderr)
        return

    records = load_records(args.path)
    if args.cmd == "compare":
        result = compare(records, args.baseline)
        print(format_compare(result))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    human = {r["run_id"]: load_human_scores(r["run_dir"]) for r in records}
    summary = summarize(records, human_scores={k: v for k, v in human.items() if v})
    print(format_summary(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
