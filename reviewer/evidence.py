"""Shared reading of ledger evidence for the judge (score.py) and the observer (falsification.py).

Evidence items in the v1 ledger have no fixed shape: plain strings, or dicts whose keys vary by
agent and run (type/modality, description/detail/finding, source, strength, numbers). Readers that
picked one key left items blank. And update_hypothesis replaces the whole evidence list, so items
an agent omitted from a later update vanish from the final `evidence` field. These helpers read
every key and recover every item ever recorded.
"""
import json
import re

# Keys tried in order for an item's kind and its main text; first non-empty wins.
_TYPE_KEYS = ("type", "modality", "kind", "evidence_type")
_TEXT_KEYS = ("description", "detail", "finding", "text", "summary", "result")


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def evidence_parts(item, default_type: str = "unknown") -> tuple[str, str]:
    """(kind, text) for one evidence item.

    text is the first non-empty text-like field followed by every other non-empty field as
    `key: value`, so an item with no description (e.g. {"source": "GWAS", "strength": "moderate"})
    still renders its content instead of an empty string.
    """
    if isinstance(item, str):
        return default_type, item.strip()
    if not isinstance(item, dict):
        return default_type, _clean(item)
    kind = next((_clean(item[k]) for k in _TYPE_KEYS if _clean(item.get(k))), default_type)
    main_key = next((k for k in _TEXT_KEYS if _clean(item.get(k))), None)
    used = {main_key} | {k for k in _TYPE_KEYS if _clean(item.get(k)) == kind}
    extras = [f"{k}: {_clean(v)}" for k, v in item.items() if k not in used and _clean(v)]
    main = _clean(item[main_key]) if main_key else ""
    return kind, "; ".join(x for x in [main, *extras] if x)


def _key(item) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False) if not isinstance(item, str) else item


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9.~\-]*", text.lower()))


# A dropped item counts as a reworded version of a final item when both have the same kind and
# share at least this share of the smaller item's tokens. Rewordings seen in saved runs overlap
# 0.74-0.81; without this check they would be shown twice and inflate the item count.
SUPERSEDED_OVERLAP = 0.6


def _superseded_by_final(item, final: list) -> bool:
    kind, text = evidence_parts(item)
    toks = _tokens(text)
    if not toks:
        return False
    for f in final:
        f_kind, f_text = evidence_parts(f)
        if f_kind.lower() != kind.lower():
            continue
        f_toks = _tokens(f_text)
        if f_toks and len(toks & f_toks) / min(len(toks), len(f_toks)) >= SUPERSEDED_OVERLAP:
            return True
    return False


FINAL, DROPPED, SUPERSEDED = "final", "dropped", "superseded"


def all_evidence(hypothesis: dict) -> list[tuple[object, str]]:
    """Every evidence item ever recorded for a hypothesis, in first-seen order, with its state.

    update_hypothesis replaces the whole evidence list, so an item can be missing from the final
    `evidence` field for two reasons the v1 ledger doesn't distinguish:
      - "superseded": the agent resent it reworded (same kind, mostly the same content as a final item);
      - "dropped":    it is simply gone — omitted from a later list, deliberately or not.
    Items still in the final list are "final".
    """
    final = hypothesis.get("evidence") or []
    seen: dict[str, object] = {}
    for update in hypothesis.get("update_history") or []:
        changes = update.get("changes") or {}
        for side in ("from", "to"):
            for item in (changes.get(side) or {}).get("evidence") or []:
                seen.setdefault(_key(item), item)
    for item in final:
        seen.setdefault(_key(item), item)
    final_keys = {_key(i) for i in final}
    out = []
    for k, item in seen.items():
        if k in final_keys:
            out.append((item, FINAL))
        else:
            out.append((item, SUPERSEDED if _superseded_by_final(item, final) else DROPPED))
    return out


def evidence_for_judging(hypothesis: dict) -> list[tuple[object, str]]:
    """all_evidence() without superseded rewordings (final and dropped items only)."""
    return [(i, s) for i, s in all_evidence(hypothesis) if s != SUPERSEDED]


def format_evidence_list(hypothesis: dict) -> str:
    """Numbered plain-text rendering of evidence_for_judging(), marking dropped items."""
    every = all_evidence(hypothesis)
    items = [(i, s) for i, s in every if s != SUPERSEDED]
    if not items:
        return "(no evidence recorded)"
    lines = []
    for n, (item, state) in enumerate(items, 1):
        kind, text = evidence_parts(item)
        flag = " [recorded earlier; not in the agent's final evidence list]" if state == DROPPED else ""
        lines.append(f"{n}. [{kind}] {text or '(empty)'}{flag}")
    dropped = sum(1 for _, s in items if s == DROPPED)
    if dropped:
        lines.append(f"({dropped} of {len(items)} items were dropped from the final list by a later update.)")
    return "\n".join(lines)
