"""The run ledger: one append-only JSON-lines file holding everything a run produced.

Every entry has `entry_id`, `time`, `kind` and `written_by`. Kinds:

- written by the harness: `run_info` (start and end of the run), `tool_call` (every tool the agent
  used, with its full output), `agent_message` (text and thinking blocks), `artifact` (workspace
  files at the end of the run);
- written by the agent through the ledger tools: `hypothesis`, `note`, `human_input`,
  `conclusion`.

Entries are never edited. The ledger server and the harness both append, so every append happens
under an exclusive lock on the file. Tool calls are copied from the CLI's output stream by
`Ledger.sync_stream`, which both writers call: the harness on every poll, the ledger server before
it links evidence to tool calls, so a call is always in the ledger before evidence that cites it.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LEDGER_FILE = "ledger.jsonl"
STREAM_FILE = "transcript.jsonl"
LEDGER_SERVER = "hypothesis-ledger"

STATUSES = ("active", "supported", "weakened", "refuted", "abandoned")
LIVE_STATUSES = ("active", "supported", "weakened")
NOTE_KINDS = ("interpretation", "plan", "debug", "other")
RELATIONS = ("supports", "contradicts", "context")
ARTIFACT_INLINE_BYTES = 200_000
ARTIFACT_HASH_MAX_BYTES = 2_000_000_000


class LedgerError(ValueError):
    """A ledger call the agent can fix; the message says how."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- CLI stream → tool_call / agent_message entries ---------------------------------------------

_CLAUDE_MCP = re.compile(r"^mcp__(?P<server>[^_].*?)__(?P<tool>.+)$")
_GEMINI_MCP = re.compile(r"^mcp_(?P<server>[a-z0-9-]+)_(?P<tool>.+)$")
# Tools that run another tool named in their arguments (ToolUniverse).
_DISPATCH_TOOLS = {"execute_tool"}


def split_tool_name(cli_name: str) -> tuple[str | None, str]:
    """('tooluniverse', 'execute_tool') from 'mcp__tooluniverse__execute_tool' or
    'mcp_tooluniverse_execute_tool'; (None, 'Bash') for built-in tools."""
    for pattern in (_CLAUDE_MCP, _GEMINI_MCP):
        m = pattern.match(cli_name or "")
        if m:
            return m.group("server"), m.group("tool")
    return None, cli_name or ""


def _result_text(content) -> str:
    """Claude tool_result content: a string, or a list of blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(parts)


class StreamParser:
    """Turns complete lines of a Claude Code or Gemini CLI stream-json file into ledger entries.

    Each entry carries a `stream_key` that is the same in every process that parses the same
    stream, so two writers can't record one call twice. State is per process.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.line_no = 0
        self.pending: dict[str, dict] = {}      # cli call id -> call without a result yet
        self.gemini_text: dict | None = None    # assistant text deltas being joined

    def read(self, final: bool = False) -> list[dict]:
        out: list[dict] = []
        if self.path.exists():
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
            end = data.rfind(b"\n")
            if final and data and not data.endswith(b"\n"):
                end = len(data) - 1
            if end >= 0:
                self.offset += end + 1
                for raw in data[: end + 1].splitlines():
                    self.line_no += 1
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    out.extend(self._record(rec))
        if final:
            out.extend(self._flush_text())
            for cli_id, call in self.pending.items():
                out.append(dict(call, output=None, error="no result recorded (run ended)",
                                stream_key=f"call:{cli_id}"))
            self.pending = {}
        return out

    def _flush_text(self) -> list[dict]:
        t, self.gemini_text = self.gemini_text, None
        return [t] if t else []

    def _record(self, rec: dict) -> list[dict]:
        kind = rec.get("type")
        out: list[dict] = []
        if not (kind == "message" and rec.get("role") == "assistant"):
            out.extend(self._flush_text())
        if kind == "assistant":                       # Claude: one content block per record
            content = (rec.get("message") or {}).get("content") or []
            for i, block in enumerate(content if isinstance(content, list) else []):
                btype = block.get("type") if isinstance(block, dict) else None
                key = f"msg:{self.line_no}:{i}"
                if btype == "text" and block.get("text", "").strip():
                    out.append({"kind": "agent_message", "visibility": "visible",
                                "text": block["text"], "stream_key": key})
                elif btype == "thinking" and block.get("thinking", "").strip():
                    out.append({"kind": "agent_message", "visibility": "thinking",
                                "text": block["thinking"], "stream_key": key})
                elif btype == "tool_use":
                    self.pending[block.get("id")] = self._call(block.get("name"), block.get("input"),
                                                               block.get("id"), None)
        elif kind == "user":                          # Claude: tool results
            content = (rec.get("message") or {}).get("content") or []
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out.extend(self._finish(block.get("tool_use_id"), _result_text(block.get("content")),
                                            "tool reported an error" if block.get("is_error") else None,
                                            None))
        elif kind == "tool_use":                      # Gemini
            self.pending[rec.get("tool_id")] = self._call(rec.get("tool_name"), rec.get("parameters"),
                                                          rec.get("tool_id"), rec.get("timestamp"))
        elif kind == "tool_result":                   # Gemini
            err = rec.get("error")
            error = (err.get("message") or err.get("type") or "error") if isinstance(err, dict) else (
                "tool reported an error" if rec.get("status") not in (None, "success") else None)
            found = self._finish(rec.get("tool_id"), rec.get("output"), error, rec.get("timestamp"))
            if isinstance(err, dict) and err.get("type"):
                for e in found:
                    e["error_type"] = err["type"]
            out.extend(found)
        elif kind == "message" and rec.get("role") == "assistant":   # Gemini text (deltas)
            text = rec.get("content") or ""
            if self.gemini_text is None:
                self.gemini_text = {"kind": "agent_message", "visibility": "visible", "text": "",
                                    "stream_key": f"msg:{self.line_no}:0"}
            self.gemini_text["text"] += text if isinstance(text, str) else json.dumps(text)
        return out

    def _call(self, cli_name, arguments, cli_id, started) -> dict:
        server, tool = split_tool_name(cli_name or "")
        arguments = arguments if isinstance(arguments, dict) else {"value": arguments}
        call = {"kind": "tool_call", "cli_call_id": cli_id, "cli_tool": cli_name, "server": server,
                "tool": tool, "arguments": arguments, "started": started}
        if tool in _DISPATCH_TOOLS and isinstance(arguments.get("tool_name"), str):
            call["tool"], call["via"] = arguments["tool_name"], tool
        return call

    def _finish(self, cli_id, output, error, finished) -> list[dict]:
        call = self.pending.pop(cli_id, None)
        if call is None or call.get("server") == LEDGER_SERVER:
            # The ledger tools' own calls are recorded as the entries they write.
            return []
        if output is not None and not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        return [dict(call, output=output, error=error, finished=finished, stream_key=f"call:{cli_id}")]


# --- The ledger ---------------------------------------------------------------------------------

class Ledger:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / LEDGER_FILE
        self._entries: list[dict] = []
        self._offset = 0
        self._stream_keys: set[str] = set()
        self._stream = StreamParser(self.run_dir / STREAM_FILE)

    # reading

    def exists(self) -> bool:
        return self.path.exists()

    def entries(self, kind: str | None = None) -> list[dict]:
        self._refresh()
        return [e for e in self._entries if kind is None or e.get("kind") == kind]

    def _refresh(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        end = data.rfind(b"\n")
        if end < 0:
            return
        for raw in data[: end + 1].splitlines():
            if raw.strip():
                e = json.loads(raw)
                self._entries.append(e)
                if e.get("stream_key"):
                    self._stream_keys.add(e["stream_key"])
        self._offset += end + 1

    # writing

    @contextmanager
    def _locked(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                self._refresh()
                yield f
            finally:
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)

    def _write(self, f, entries: list[dict], written_by: str) -> list[dict]:
        written = []
        for e in entries:
            full = {"entry_id": f"L{len(self._entries) + len(written) + 1:05d}", "time": now(),
                    "kind": e["kind"], "written_by": written_by,
                    **{k: v for k, v in e.items() if k != "kind"}}
            f.write(json.dumps(full, ensure_ascii=False) + "\n")
            written.append(full)
        f.flush()
        return written

    def append(self, entries: list[dict], written_by: str) -> list[dict]:
        with self._locked() as f:
            return self._write(f, entries, written_by)

    def transact(self, build, written_by: str) -> list[dict]:
        """Run build(ids) under the lock and append the entries it returns.

        build sees the ledger as it is at that moment (self.entries()), so ids it hands out and
        checks it makes can't race with another writer.
        """
        with self._locked() as f:
            entries = build(IdCounter(self._entries))
            return self._write(f, entries, written_by)

    def sync_stream(self, final: bool = False) -> int:
        """Copy tool calls and agent messages from the CLI stream that aren't in the ledger yet."""
        new = self._stream.read(final=final)
        if not new:
            return 0
        with self._locked() as f:
            fresh = [e for e in new if e["stream_key"] not in self._stream_keys]
            ids = IdCounter(self._entries)
            for e in fresh:
                if e["kind"] == "tool_call":
                    e["tool_call_id"] = ids.next("T")
            self._write(f, fresh, "harness")
            return len(fresh)

    # views used by the server and readers

    def run_info(self, phase: str = "start") -> dict | None:
        found = [e for e in self.entries("run_info") if e.get("phase") == phase]
        return found[-1] if found else None

    def task_rules(self) -> dict:
        """Hypothesis attributes and constraints declared by the task (empty if none)."""
        info = self.run_info("start")
        if info and isinstance(info.get("task_rules"), dict):
            return info["task_rules"]
        spec = self.run_dir / "task_spec.yaml"
        if spec.exists():
            import yaml
            return task_rules_from_spec(yaml.safe_load(spec.read_text(encoding="utf-8")) or {})
        return {"hypothesis_attributes": [], "hypothesis_constraints": []}

    def hypothesis_entries(self, hypothesis_id: str | None = None) -> list[dict]:
        return [e for e in self.entries("hypothesis")
                if hypothesis_id is None or e.get("hypothesis_id") == hypothesis_id]

    def hypotheses(self) -> dict[str, list[dict]]:
        """hypothesis_id -> its entries in order."""
        out: dict[str, list[dict]] = {}
        for e in self.entries("hypothesis"):
            out.setdefault(e["hypothesis_id"], []).append(e)
        return out

    def evidence_items(self) -> dict[str, dict]:
        """evidence_id -> the item as first recorded (with the hypothesis and entry it came from)."""
        out = {}
        for e in self.entries("hypothesis"):
            for item in e.get("evidence", []):
                if "evidence_id" in item:
                    out[item["evidence_id"]] = dict(item, recorded_in=e["entry_id"],
                                                    recorded_for=e["hypothesis_id"])
        return out

    def tool_calls(self) -> dict[str, dict]:
        return {e["tool_call_id"]: e for e in self.entries("tool_call")}

    def conclusion(self) -> dict | None:
        found = self.entries("conclusion")
        return found[-1] if found else None


class IdCounter:
    """Hands out H1, E1, T1, … continuing from what the ledger already holds."""

    def __init__(self, entries: list[dict]):
        self.n = {"H": 0, "E": 0, "T": 0}
        for e in entries:
            if e.get("kind") == "hypothesis" and e.get("action") == "create":
                self.n["H"] += 1
            if e.get("kind") == "hypothesis":
                self.n["E"] += sum(1 for item in e.get("evidence", []) if "evidence_id" in item)
            if e.get("kind") == "tool_call":
                self.n["T"] += 1

    def next(self, prefix: str) -> str:
        self.n[prefix] += 1
        return f"{prefix}{self.n[prefix]}"


def task_rules_from_spec(spec: dict) -> dict:
    return {"hypothesis_attributes": list(spec.get("hypothesis_attributes") or []),
            "hypothesis_constraints": list(spec.get("hypothesis_constraints") or [])}


# --- Linking evidence to tool calls (at write time) ---------------------------------------------

def link_source(source: dict, calls: list[dict]) -> tuple[list[str], str, list[str]]:
    """Find the recorded tool calls an evidence item's `source` names.

    Returns (tool_call_ids, link, candidate ids). link is exact (one call matched), probable
    (several matched; the most recent is linked), citation (no tool, a citation given) or none.
    """
    source = source if isinstance(source, dict) else {}
    named_ids = [i for i in source.get("tool_call_ids") or [] if isinstance(i, str)]
    known = {c["tool_call_id"] for c in calls}
    if named_ids:
        good = [i for i in named_ids if i in known]
        if good:
            return good, "exact", good
    tool = str(source.get("tool") or "").strip().lower()
    match = str(source.get("match") or "").strip().lower()
    if tool or match:
        def tool_ok(c):
            names = {str(c.get(k) or "").lower() for k in ("tool", "cli_tool", "via")}
            return not tool or tool in names
        def match_ok(c):
            return not match or match in json.dumps(c.get("arguments"), ensure_ascii=False).lower()
        found = [c["tool_call_id"] for c in calls if tool_ok(c) and match_ok(c)]
        if not found and tool and match:
            # The agent may name the tool loosely; the argument text is the stronger signal.
            found = [c["tool_call_id"] for c in calls if match_ok(c)]
            if found:
                return [found[-1]], "probable", found
        if len(found) == 1:
            return found, "exact", found
        if found:
            return [found[-1]], "probable", found
    if source.get("citation"):
        return [], "citation", []
    return [], "none", []


# --- Workspace files at the end of a run --------------------------------------------------------

def artifact_entries(root: Path, skip: tuple[str, ...] = (), prefix: str = "") -> list[dict]:
    """One `artifact` entry per file under root: path, size, sha256, and text content if small."""
    root = Path(root)
    out = []
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if not p.is_file() or any(rel == s or rel.startswith(s.rstrip("/") + "/") for s in skip):
            continue
        size = p.stat().st_size
        entry = {"kind": "artifact", "path": prefix + rel, "bytes": size}
        if size <= ARTIFACT_HASH_MAX_BYTES:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            entry["sha256"] = h.hexdigest()
        if size <= ARTIFACT_INLINE_BYTES:
            try:
                entry["content"] = p.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                pass
        out.append(entry)
    return out
