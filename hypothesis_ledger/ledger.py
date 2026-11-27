"""The run ledger: one append-only JSON-lines file holding everything a run produced.

Every entry has `entry_id`, `time`, `kind` and `written_by`. Kinds:

- written by the harness: `run_info` (start and end of the run), `tool_call` (every tool the agent
  used, with its full output), `agent_message` (text and thinking blocks), `artifact` (workspace
  files at the end of the run), `harness_message` (anything the harness told the agent during
  the run, such as the turn-budget notice);
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


_PERSISTED = "<persisted-output>"


def _full_result(value) -> str | None:
    """The whole result from a Claude Code record's `tool_use_result`: text blocks (MCP tools) or
    a dict with stdout/stderr (Bash)."""
    if isinstance(value, list):
        return _result_text(value) or None
    if isinstance(value, dict):
        parts = [value.get("stdout") or "", value.get("stderr") or ""]
        return "\n".join(p for p in parts if p) or None
    return value if isinstance(value, str) and value and not value.lstrip().startswith(_PERSISTED) else None


# An MCP result over Claude Code's token limit is not shown at all: the agent gets this message and
# the result is saved to a file, the only place it exists.
_SAVED = re.compile(r"^Error: result \(.*?\) exceeds maximum allowed tokens\. "
                    r"Output has been saved to (\S+?)\.?\s*$", re.M)
_PERSISTED_PATH = re.compile(r"Full output saved to: (\S+)")


def _saved_file(text: str, base: Path) -> str | None:
    """The result Claude Code saved to a file, read only from inside `base` (the run directory)."""
    m = _SAVED.search(text) if text.lstrip().startswith("Error: result") else (
        _PERSISTED_PATH.search(text) if text.lstrip().startswith(_PERSISTED) else None)
    if not m:
        return None
    try:
        f = Path(m.group(1)).resolve()
        if not f.is_relative_to(base.resolve()) or not f.is_file():
            return None
        raw = f.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    if f.suffix == ".json":
        try:
            val = json.loads(raw)
            return _full_result(val) or raw
        except json.JSONDecodeError:
            pass
    return raw or None


# Gemini CLI's stream carries a tool's display text (`resultDisplay`), which for some tools is only
# a summary: google_web_search gives 'Search results for "…" returned.' while the model got the
# results. What the model got is in the CLI's own session file, harness/gemini-home/.gemini/tmp/
# <project>/chats/session-*.jsonl, in the call's functionResponse.
_GEMINI_CHATS = ("harness", "gemini-home", ".gemini", "tmp", "*", "chats", "*.jsonl")
_HOOK_CONTEXT = re.compile(r"\s*<hook_context>.*?</hook_context>\s*", re.S)
_GEMINI_SUMMARY_TOOLS = {"google_web_search", "web_fetch"}   # wait for the session file


def _gemini_session_output(run_dir: Path, cli_id: str) -> str | None:
    """What the model was given for Gemini call `cli_id`, from the CLI's session file; our own
    turn-budget hook text is removed. None if the session file doesn't have it (yet)."""
    if not cli_id:
        return None
    best = None
    for f in Path(run_dir).glob("/".join(_GEMINI_CHATS)):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if cli_id not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            responses = [p.get("functionResponse") for c in rec.get("toolCalls") or []
                         if isinstance(c, dict) and c.get("id") == cli_id
                         for p in (c.get("result") or []) if isinstance(p, dict)]
            if isinstance(rec.get("content"), list):
                responses += [p.get("functionResponse") for p in rec["content"] if isinstance(p, dict)]
            for r in responses:
                if not isinstance(r, dict) or r.get("id") != cli_id:
                    continue
                resp = r.get("response")
                text = resp.get("output") if isinstance(resp, dict) and isinstance(resp.get("output"), str) \
                    else json.dumps(resp, ensure_ascii=False)
                text = _HOOK_CONTEXT.sub("\n", text).strip()
                if text and (best is None or len(text) > len(best)):
                    best = text
    return best


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
        self.awaiting: dict[str, dict] = {}     # Gemini results waiting for the CLI's session file

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
        for rec in list(self.awaiting.values()):
            out.extend(self._gemini_result(rec, final=final))
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
            results = [b for b in (content if isinstance(content, list) else [])
                       if isinstance(b, dict) and b.get("type") == "tool_result"]
            for block in results:
                text = _result_text(block.get("content"))
                head = text.lstrip()
                persisted = head.startswith((_PERSISTED, "Error: result")) and len(results) == 1
                if persisted:
                    # Claude Code replaces a large result with a 2 KB preview and a file path (the
                    # record's tool_use_result still holds the whole result), or, over its token
                    # limit, with only the path to the file it saved.
                    full = (_full_result(rec.get("tool_use_result")) if head.startswith(_PERSISTED)
                            else None) or _saved_file(text, self.path.parent)
                    persisted = bool(full)
                    text = full or text
                found = self._finish(block.get("tool_use_id"), text,
                                     "tool reported an error" if block.get("is_error") else None, None)
                for e in found:
                    if persisted:
                        e["output_persisted_by_cli"] = True
                out.extend(found)
        elif kind == "tool_use":                      # Gemini
            self.pending[rec.get("tool_id")] = self._call(rec.get("tool_name"), rec.get("parameters"),
                                                          rec.get("tool_id"), rec.get("timestamp"))
        elif kind == "tool_result":                   # Gemini
            out.extend(self._gemini_result(rec, final=False))
        elif kind == "message" and rec.get("role") == "assistant":   # Gemini text (deltas)
            text = rec.get("content") or ""
            if self.gemini_text is None:
                self.gemini_text = {"kind": "agent_message", "visibility": "visible", "text": "",
                                    "stream_key": f"msg:{self.line_no}:0"}
            self.gemini_text["text"] += text if isinstance(text, str) else json.dumps(text)
        return out

    def _gemini_result(self, rec: dict, final: bool) -> list[dict]:
        cli_id = rec.get("tool_id")
        shown = rec.get("output")
        shown = shown if isinstance(shown, str) or shown is None else json.dumps(shown, ensure_ascii=False)
        full = _gemini_session_output(self.path.parent, cli_id)
        call = self.pending.get(cli_id) or {}
        if full is None and not final and call.get("cli_tool") in _GEMINI_SUMMARY_TOOLS:
            self.awaiting[cli_id] = rec          # session file not written yet: try again later
            return []
        self.awaiting.pop(cli_id, None)
        # The display text is kept when it is the whole result: shell output (what the model got only
        # adds the CLI's wrapper) and large MCP results (the model got a shortened copy). Only an
        # empty or summary display, shorter than what the model got and not part of it, is replaced.
        s = (shown or "").strip()
        replaced = bool(full) and (not s or (s not in full and len(full) > len(s)))
        err = rec.get("error")
        error = (err.get("message") or err.get("type") or "error") if isinstance(err, dict) else (
            "tool reported an error" if rec.get("status") not in (None, "success") else None)
        found = self._finish(cli_id, full if replaced else shown, error, rec.get("timestamp"))
        for e in found:
            if isinstance(err, dict) and err.get("type"):
                e["error_type"] = err["type"]
            if replaced:
                e["output_from_cli_session"] = True
                e["output_display"] = shown
            elif full is None and call.get("cli_tool") in _GEMINI_SUMMARY_TOOLS:
                e["output_display_only"] = True
        return found

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
        if error is None and _reports_failure(output):
            error = "tool returned an error status"
        return [dict(call, output=output, error=error, finished=finished, stream_key=f"call:{cli_id}")]


_ERROR_FIELDS = {"status", "success", "error", "error_details", "message", "details", "next_steps",
                 "missing_tools", "retriable"}


def _reports_failure(output) -> bool:
    """True if a tool's output is a JSON object saying it failed.

    ToolUniverse reports failures inside a normal result: {"status": "error", ...} or
    {"success": false, ...}, sometimes wrapped as {"result": "<that JSON>"} by the MCP layer.
    Such an output with any other field (data) is a partial result, not a failure.
    """
    for _ in range(2):
        if not isinstance(output, str) or not output.lstrip().startswith("{"):
            return False
        try:
            obj = json.loads(output)
        except ValueError:
            return False
        if not isinstance(obj, dict):
            return False
        if obj.get("status") == "error" or obj.get("success") is False:
            # A partial result (e.g. LiteratureSearchTool's articles without its summary) still
            # carries data; only an output with nothing but error fields is a failure.
            return set(obj) <= _ERROR_FIELDS
        output = obj.get("result") if len(obj) == 1 else None
    return False


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

_WORD = re.compile(r"[A-Za-z0-9_.\-]+")       # keeps 'deg.py', 'gene_symbol', '218acb0f-9f2f'
_PART = re.compile(r"[A-Za-z0-9]+")            # and their parts: 'deg', 'py', '218acb0f'
_SCI = r"(?:\s*[eE]\s*[-+]?\d+|\s*[x×]\s*10\s*\^?\s*[-+−⁻]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+)?"
_SCI_REQUIRED = r"(?:\s*[eE]\s*[-+]?\d+|\s*[x×]\s*10\s*\^?\s*[-+−⁻]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
# Decimals ('0.226', '6.3e-10') and whole numbers in scientific notation ('1e-160', '5×10⁻⁸');
# plain integers are too common to identify a call.
_NUMBER = re.compile(r"-?\d+\.\d+" + _SCI + r"|(?<![\w.])-?\d+" + _SCI_REQUIRED)
# Identifiers such as rs10168266, GCST90270940, ENSG00000163687: letters then 3+ digits.
_IDENTIFIER = re.compile(r"\b[A-Za-z]{1,10}[_:-]?\d{3,}\b")
_OUTPUT_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻−", "0123456789--")
MATCH_MIN_SHARE = 0.5   # share of the words in source.match that must appear in a call's arguments


def _words(text: str) -> set[str]:
    return {w.strip(".-").lower() for w in _WORD.findall(text or "") if len(w.strip(".-")) > 1}


def _all_words(text: str) -> set[str]:
    return _words(text) | {p.lower() for p in _PART.findall(text or "") if len(p) > 1}


def _parse_number(text: str) -> tuple[float, float]:
    """('6.3×10⁻¹⁰') -> (6.3e-10, half a unit in the last written place)."""
    t = text.translate(_SUPERSCRIPT).replace(" ", "")
    m = re.match(r"(-?\d+(?:\.(\d+))?)(?:[eE]([-+]?\d+)|[x×]10\^?([-+]?\d+))?$", t)
    if not m or (m.group(2) is None and m.group(3) is None and m.group(4) is None):
        return float("nan"), 0.0
    exp = int(m.group(3) or m.group(4) or 0)
    places = len(m.group(2) or "")
    return float(m.group(1)) * 10 ** exp, 0.5 * 10 ** (exp - places)


_numbers_cache: dict[tuple, set[float]] = {}


def _output_numbers(call: dict) -> set[float]:
    key = (call.get("entry_id"), call.get("tool_call_id"), len(call.get("output") or ""))
    if key not in _numbers_cache:
        nums = set()
        for t in _OUTPUT_NUMBER.findall(call.get("output") or ""):
            try:
                nums.add(float(t))
            except ValueError:
                pass
        _numbers_cache[key] = nums
    return _numbers_cache[key]


def _rounds_to(text: str, shown: set[float]) -> bool:
    """A number as the agent wrote it ('-0.117', '6.3e-10') matches an output value that rounds
    or truncates to it (agents do both: 1.3757e-263 was written as 1.37e-263)."""
    target, tol = _parse_number(text)
    if tol <= 0:
        return False
    eps = tol * 1e-9
    for x in shown:
        if abs(x - target) <= tol + eps:                                  # rounded
            return True
        if (x >= 0) == (target >= 0) and 0 <= abs(x) - abs(target) < 2 * tol + eps:   # truncated
            return True
    return False


def link_source(source: dict, calls: list[dict], finding: str = "") -> tuple[list[str], str, list[str], list[str]]:
    """Find the recorded tool calls an evidence item came from.

    Signals, applied in turn to narrow the candidates:
      tool     the call's tool name equals source.tool;
      match    most words of source.match appear in the call's arguments;
      identifiers  identifiers in the finding (rs10168266, GCST90270940) appear in the output
               (the most of them wins; alone, at least two);
      numbers  the finding's decimal and scientific-notation numbers appear in the call's output,
               rounded or truncated (the most of them wins; with no other signal, at least two).
    Ids the agent names in source.tool_call_ids are kept only if they pass the same checks.
    Returns (tool_call_ids, link, candidate ids, signals that matched). link is exact (one call
    left after at least one signal matched), probable (several left; the most recent is linked),
    citation (no call matched, a citation given) or none. Failed calls are dropped from the
    candidates when a successful one is left.
    """
    source = source if isinstance(source, dict) else {}
    known = {c["tool_call_id"] for c in calls}
    numbers = set(_NUMBER.findall(finding or ""))

    def shows_numbers(c) -> int:
        shown = _output_numbers(c)
        return sum(1 for n in numbers if _rounds_to(n, shown))

    idents = {i.lower() for i in _IDENTIFIER.findall(finding or "")}

    def shows_idents(c) -> int:
        out = (c.get("output") or "").lower()
        return sum(1 for i in idents if re.search(r"(?<![a-z0-9])" + re.escape(i) + r"(?![a-z0-9])", out))

    # Ids the agent names are checked like any other signal: agents guess them (Gemini shake-down
    # 5 named T13-T15 for calls recorded as T15-T17). Named calls are kept only if none failed and,
    # when the finding has two or more numbers, each shows at least two of them.
    named_ids = [i for i in source.get("tool_call_ids") or [] if i in known]
    rejected = []
    if named_ids:
        named = [c for c in calls if c["tool_call_id"] in named_ids]
        if all(not c.get("error") and (len(numbers) < 2 or shows_numbers(c) >= 2)
               and (not idents or shows_idents(c) >= 1) for c in named):
            return named_ids, "exact", named_ids, ["tool_call_ids"]
        rejected = ["tool_call_ids rejected: " + ", ".join(named_ids)]

    candidates, signals = list(calls), []
    tool = str(source.get("tool") or "").strip().lower()
    if tool:
        by_name = [c for c in candidates
                   if tool in {str(c.get(k) or "").lower() for k in ("tool", "cli_tool", "via")}]
        if by_name:
            candidates, signals = by_name, signals + ["tool"]
    by_tool = list(candidates)
    words = _words(str(source.get("match") or ""))
    if words:
        def share(c):
            return len(words & _all_words(json.dumps(c.get("arguments"), ensure_ascii=False))) / len(words)
        best = max((share(c) for c in candidates), default=0)
        if best >= MATCH_MIN_SHARE:
            candidates = [c for c in candidates if share(c) == best]
            signals.append("match")
    if idents:
        # Identifiers the finding names (SNP, study, gene ids) narrow like numbers do: the calls
        # showing the most of them; alone, at least two must be shown.
        best = max((shows_idents(c) for c in candidates), default=0)
        if best >= (1 if signals else 2):
            candidates = [c for c in candidates if shows_idents(c) == best]
            signals.append("identifiers")
    if numbers:
        found = shows_numbers
        best = max((found(c) for c in candidates), default=0)
        # The words of `match` can point at the wrong call: the agent names the script it wrote
        # in one call and ran in another. If the calls the words kept show fewer than two of the
        # finding's numbers and a call they ruled out shows at least two, the numbers win. (A
        # finding that combines two calls' numbers keeps the call its words name.)
        wide = max((found(c) for c in by_tool), default=0)
        if "match" in signals and best < 2 <= wide:
            candidates, best = by_tool, wide
            signals.remove("match")
        # Alone, numbers must match at least two values: one common value (e.g. -0.50) is
        # too easily found in an unrelated output.
        if best >= (min(2, len(numbers)) if signals else 2):
            candidates = [c for c in candidates if found(c) == best]
            signals.append("numbers")
    if signals and candidates:
        # A call that failed produced no finding; keep it only if nothing else is left.
        succeeded = [c for c in candidates if not c.get("error")]
        candidates = succeeded or candidates
        ids = [c["tool_call_id"] for c in candidates]
        if len(ids) == 1:
            return ids, "exact", ids, signals + rejected
        return [ids[-1]], "probable", ids, signals + rejected
    if source.get("citation"):
        return [], "citation", [], rejected
    return [], "none", [], rejected


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
