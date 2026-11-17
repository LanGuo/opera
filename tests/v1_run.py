"""Build runs in the format recorded before ledger.jsonl (ledger.json, step_log.jsonl,
run_log.jsonl, final_report.json), for tests of readers that still accept old runs."""
import json

from hypothesis_ledger.workspace import Workspace


class V1Workspace(Workspace):
    def __init__(self, run_dir: str):
        super().__init__(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)
        (self.run_dir / "results").mkdir(exist_ok=True)
        if not self.ledger_path.exists():
            self._write_ledger({"hypotheses": {}, "tool_call_count": 0})

    def _write_ledger(self, data: dict):
        self.ledger_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def update_ledger(self, fn):
        data = self.read_ledger()
        fn(data)
        self._write_ledger(data)

    def increment_tool_call_count(self) -> int:
        data = self.read_ledger()
        data["tool_call_count"] = data.get("tool_call_count", 0) + 1
        self._write_ledger(data)
        return data["tool_call_count"]

    def _append(self, path, entry: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_run(self, entry: dict):
        self._append(self.run_log_path, entry)

    def log_step(self, entry: dict):
        self._append(self.step_log_path, entry)

    def write_final_report(self, report: dict):
        self.final_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
