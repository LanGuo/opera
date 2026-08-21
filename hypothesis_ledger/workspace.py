import json
from pathlib import Path
from typing import Optional


class Workspace:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.ledger_path = self.run_dir / "ledger.json"
        self.run_log_path = self.run_dir / "run_log.jsonl"
        self.step_log_path = self.run_dir / "step_log.jsonl"
        self.final_report_path = self.run_dir / "final_report.json"
        self._init()

    def _init(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)
        (self.run_dir / "results").mkdir(exist_ok=True)
        if not self.ledger_path.exists():
            self._write_ledger({"hypotheses": {}, "tool_call_count": 0})

    def read_ledger(self) -> dict:
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def _write_ledger(self, data: dict):
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.ledger_path)

    def update_ledger(self, fn):
        data = self.read_ledger()
        fn(data)
        self._write_ledger(data)

    def increment_tool_call_count(self) -> int:
        result = [0]
        def _inc(data):
            data["tool_call_count"] = data.get("tool_call_count", 0) + 1
            result[0] = data["tool_call_count"]
        self.update_ledger(_inc)
        return result[0]

    def _append_jsonl(self, path: Path, entry: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_run(self, entry: dict):
        self._append_jsonl(self.run_log_path, entry)

    def log_step(self, entry: dict):
        self._append_jsonl(self.step_log_path, entry)

    def write_final_report(self, report: dict):
        tmp = self.final_report_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        tmp.replace(self.final_report_path)

    @property
    def task_spec_path(self) -> Path:
        return self.run_dir / "task_spec.yaml"

    def read_task_spec(self) -> Optional[dict]:
        if not self.task_spec_path.exists():
            return None
        import yaml
        return yaml.safe_load(self.task_spec_path.read_text(encoding="utf-8"))
