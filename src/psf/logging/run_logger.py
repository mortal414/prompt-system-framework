from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from ..engine import ExecutionResult, NodeRunRecord
from ..models import SystemDef


class RunLogger:
    def __init__(self, root: Path, system: SystemDef, *, run_id: str | None = None) -> None:
        self.root = root
        self.system = system
        self.run_id = run_id or _generate_run_id()
        self.run_dir = self.root / system.id / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.system_snapshot_path = self.run_dir / "system.yaml"
        self.input_snapshot_path = self.run_dir / "input.json"
        self.nodes_log_path = self.run_dir / "nodes.jsonl"
        self.summary_path = self.run_dir / "run.json"

    def snapshot_system(self) -> None:
        data = self.system.model_dump(by_alias=True)
        with self.system_snapshot_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)

    def snapshot_input(self, payload: dict[str, Any]) -> None:
        with self.input_snapshot_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def log_node(self, record: NodeRunRecord) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_id": record.node_id,
            "prompt": record.prompt,
            "raw_output": record.raw_output,
            "parsed_output": record.parsed_output,
            "provider": record.provider,
            "model": record.model,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.total_tokens,
            "duration_ms": record.duration_ms,
            "error": record.error,
            "provider_metadata": record.provider_metadata,
        }
        with self.nodes_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def finalize(
        self,
        result: ExecutionResult,
        *,
        config_path: Optional[Path] = None,
    ) -> None:
        summary = {
            "system_id": self.system.id,
            "system_version": self.system.version,
            "system_label": self.system.label,
            "entry_node_id": self.system.entry_node_id,
            "run_id": self.run_id,
            "status": result.status,
            "error": result.error,
            "started_at": result.context.get("runtime", {}).get("started_at"),
            "finished_at": result.context.get("runtime", {}).get("finished_at"),
            "final_output": result.final_output,
            "metrics": result.metrics,
            "models_used": result.metrics.get("models_used"),
            "artifacts": {
                "system": str(self.system_snapshot_path),
                "input": str(self.input_snapshot_path),
                "nodes_log": str(self.nodes_log_path),
            },
        }
        if config_path:
            summary["source_config"] = str(config_path)
        with self.summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)


def _generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    return f"{timestamp}_{suffix}"
