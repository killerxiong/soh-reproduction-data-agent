from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


class AgentTrace:
    """Append-only JSONL trace for the full Agent run."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        stage: str,
        status: str,
        message: str = "",
        tool: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            "status": status,
            "tool": tool,
            "message": message,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "error": error,
            "extra": extra or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def start(self, stage: str, message: str = "", tool: Optional[str] = None, inputs: Optional[Dict[str, Any]] = None) -> None:
        self.log(stage=stage, status="start", message=message, tool=tool, inputs=inputs)

    def success(self, stage: str, message: str = "", tool: Optional[str] = None, outputs: Optional[Dict[str, Any]] = None) -> None:
        self.log(stage=stage, status="success", message=message, tool=tool, outputs=outputs)

    def fail(self, stage: str, exc: Exception, tool: Optional[str] = None) -> None:
        self.log(
            stage=stage,
            status="failed",
            tool=tool,
            error=str(exc),
            extra={"traceback": traceback.format_exc()},
        )
