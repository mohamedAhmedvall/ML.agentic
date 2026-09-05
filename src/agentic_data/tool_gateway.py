from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


AVAILABLE_TOOLS = (
    "data.inspect_csv",
    "file.read_text",
    "file.write_text",
    "python.run",
)


class ToolGatewayError(RuntimeError):
    """Raised when a requested tool is unavailable or violates the workspace policy."""


@dataclass(frozen=True)
class ToolResult:
    tool: str
    output: dict[str, Any]


class ToolGateway:
    """Small deterministic execution gateway for ML.agentic agents.

    Named capabilities are exposed instead of arbitrary shell access. File tools resolve
    every path below the configured workspace. ``python.run`` executes in that workspace
    with isolated Python startup, but is not an OS/container security boundary.
    """

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "file.read_text": self._read_text,
            "file.write_text": self._write_text,
            "data.inspect_csv": self._inspect_csv,
            "python.run": self._run_python,
        }

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def execute(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            handler = self._tools[tool]
        except KeyError as exc:
            raise ToolGatewayError(f"unknown tool: {tool}") from exc
        return ToolResult(tool=tool, output=handler(arguments))

    def _path(self, raw: str) -> Path:
        candidate = (self.workspace / raw).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolGatewayError("path escapes workspace") from exc
        return candidate

    def _read_text(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(args["path"]))
        max_chars = min(max(int(args.get("max_chars", 50_000)), 1), 200_000)
        text = path.read_text(encoding="utf-8")
        return {
            "path": str(path.relative_to(self.workspace)),
            "content": text[:max_chars],
            "truncated": len(text) > max_chars,
        }

    def _write_text(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(args["path"]))
        content = str(args.get("content", ""))
        if len(content) > 1_000_000:
            raise ToolGatewayError("content exceeds 1 MB limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": str(path.relative_to(self.workspace)), "bytes": len(content.encode("utf-8"))}

    def _inspect_csv(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(args["path"]))
        sample_rows = min(max(int(args.get("sample_rows", 5)), 1), 20)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            total = 0
            for row in reader:
                total += 1
                if len(rows) < sample_rows:
                    rows.append(row)
        return {
            "path": str(path.relative_to(self.workspace)),
            "columns": reader.fieldnames or [],
            "row_count": total,
            "sample": rows,
        }

    def _run_python(self, args: dict[str, Any]) -> dict[str, Any]:
        code = str(args.get("code", ""))
        if not code.strip():
            raise ToolGatewayError("python code is empty")
        if len(code) > 100_000:
            raise ToolGatewayError("python code exceeds limit")
        timeout = min(max(int(args.get("timeout_seconds", 30)), 1), 120)
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-100_000:],
            "stderr": proc.stderr[-100_000:],
        }


def tool_manifest() -> str:
    return json.dumps({"tools": AVAILABLE_TOOLS})
