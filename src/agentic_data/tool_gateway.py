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


class DockerToolGateway(ToolGateway):
    """Python execution for the local web runner; never falls back to host Python."""

    def _run_python(self, args: dict[str, Any]) -> dict[str, Any]:
        import os
        import shutil
        import uuid
        code = str(args.get('code', ''))
        if not code.strip() or len(code) > 100_000:
            raise ToolGatewayError('Python code must contain 1 to 100000 characters')
        if not shutil.which('docker'):
            raise ToolGatewayError('Docker is required for Python execution from the dashboard')
        timeout = min(max(int(args.get('timeout_seconds', 30)), 1), 120)
        name = 'ml-agentic-' + uuid.uuid4().hex
        # Output files live on the bounded container tmpfs, never in host memory.
        wrapper = '''import json,subprocess,sys,tempfile
with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
 p=subprocess.run([sys.executable,'-I','-c',sys.stdin.read()],stdout=out,stderr=err)
 def tail(f):
  f.seek(0,2); f.seek(max(0,f.tell()-100000)); return f.read().decode('utf-8','replace')
 print(json.dumps(dict(returncode=p.returncode,stdout=tail(out),stderr=tail(err))))
'''
        command = ['docker', 'run', '--rm', '--pull=never', '--name', name, '-i',
                   '--network=none', '--read-only', '--cap-drop=ALL',
                   '--security-opt=no-new-privileges', '--memory=512m', '--cpus=1',
                   '--pids-limit=64', '--ulimit=fsize=16777216:16777216',
                   '--tmpfs=/tmp:rw,nosuid,noexec,size=64m',
                   *(['--user', f'{os.getuid()}:{os.getgid()}'] if hasattr(os, 'getuid') else []),
                   '--mount', f'type=bind,source={self.workspace},target=/workspace',
                   '--workdir=/workspace', os.environ.get('ML_AGENTIC_PYTHON_IMAGE', 'python:3.12-slim'),
                   'python', '-I', '-c', wrapper]
        try:
            result = subprocess.run(command, input=code, text=True, capture_output=True, timeout=timeout)
            if result.returncode:
                raise ToolGatewayError('Docker execution failed: ' + result.stderr[-2000:])
            return json.loads(result.stdout)
        except subprocess.TimeoutExpired as exc:
            raise ToolGatewayError('Python execution timed out') from exc
        finally:
            subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=10, check=False)
