from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category(path: Path, mime_type: str) -> str:
    suffix = path.suffix.lower()
    if mime_type.startswith("image/"):
        return "visualization"
    if mime_type.startswith("text/") or suffix in {".md", ".pdf", ".html", ".docx"}:
        return "report"
    if suffix in {".csv", ".parquet", ".jsonl", ".xlsx", ".feather"}:
        return "data"
    if suffix in {".pkl", ".joblib", ".onnx", ".pt", ".pth", ".keras", ".h5"}:
        return "model"
    if suffix in {".py", ".ipynb", ".sql", ".r"}:
        return "code"
    if suffix in {".json", ".yaml", ".yml", ".toml"}:
        return "metadata"
    return "file"


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    run_id: str
    path: str
    name: str
    size_bytes: int
    sha256: str
    mime_type: str
    category: str
    created_by: str | None
    tool: str | None
    status: str
    created_at: str
    promoted_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArtifactRegistry:
    """Persistent registry for artifacts produced by ML.agentic runs."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).expanduser().resolve()
        self.registry_file = self.project_root / "artifacts.jsonl"
        self.promoted_dir = self.project_root / "artifacts"
        self.promoted_dir.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        run_id: str,
        workspace: str | Path,
        relative_path: str,
        *,
        created_by: str | None = None,
        tool: str | None = None,
        status: str = "generated",
    ) -> ArtifactRecord:
        workspace_path = Path(workspace).expanduser().resolve()
        path = (workspace_path / relative_path).resolve()
        try:
            path.relative_to(workspace_path)
        except ValueError as exc:
            raise ValueError("artifact must be inside the run workspace") from exc
        if not path.is_file():
            raise FileNotFoundError(f"artifact not found: {path}")

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        record = ArtifactRecord(
            id=f"art_{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            path=path.relative_to(self.project_root).as_posix(),
            name=path.name,
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
            mime_type=mime_type,
            category=_category(path, mime_type),
            created_by=created_by,
            tool=tool,
            status=status,
            created_at=_now(),
        )
        self._append(record.as_dict())
        return record

    def list(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if not self.registry_file.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self.registry_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if run_id is None or record.get("run_id") == run_id:
                records.append(record)
        return records

    def get(self, artifact_id: str) -> dict[str, Any]:
        for record in reversed(self.list()):
            if record.get("id") == artifact_id:
                return record
        raise KeyError(f"unknown artifact: {artifact_id}")

    def promote(self, artifact_id: str, name: str | None = None) -> dict[str, Any]:
        record = self.get(artifact_id)
        source = (self.project_root / record["path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"artifact file missing: {source}")
        target_name = name or source.name
        if Path(target_name).name != target_name:
            raise ValueError("promotion name must be a file name, not a path")
        target = self.promoted_dir / target_name
        if target.exists():
            target = self.promoted_dir / f"{target.stem}-{artifact_id[-6:]}{target.suffix}"
        shutil.copy2(source, target)
        promoted = dict(record)
        promoted["status"] = "promoted"
        promoted["promoted_path"] = target.relative_to(self.project_root).as_posix()
        promoted["promoted_at"] = _now()
        self._append(promoted)
        return promoted

    def _append(self, record: dict[str, Any]) -> None:
        with self.registry_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
