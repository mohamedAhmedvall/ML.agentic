from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import RuntimeEvent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-").lower()
    return cleaned or "project"


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    root: Path

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def events_file(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def metadata_file(self) -> Path:
        return self.root / "project.json"


class ProjectStore:
    def __init__(self, base_dir: str | Path = ".ml-agentic/projects"):
        self.base_dir = Path(base_dir)

    def create(self, name: str, path: str | Path | None = None) -> Project:
        root = Path(path) if path else self.base_dir / f"{_slug(name)}-{uuid.uuid4().hex[:8]}"
        root = root.expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"project directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        project = Project(id=f"proj_{uuid.uuid4().hex[:12]}", name=name.strip(), root=root)
        for directory in (project.datasets_dir, project.runs_dir, project.artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "id": project.id,
            "name": project.name,
            "created_at": _now(),
            "version": 1,
        }
        project.metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.append_event(project, "project.created", {"name": project.name})
        return project

    def open(self, path: str | Path) -> Project:
        root = Path(path).expanduser().resolve()
        metadata_file = root / "project.json"
        if not metadata_file.is_file():
            raise FileNotFoundError(f"not an ML.agentic project: {root}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        return Project(id=metadata["id"], name=metadata["name"], root=root)

    def add_dataset(self, project: Project, source: str | Path) -> Path:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"dataset not found: {source_path}")
        target = project.datasets_dir / source_path.name
        if target.exists():
            target = project.datasets_dir / f"{source_path.stem}-{uuid.uuid4().hex[:6]}{source_path.suffix}"
        shutil.copy2(source_path, target)
        self.append_event(project, "dataset.added", {"path": str(target.relative_to(project.root))})
        return target

    def append_event(self, project: Project, event_type: str, payload: dict[str, Any]) -> None:
        event = {"time": _now(), "type": event_type, "payload": payload}
        self._write_event(project, event)

    def append_runtime_event(self, project: Project, event: RuntimeEvent) -> None:
        self._write_event(project, event.as_dict())

    def _write_event(self, project: Project, event: dict[str, Any]) -> None:
        with project.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def link_run(self, project: Project, run_id: str, summary: dict[str, Any]) -> Path:
        run_dir = project.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_file = run_dir / "run.json"
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self.append_event(project, "run.completed", {"run_id": run_id, "status": summary.get("status")})
        return summary_file
