import json
import tempfile
import unittest
from pathlib import Path

from agentic_data.project_store import ProjectStore
from agentic_data.web_app import project_snapshot


class WebAppTests(unittest.TestCase):
    def test_project_snapshot_exposes_project_runs_events_and_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demo"
            store = ProjectStore(Path(tmp) / "managed")
            project = store.create("Demo", root)
            (project.datasets_dir / "input.csv").write_text("x\n1\n", encoding="utf-8")
            run_dir = project.runs_dir / "run_1"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(json.dumps({"run_id": "run_1", "status": "succeeded"}), encoding="utf-8")
            store.append_event(project, "run.started", {"run_id": "run_1"})

            snapshot = project_snapshot(project.root)
            self.assertEqual(snapshot["project"]["name"], "Demo")
            self.assertEqual(snapshot["datasets"][0]["name"], "input.csv")
            self.assertEqual(snapshot["runs"][0]["run_id"], "run_1")
            self.assertTrue(any(event["type"] == "run.started" for event in snapshot["events"]))


if __name__ == "__main__":
    unittest.main()
