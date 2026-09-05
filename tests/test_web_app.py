import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agentic_data.project_store import ProjectStore
from agentic_data.providers import ProviderName, ProviderRequest, ProviderResponse, ProviderUsage
from agentic_data.web_app import app, create_project, generate_plan_preview, project_snapshot


class PlannerAdapter:
    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            output={
                "id": "wf_web",
                "objective": "inspect then report",
                "nodes": [
                    {
                        "id": "inspect",
                        "role": "inspector",
                        "depends_on": [],
                        "harness": {"provider": "auto", "model": "auto", "tools": ["data.inspect_csv"]},
                    },
                    {
                        "id": "report",
                        "role": "reporter",
                        "depends_on": ["inspect"],
                        "harness": {"provider": "auto", "model": "auto", "tools": ["file.write_text"]},
                    },
                ],
            },
            usage=ProviderUsage(15, 8),
            provider=ProviderName.OLLAMA,
            model=request.model,
        )


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

    def test_create_project_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            created = create_project("UI project", str(Path(tmp) / "project"), str(Path(tmp) / "managed"))
            self.assertEqual(created["name"], "UI project")
            self.assertTrue((Path(created["path"]) / "project.json").is_file())

    def test_generate_plan_preview_persists_draft_and_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ProjectStore().create("Plan", Path(tmp) / "project")
            (project.datasets_dir / "data.csv").write_text("x\n1\n2\n", encoding="utf-8")
            draft = generate_plan_preview(
                project.root,
                "understand the dataset",
                "data.csv",
                provider="ollama",
                model="qwen3",
                adapters={ProviderName.OLLAMA: PlannerAdapter()},
            )
            self.assertEqual(draft["workflow"]["nodes"][1]["depends_on"], ["inspect"])
            self.assertEqual(draft["source_dataset"], "data.csv")
            self.assertTrue((project.root / "draft_workflow.json").is_file())
            snapshot = project_snapshot(project.root)
            self.assertEqual(snapshot["draft"]["workflow"]["id"], "wf_web")
            self.assertTrue(any(event["type"] == "workflow.planned" for event in snapshot["events"]))

    def test_web_api_creates_project_and_plans_dag(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(app)
            project_path = str(Path(tmp) / "project")
            create_response = client.post("/api/projects", json={"name": "API project", "path": project_path})
            self.assertEqual(create_response.status_code, 200)
            project = ProjectStore().open(project_path)
            (project.datasets_dir / "data.csv").write_text("x\n1\n", encoding="utf-8")

            with patch("agentic_data.web_app._adapters", return_value={ProviderName.OLLAMA: PlannerAdapter()}):
                response = client.post(
                    "/api/plan",
                    json={
                        "project_path": project_path,
                        "problem": "analyze data",
                        "dataset": "data.csv",
                        "provider": "ollama",
                        "model": "qwen3",
                    },
                )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["workflow"]["id"], "wf_web")
            self.assertEqual(payload["workflow"]["nodes"][0]["role"], "inspector")


if __name__ == "__main__":
    unittest.main()
