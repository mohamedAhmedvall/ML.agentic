import json
import tempfile
import unittest
from pathlib import Path

from agentic_data.artifacts import ArtifactRegistry
from agentic_data.contracts import AgentNode, Harness, NodeState, Workflow
from agentic_data.project_store import ProjectStore
from agentic_data.providers import ProviderName, ProviderRequest, ProviderResponse, ProviderUsage
from agentic_data.run_manager import RunLimits, RunManager


class WorkflowAdapter:
    def __init__(self):
        self.turns = {}
        self.requests = []

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        role = request.instructions.split("Tu es l'agent ", 1)[1].split(" du workflow", 1)[0]
        turn = self.turns.get(role, 0)
        self.turns[role] = turn + 1

        if role == "inspector":
            output = (
                {"tool_call": {"name": "data.inspect_csv", "arguments": {"path": "input.csv", "sample_rows": 2}}}
                if turn == 0
                else {"profile": "validated", "rows": 3}
            )
        elif role == "analyst":
            output = (
                {
                    "tool_call": {
                        "name": "python.run",
                        "arguments": {
                            "code": (
                                "from pathlib import Path\n"
                                "Path('analysis.csv').write_text('metric,value\\nmean,20\\n', encoding='utf-8')\n"
                                "print('analysis complete')\n"
                            )
                        },
                    }
                }
                if turn == 0
                else {"analysis": "complete", "artifact": "analysis.csv"}
            )
        elif role == "reporter":
            output = (
                {
                    "tool_call": {
                        "name": "file.write_text",
                        "arguments": {"path": "report.md", "content": "# Analysis\nWorkflow completed."},
                    }
                }
                if turn == 0
                else {"report": "report.md", "status": "ready"}
            )
        else:
            raise AssertionError(f"unexpected role: {role}")

        return ProviderResponse(
            output=output,
            usage=ProviderUsage(12, 6),
            provider=ProviderName.OLLAMA,
            model=request.model,
        )


class EndToEndWorkflowTests(unittest.TestCase):
    def test_project_workflow_tools_events_handoffs_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ProjectStore(root / "projects")
            project = store.create("Generic analysis", root / "project")
            registry = ArtifactRegistry(project.root)
            adapter = WorkflowAdapter()
            manager = RunManager({ProviderName.OLLAMA: adapter}, workspace_root=project.runs_dir)
            events = []

            manager.events.subscribe(events.append)
            manager.events.subscribe(lambda event: store.append_runtime_event(project, event))

            workflow = Workflow(
                "wf_end_to_end",
                "analyze a tabular dataset and produce reusable artifacts",
                (
                    AgentNode(
                        "inspect",
                        "inspector",
                        harness=Harness("qwen3", provider="ollama", tools=("data.inspect_csv",)),
                    ),
                    AgentNode(
                        "analyze",
                        "analyst",
                        depends_on=("inspect",),
                        harness=Harness("qwen3", provider="ollama", tools=("python.run",)),
                    ),
                    AgentNode(
                        "report",
                        "reporter",
                        depends_on=("analyze",),
                        harness=Harness("qwen3", provider="ollama", tools=("file.write_text",)),
                    ),
                ),
            )

            run = manager.start(workflow, RunLimits(max_tokens=10_000, max_model_turns=10))
            (run.workspace / "input.csv").write_text("value\n10\n20\n30\n", encoding="utf-8")

            def register_artifact(event):
                if event.type == "artifact.created":
                    registry.register(
                        run.id,
                        run.workspace,
                        event.payload["path"],
                        created_by=event.node_id,
                        tool=event.payload.get("tool"),
                    )

            manager.events.subscribe(register_artifact)
            outcome = manager.execute_until_blocked(run.id)

            self.assertEqual(outcome["status"], "completed")
            self.assertEqual(run.results["inspect"].state, NodeState.SUCCEEDED)
            self.assertEqual(run.results["analyze"].state, NodeState.SUCCEEDED)
            self.assertEqual(run.results["report"].state, NodeState.SUCCEEDED)
            self.assertTrue((run.workspace / "analysis.csv").is_file())
            self.assertTrue((run.workspace / "report.md").is_file())

            artifacts = registry.list(run.id)
            names = {artifact["name"] for artifact in artifacts}
            self.assertEqual(names, {"analysis.csv", "report.md"})
            self.assertEqual({artifact["created_by"] for artifact in artifacts}, {"analyze", "report"})

            analyze_requests = [r for r in adapter.requests if "l'agent analyst" in r.instructions]
            self.assertEqual(analyze_requests[0].input[0]["dependencies"]["inspect"]["profile"], "validated")
            report_requests = [r for r in adapter.requests if "l'agent reporter" in r.instructions]
            self.assertEqual(report_requests[0].input[0]["dependencies"]["analyze"]["analysis"], "complete")

            event_types = [event.type for event in events]
            self.assertEqual(event_types[0], "run.started")
            self.assertEqual(event_types[-1], "run.completed")
            self.assertEqual(event_types.count("agent.completed"), 3)
            self.assertEqual(event_types.count("tool.called"), 3)
            self.assertEqual(event_types.count("tool.completed"), 3)
            self.assertEqual(event_types.count("artifact.created"), 2)

            run_started = events[0]
            self.assertEqual(run_started.payload["nodes"][1]["depends_on"], ["inspect"])
            self.assertEqual(run_started.payload["nodes"][2]["depends_on"], ["analyze"])

            persisted = [json.loads(line) for line in project.events_file.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event.get("type") == "run.completed" for event in persisted))
            self.assertTrue(any(event.get("type") == "artifact.created" for event in persisted))


if __name__ == "__main__":
    unittest.main()
