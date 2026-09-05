import tempfile
import unittest

from agentic_data.contracts import AgentNode, Harness, Workflow
from agentic_data.providers import ProviderName, ProviderRequest, ProviderResponse, ProviderUsage
from agentic_data.run_manager import RunLimits, RunManager


class ToolCallingAdapter:
    def __init__(self):
        self.outputs = [
            {"tool_call": {"name": "file.write_text", "arguments": {"path": "result.txt", "content": "done"}}},
            {"status": "complete", "artifact": "result.txt"},
        ]

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            output=self.outputs.pop(0),
            usage=ProviderUsage(10, 5),
            provider=ProviderName.OLLAMA,
            model=request.model,
        )


class EventStreamTests(unittest.TestCase):
    def test_runtime_emits_observable_agent_tool_and_artifact_events(self):
        with tempfile.TemporaryDirectory() as tempdir:
            manager = RunManager({ProviderName.OLLAMA: ToolCallingAdapter()}, workspace_root=tempdir)
            events = []
            manager.events.subscribe(events.append)
            workflow = Workflow(
                "generic-analysis",
                "produce a useful artifact",
                (
                    AgentNode(
                        "worker",
                        "analyst",
                        harness=Harness("qwen3", provider="ollama", tools=("file.write_text",)),
                    ),
                ),
            )
            run = manager.start(workflow, RunLimits(max_tokens=5000, max_model_turns=4))
            result = manager.execute_until_blocked(run.id)

            self.assertEqual(result["status"], "completed")
            event_types = [event.type for event in events]
            self.assertEqual(event_types[0], "run.started")
            self.assertIn("agent.started", event_types)
            self.assertIn("tool.called", event_types)
            self.assertIn("tool.completed", event_types)
            self.assertIn("artifact.created", event_types)
            self.assertIn("agent.completed", event_types)
            self.assertEqual(event_types[-1], "run.completed")

            artifact_event = next(event for event in events if event.type == "artifact.created")
            self.assertEqual(artifact_event.node_id, "worker")
            self.assertEqual(artifact_event.payload["path"], "result.txt")
            self.assertTrue((run.workspace / "result.txt").is_file())


if __name__ == "__main__":
    unittest.main()
