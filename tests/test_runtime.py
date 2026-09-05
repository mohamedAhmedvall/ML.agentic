import tempfile
import unittest

from agentic_data.contracts import AgentNode, Harness, NodeState, Workflow
from agentic_data.providers import ProviderName, ProviderRequest, ProviderResponse, ProviderUsage
from agentic_data.run_manager import RunLimits, RunManager


class FakeAdapter:
    def __init__(self, provider=ProviderName.OLLAMA):
        self.provider = provider

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            output={"model": request.model},
            usage=ProviderUsage(100, 50),
            provider=self.provider,
            model=request.model,
        )


class ToolCallingAdapter:
    def __init__(self, outputs, provider=ProviderName.OLLAMA):
        self.outputs = list(outputs)
        self.provider = provider
        self.requests = []

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        output = self.outputs.pop(0)
        return ProviderResponse(
            output=output,
            usage=ProviderUsage(10, 5),
            provider=self.provider,
            model=request.model,
        )


class RuntimeTests(unittest.TestCase):
    def workflow(self):
        return Workflow(
            "demo",
            "predict churn",
            (
                AgentNode("frame", "framer", harness=Harness("auto", provider="auto")),
                AgentNode(
                    "train",
                    "modeler",
                    depends_on=("frame",),
                    harness=Harness("qwen3", provider="ollama"),
                ),
            ),
        )

    def manager(self, adapters):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        return RunManager(adapters, workspace_root=tempdir.name)

    def test_selected_provider_runs_entire_workflow(self):
        manager = self.manager(
            {ProviderName.ANTHROPIC_CLAUDE: FakeAdapter(ProviderName.ANTHROPIC_CLAUDE), ProviderName.OLLAMA: FakeAdapter()}
        )
        run = manager.start(
            self.workflow(), RunLimits(max_tokens=5000, max_model_turns=3),
            default_provider=ProviderName.ANTHROPIC_CLAUDE,
        )
        result = manager.execute_until_blocked(run.id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(run.results["frame"].state, NodeState.SUCCEEDED)
        self.assertEqual(run.results["train"].state, NodeState.SUCCEEDED)
        self.assertEqual(run.budget.used_tokens, 300)

    def test_approval_gate_blocks_execution(self):
        workflow = Workflow(
            "approval",
            "publish model",
            (AgentNode("deploy", "deployer", harness=Harness("qwen3", provider="ollama", approval="always")),),
        )
        manager = self.manager({ProviderName.OLLAMA: FakeAdapter()})
        run = manager.start(workflow, RunLimits(max_tokens=5000))
        self.assertEqual(manager.execute(run.id, "deploy")["status"], "approval_required")
        manager.approve(run.id, "deploy")
        self.assertEqual(manager.execute(run.id, "deploy")["status"], "succeeded")

    def test_model_turn_limit_is_hard(self):
        manager = self.manager({ProviderName.OPENAI_CODEX: FakeAdapter(ProviderName.OPENAI_CODEX)})
        run = manager.start(self.workflow(), RunLimits(max_tokens=5000, max_model_turns=1))
        manager.execute(run.id, "frame")
        with self.assertRaisesRegex(RuntimeError, "limit"):
            manager.execute(run.id, "train")

    def test_cycle_is_rejected(self):
        workflow = Workflow(
            "cycle", "x", (AgentNode("a", "x", ("b",)), AgentNode("b", "x", ("a",)))
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.manager({}).start(workflow)

    def test_agent_can_call_allowlisted_tool_and_continue(self):
        adapter = ToolCallingAdapter(
            [
                {"tool_call": {"name": "file.write_text", "arguments": {"path": "artifact.txt", "content": "hello"}}},
                {"artifact": "artifact.txt", "status": "done"},
            ]
        )
        workflow = Workflow(
            "tools",
            "create an artifact",
            (
                AgentNode(
                    "writer",
                    "writer",
                    harness=Harness("qwen3", provider="ollama", tools=("file.write_text",)),
                ),
            ),
        )
        manager = self.manager({ProviderName.OLLAMA: adapter})
        run = manager.start(workflow, RunLimits(max_tokens=5000, max_model_turns=4))
        result = manager.execute(run.id, "writer")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual((run.workspace / "artifact.txt").read_text(), "hello")
        history = adapter.requests[1].input[0]["tool_history"]
        self.assertEqual(history[0]["tool"], "file.write_text")
        self.assertEqual(run.model_turns, 2)

    def test_agent_cannot_call_tool_outside_allowlist(self):
        adapter = ToolCallingAdapter(
            [{"tool_call": {"name": "python.run", "arguments": {"code": "print('no')"}}}]
        )
        workflow = Workflow(
            "tools-denied",
            "deny undeclared tools",
            (AgentNode("reader", "reader", harness=Harness("qwen3", provider="ollama", tools=("file.read_text",))),),
        )
        manager = self.manager({ProviderName.OLLAMA: adapter})
        run = manager.start(workflow, RunLimits(max_tokens=5000))
        result = manager.execute(run.id, "reader")
        self.assertEqual(result["status"], "failed")
        self.assertIn("not allowed", result["error"])
        self.assertEqual(run.results["reader"].state, NodeState.FAILED)


if __name__ == "__main__":
    unittest.main()
