import json
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

    def test_selected_provider_runs_entire_workflow(self):
        manager = RunManager(
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
        manager = RunManager({ProviderName.OLLAMA: FakeAdapter()})
        run = manager.start(workflow, RunLimits(max_tokens=5000))
        self.assertEqual(manager.execute(run.id, "deploy")["status"], "approval_required")
        manager.approve(run.id, "deploy")
        self.assertEqual(manager.execute(run.id, "deploy")["status"], "succeeded")

    def test_model_turn_limit_is_hard(self):
        manager = RunManager({ProviderName.OPENAI_CODEX: FakeAdapter(ProviderName.OPENAI_CODEX)})
        run = manager.start(self.workflow(), RunLimits(max_tokens=5000, max_model_turns=1))
        manager.execute(run.id, "frame")
        with self.assertRaisesRegex(RuntimeError, "limit"):
            manager.execute(run.id, "train")

    def test_cycle_is_rejected(self):
        workflow = Workflow(
            "cycle", "x", (AgentNode("a", "x", ("b",)), AgentNode("b", "x", ("a",)))
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            RunManager({}).start(workflow)


if __name__ == "__main__":
    unittest.main()
