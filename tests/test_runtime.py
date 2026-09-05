import json
import unittest

from runtime_impl.contracts import AgentNode, Harness, NodeState, Workflow
from runtime_impl.providers import ProviderName, ProviderRequest, ProviderResponse, ProviderUsage
from runtime_impl.run_manager import RunLimits, RunManager
from runtime_impl.runners import ChatGPTHostAdapter


class FakeAdapter:
    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            output={"model": request.model},
            usage=ProviderUsage(100, 50),
            provider=ProviderName.OLLAMA,
            model=request.model,
        )


class RuntimeTests(unittest.TestCase):
    def workflow(self):
        return Workflow(
            "demo",
            "predict churn",
            (
                AgentNode("frame", "framer", harness=Harness("host", provider="chatgpt_host")),
                AgentNode(
                    "train",
                    "modeler",
                    depends_on=("frame",),
                    harness=Harness("qwen3", provider="ollama"),
                ),
            ),
        )

    def test_host_turn_then_local_turn(self):
        manager = RunManager(
            {ProviderName.CHATGPT_HOST: ChatGPTHostAdapter(), ProviderName.OLLAMA: FakeAdapter()}
        )
        run = manager.start(self.workflow(), RunLimits(max_tokens=5000, max_model_turns=3))
        envelope = manager.execute(run.id, "frame")
        self.assertEqual(envelope["status"], "host_turn_required")
        manager.complete_host(run.id, "frame", {"metric": "recall"}, 80, 20)
        result = manager.execute(run.id, "train")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(run.results["train"].state, NodeState.SUCCEEDED)
        self.assertEqual(run.budget.used_tokens, 298)

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
        manager = RunManager({ProviderName.CHATGPT_HOST: ChatGPTHostAdapter()})
        run = manager.start(self.workflow(), RunLimits(max_tokens=5000, max_model_turns=1))
        manager.execute(run.id, "frame")
        with self.assertRaisesRegex(RuntimeError, "limit"):
            manager.execute(run.id, "frame")

    def test_cycle_is_rejected(self):
        workflow = Workflow(
            "cycle", "x", (AgentNode("a", "x", ("b",)), AgentNode("b", "x", ("a",)))
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            RunManager({}).start(workflow)


if __name__ == "__main__":
    unittest.main()
