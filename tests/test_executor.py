import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from agentic_data import AgentNode, Harness, NodeState, Workflow, WorkflowExecutor


class WorkflowExecutorTests(unittest.TestCase):
    def test_dependencies_are_available_to_descendants(self):
        workflow = Workflow(
            id="demo",
            objective="demo",
            nodes=(
                AgentNode("profile", "profiler"),
                AgentNode("report", "reporter", depends_on=("profile",)),
            ),
        )

        results = WorkflowExecutor(
            lambda node, deps: {"inputs": sorted(deps), "role": node.role}
        ).execute(workflow)

        self.assertEqual(results["profile"].state, NodeState.SUCCEEDED)
        self.assertEqual(results["report"].output["inputs"], ["profile"])

    def test_failure_skips_descendants_but_not_independent_branch(self):
        workflow = Workflow(
            id="demo",
            objective="demo",
            nodes=(
                AgentNode("broken", "analyst"),
                AgentNode("child", "reporter", depends_on=("broken",)),
                AgentNode("independent", "profiler"),
            ),
        )

        def runner(node, _deps):
            if node.id == "broken":
                raise RuntimeError("bad input")
            return {"ok": True}

        results = WorkflowExecutor(runner).execute(workflow)
        self.assertEqual(results["broken"].state, NodeState.FAILED)
        self.assertEqual(results["child"].state, NodeState.SKIPPED)
        self.assertEqual(results["independent"].state, NodeState.SUCCEEDED)

    def test_retry_policy(self):
        attempts = 0

        def flaky(_node, _deps):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary")
            return {"ok": True}

        workflow = Workflow(
            id="retry",
            objective="demo",
            nodes=(AgentNode("agent", "analyst", harness=Harness("model", max_retries=1)),),
        )
        result = WorkflowExecutor(flaky).execute(workflow)["agent"]
        self.assertEqual(result.state, NodeState.SUCCEEDED)
        self.assertEqual(result.attempts, 2)

    def test_cycle_is_rejected(self):
        workflow = Workflow(
            id="cycle",
            objective="demo",
            nodes=(
                AgentNode("a", "one", depends_on=("b",)),
                AgentNode("b", "two", depends_on=("a",)),
            ),
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            WorkflowExecutor(lambda *_: {}).execute(workflow)


if __name__ == "__main__":
    unittest.main()

