from __future__ import annotations

from collections.abc import Callable

from .contracts import AgentNode, NodeResult, NodeState, Workflow

Runner = Callable[[AgentNode, dict[str, NodeResult]], dict]


class WorkflowExecutor:
    """Small synchronous reference kernel for dependency semantics.

    Production execution will dispatch ready nodes to isolated workers. Keeping
    this kernel free of model and container frameworks makes its behavior easy
    to test and reuse.
    """

    def __init__(self, runner: Runner):
        self.runner = runner

    def execute(self, workflow: Workflow) -> dict[str, NodeResult]:
        nodes = workflow.node_map()
        self._assert_acyclic(nodes)
        results: dict[str, NodeResult] = {}
        pending = set(nodes)

        while pending:
            progressed = False
            for node_id in tuple(pending):
                node = nodes[node_id]
                if not all(dep in results for dep in node.depends_on):
                    continue

                dependency_results = {dep: results[dep] for dep in node.depends_on}
                if any(result.state != NodeState.SUCCEEDED for result in dependency_results.values()):
                    results[node_id] = NodeResult(node_id=node_id, state=NodeState.SKIPPED)
                else:
                    results[node_id] = self._run_with_retries(node, dependency_results)
                pending.remove(node_id)
                progressed = True

            if not progressed:
                raise RuntimeError("workflow could not make progress")

        return results

    def _run_with_retries(self, node: AgentNode, dependencies: dict[str, NodeResult]) -> NodeResult:
        attempts = 0
        last_error: Exception | None = None
        while attempts <= node.harness.max_retries:
            attempts += 1
            try:
                output = self.runner(node, dependencies)
                return NodeResult(node_id=node.id, state=NodeState.SUCCEEDED, output=output, attempts=attempts)
            except Exception as exc:  # executor boundary turns runner errors into state
                last_error = exc
        return NodeResult(
            node_id=node.id,
            state=NodeState.FAILED,
            error=str(last_error),
            attempts=attempts,
        )

    @staticmethod
    def _assert_acyclic(nodes: dict[str, AgentNode]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("workflow contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency in nodes[node_id].depends_on:
                visit(dependency)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in nodes:
            visit(node_id)

