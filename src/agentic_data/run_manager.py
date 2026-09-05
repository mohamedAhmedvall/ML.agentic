from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .contracts import AgentNode, NodeResult, NodeState, Workflow
from .providers import MeasurementQuality, ProviderName, ProviderRequest, ProviderUsage
from .runners import HostExecutionRequired
from .token_budget import TokenBudget


@dataclass(frozen=True)
class RunLimits:
    max_tokens: int = 24_000
    max_cost_micros: int = 0
    max_model_turns: int = 12


@dataclass
class ManagedRun:
    id: str
    workflow: Workflow
    limits: RunLimits
    budget: TokenBudget
    results: dict[str, NodeResult] = field(default_factory=dict)
    model_turns: int = 0
    approvals: set[str] = field(default_factory=set)


class RunManager:
    def __init__(self, adapters: dict[ProviderName, Any]):
        self.adapters = adapters
        self.runs: dict[str, ManagedRun] = {}

    def start(self, workflow: Workflow, limits: RunLimits | None = None) -> ManagedRun:
        workflow.node_map()
        self._assert_acyclic(workflow)
        limits = limits or RunLimits()
        run = ManagedRun(
            id=f"run_{uuid.uuid4().hex[:12]}",
            workflow=workflow,
            limits=limits,
            budget=TokenBudget(limits.max_tokens, limits.max_cost_micros),
        )
        self.runs[run.id] = run
        return run

    def ready(self, run_id: str) -> list[AgentNode]:
        run = self.runs[run_id]
        ready = []
        for node in run.workflow.nodes:
            if node.id in run.results:
                continue
            deps = [run.results.get(dep) for dep in node.depends_on]
            if any(dep and dep.state != NodeState.SUCCEEDED for dep in deps):
                run.results[node.id] = NodeResult(node.id, NodeState.SKIPPED)
            elif all(dep is not None for dep in deps):
                ready.append(node)
        return ready

    def prepare(self, run_id: str, node_id: str) -> ProviderRequest:
        run = self.runs[run_id]
        node = next(node for node in self.ready(run_id) if node.id == node_id)
        dependencies = {dep: run.results[dep].output for dep in node.depends_on}
        return ProviderRequest(
            model=node.harness.model,
            instructions=(
                f"Tu es l'agent {node.role} du workflow Orbia. Retourne uniquement un objet JSON. "
                f"Objectif métier: {run.workflow.objective}"
            ),
            input=[{"dependencies": dependencies, "allowed_tools": list(node.harness.tools)}],
            max_output_tokens=min(2_000, run.budget.remaining_tokens),
        )

    def execute(self, run_id: str, node_id: str) -> dict[str, Any]:
        run = self.runs[run_id]
        if run.model_turns >= run.limits.max_model_turns:
            raise RuntimeError("model turn limit reached")
        node = next(node for node in run.workflow.nodes if node.id == node_id)
        if node.harness.approval != "never" and node_id not in run.approvals:
            return {"status": "approval_required", "node_id": node_id, "policy": node.harness.approval}
        provider = ProviderName(node.harness.provider)
        request = self.prepare(run_id, node_id)
        run.budget.assert_capacity(estimate_request(request), request.max_output_tokens)
        run.model_turns += 1
        try:
            response = self.adapters[provider].invoke(request)
        except HostExecutionRequired:
            return {
                "status": "host_turn_required",
                "provider": provider,
                "instructions": request.instructions,
                "input": request.input,
                "max_output_tokens": request.max_output_tokens,
            }
        run.budget.record(response.usage)
        run.results[node_id] = NodeResult(node_id, NodeState.SUCCEEDED, response.output)
        return {"status": "succeeded", "output": response.output, "usage": usage_dict(response.usage)}

    def complete_host(
        self, run_id: str, node_id: str, output: dict[str, Any], input_tokens: int, output_tokens: int
    ) -> None:
        run = self.runs[run_id]
        if node_id not in {node.id for node in self.ready(run_id)}:
            raise ValueError("node is not ready")
        usage = ProviderUsage(
            input_tokens=max(128, input_tokens),
            output_tokens=max(1, output_tokens or (len(str(output)) + 3) // 4),
            measurement=MeasurementQuality.ESTIMATED,
        )
        run.budget.record(usage)
        run.results[node_id] = NodeResult(node_id, NodeState.SUCCEEDED, output)

    def approve(self, run_id: str, node_id: str) -> None:
        run = self.runs[run_id]
        if node_id not in {node.id for node in self.ready(run_id)}:
            raise ValueError("node is not ready")
        run.approvals.add(node_id)

    @staticmethod
    def _assert_acyclic(workflow: Workflow) -> None:
        nodes = workflow.node_map()
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


def estimate_request(request: ProviderRequest) -> int:
    return max(1, (len(request.instructions) + len(str(request.input)) + 3) // 4)


def usage_dict(usage: ProviderUsage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "measurement": usage.measurement,
    }
