from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .contracts import AgentNode, NodeResult, NodeState, Workflow
from .providers import ProviderName, ProviderRequest, ProviderUsage
from .runners import ProviderUnavailable
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
    default_provider: ProviderName = ProviderName.OPENAI_CODEX


class RunManager:
    def __init__(self, adapters: dict[ProviderName, Any]):
        self.adapters = adapters
        self.runs: dict[str, ManagedRun] = {}

    def start(self, workflow: Workflow, limits: RunLimits | None = None, default_provider: ProviderName = ProviderName.OPENAI_CODEX) -> ManagedRun:
        workflow.node_map()
        self._assert_acyclic(workflow)
        limits = limits or RunLimits()
        run = ManagedRun(id=f"run_{uuid.uuid4().hex[:12]}", workflow=workflow, limits=limits, budget=TokenBudget(limits.max_tokens, limits.max_cost_micros), default_provider=default_provider)
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
            instructions=(f"Tu es l'agent {node.role} du workflow ML.agentic. Retourne uniquement un objet JSON. " f"Objectif métier: {run.workflow.objective}"),
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
        provider = run.default_provider if node.harness.provider == "auto" else ProviderName(node.harness.provider)
        request = self.prepare(run_id, node_id)
        run.budget.assert_capacity(estimate_request(request), request.max_output_tokens)
        run.model_turns += 1
        candidates = [provider]
        if node.harness.fallback_provider:
            candidates.append(ProviderName(node.harness.fallback_provider))
        last_error = None
        for candidate in candidates:
            try:
                response = self.adapters[candidate].invoke(request)
                break
            except (ProviderUnavailable, KeyError) as exc:
                last_error = exc
        else:
            run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=str(last_error))
            return {"status": "failed", "node_id": node_id, "error": str(last_error)}
        run.budget.record(response.usage)
        run.results[node_id] = NodeResult(node_id, NodeState.SUCCEEDED, response.output)
        return {"status": "succeeded", "output": response.output, "usage": usage_dict(response.usage)}

    def execute_until_blocked(self, run_id: str) -> dict[str, Any]:
        while True:
            ready = self.ready(run_id)
            if not ready:
                run = self.runs[run_id]
                failed = [node_id for node_id, result in run.results.items() if result.state == NodeState.FAILED]
                done = len(run.results) == len(run.workflow.nodes) and not failed
                return {"status": "completed" if done else "failed", "failed": failed}
            progressed = False
            approvals = []
            for node in ready:
                result = self.execute(run_id, node.id)
                if result["status"] == "approval_required":
                    approvals.append(node.id)
                else:
                    progressed = True
                if result["status"] == "failed":
                    return result
            if approvals and not progressed:
                return {"status": "approval_required", "nodes": approvals}

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
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "cached_input_tokens": usage.cached_input_tokens, "measurement": usage.measurement}
