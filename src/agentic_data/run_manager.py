from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import AgentNode, NodeResult, NodeState, Workflow
from .providers import ProviderName, ProviderRequest, ProviderUsage
from .runners import ProviderUnavailable
from .token_budget import TokenBudget
from .tool_gateway import ToolGateway, ToolGatewayError


@dataclass(frozen=True)
class RunLimits:
    max_tokens: int = 24_000
    max_cost_micros: int = 0
    max_model_turns: int = 12
    max_tool_calls_per_agent: int = 8


@dataclass
class ManagedRun:
    id: str
    workflow: Workflow
    limits: RunLimits
    budget: TokenBudget
    workspace: Path
    results: dict[str, NodeResult] = field(default_factory=dict)
    model_turns: int = 0
    approvals: set[str] = field(default_factory=set)
    default_provider: ProviderName = ProviderName.OPENAI_CODEX


class RunManager:
    def __init__(self, adapters: dict[ProviderName, Any], workspace_root: str | Path = ".ml-agentic/runs"):
        self.adapters = adapters
        self.runs: dict[str, ManagedRun] = {}
        self.workspace_root = Path(workspace_root)

    def start(
        self,
        workflow: Workflow,
        limits: RunLimits | None = None,
        default_provider: ProviderName = ProviderName.OPENAI_CODEX,
    ) -> ManagedRun:
        workflow.node_map()
        self._assert_acyclic(workflow)
        limits = limits or RunLimits()
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        workspace = self.workspace_root / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        run = ManagedRun(
            id=run_id,
            workflow=workflow,
            limits=limits,
            budget=TokenBudget(limits.max_tokens, limits.max_cost_micros),
            workspace=workspace,
            default_provider=default_provider,
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

    def prepare(
        self,
        run_id: str,
        node_id: str,
        tool_history: list[dict[str, Any]] | None = None,
    ) -> ProviderRequest:
        run = self.runs[run_id]
        node = next(node for node in self.ready(run_id) if node.id == node_id)
        dependencies = {dep: run.results[dep].output for dep in node.depends_on}
        return ProviderRequest(
            model=node.harness.model,
            instructions=(
                f"Tu es l'agent {node.role} du workflow ML.agentic. Retourne uniquement un objet JSON. "
                f"Objectif métier: {run.workflow.objective}. "
                "Si tu dois utiliser un outil autorisé, retourne exactement "
                '{"tool_call":{"name":"nom.outil","arguments":{...}}}. '
                "Après réception du résultat de l'outil, poursuis le travail. "
                "Sinon retourne directement ton résultat final JSON. N'invente jamais un outil."
            ),
            input=[
                {
                    "dependencies": dependencies,
                    "available_tools": list(node.harness.tools),
                    "tool_history": tool_history or [],
                    "workspace": ".",
                }
            ],
            max_output_tokens=min(2_000, run.budget.remaining_tokens),
        )

    def execute(self, run_id: str, node_id: str) -> dict[str, Any]:
        run = self.runs[run_id]
        node = next(node for node in run.workflow.nodes if node.id == node_id)
        if node.harness.approval != "never" and node_id not in run.approvals:
            return {"status": "approval_required", "node_id": node_id, "policy": node.harness.approval}

        provider = run.default_provider if node.harness.provider == "auto" else ProviderName(node.harness.provider)
        candidates = [provider]
        if node.harness.fallback_provider:
            candidates.append(ProviderName(node.harness.fallback_provider))

        gateway = ToolGateway(run.workspace)
        tool_history: list[dict[str, Any]] = []
        tool_calls = 0

        while True:
            if run.model_turns >= run.limits.max_model_turns:
                raise RuntimeError("model turn limit reached")
            request = self.prepare(run_id, node_id, tool_history)
            run.budget.assert_capacity(estimate_request(request), request.max_output_tokens)
            run.model_turns += 1

            response = None
            last_error = None
            for candidate in candidates:
                try:
                    response = self.adapters[candidate].invoke(request)
                    break
                except (ProviderUnavailable, KeyError) as exc:
                    last_error = exc
            if response is None:
                run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=str(last_error))
                return {"status": "failed", "node_id": node_id, "error": str(last_error)}

            run.budget.record(response.usage)
            call = response.output.get("tool_call") if isinstance(response.output, dict) else None
            if not isinstance(call, dict):
                run.results[node_id] = NodeResult(node_id, NodeState.SUCCEEDED, response.output)
                return {
                    "status": "succeeded",
                    "output": response.output,
                    "usage": usage_dict(response.usage),
                    "tool_calls": tool_calls,
                }

            tool_calls += 1
            if tool_calls > run.limits.max_tool_calls_per_agent:
                error = "tool call limit reached"
                run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=error)
                return {"status": "failed", "node_id": node_id, "error": error}

            name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                error = "invalid tool_call payload"
                run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=error)
                return {"status": "failed", "node_id": node_id, "error": error}
            if name not in node.harness.tools:
                error = f"tool not allowed for agent: {name}"
                run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=error)
                return {"status": "failed", "node_id": node_id, "error": error}

            try:
                tool_result = gateway.execute(name, arguments)
            except (ToolGatewayError, OSError, ValueError, KeyError) as exc:
                error = f"tool execution failed: {exc}"
                run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=error)
                return {"status": "failed", "node_id": node_id, "error": error}

            tool_history.append(
                {
                    "tool": name,
                    "arguments": arguments,
                    "result": tool_result.output,
                }
            )

    def execute_until_blocked(self, run_id: str) -> dict[str, Any]:
        """Run every ready node until completion, approval, failure or a hard limit."""
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
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "measurement": usage.measurement,
    }
