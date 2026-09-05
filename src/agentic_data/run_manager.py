from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import AgentNode, NodeResult, NodeState, Workflow
from .events import EventBus
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
    def __init__(
        self,
        adapters: dict[ProviderName, Any],
        workspace_root: str | Path = ".ml-agentic/runs",
        event_bus: EventBus | None = None,
    ):
        self.adapters = adapters
        self.runs: dict[str, ManagedRun] = {}
        self.workspace_root = Path(workspace_root)
        self.events = event_bus or EventBus()

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
        self.events.emit(
            "run.started",
            run.id,
            {
                "workflow_id": workflow.id,
                "objective": workflow.objective,
                "nodes": [node.id for node in workflow.nodes],
                "provider": default_provider.value,
            },
        )
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
                self.events.emit(
                    "agent.skipped",
                    run.id,
                    {"role": node.role, "reason": "dependency did not succeed"},
                    node_id=node.id,
                )
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
            self.events.emit(
                "agent.awaiting_approval",
                run.id,
                {"role": node.role, "policy": node.harness.approval},
                node_id=node.id,
            )
            return {"status": "approval_required", "node_id": node_id, "policy": node.harness.approval}

        provider = run.default_provider if node.harness.provider == "auto" else ProviderName(node.harness.provider)
        candidates = [provider]
        if node.harness.fallback_provider:
            candidates.append(ProviderName(node.harness.fallback_provider))

        self.events.emit(
            "agent.started",
            run.id,
            {
                "role": node.role,
                "provider": provider.value,
                "model": node.harness.model,
                "depends_on": list(node.depends_on),
                "tools": list(node.harness.tools),
            },
            node_id=node.id,
        )
        gateway = ToolGateway(run.workspace)
        tool_history: list[dict[str, Any]] = []
        tool_calls = 0

        while True:
            if run.model_turns >= run.limits.max_model_turns:
                self.events.emit(
                    "agent.failed",
                    run.id,
                    {"role": node.role, "error": "model turn limit reached"},
                    node_id=node.id,
                )
                raise RuntimeError("model turn limit reached")
            request = self.prepare(run_id, node_id, tool_history)
            run.budget.assert_capacity(estimate_request(request), request.max_output_tokens)
            run.model_turns += 1

            response = None
            last_error = None
            selected_provider = None
            for candidate in candidates:
                try:
                    response = self.adapters[candidate].invoke(request)
                    selected_provider = candidate
                    break
                except (ProviderUnavailable, KeyError) as exc:
                    last_error = exc
            if response is None:
                error = str(last_error)
                run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=error)
                self.events.emit(
                    "agent.failed",
                    run.id,
                    {"role": node.role, "error": error},
                    node_id=node.id,
                )
                return {"status": "failed", "node_id": node_id, "error": error}

            run.budget.record(response.usage)
            self.events.emit(
                "model.turn.completed",
                run.id,
                {
                    "provider": selected_provider.value if selected_provider else response.provider.value,
                    "model": response.model,
                    "usage": usage_dict(response.usage),
                    "turn": run.model_turns,
                },
                node_id=node.id,
            )
            call = response.output.get("tool_call") if isinstance(response.output, dict) else None
            if not isinstance(call, dict):
                run.results[node_id] = NodeResult(node_id, NodeState.SUCCEEDED, response.output)
                self.events.emit(
                    "agent.completed",
                    run.id,
                    {
                        "role": node.role,
                        "output": response.output,
                        "tool_calls": tool_calls,
                    },
                    node_id=node.id,
                )
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
                self.events.emit("agent.failed", run.id, {"role": node.role, "error": error}, node_id=node.id)
                return {"status": "failed", "node_id": node_id, "error": error}

            name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                error = "invalid tool_call payload"
                run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=error)
                self.events.emit("agent.failed", run.id, {"role": node.role, "error": error}, node_id=node.id)
                return {"status": "failed", "node_id": node_id, "error": error}
            if name not in node.harness.tools:
                error = f"tool not allowed for agent: {name}"
                run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=error)
                self.events.emit("agent.failed", run.id, {"role": node.role, "error": error}, node_id=node.id)
                return {"status": "failed", "node_id": node_id, "error": error}

            self.events.emit(
                "tool.called",
                run.id,
                {"tool": name, "arguments": arguments, "call_index": tool_calls},
                node_id=node.id,
            )
            before = _workspace_files(run.workspace)
            try:
                tool_result = gateway.execute(name, arguments)
            except (ToolGatewayError, OSError, ValueError, KeyError) as exc:
                error = f"tool execution failed: {exc}"
                run.results[node_id] = NodeResult(node_id, NodeState.FAILED, error=error)
                self.events.emit(
                    "tool.failed",
                    run.id,
                    {"tool": name, "error": str(exc), "call_index": tool_calls},
                    node_id=node.id,
                )
                self.events.emit("agent.failed", run.id, {"role": node.role, "error": error}, node_id=node.id)
                return {"status": "failed", "node_id": node_id, "error": error}

            self.events.emit(
                "tool.completed",
                run.id,
                {"tool": name, "result": tool_result.output, "call_index": tool_calls},
                node_id=node.id,
            )
            after = _workspace_files(run.workspace)
            for artifact in sorted(after - before):
                self.events.emit(
                    "artifact.created",
                    run.id,
                    {"path": artifact, "created_by": node.id, "tool": name},
                    node_id=node.id,
                )

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
                status = "completed" if done else "failed"
                self.events.emit(
                    f"run.{status}",
                    run.id,
                    {
                        "failed": failed,
                        "model_turns": run.model_turns,
                        "used_tokens": run.budget.used_tokens,
                    },
                )
                return {"status": status, "failed": failed}
            progressed = False
            approvals = []
            for node in ready:
                result = self.execute(run_id, node.id)
                if result["status"] == "approval_required":
                    approvals.append(node.id)
                else:
                    progressed = True
                if result["status"] == "failed":
                    self.events.emit("run.failed", run_id, {"failed": [node.id], "error": result.get("error")})
                    return result
            if approvals and not progressed:
                self.events.emit("run.awaiting_approval", run_id, {"nodes": approvals})
                return {"status": "approval_required", "nodes": approvals}

    def approve(self, run_id: str, node_id: str) -> None:
        run = self.runs[run_id]
        if node_id not in {node.id for node in self.ready(run_id)}:
            raise ValueError("node is not ready")
        run.approvals.add(node_id)
        self.events.emit("agent.approved", run.id, {}, node_id=node_id)

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


def _workspace_files(workspace: Path) -> set[str]:
    return {
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file()
    }


def estimate_request(request: ProviderRequest) -> int:
    return max(1, (len(request.instructions) + len(str(request.input)) + 3) // 4)


def usage_dict(usage: ProviderUsage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "measurement": usage.measurement,
    }
