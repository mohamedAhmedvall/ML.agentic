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
    def __init__(self, adapters: dict[ProviderName, Any], workspace_root: str | Path = ".ml-agentic/runs", event_bus: EventBus | None = None):
        self.adapters = adapters
        self.runs: dict[str, ManagedRun] = {}
        self.workspace_root = Path(workspace_root)
        self.events = event_bus or EventBus()

    def start(self, workflow: Workflow, limits: RunLimits | None = None, default_provider: ProviderName = ProviderName.OPENAI_CODEX) -> ManagedRun:
        workflow.node_map(); self._assert_acyclic(workflow)
        limits = limits or RunLimits(); run_id = f"run_{uuid.uuid4().hex[:12]}"; workspace = self.workspace_root / run_id; workspace.mkdir(parents=True, exist_ok=True)
        run = ManagedRun(run_id, workflow, limits, TokenBudget(limits.max_tokens, limits.max_cost_micros), workspace, default_provider=default_provider)
        self.runs[run.id] = run
        self.events.emit("run.started", run.id, {"workflow_id": workflow.id, "objective": workflow.objective,
            "nodes": [{"id": n.id, "role": n.role, "depends_on": list(n.depends_on), "provider": n.harness.provider, "model": n.harness.model, "tools": list(n.harness.tools), "approval": n.harness.approval} for n in workflow.nodes],
            "provider": default_provider.value})
        return run

    def ready(self, run_id: str) -> list[AgentNode]:
        run = self.runs[run_id]; ready = []
        for node in run.workflow.nodes:
            if node.id in run.results: continue
            deps = [run.results.get(dep) for dep in node.depends_on]
            if any(dep and dep.state != NodeState.SUCCEEDED for dep in deps):
                run.results[node.id] = NodeResult(node.id, NodeState.SKIPPED); self.events.emit("agent.skipped", run.id, {"role": node.role, "reason": "dependency did not succeed"}, node_id=node.id)
            elif all(dep is not None for dep in deps): ready.append(node)
        return ready

    def prepare(self, run_id: str, node_id: str, tool_history: list[dict[str, Any]] | None = None) -> ProviderRequest:
        run = self.runs[run_id]; node = next(node for node in self.ready(run_id) if node.id == node_id); dependencies = {dep: run.results[dep].output for dep in node.depends_on}
        return ProviderRequest(model=node.harness.model, instructions=(f"Tu es l'agent {node.role} du workflow ML.agentic. Retourne uniquement un objet JSON. Objectif métier: {run.workflow.objective}. Si tu dois utiliser un outil autorisé, retourne exactement " '{"tool_call":{"name":"nom.outil","arguments":{...}}}. ' "Après réception du résultat de l'outil, poursuis le travail. Sinon retourne directement ton résultat final JSON. N'invente jamais un outil."), input=[{"dependencies": dependencies, "available_tools": list(node.harness.tools), "tool_history": tool_history or [], "workspace": "."}])

    def execute(self, run_id: str, node_id: str) -> NodeResult:
        run = self.runs[run_id]; node = next(node for node in self.ready(run_id) if node.id == node_id)
        if node.harness.approval != "never" and node_id not in run.approvals:
            result = NodeResult(node.id, NodeState.AWAITING_APPROVAL, attempts=0); run.results[node.id] = result; self.events.emit("agent.awaiting_approval", run.id, {"role": node.role, "approval": node.harness.approval}, node_id=node.id); return result
        provider_name = ProviderName(node.harness.provider) if node.harness.provider != "auto" else run.default_provider
        adapter = self.adapters.get(provider_name)
        if adapter is None and node.harness.fallback_provider: provider_name = ProviderName(node.harness.fallback_provider); adapter = self.adapters.get(provider_name)
        if adapter is None: raise ProviderUnavailable(f"no adapter configured for {provider_name.value}")
        gateway = ToolGateway(run.workspace); tool_history: list[dict[str, Any]] = []; self.events.emit("agent.started", run.id, {"role": node.role, "provider": provider_name.value, "model": node.harness.model, "depends_on": list(node.depends_on), "tools": list(node.harness.tools)}, node_id=node.id)
        while True:
            if run.model_turns >= run.limits.max_model_turns:
                self.events.emit("agent.failed", run.id, {"role": node.role, "error": "model turn limit reached"}, node_id=node.id); raise RuntimeError("model turn limit reached")
            request = self.prepare(run_id, node_id, tool_history); response = adapter.invoke(request); run.model_turns += 1; run.budget.record(response.usage)
            self.events.emit("model.turn.completed", run.id, {"provider": provider_name.value, "model": node.harness.model, "usage": self.usage_dict(response.usage), "turn": run.model_turns}, node_id=node.id)
            call = response.output.get("tool_call") if isinstance(response.output, dict) else None
            if not call:
                result = NodeResult(node.id, NodeState.SUCCEEDED, output=response.output); run.results[node.id] = result; self.events.emit("agent.completed", run.id, {"role": node.role, "output": response.output, "tool_calls": len(tool_history)}, node_id=node.id); return result
            if len(tool_history) >= run.limits.max_tool_calls_per_agent:
                result = NodeResult(node.id, NodeState.FAILED, error="tool call limit reached"); run.results[node.id] = result; self.events.emit("agent.failed", run.id, {"role": node.role, "error": result.error}, node_id=node.id); return result
            try:
                tool_name = str(call["name"]); arguments = dict(call.get("arguments", {}))
                if tool_name not in node.harness.tools: raise ToolGatewayError(f"tool not allowed for agent: {tool_name}")
                before = self._workspace_files(run.workspace); self.events.emit("tool.called", run.id, {"tool": tool_name, "arguments": arguments, "call_index": len(tool_history)+1}, node_id=node.id)
                tool_result = gateway.execute(tool_name, arguments); tool_history.append({"tool": tool_name, "arguments": arguments, "result": tool_result.output}); self.events.emit("tool.completed", run.id, {"tool": tool_name, "result": tool_result.output, "call_index": len(tool_history)}, node_id=node.id)
                after = self._workspace_files(run.workspace)
                for path in sorted(after-before): self.events.emit("artifact.created", run.id, {"path": path, "created_by": node.id, "tool": tool_name}, node_id=node.id)
            except (ToolGatewayError, OSError, ValueError, KeyError) as exc:
                result = NodeResult(node.id, NodeState.FAILED, error=str(exc)); run.results[node.id] = result; self.events.emit("tool.failed", run.id, {"tool": call.get("name") if isinstance(call, dict) else None, "error": str(exc)}, node_id=node.id); self.events.emit("agent.failed", run.id, {"role": node.role, "error": str(exc)}, node_id=node.id); return result

    def execute_until_blocked(self, run_id: str) -> str:
        run = self.runs[run_id]
        while True:
            ready = self.ready(run_id)
            if not ready:
                states = [r.state for r in run.results.values()]
                status = "failed" if NodeState.FAILED in states else "completed" if len(run.results) == len(run.workflow.nodes) else "blocked"
                self.events.emit("run.failed" if status == "failed" else "run.completed" if status == "completed" else "run.blocked", run.id, {"status": status}); return status
            progressed = False
            for node in ready:
                result = self.execute(run_id, node.id)
                if result.state == NodeState.AWAITING_APPROVAL:
                    self.events.emit("run.awaiting_approval", run.id, {"node_id": node.id}, node_id=node.id); return "awaiting_approval"
                progressed = True
                if result.state == NodeState.FAILED: self.events.emit("run.failed", run.id, {"status": "failed", "node_id": node.id}, node_id=node.id); return "failed"
            if not progressed: return "blocked"

    def approve(self, run_id: str, node_id: str) -> None:
        run = self.runs[run_id]; existing = run.results.get(node_id)
        if existing and existing.state == NodeState.AWAITING_APPROVAL: del run.results[node_id]
        run.approvals.add(node_id); self.events.emit("agent.approved", run.id, {}, node_id=node_id)

    def run_status(self, run_id: str) -> dict[str, Any]:
        run = self.runs[run_id]
        return {"run_id": run.id, "workflow_id": run.workflow.id, "objective": run.workflow.objective, "workspace": str(run.workspace), "ready": [n.id for n in self.ready(run_id)], "results": {k: {"state": v.state.value, "output": v.output, "error": v.error} for k,v in run.results.items()}, "usage": run.budget.snapshot(), "model_turns": run.model_turns}

    @staticmethod
    def usage_dict(usage: ProviderUsage) -> dict[str, Any]:
        return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "cost_micros": usage.cost_micros}

    @staticmethod
    def _workspace_files(workspace: Path) -> set[str]:
        return {str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file()}

    @staticmethod
    def _assert_acyclic(workflow: Workflow) -> None:
        nodes = workflow.node_map(); visiting: set[str] = set(); visited: set[str] = set()
        def visit(node_id: str) -> None:
            if node_id in visiting: raise ValueError("workflow contains a dependency cycle")
            if node_id in visited: return
            visiting.add(node_id)
            for dep in nodes[node_id].depends_on: visit(dep)
            visiting.remove(node_id); visited.add(node_id)
        for node_id in nodes: visit(node_id)
