from __future__ import annotations

from typing import Any

from .contracts import AgentNode, Harness, Workflow
from .providers import ProviderName, ProviderRequest
from .tool_gateway import AVAILABLE_TOOLS


def parse_workflow(data: dict[str, Any]) -> Workflow:
    nodes: list[AgentNode] = []
    for item in data["nodes"]:
        config = item.get("harness", {})
        nodes.append(
            AgentNode(
                id=item["id"],
                role=item["role"],
                depends_on=tuple(item.get("depends_on", [])),
                harness=Harness(
                    model=config.get("model", "auto"),
                    provider=config.get("provider", "auto"),
                    fallback_provider=config.get("fallback_provider"),
                    tools=tuple(config.get("tools", [])),
                    approval=config.get("approval", "never"),
                    max_retries=int(config.get("max_retries", 0)),
                    timeout_seconds=int(config.get("timeout_seconds", 300)),
                    network=config.get("network", "deny"),
                ),
            )
        )
    workflow = Workflow(id=data["id"], objective=data["objective"], nodes=tuple(nodes))
    workflow.node_map()
    return workflow


def workflow_to_dict(workflow: Workflow) -> dict[str, Any]:
    return {
        "id": workflow.id,
        "objective": workflow.objective,
        "nodes": [
            {
                "id": node.id,
                "role": node.role,
                "depends_on": list(node.depends_on),
                "harness": {
                    "model": node.harness.model,
                    "provider": node.harness.provider,
                    "fallback_provider": node.harness.fallback_provider,
                    "tools": list(node.harness.tools),
                    "approval": node.harness.approval,
                    "max_retries": node.harness.max_retries,
                    "timeout_seconds": node.harness.timeout_seconds,
                    "network": node.harness.network,
                },
            }
            for node in workflow.nodes
        ],
    }


def plan_workflow(
    problem: str,
    dataset_name: str,
    provider: ProviderName,
    model: str,
    adapters: dict[ProviderName, Any],
) -> tuple[Workflow, Any]:
    available_tools = list(AVAILABLE_TOOLS)
    request = ProviderRequest(
        model=model,
        instructions=(
            "Tu es le planificateur ML.agentic. Transforme le problème en DAG data/ML exécutable. "
            "Le dataset est disponible dans le workspace sous le nom indiqué. Retourne uniquement un objet JSON "
            "avec id, objective et nodes. Chaque node contient id, role, depends_on et harness. harness contient "
            "provider='auto', model='auto', tools, approval, max_retries, timeout_seconds et network. "
            "N'utilise que les outils fournis. Maximum 24 nodes. Ne demande pas de réseau. "
            "Le workflow peut couvrir exploration, préparation, analyse, modélisation, évaluation ou reporting selon le besoin."
        ),
        input=[{"problem": problem, "dataset": dataset_name, "available_tools": available_tools}],
        max_output_tokens=4_000,
    )
    response = adapters[provider].invoke(request)
    raw = response.output.get("workflow", response.output) if isinstance(response.output, dict) else response.output
    if not isinstance(raw, dict):
        raise ValueError("planner did not return a workflow object")
    raw.setdefault("id", "wf_generated")
    raw.setdefault("objective", f"{problem} Dataset: {dataset_name}")
    if len(raw.get("nodes", [])) > 24:
        raise ValueError("planner exceeded the 24-node limit")
    allowed = set(available_tools)
    for node in raw.get("nodes", []):
        tools = set(node.get("harness", {}).get("tools", []))
        unknown = tools - allowed
        if unknown:
            raise ValueError(f"planner requested unknown tools: {sorted(unknown)}")
    return parse_workflow(raw), response.usage
