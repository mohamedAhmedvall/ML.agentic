from __future__ import annotations

import json
import os
from typing import Any

from mcp.server import MCPServer

from .contracts import AgentNode, Harness, Workflow
from .providers import ProviderName, ProviderRequest
from .run_manager import RunLimits, RunManager
from .runners import ClaudeAdapter, CodexAdapter, CopilotAdapter, OllamaAdapter


mcp = MCPServer("Orbia Agentic Data Platform")
manager = RunManager(
    {
        ProviderName.OPENAI_CODEX: CodexAdapter(),
        ProviderName.GITHUB_COPILOT: CopilotAdapter(),
        ProviderName.ANTHROPIC_CLAUDE: ClaudeAdapter(),
        ProviderName.OLLAMA: OllamaAdapter(os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")),
    }
)


def parse_workflow(raw: str) -> Workflow:
    data = json.loads(raw)
    nodes = []
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
    return Workflow(id=data["id"], objective=data["objective"], nodes=tuple(nodes))


def _plan_problem(problem: str, provider: ProviderName, model: str = "auto") -> tuple[Workflow, Any]:
    """Ask the selected provider to propose a bounded DAG; Orbia validates the result."""
    if not 10 <= len(problem.strip()) <= 20_000:
        raise ValueError("problem must contain between 10 and 20000 characters")
    request = ProviderRequest(
        model=model,
        instructions=(
            "Tu es le planificateur Orbia. Transforme le problème en DAG de data science exécutable. "
            "Retourne un objet JSON avec id, objective et nodes. Chaque node contient id, role, "
            "depends_on et harness. Utilise provider='auto', model='auto', une allowlist tools minimale, "
            "approval='never' sauf action irréversible, max_retries<=2, network='deny'. Maximum 24 nodes."
        ),
        input=[{"problem": problem}],
        max_output_tokens=4_000,
    )
    response = manager.adapters[provider].invoke(request)
    plan = response.output.get("workflow", response.output)
    if not isinstance(plan, dict):
        raise ValueError("planner did not return a workflow object")
    plan.setdefault("id", "wf_generated")
    plan.setdefault("objective", problem)
    if len(plan.get("nodes", [])) > 24:
        raise ValueError("planner exceeded the 24-node limit")
    return parse_workflow(json.dumps(plan)), response.usage


@mcp.tool()
def platform_manifest() -> dict[str, Any]:
    """Return Orbia's providers and non-negotiable execution boundaries."""
    return {
        "providers": {
            "openai_codex": "local Codex CLI using ChatGPT subscription authentication",
            "github_copilot": "local GitHub Copilot SDK using the signed-in subscriber",
            "anthropic_claude": "local Claude Code CLI using Anthropic authentication",
            "ollama": "local Ollama HTTP runtime",
        },
        "boundaries": [
            "dependency-aware DAG",
            "token and model-turn budgets",
            "tool allowlist per agent",
            "no secret in workflow JSON",
            "human approval represented in the harness",
        ],
    }


@mcp.tool()
def start_workflow(
    workflow_json: str,
    provider: str = "openai_codex",
    max_tokens: int = 24000,
    max_model_turns: int = 12,
) -> dict:
    """Validate a workflow and start a controlled run. Returns the first ready agents."""
    workflow = parse_workflow(workflow_json)
    run = manager.start(
        workflow,
        RunLimits(max_tokens=max_tokens, max_model_turns=max_model_turns),
        default_provider=ProviderName(provider),
    )
    return {"run_id": run.id, "ready": [node.id for node in manager.ready(run.id)]}


@mcp.tool()
def solve_problem(
    problem: str,
    provider: str = "openai_codex",
    model: str = "auto",
    max_tokens: int = 24000,
    max_model_turns: int = 12,
) -> dict:
    """Plan a data-science DAG from one problem and execute it autonomously until blocked."""
    selected = ProviderName(provider)
    workflow, planner_usage = _plan_problem(problem, selected, model)
    run = manager.start(
        workflow,
        RunLimits(max_tokens=max_tokens, max_model_turns=max_model_turns),
        default_provider=selected,
    )
    run.budget.record(planner_usage)
    run.model_turns = 1
    outcome = manager.execute_until_blocked(run.id)
    return {"workflow_id": workflow.id, **outcome, **_run_status(run.id)}


@mcp.tool()
def run_agent(run_id: str, node_id: str) -> dict:
    """Execute one ready agent with its configured provider."""
    return manager.execute(run_id, node_id)


@mcp.tool()
def run_autonomous(run_id: str) -> dict:
    """Let Orbia execute the DAG until completion, approval, failure or a hard limit."""
    outcome = manager.execute_until_blocked(run_id)
    return {**outcome, **_run_status(run_id)}


@mcp.tool()
def approve_agent(run_id: str, node_id: str) -> dict:
    """Approve one ready agent whose harness requires human validation."""
    manager.approve(run_id, node_id)
    return _run_status(run_id)


@mcp.tool()
def run_status(run_id: str) -> dict:
    """Return completed nodes, ready nodes and remaining token budget."""
    return _run_status(run_id)


def _run_status(run_id: str) -> dict:
    run = manager.runs[run_id]
    return {
        "run_id": run.id,
        "completed": {node_id: result.state for node_id, result in run.results.items()},
        "ready": [node.id for node in manager.ready(run_id)],
        "model_turns": run.model_turns,
        "max_model_turns": run.limits.max_model_turns,
        "used_tokens": run.budget.used_tokens,
        "remaining_tokens": run.budget.remaining_tokens,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
