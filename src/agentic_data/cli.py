from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import ArtifactRegistry
from .contracts import AgentNode, Harness, Workflow
from .project_store import ProjectStore
from .providers import ProviderName, ProviderRequest
from .run_manager import RunLimits, RunManager
from .runners import ClaudeAdapter, CodexAdapter, CopilotAdapter, OllamaAdapter
from .tool_gateway import ToolGateway


def _adapters() -> dict[ProviderName, Any]:
    return {
        ProviderName.OPENAI_CODEX: CodexAdapter(),
        ProviderName.GITHUB_COPILOT: CopilotAdapter(),
        ProviderName.ANTHROPIC_CLAUDE: ClaudeAdapter(),
        ProviderName.OLLAMA: OllamaAdapter(),
    }


def _parse_workflow(data: dict[str, Any]) -> Workflow:
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
    return Workflow(id=data["id"], objective=data["objective"], nodes=tuple(nodes))


def _plan(
    problem: str,
    dataset_name: str,
    provider: ProviderName,
    model: str,
    adapters: dict[ProviderName, Any],
) -> tuple[Workflow, Any]:
    available_tools = list(ToolGateway(".ml-agentic/planner-manifest").tools)
    request = ProviderRequest(
        model=model,
        instructions=(
            "Tu es le planificateur ML.agentic. Transforme le problème en DAG de data science exécutable. "
            "Le dataset est disponible dans le workspace sous le nom indiqué. "
            "Retourne uniquement un objet JSON avec id, objective et nodes. Chaque node contient id, role, "
            "depends_on et harness. harness contient provider='auto', model='auto', tools, approval, "
            "max_retries et network. N'utilise que les outils fournis. Préfère data.inspect_csv pour explorer, "
            "python.run pour les calculs/modèles et file.write_text pour créer les rapports. "
            "Maximum 24 nodes. Ne demande pas de réseau."
        ),
        input=[{"problem": problem, "dataset": dataset_name, "available_tools": available_tools}],
        max_output_tokens=4_000,
    )
    response = adapters[provider].invoke(request)
    raw = response.output.get("workflow", response.output)
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
    return _parse_workflow(raw), response.usage


def _artifacts(workspace: Path, dataset_name: str) -> list[str]:
    return sorted(
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file() and path.name not in {dataset_name, "run.json"}
    )


def init_command(args: argparse.Namespace) -> dict[str, Any]:
    project = ProjectStore(args.projects_root).create(args.name, args.path)
    return {
        "project_id": project.id,
        "name": project.name,
        "path": str(project.root),
        "directories": ["datasets", "runs", "artifacts"],
        "events": str(project.events_file),
        "artifact_registry": str(project.root / "artifacts.jsonl"),
    }


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.data).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"dataset not found: {source}")
    if source.suffix.lower() != ".csv":
        raise ValueError("the first ML.agentic data runner currently accepts CSV files only")

    store = ProjectStore(args.projects_root)
    project = store.open(args.project) if args.project else None
    registry = ArtifactRegistry(project.root) if project else None
    if project:
        source = store.add_dataset(project, source)

    provider = ProviderName(args.provider)
    adapters = _adapters()
    dataset_name = "input.csv"
    workflow, planner_usage = _plan(args.problem, dataset_name, provider, args.model, adapters)
    workspace_root = project.runs_dir if project else Path(args.workspace_root)
    manager = RunManager(adapters, workspace_root=workspace_root)
    if project:
        manager.events.subscribe(lambda event: store.append_runtime_event(project, event))

    run = manager.start(
        workflow,
        RunLimits(max_tokens=args.max_tokens, max_model_turns=args.max_model_turns),
        default_provider=provider,
    )

    if registry:
        def register_artifact(event: Any) -> None:
            if event.type != "artifact.created" or event.run_id != run.id:
                return
            payload = event.payload
            registry.register(
                run.id,
                run.workspace,
                payload["path"],
                created_by=event.node_id or payload.get("created_by"),
                tool=payload.get("tool"),
            )

        manager.events.subscribe(register_artifact)

    shutil.copy2(source, run.workspace / dataset_name)
    run.budget.record(planner_usage)
    run.model_turns = 1

    outcome = manager.execute_until_blocked(run.id)
    artifact_records = registry.list(run.id) if registry else []
    summary = {
        "project_id": project.id if project else None,
        "run_id": run.id,
        "status": outcome["status"],
        "workflow_id": workflow.id,
        "workspace": str(run.workspace.resolve()),
        "dataset": dataset_name,
        "artifacts": artifact_records if registry else _artifacts(run.workspace, dataset_name),
        "model_turns": run.model_turns,
        "used_tokens": run.budget.used_tokens,
        "nodes": {
            node_id: {"state": result.state, "output": result.output, "error": result.error}
            for node_id, result in run.results.items()
        },
    }
    (run.workspace / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return summary


def artifacts_command(args: argparse.Namespace) -> dict[str, Any]:
    project = ProjectStore(args.projects_root).open(args.project)
    records = ArtifactRegistry(project.root).list(args.run_id)
    return {"project_id": project.id, "count": len(records), "artifacts": records}


def promote_command(args: argparse.Namespace) -> dict[str, Any]:
    project = ProjectStore(args.projects_root).open(args.project)
    registry = ArtifactRegistry(project.root)
    promoted = registry.promote(args.artifact_id, args.name)
    ProjectStore(args.projects_root).append_event(
        project,
        "artifact.promoted",
        {
            "artifact_id": args.artifact_id,
            "promoted_path": promoted["promoted_path"],
            "run_id": promoted["run_id"],
        },
    )
    return {"project_id": project.id, "artifact": promoted}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-agentic", description="Controlled agentic data-science runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create a persistent ML.agentic project")
    init.add_argument("name", help="Project name")
    init.add_argument("--path", help="Explicit project directory")
    init.add_argument("--projects-root", default=".ml-agentic/projects")

    run = sub.add_parser("run", help="Plan and execute a data-science workflow on a CSV")
    run.add_argument("--data", required=True, help="Path to the input CSV")
    run.add_argument("--problem", required=True, help="Business/data-science objective")
    run.add_argument("--project", help="Existing ML.agentic project directory")
    run.add_argument(
        "--provider",
        default="openai_codex",
        choices=[provider.value for provider in ProviderName],
    )
    run.add_argument("--model", default="auto")
    run.add_argument("--max-tokens", type=int, default=24_000)
    run.add_argument("--max-model-turns", type=int, default=12)
    run.add_argument("--workspace-root", default=".ml-agentic/runs")
    run.add_argument("--projects-root", default=".ml-agentic/projects")

    artifacts = sub.add_parser("artifacts", help="List project artifacts")
    artifacts.add_argument("--project", required=True, help="Existing ML.agentic project directory")
    artifacts.add_argument("--run-id", help="Only artifacts from one run")
    artifacts.add_argument("--projects-root", default=".ml-agentic/projects")

    promote = sub.add_parser("promote", help="Promote a run artifact to the project artifact directory")
    promote.add_argument("artifact_id", help="Artifact registry id")
    promote.add_argument("--project", required=True, help="Existing ML.agentic project directory")
    promote.add_argument("--name", help="Optional promoted file name")
    promote.add_argument("--projects-root", default=".ml-agentic/projects")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init":
        result = init_command(args)
    elif args.command == "run":
        result = run_command(args)
    elif args.command == "artifacts":
        result = artifacts_command(args)
    elif args.command == "promote":
        result = promote_command(args)
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
