# Orbia — Agentic Data Platform

Orbia is an agent-native data science workspace. Its deterministic control plane—not a model provider—owns dependencies, budgets, approvals and run state. The user presents a problem, selects a default provider, and Orbia lets the agents execute the resulting DAG autonomously.

## Working MVP

- Dependency-aware workflow DAG.
- MCP tools for starting a run, executing one agent, running autonomously and inspecting status.
- Four peer providers: Codex with ChatGPT subscription sign-in, GitHub Copilot, Claude Code and Ollama.
- Real local Ollama adapter through `POST /api/chat`.
- Real GitHub Copilot SDK adapter using the locally signed-in Copilot user.
- Per-run token and model-turn limits.
- Per-agent provider, model, tool allowlist and approval gate.
- Structured results shared between dependent agents.

## Repository map

- `src/agentic_data/` — contracts, budgets, orchestration, MCP server and provider adapters
- `specs/workflow.example.json` — portable workflow and harness configuration
- `tests/` — dependency, budget, approval and routing tests
- `prototype/` — dependency-free interface concept
- `docs/` — product, architecture, providers and roadmap

## Install the runtime

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[runtime]"
python -m copilot download-runtime
```

Authenticate only the providers you want to expose on the runner:

```bash
codex login
claude auth login
```

Sign in to GitHub Copilot once with the Copilot CLI. For Ollama, start the local service and pull the model configured in the workflow. No credential belongs in workflow JSON.

## Start the MCP server

Local stdio mode:

```bash
orbia-mcp
```

Streamable HTTP for ChatGPT/plugin development:

```bash
mcp run src/agentic_data/mcp_server.py --transport streamable-http
```

MCP is an optional control interface, not the orchestrator. A UI or any compatible client can call `solve_problem` with a plain-language problem and provider: the selected planner proposes a DAG, Orbia validates it, then executes it until completion or a control gate. Existing DAGs can use `start_workflow`, then `run_autonomous`. Nodes configured with `auto` use the provider selected when the run starts; individual nodes can override it and declare a fallback.

## Global operation

1. The user describes the business/data problem and selects a provider.
2. The selected provider proposes a workflow DAG; Orbia validates its shape, dependencies and 24-node ceiling.
3. The scheduler releases nodes whose dependencies are satisfied.
4. Each agent receives only its role, dependency outputs, allowed tools and budget.
5. Orbia continues until the DAG completes, a human gate is reached, a provider fails without fallback, or a hard budget is exhausted.

## Security defaults

Provider calls only generate structured output. Agent tool execution is not silently enabled: a tool must be implemented, included in the agent allowlist, and pass the configured approval policy. Secrets stay in the runner environment and never enter workflow JSON or logs.

## Test

```bash
python -m unittest discover -s tests -v
```
