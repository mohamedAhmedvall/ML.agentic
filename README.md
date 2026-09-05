# ML.agentic — Agentic Data Platform

ML.agentic is an agent-native data science workspace. Its deterministic control plane—not a model provider—owns dependencies, budgets, approvals, tools and run state. The user presents a problem, selects a default provider, and ML.agentic lets agents execute a validated DAG autonomously.

## Working MVP

- Dependency-aware workflow DAG.
- MCP tools for starting a run, executing one agent, running autonomously and inspecting status.
- Four peer providers: Codex with ChatGPT subscription sign-in, GitHub Copilot, Claude Code and Ollama.
- Real local Ollama and GitHub Copilot adapters.
- Per-run token and model-turn limits.
- Per-agent provider, model, tool allowlist and approval gate.
- Structured results shared between dependent agents.
- Controlled local Tool Gateway with workspace-scoped file I/O, CSV inspection and Python execution.

## Repository map

- `src/agentic_data/` — contracts, budgets, orchestration, MCP server, Tool Gateway and provider adapters
- `specs/workflow.example.json` — portable workflow and harness configuration
- `tests/` — dependency, budget, approval, routing and tool tests
- `prototype/` — dependency-free interface concept
- `docs/` — product, architecture, providers and roadmap

The internal Python namespace remains `agentic_data` for compatibility; the product and package are ML.agentic / `ml-agentic`.

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

## Start the MCP server

```bash
ml-agentic-mcp
```

Or with streamable HTTP:

```bash
mcp run src/agentic_data/mcp_server.py --transport streamable-http
```

MCP is an optional control interface, not the orchestrator. A UI or compatible client can call `solve_problem`; the selected planner proposes a DAG, ML.agentic validates it, then executes it until completion or a control gate.

## Tool Gateway

`ToolGateway` is the execution boundary between agents and the machine. It exposes named capabilities rather than unrestricted shell access. The first capabilities are:

- `file.read_text`
- `file.write_text`
- `data.inspect_csv`
- `python.run`

Every path is constrained to a configured workspace. `python.run` executes with isolated Python mode, a bounded timeout and captured output. Agent-to-gateway wiring is the next runtime milestone; the gateway itself is intentionally deterministic and provider-independent.

## Global operation

1. The user describes the business/data problem and selects a provider.
2. The selected provider proposes a workflow DAG; ML.agentic validates its shape, dependencies and 24-node ceiling.
3. The scheduler releases nodes whose dependencies are satisfied.
4. Each agent receives only its role, dependency outputs, allowed tools and budget.
5. Tool requests must pass through the ML.agentic Tool Gateway and the agent allowlist.
6. ML.agentic continues until the DAG completes, a human gate is reached, a provider fails without fallback, or a hard budget is exhausted.

## Security defaults

Providers do not receive unrestricted machine access. A tool must be implemented by the gateway, included in the agent allowlist and pass the configured approval policy. Workspace paths cannot escape their sandbox. Secrets stay in the runner environment and never enter workflow JSON or logs.

## Test

```bash
python -m unittest discover -s tests -v
```
