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
- End-to-end `ml-agentic run` command for CSV-based data-science workflows.

## Repository map

- `src/agentic_data/` — contracts, budgets, orchestration, CLI, MCP server, Tool Gateway and provider adapters
- `specs/workflow.example.json` — portable workflow and harness configuration
- `tests/` — dependency, budget, approval, routing, tool and CLI tests
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

## Run a real CSV workflow

```bash
ml-agentic run \
  --data clients.csv \
  --problem "Prédire le churn à 30 jours et produire un rapport avec les métriques du modèle" \
  --provider openai_codex
```

The planner receives the business objective, the dataset name and the exact Tool Gateway manifest. ML.agentic creates an isolated run workspace, copies the source dataset to `input.csv`, validates the generated DAG, executes agents and tool calls, then writes `run.json` with the final run summary.

Generated files such as reports, scripts, metrics or model artifacts remain in:

```text
.ml-agentic/runs/<run_id>/
```

The first CLI data runner accepts CSV files only. This is deliberate while dataset ingestion and artifact contracts are stabilized.

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

Every path is constrained to a configured workspace. `python.run` executes with isolated Python mode, a bounded timeout and captured output. Agents request tools through a provider-independent JSON protocol, and ML.agentic checks every request against the node's tool allowlist before execution.

## Global operation

1. The user describes the business/data problem and selects a provider.
2. The selected provider proposes a workflow DAG; ML.agentic validates its shape, dependencies, tool names and 24-node ceiling.
3. The scheduler releases nodes whose dependencies are satisfied.
4. Each agent receives only its role, dependency outputs, allowed tools and budget.
5. Tool requests pass through the ML.agentic Tool Gateway and the agent allowlist.
6. Tool results are returned to the agent until it emits a final structured result or hits a hard limit.
7. ML.agentic continues until the DAG completes, a human gate is reached, a provider fails without fallback, or a hard budget is exhausted.

## Security defaults

Providers do not receive unrestricted machine access. A tool must be implemented by the gateway, included in the agent allowlist and pass the configured approval policy. Workspace paths cannot escape their configured workspace. `python.run` is process-isolated and bounded, but it is not yet a hardened OS/container sandbox. Secrets stay in the runner environment and never enter workflow JSON or logs.

## Test

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the unit suite on Python 3.11 and 3.12 for pushes and pull requests.
