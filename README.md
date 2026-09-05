# Orbia — Agentic Data Platform

Orbia is an agent-native data science workspace. ChatGPT can be the reasoning host through the user's existing subscription; Orbia does **not** reuse browser cookies and does **not** require an OpenAI API key.

## Working MVP

- Dependency-aware workflow DAG.
- MCP tools for starting a run, requesting the next agent, recording ChatGPT-hosted results and inspecting status.
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

Sign in to GitHub Copilot once with the Copilot CLI. For Ollama, start the local service and pull the model configured in the workflow.

## Start the MCP server

Local stdio mode:

```bash
orbia-mcp
```

Streamable HTTP for ChatGPT/plugin development:

```bash
mcp run src/agentic_data/mcp_server.py --transport streamable-http
```

Connect the resulting `/mcp` endpoint to ChatGPT. ChatGPT executes `chatgpt_host` nodes in the current subscription session; `github_copilot` and `ollama` nodes run on the machine hosting Orbia.

## Security defaults

Provider calls only generate structured output. Agent tool execution is not silently enabled: a tool must be implemented, included in the agent allowlist, and pass the configured approval policy. Secrets stay in the runner environment and never enter workflow JSON or logs.

## Test

```bash
python -m unittest discover -s tests -v
```
