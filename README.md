# Agentic Data Platform

An agent-native data science workspace: teams describe an objective, assemble or generate a dependency-aware workflow, configure each agent's harness, execute it safely, and inspect every result and artifact.

## Product thesis

Classic data platforms organize recipes, notebooks, and pipelines. This project organizes **delegated work**: agents plan, profile, transform, model, review, and report while humans keep control of tools, data access, budgets, and approvals.

The first useful slice is intentionally narrow:

1. Start from a dataset and a business objective.
2. Generate or edit a DAG of specialized agents.
3. Configure each agent: model, prompt, tools, permissions, limits, retry policy, and approval gate.
4. Execute nodes only when their dependencies succeeded.
5. Inspect logs, structured outputs, artifacts, lineage, cost, and failures.
6. Re-run one node and only the descendants invalidated by its new output.

## Repository map

- `docs/PRODUCT.md` — users, differentiator, MVP and non-goals
- `docs/ARCHITECTURE.md` — control plane, execution plane and security model
- `docs/ROADMAP.md` — prioritized delivery plan
- `specs/workflow.example.json` — portable workflow contract
- `src/agentic_data/` — dependency-aware orchestration kernel
- `tests/` — executable behavior tests
- `prototype/` — dependency-free interactive UI concept

## Run the kernel tests

```bash
python -m unittest discover -s tests -v
```

## Open the interface prototype

Open `prototype/index.html` in a browser. Select a node to edit its model, approval mode, retry count and tool allowlist, then run the workflow.

## MVP stack recommendation

| Layer | MVP | Scale-up path |
|---|---|---|
| Web | Next.js + React Flow | Same |
| API/control plane | FastAPI + Pydantic | Split orchestration services |
| State | PostgreSQL | Same, with event/outbox pattern |
| Queue | Redis + worker | NATS or managed queue |
| Artifacts | MinIO/S3 | Object storage + catalog |
| Execution | Ephemeral Docker containers | Kubernetes Jobs + gVisor/Firecracker |
| Agent runtime | Thin in-house harness | Pluggable adapters; avoid framework lock-in |
| Observability | OpenTelemetry | Central traces, evaluations and cost analytics |

## Working principles

- Generated code never runs inside the API process.
- Network is denied by default and enabled per tool/domain.
- Agents exchange versioned artifacts and structured contracts, not hidden chat history.
- Every run is reproducible from workflow version, dataset version, environment and seed.
- Human approval is a first-class node state, not an afterthought.

