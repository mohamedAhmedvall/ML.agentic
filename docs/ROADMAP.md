# Delivery roadmap

## Phase 0 — prove the kernel (current repository)

- Workflow/node contracts.
- Dependency-aware execution behavior.
- Interactive harness concept.
- Example portable workflow.

Exit: the team agrees on node lifecycle, artifact contract and the first end-to-end use case.

## Phase 1 — vertical slice

- Next.js workflow studio with React Flow.
- FastAPI CRUD for projects, workflows and versions.
- PostgreSQL schema and migrations.
- CSV/Parquet upload to MinIO.
- Planner creates a six-node DAG from an objective.
- Docker worker executes profiler and Python-analysis nodes.
- Run view streams states/logs and renders artifacts.

Exit: one user produces a reproducible churn-analysis report from a real dataset.

## Phase 2 — team-ready alpha

- Authentication, project roles and secret references.
- Tool catalogue with JSON Schema inputs/outputs.
- Human approval inbox.
- Targeted retries and downstream invalidation.
- Dataset and artifact lineage view.
- Cost, token, CPU and duration budgets.

Exit: a small data team can collaborate without sharing local notebooks or credentials.

## Phase 3 — governed execution

- Kubernetes Job executor with stronger sandboxing.
- Organization policies and environment promotion.
- Scheduled runs, webhooks and connector SDK.
- Evaluation suites and regression comparison.
- OpenTelemetry dashboards and audit export.

Exit: approved workflows can run repeatedly against governed data.

## First backlog, ordered

1. Freeze workflow/node/result JSON schemas.
2. Implement cycle detection and state transitions.
3. Define artifact storage and hashing convention.
4. Build workflow canvas and harness side panel.
5. Add event-streaming run API.
6. Implement Docker executor boundary.
7. Ship profiler agent with deterministic outputs.
8. Add generated Python agent behind approval.
9. Add reviewer agent with evidence citations.
10. Complete the churn-analysis golden path.

