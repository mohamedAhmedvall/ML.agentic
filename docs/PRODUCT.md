# Product brief

## Problem

Data science work is fragmented across tickets, notebooks, SQL editors, orchestration tools and review meetings. LLM assistants can generate code, but teams cannot safely delegate an end-to-end analysis because dependencies, permissions, reproducibility, review and lineage are missing.

## Product promise

Turn a business objective into an observable, editable and governed team of data agents. The platform should make autonomous work as inspectable as a production data pipeline.

## Primary users

- Data scientist: delegates profiling, experiment setup, diagnostics and reporting.
- Data engineer: controls connectors, environments, schemas and execution policy.
- ML engineer: validates evaluation, packaging and reproducibility.
- Analytics lead: reviews the plan, evidence, cost and final decision artifact.
- Platform/admin team: publishes approved tools and organization policies.

## Core object model

- **Project**: team boundary, data sources, secrets and policies.
- **Workflow version**: immutable DAG definition.
- **Agent node**: role plus its complete harness configuration.
- **Tool**: typed capability with schema, permissions and audit metadata.
- **Run**: execution of one workflow version with concrete inputs.
- **Artifact**: dataset, profile, chart, model, metric, report or code bundle.
- **Approval**: a human decision bound to a node input/output hash.

## The differentiator

The workflow canvas is not only a graph of transformations. Every node exposes an **agent harness**:

- role and system instructions;
- model/provider and fallback;
- allowed tools and data scopes;
- network and filesystem policy;
- token, money, time, CPU and memory budgets;
- retry and escalation policy;
- memory/context sources;
- expected structured output contract;
- evaluation rules and human approval gate.

## MVP scenario

A team uploads a CSV or Parquet dataset and asks: “Identify the drivers of churn and produce a validated executive report.” The planner proposes a DAG:

`Profile data → Clean data → Explore drivers → Train baseline → Validate evidence → Produce report`

The user edits the graph and harnesses, approves code execution, starts the run, watches node states, opens artifacts and retries only failed or invalidated descendants.

## MVP capabilities

### Must have

- One workspace and project with CSV/Parquet upload.
- Visual DAG with typed dependencies and cycle validation.
- Six built-in agent roles: planner, profiler, transformer, analyst, modeler, reviewer/reporting.
- Harness editor for model, instructions, tools, budgets, retries and approvals.
- Isolated Python execution with package allowlist and network off by default.
- Live node status, logs, output preview, artifacts and error explanation.
- Deterministic lineage and targeted re-run.
- Export/import workflow as JSON.

### Not in the first MVP

- Dozens of enterprise connectors.
- AutoML catalogue parity with Dataiku.
- Real-time streaming.
- Custom Kubernetes deployment UI.
- Marketplace for community agents.
- Fully autonomous production deployment.

## Success measures

- A new user reaches a first evidence-backed report in under 15 minutes.
- At least 80% of node failures are diagnosable from the run view without server access.
- Re-running one changed node avoids all unaffected work.
- Every result can be traced to input artifacts, code, environment and approvals.
- Zero generated-code execution inside the control-plane process.

## Strategic wedge

Do not start as “Dataiku plus chat.” Start as the best environment for **reviewable multi-agent analytical work**. Once execution, contracts and lineage are trusted, add connectors, schedules, shared templates and production promotion.

