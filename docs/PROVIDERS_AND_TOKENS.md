# OpenAI, Copilot and token control

## Provider boundary

Orbia owns the workflow, state machine, tool permissions, artifacts, approvals and budgets. Providers receive one bounded turn through a normalized adapter and return structured output plus normalized usage.

### OpenAI adapter

- Server-side OpenAI API key.
- Responses API for structured turns and tool calls.
- Prompt-cache metadata is normalized into cached input tokens.
- Recommended for business framing, evaluation, review and reporting.

### GitHub Copilot adapter

- GitHub Copilot SDK session, authenticated through a supported GitHub method.
- Hooks map Copilot tool and permission requests to Orbia's policy engine.
- Recommended for code generation, repository-aware work and data/ML implementation.

The workflow contract stores logical capabilities, never provider-specific tool payloads. A provider outage or budget decision can therefore use the configured fallback without changing the DAG.

## Token optimization that remains observable

1. Set hard budgets per run, phase, agent and turn.
2. Refuse a turn before dispatch when estimated input plus maximum output exceeds the remaining budget.
3. Send artifact IDs, schemas and relevant slices instead of copying complete datasets or notebooks.
4. Keep a short recent window and a versioned structured summary.
5. Cache stable system instructions, tool schemas and project policy.
6. Route coding turns to Copilot and business/review turns to OpenAI.
7. Stop after explicit tool-call and loop limits.
8. Record raw, cached, output and billable tokens for every provider call.
9. Require approval before automatically raising a budget.

Optimization must never silently remove required evidence. Context reduction emits an event describing what was summarized, referenced or dropped.

## MVP connection sequence

1. Implement the OpenAI Responses adapter and usage normalization.
2. Implement the Copilot SDK adapter and permission hooks.
3. Add a secret-reference store; secrets never enter workflow JSON or logs.
4. Add provider health checks and a no-side-effect test turn.
5. Add per-model price configuration and cost reconciliation.
6. Add fallback only for compatible output schemas and tool capabilities.

