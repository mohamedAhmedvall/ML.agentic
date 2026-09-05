# Providers and token control

## Orbia control plane

Orbia owns workflow state, DAG scheduling, tool policies, budgets and approvals. Providers only execute bounded agent turns. MCP is one optional control interface and ChatGPT is never required to run the workflow.

## OpenAI Codex

`openai_codex` invokes `codex exec --json` on the runner. The operator can authenticate with `codex login` using a ChatGPT subscription; no OpenAI API key is stored in Orbia. JSONL usage events are normalized into the common budget ledger.

## GitHub Copilot SDK

`github_copilot` uses the official Python SDK and the locally signed-in Copilot account. SDK usage is normalized as estimated tokens when exact counters are unavailable.

## Anthropic Claude

`anthropic_claude` invokes Claude Code in non-interactive JSON mode. Authentication remains local to Claude Code. Orbia supplies the prompt boundary, maximum turns and permission policy.

## Ollama

`ollama` calls the local `POST /api/chat` endpoint. Prompt, cached and generated token counters returned by Ollama are recorded exactly.

## Selection and fallback

| Scope | Provider |
|---|---|
| Run default | Selected by the user at launch |
| Agent override | Set in the agent harness |
| Fallback | Explicitly set in the agent harness |

Fallback is explicit and never changes tool permissions or approval requirements.
