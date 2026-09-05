# Providers and token control

## ChatGPT host — no OpenAI API

`chatgpt_host` means that the current ChatGPT conversation performs the reasoning turn and calls Orbia's MCP tools. Orbia returns a bounded prompt envelope, then records the structured result. No OpenAI API key, ChatGPT cookie, or unofficial browser automation is used.

ChatGPT token values are marked `estimated` because the MCP server does not receive an API billing record. Hard safety is therefore also enforced with model-turn limits, output limits, dependency gates and approvals.

## GitHub Copilot SDK

`github_copilot` uses the official Python SDK and the locally signed-in Copilot account. The adapter disables built-in tools for provider turns; Orbia-owned tools must be added explicitly behind policy checks. SDK usage is normalized as estimated tokens when exact counters are unavailable.

## Ollama

`ollama` calls the local `POST /api/chat` endpoint. Prompt, cached and generated token counters returned by Ollama are recorded exactly. Set `OLLAMA_URL` only when the runner is intentionally exposed elsewhere.

## Routing

| Task | Primary | Fallback |
|---|---|---|
| Business framing, review, reporting | ChatGPT host | Ollama |
| Data analysis, code generation | GitHub Copilot | Ollama |

Fallback is explicit and never changes tool permissions or approval requirements.

