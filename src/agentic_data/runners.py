from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from urllib.request import Request, urlopen

from .providers import (
    MeasurementQuality,
    ProviderName,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
)


class ProviderUnavailable(RuntimeError):
    pass


class HostExecutionRequired(RuntimeError):
    """Raised when the current ChatGPT conversation must execute the turn."""


def estimate_tokens(value: str) -> int:
    return max(1, (len(value) + 3) // 4)


@dataclass
class OllamaAdapter:
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 300
    name: ProviderName = ProviderName.OLLAMA

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        payload = {
            "model": request.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": json.dumps(request.input, ensure_ascii=False)},
            ],
            "options": {"num_predict": request.max_output_tokens},
        }
        http_request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read())
        except OSError as exc:
            raise ProviderUnavailable(f"Ollama indisponible sur {self.base_url}") from exc

        content = body.get("message", {}).get("content", "")
        try:
            output = json.loads(content)
        except json.JSONDecodeError:
            output = {"text": content}
        return ProviderResponse(
            output=output,
            usage=ProviderUsage(
                input_tokens=int(body.get("prompt_eval_count", 0)),
                output_tokens=int(body.get("eval_count", 0)),
                cached_input_tokens=int(body.get("prompt_eval_cached_count", 0)),
            ),
            provider=self.name,
            model=request.model,
        )


@dataclass
class CopilotAdapter:
    timeout_seconds: int = 300
    name: ProviderName = ProviderName.GITHUB_COPILOT

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._invoke(request))
        raise RuntimeError("CopilotAdapter.invoke must run outside an active asyncio loop")

    async def _invoke(self, request: ProviderRequest) -> ProviderResponse:
        try:
            from copilot import CopilotClient
            from copilot.session_events import AssistantMessageData, SessionIdleData
        except ImportError as exc:
            raise ProviderUnavailable("Installer l'option runtime: pip install -e '.[runtime]'") from exc

        messages: list[str] = []
        done = asyncio.Event()
        prompt = json.dumps(request.input, ensure_ascii=False)
        async with CopilotClient(use_logged_in_user=True) as client:
            async with await client.create_session(
                model=request.model,
                available_tools=[],
                system_message={"mode": "append", "content": request.instructions},
            ) as session:
                def on_event(event):
                    match event.data:
                        case AssistantMessageData() as data:
                            messages.append(data.content)
                        case SessionIdleData():
                            done.set()

                session.on(on_event)
                await session.send(prompt)
                await asyncio.wait_for(done.wait(), timeout=self.timeout_seconds)

        content = "\n".join(messages)
        try:
            output = json.loads(content)
        except json.JSONDecodeError:
            output = {"text": content}
        return ProviderResponse(
            output=output,
            usage=ProviderUsage(
                input_tokens=estimate_tokens(request.instructions + prompt),
                output_tokens=estimate_tokens(content),
                measurement=MeasurementQuality.ESTIMATED,
            ),
            provider=self.name,
            model=request.model,
        )


class ChatGPTHostAdapter:
    name = ProviderName.CHATGPT_HOST

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        raise HostExecutionRequired(
            "Ce tour doit être exécuté par ChatGPT, puis enregistré avec record_agent_result."
        )

