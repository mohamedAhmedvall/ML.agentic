from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class ProviderName(StrEnum):
    OPENAI = "openai"
    GITHUB_COPILOT = "github_copilot"


class TaskKind(StrEnum):
    BUSINESS_REASONING = "business_reasoning"
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    MODEL_REVIEW = "model_review"
    REPORTING = "reporting"


@dataclass(frozen=True)
class ProviderRequest:
    model: str
    instructions: str
    input: list[dict[str, Any]]
    tools: tuple[dict[str, Any], ...] = ()
    max_output_tokens: int = 2_000
    cache_key: str | None = None


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cost_micros: int = 0

    @property
    def billable_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens) + self.output_tokens


@dataclass(frozen=True)
class ProviderResponse:
    output: dict[str, Any]
    usage: ProviderUsage
    provider: ProviderName
    model: str
    trace_id: str | None = None


class ProviderAdapter(Protocol):
    name: ProviderName

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        """Execute one structured model turn and return normalized usage."""


@dataclass(frozen=True)
class Route:
    primary: ProviderName
    fallback: ProviderName
    reason: str


class ProviderRouter:
    """Explicit routing policy; no provider choice is hidden in prompts."""

    def route(self, task: TaskKind) -> Route:
        if task in {TaskKind.CODE_GENERATION, TaskKind.DATA_ANALYSIS}:
            return Route(
                primary=ProviderName.GITHUB_COPILOT,
                fallback=ProviderName.OPENAI,
                reason="code-oriented task",
            )
        return Route(
            primary=ProviderName.OPENAI,
            fallback=ProviderName.GITHUB_COPILOT,
            reason="reasoning or reporting task",
        )

