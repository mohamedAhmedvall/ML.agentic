from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class ProviderName(StrEnum):
    OPENAI_CODEX = "openai_codex"
    GITHUB_COPILOT = "github_copilot"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    OLLAMA = "ollama"


class MeasurementQuality(StrEnum):
    EXACT = "exact"
    ESTIMATED = "estimated"


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
    measurement: MeasurementQuality = MeasurementQuality.EXACT

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
    """Optional auto-routing policy. ML.agentic remains the orchestrator."""

    def route(self, task: TaskKind, default: ProviderName) -> Route:
        if task in {TaskKind.CODE_GENERATION, TaskKind.DATA_ANALYSIS}:
            fallback = ProviderName.OLLAMA if default != ProviderName.OLLAMA else ProviderName.GITHUB_COPILOT
            return Route(default, fallback, "user-selected provider; local fallback")
        fallback = ProviderName.OLLAMA if default != ProviderName.OLLAMA else ProviderName.OPENAI_CODEX
        return Route(default, fallback, "user-selected provider")
