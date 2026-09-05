from __future__ import annotations

from dataclasses import dataclass, field

from .providers import ProviderUsage


class TokenBudgetExceeded(RuntimeError):
    pass


@dataclass
class TokenBudget:
    max_tokens: int
    max_cost_micros: int
    used_tokens: int = 0
    used_cost_micros: int = 0
    cached_tokens: int = 0
    events: list[ProviderUsage] = field(default_factory=list)

    def assert_capacity(self, estimated_input: int, max_output: int) -> None:
        requested = estimated_input + max_output
        if self.used_tokens + requested > self.max_tokens:
            remaining = max(0, self.max_tokens - self.used_tokens)
            raise TokenBudgetExceeded(f"turn needs up to {requested} tokens but only {remaining} remain")

    def record(self, usage: ProviderUsage) -> None:
        next_tokens = self.used_tokens + usage.billable_tokens
        next_cost = self.used_cost_micros + usage.cost_micros
        if next_tokens > self.max_tokens:
            raise TokenBudgetExceeded("provider response exceeded token budget")
        if next_cost > self.max_cost_micros:
            raise TokenBudgetExceeded("provider response exceeded cost budget")
        self.used_tokens = next_tokens
        self.used_cost_micros = next_cost
        self.cached_tokens += usage.cached_input_tokens
        self.events.append(usage)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)


@dataclass(frozen=True)
class ContextPolicy:
    keep_recent_turns: int = 4
    structured_summary_tokens: int = 800
    inline_artifact_bytes: int = 0
    use_artifact_references: bool = True
    enable_prompt_cache: bool = True

    def validate(self) -> None:
        if self.keep_recent_turns < 0:
            raise ValueError("keep_recent_turns cannot be negative")
        if self.structured_summary_tokens < 0:
            raise ValueError("structured_summary_tokens cannot be negative")
        if self.inline_artifact_bytes and self.use_artifact_references:
            raise ValueError("referenced artifacts must not be duplicated inline")

