import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from agentic_data.providers import ProviderName, ProviderRouter, ProviderUsage, TaskKind
from agentic_data.token_budget import ContextPolicy, TokenBudget, TokenBudgetExceeded


class TokenBudgetTests(unittest.TestCase):
    def test_cached_input_is_not_counted_twice(self):
        budget = TokenBudget(max_tokens=10_000, max_cost_micros=1_000_000)
        budget.record(ProviderUsage(6_000, 1_000, cached_input_tokens=4_000))
        self.assertEqual(budget.used_tokens, 3_000)
        self.assertEqual(budget.cached_tokens, 4_000)

    def test_turn_is_blocked_before_worst_case_overflow(self):
        budget = TokenBudget(max_tokens=5_000, max_cost_micros=1_000_000, used_tokens=3_500)
        with self.assertRaises(TokenBudgetExceeded):
            budget.assert_capacity(estimated_input=1_000, max_output=1_000)

    def test_cost_is_a_hard_boundary(self):
        budget = TokenBudget(max_tokens=50_000, max_cost_micros=100)
        with self.assertRaisesRegex(TokenBudgetExceeded, "cost"):
            budget.record(ProviderUsage(100, 50, cost_micros=101))

    def test_provider_routing_is_explicit(self):
        router = ProviderRouter()
        self.assertEqual(router.route(TaskKind.CODE_GENERATION).primary, ProviderName.GITHUB_COPILOT)
        self.assertEqual(router.route(TaskKind.BUSINESS_REASONING).primary, ProviderName.OPENAI)

    def test_artifact_context_cannot_be_both_referenced_and_inlined(self):
        with self.assertRaises(ValueError):
            ContextPolicy(inline_artifact_bytes=100, use_artifact_references=True).validate()


if __name__ == "__main__":
    unittest.main()

