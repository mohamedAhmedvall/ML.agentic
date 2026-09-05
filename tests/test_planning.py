import unittest

from agentic_data.planning import parse_workflow, plan_workflow, workflow_to_dict
from agentic_data.providers import ProviderName, ProviderRequest, ProviderResponse, ProviderUsage


class PlannerAdapter:
    def __init__(self, output):
        self.output = output
        self.requests = []

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            output=self.output,
            usage=ProviderUsage(20, 10),
            provider=ProviderName.OLLAMA,
            model=request.model,
        )


class PlanningTests(unittest.TestCase):
    def test_plan_workflow_builds_valid_generic_dag(self):
        adapter = PlannerAdapter(
            {
                "workflow": {
                    "id": "wf_generic",
                    "objective": "analyze dataset",
                    "nodes": [
                        {
                            "id": "inspect",
                            "role": "inspector",
                            "depends_on": [],
                            "harness": {"tools": ["data.inspect_csv"], "provider": "auto", "model": "auto"},
                        },
                        {
                            "id": "report",
                            "role": "reporter",
                            "depends_on": ["inspect"],
                            "harness": {"tools": ["file.write_text"], "provider": "auto", "model": "auto"},
                        },
                    ],
                }
            }
        )
        workflow, usage = plan_workflow(
            "understand the dataset",
            "input.csv",
            ProviderName.OLLAMA,
            "qwen3",
            {ProviderName.OLLAMA: adapter},
        )
        self.assertEqual(workflow.id, "wf_generic")
        self.assertEqual(workflow.nodes[1].depends_on, ("inspect",))
        self.assertEqual(usage.billable_tokens, 30)
        self.assertIn("available_tools", adapter.requests[0].input[0])
        self.assertEqual(workflow_to_dict(workflow)["nodes"][1]["harness"]["tools"], ["file.write_text"])

    def test_plan_rejects_unknown_tool(self):
        adapter = PlannerAdapter(
            {
                "id": "bad",
                "objective": "bad plan",
                "nodes": [
                    {
                        "id": "x",
                        "role": "worker",
                        "harness": {"tools": ["shell.exec"], "provider": "auto", "model": "auto"},
                    }
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown tools"):
            plan_workflow("do work", "input.csv", ProviderName.OLLAMA, "qwen3", {ProviderName.OLLAMA: adapter})

    def test_parse_rejects_unknown_dependency(self):
        with self.assertRaisesRegex(ValueError, "unknown dependencies"):
            parse_workflow(
                {
                    "id": "bad-dependency",
                    "objective": "x",
                    "nodes": [
                        {
                            "id": "report",
                            "role": "reporter",
                            "depends_on": ["missing"],
                            "harness": {"tools": []},
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
