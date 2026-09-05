import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_data.cli import run_command
from agentic_data.contracts import AgentNode, Harness, Workflow
from agentic_data.providers import ProviderName, ProviderRequest, ProviderResponse, ProviderUsage


class ToolCallingAdapter:
    def __init__(self):
        self.calls = 0

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            output = {
                "tool_call": {
                    "name": "file.write_text",
                    "arguments": {"path": "report.md", "content": "# Result\nCSV processed."},
                }
            }
        else:
            output = {"summary": "done"}
        return ProviderResponse(
            output=output,
            usage=ProviderUsage(20, 10),
            provider=ProviderName.OPENAI_CODEX,
            model="auto",
        )


class CliTests(unittest.TestCase):
    def test_run_copies_csv_and_collects_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "customers.csv"
            source.write_text("id,churn\n1,0\n2,1\n", encoding="utf-8")
            workflow = Workflow(
                "wf_test",
                "analyze churn Dataset: input.csv",
                (
                    AgentNode(
                        "report",
                        "analyst",
                        harness=Harness("auto", tools=("file.write_text",)),
                    ),
                ),
            )
            adapter = ToolCallingAdapter()
            args = argparse.Namespace(
                data=str(source),
                problem="analyze churn",
                provider="openai_codex",
                model="auto",
                max_tokens=5000,
                max_model_turns=5,
                workspace_root=str(root / "runs"),
            )
            with patch("agentic_data.cli._adapters", return_value={ProviderName.OPENAI_CODEX: adapter}), patch(
                "agentic_data.cli._plan", return_value=(workflow, ProviderUsage(10, 5))
            ):
                result = run_command(args)

            workspace = Path(result["workspace"])
            self.assertEqual(result["status"], "completed")
            self.assertTrue((workspace / "input.csv").is_file())
            self.assertTrue((workspace / "report.md").is_file())
            self.assertTrue((workspace / "run.json").is_file())
            self.assertIn("report.md", result["artifacts"])
            self.assertEqual(result["nodes"]["report"]["output"], {"summary": "done"})


if __name__ == "__main__":
    unittest.main()
