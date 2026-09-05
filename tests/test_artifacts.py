import tempfile
import unittest
from pathlib import Path

from agentic_data.artifacts import ArtifactRegistry


class ArtifactRegistryTests(unittest.TestCase):
    def test_register_list_and_promote(self):
        with tempfile.TemporaryDirectory() as tempdir:
            project = Path(tempdir)
            workspace = project / "runs" / "run_demo"
            workspace.mkdir(parents=True)
            artifact = workspace / "report.md"
            artifact.write_text("# report\n", encoding="utf-8")

            registry = ArtifactRegistry(project)
            record = registry.register(
                "run_demo",
                workspace,
                "report.md",
                created_by="reporter",
                tool="file.write_text",
            )

            self.assertEqual(record.category, "report")
            self.assertEqual(record.created_by, "reporter")
            self.assertEqual(record.tool, "file.write_text")
            self.assertEqual(len(record.sha256), 64)
            self.assertEqual(registry.list("run_demo")[0]["id"], record.id)

            promoted = registry.promote(record.id, "final-report.md")
            self.assertEqual(promoted["status"], "promoted")
            self.assertEqual(promoted["promoted_path"], "artifacts/final-report.md")
            self.assertTrue((project / "artifacts" / "final-report.md").is_file())

    def test_register_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as tempdir:
            project = Path(tempdir)
            workspace = project / "runs" / "run_demo"
            workspace.mkdir(parents=True)
            outside = project / "outside.txt"
            outside.write_text("no", encoding="utf-8")
            registry = ArtifactRegistry(project)
            with self.assertRaisesRegex(ValueError, "inside the run workspace"):
                registry.register("run_demo", workspace, "../../outside.txt")


if __name__ == "__main__":
    unittest.main()
