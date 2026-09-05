import tempfile
import unittest
from pathlib import Path

from agentic_data.tool_gateway import ToolGateway, ToolGatewayError


class ToolGatewayTests(unittest.TestCase):
    def test_csv_inspection_and_file_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            gateway = ToolGateway(tmp)
            gateway.execute("file.write_text", {"path": "clients.csv", "content": "id,churn\n1,0\n2,1\n"})
            result = gateway.execute("data.inspect_csv", {"path": "clients.csv"}).output
            self.assertEqual(result["columns"], ["id", "churn"])
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["sample"][1]["churn"], "1")

    def test_python_runs_inside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            gateway = ToolGateway(tmp)
            result = gateway.execute("python.run", {"code": "from pathlib import Path; Path('artifact.txt').write_text('ok'); print('done')"}).output
            self.assertEqual(result["returncode"], 0)
            self.assertIn("done", result["stdout"])
            self.assertEqual(Path(tmp, "artifact.txt").read_text(), "ok")

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            gateway = ToolGateway(tmp)
            with self.assertRaises(ToolGatewayError):
                gateway.execute("file.read_text", {"path": "../secret.txt"})

    def test_unknown_tool_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            gateway = ToolGateway(tmp)
            with self.assertRaises(ToolGatewayError):
                gateway.execute("shell.exec", {"command": "echo nope"})


if __name__ == "__main__":
    unittest.main()
