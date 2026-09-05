import json
import tempfile
import unittest
from pathlib import Path

from agentic_data.project_store import ProjectStore
from agentic_data.web_app import EventTail, _event_stream, project_snapshot


class WebAppTests(unittest.TestCase):
    def test_project_snapshot_exposes_project_runs_events_and_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demo"
            store = ProjectStore(Path(tmp) / "managed")
            project = store.create("Demo", root)
            (project.datasets_dir / "input.csv").write_text("x\n1\n", encoding="utf-8")
            run_dir = project.runs_dir / "run_1"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(json.dumps({"run_id": "run_1", "status": "succeeded"}), encoding="utf-8")
            store.append_event(project, "run.started", {"run_id": "run_1"})

            snapshot = project_snapshot(project.root)
            self.assertEqual(snapshot["project"]["name"], "Demo")
            self.assertEqual(snapshot["datasets"][0]["name"], "input.csv")
            self.assertEqual(snapshot["runs"][0]["run_id"], "run_1")
            self.assertTrue(any(event["type"] == "run.started" for event in snapshot["events"]))


class EventTailTests(unittest.TestCase):
    def test_partial_utf8_record_is_delivered_once_when_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_bytes(b'{"old":true}\n')
            tail = EventTail(path)
            record = json.dumps({"message": "terminé"}, ensure_ascii=False).encode()
            cut = record.index("é".encode()) + 1
            with path.open("ab") as handle:
                handle.write(record[:cut])
            self.assertEqual(tail.read(), [])
            with path.open("ab") as handle:
                handle.write(record[cut:] + b"\n")
            self.assertEqual(tail.read(), [{"message": "terminé"}])
            self.assertEqual(tail.read(), [])

    def test_partial_record_at_connection_does_not_replay_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_bytes(b'{"old":true}\n{"new":')
            tail = EventTail(path)
            self.assertEqual(tail.read(), [])
            with path.open("ab") as handle:
                handle.write(b'true}\n')
            self.assertEqual(tail.read(), [{"new": True}])

    def test_missing_replaced_and_truncated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            tail = EventTail(path)
            self.assertEqual(tail.read(), [])
            path.write_bytes(b'{"first_event":true}\n')
            self.assertEqual(tail.read(), [{"first_event": True}])
            path.write_bytes(b'{"short":1}\n')
            self.assertEqual(tail.read(), [{"short": 1}])
            replacement = path.with_suffix(".new")
            replacement.write_bytes(b'{"replacement_is_longer":true}\n')
            replacement.replace(path)
            self.assertEqual(tail.read(), [{"replacement_is_longer": True}])

    def test_corrupt_complete_record_does_not_block_following_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            tail = EventTail(path)
            path.write_bytes(b'broken\n[]\n{"valid":true}\n')
            self.assertEqual(tail.read(), [{"valid": True}])


class EventStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_event_is_streamed_after_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ProjectStore().create("Live", Path(tmp) / "project")
            stream = _event_stream(str(project.root))
            try:
                self.assertIn("event: ready", await anext(stream))
                ProjectStore().append_event(project, "agent.completed", {"result": "ok"})
                event = await anext(stream)
                self.assertIn("event: runtime", event)
                self.assertIn("agent.completed", event)
                self.assertIn("keepalive", await anext(stream))
            finally:
                await stream.aclose()


if __name__ == "__main__":
    unittest.main()
