import json
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from agentic_data.project_store import ProjectStore
from agentic_data.providers import ProviderName, ProviderResponse, ProviderUsage
from agentic_data.run_service import RunService
from agentic_data.web_app import build_app
from agentic_data.tool_gateway import DockerToolGateway, ToolGatewayError


class Adapter:
    def __init__(self):
        self.requests = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def invoke(self, request):
        self.requests.append(request)
        if self.block:
            self.started.set()
            assert self.release.wait(3)
        if 'reporter' in request.instructions:
            output = {'report': 'report.md'} if request.input[0]['tool_history'] else {
                'tool_call': {'name': 'file.write_text', 'arguments': {'path': 'report.md', 'content': '# Result'}}}
        else:
            output = {'rows': 2}
        return ProviderResponse(output=output, usage=ProviderUsage(10, 5), provider=ProviderName.OLLAMA, model=request.model)


def setup_project(root, approval='never'):
    project = ProjectStore().create('Analysis', root / 'project')
    (project.datasets_dir / 'data.csv').write_text('x\n1\n2\n')
    draft = {'provider': 'ollama', 'model': 'chosen-model', 'source_dataset': 'data.csv',
             'planner_usage': {'input_tokens': 20, 'output_tokens': 10},
             'workflow': {'id': 'wf', 'objective': 'Inspect and report', 'nodes': [
                 {'id': 'inspect', 'role': 'inspector', 'harness': {'model': 'auto'}},
                 {'id': 'report', 'role': 'reporter', 'depends_on': ['inspect'],
                  'harness': {'model': 'auto', 'approval': approval, 'tools': ['file.write_text']}}]}}
    (project.root / 'draft_workflow.json').write_text(json.dumps(draft))
    return project


def settle(service):
    with service.lock:
        workers = list(service.active.values())
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()


class RunServiceTests(unittest.TestCase):
    def test_approval_resume_after_restart_keeps_results_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = setup_project(Path(tmp), 'always')
            adapter = Adapter()
            service = RunService({ProviderName.OLLAMA: adapter})
            rid = service.launch(p.root)['run_id']
            settle(service)
            summary = json.loads((p.runs_dir / rid / 'run.json').read_text())
            self.assertEqual(summary['status'], 'approval_required')
            self.assertEqual(summary['used_tokens'], 45)
            restarted = RunService({ProviderName.OLLAMA: adapter})
            restarted.resume(p.root, rid, 'report')
            settle(restarted)
            summary = json.loads((p.runs_dir / rid / 'run.json').read_text())
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['used_tokens'], 75)
            self.assertEqual(summary['model_turns'], 4)
            self.assertEqual(len(adapter.requests), 3)
            self.assertEqual(adapter.requests[1].input[0]['dependencies']['inspect'], {'rows': 2})
            self.assertTrue(all(r.model == 'chosen-model' for r in adapter.requests))
            self.assertTrue((p.runs_dir / rid / 'report.md').is_file())

    def test_pause_is_at_agent_boundary_and_duplicate_resume_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = setup_project(Path(tmp))
            adapter = Adapter(); adapter.block = True
            service = RunService({ProviderName.OLLAMA: adapter})
            rid = service.launch(p.root)['run_id']
            self.assertTrue(adapter.started.wait(2))
            with self.assertRaises(ValueError):
                service.resume(p.root, rid)
            service.pause(p.root, rid)
            adapter.block = False; adapter.release.set()
            settle(service)
            self.assertEqual(len(adapter.requests), 1)
            restarted = RunService({ProviderName.OLLAMA: adapter})
            restarted.resume(p.root, rid); settle(restarted)
            self.assertEqual(len(adapter.requests), 3)

    def test_interrupted_agent_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = setup_project(Path(tmp), 'always')
            service = RunService({ProviderName.OLLAMA: Adapter()})
            rid = service.launch(p.root)['run_id']; settle(service)
            file = p.runs_dir / (rid + '.state.json')
            state = json.loads(file.read_text()); state['current_node'] = 'report'; file.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'interrupted'):
                RunService({}).resume(p.root, rid)

    def test_http_launch_download_and_csrf(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = setup_project(Path(tmp))
            service = RunService({ProviderName.OLLAMA: Adapter()})
            client = TestClient(build_app(service), base_url='http://localhost')
            body = {'project_path': str(p.root)}
            self.assertEqual(client.post('/api/runs', json=body).status_code, 403)
            html = client.get('/').text
            token = re.search('name="control-token" content="([^"]+)"', html)[1]
            client.headers['X-ML-Agentic-Token'] = token
            response = client.post('/api/runs', json=body)
            self.assertEqual(response.status_code, 200, response.text)
            settle(service)
            snapshot = client.get('/api/project', params={'path': str(p.root)}).json()
            self.assertEqual(snapshot['runs'][0]['status'], 'completed')
            artifact = snapshot['artifacts'][0]
            download = client.get('/api/artifacts/' + artifact['id'], params={'path': str(p.root)})
            self.assertEqual(download.text, '# Result')
            self.assertIn('attachment', download.headers['content-disposition'])
            self.assertEqual(client.post('/api/datasets', json=dict(body, name='../escape.csv', content='x')).status_code, 400)

    def test_docker_unavailable_never_executes_host_python(self):
        with tempfile.TemporaryDirectory() as tmp, patch('shutil.which', return_value=None), patch('subprocess.run') as run:
            with self.assertRaisesRegex(ToolGatewayError, 'Docker is required'):
                DockerToolGateway(tmp).execute('python.run', {'code': 'print(1)'})
            run.assert_not_called()
