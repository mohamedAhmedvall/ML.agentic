"""Local dashboard runner with durable checkpoints at agent boundaries.

An interrupted agent is never replayed automatically: its tools may have already
produced side effects. Paused and approval-gated runs can resume after restart.
"""
from __future__ import annotations

import json
import shutil
import threading
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .artifacts import ArtifactRegistry
from .contracts import NodeResult, NodeState
from .planning import parse_workflow, workflow_to_dict
from .project_store import ProjectStore
from .providers import ProviderName, ProviderUsage
from .run_manager import ManagedRun, RunLimits, RunManager
from .token_budget import TokenBudget
from .tool_gateway import AVAILABLE_TOOLS, DockerToolGateway


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding='utf-8')
    temporary.replace(path)


class RunService:
    def __init__(self, adapters, gateway_factory=None):
        self.adapters = adapters
        self.gateway_factory = gateway_factory or DockerToolGateway
        self.lock = threading.RLock()
        self.active: dict[tuple[str, str], threading.Thread] = {}
        self.pauses: dict[tuple[str, str], threading.Event] = {}

    def _paths(self, project_path, run_id):
        project = ProjectStore().open(project_path)
        if not run_id.startswith('run_') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in run_id):
            raise ValueError('invalid run id')
        return project, project.runs_dir / run_id

    def launch(self, project_path, max_tokens=24000, max_model_turns=12):
        with self.lock:
            project = ProjectStore().open(project_path)
            draft = json.loads((project.root / 'draft_workflow.json').read_text())
            workflow = parse_workflow(draft['workflow'])
            if not 1 <= len(workflow.nodes) <= 24:
                raise ValueError('workflow must contain 1 to 24 agents')
            RunManager._assert_acyclic(workflow)
            if not 1 <= max_tokens <= 1_000_000 or not 1 <= max_model_turns <= 1000:
                raise ValueError('invalid execution budget')
            for node in workflow.nodes:
                if set(node.harness.tools) - set(AVAILABLE_TOOLS):
                    raise ValueError('workflow contains unknown tools')
            # Resolve auto against the model actually selected during planning.
            workflow = replace(workflow, nodes=tuple(replace(n, harness=replace(n.harness,
                model=draft['model'] if n.harness.model == 'auto' else n.harness.model)) for n in workflow.nodes))
            source = (project.datasets_dir / draft['source_dataset']).resolve()
            if not source.is_relative_to(project.datasets_dir.resolve()) or not source.is_file():
                raise ValueError('dataset must belong to project')
            limits = RunLimits(max_tokens=max_tokens, max_model_turns=max_model_turns)
            budget = TokenBudget(limits.max_tokens, limits.max_cost_micros)
            budget.record(ProviderUsage(**draft['planner_usage']))
            manager = self._manager(project)
            run = manager.start(workflow, limits, ProviderName(draft['provider']))
            run.budget = budget
            run.model_turns = 1
            shutil.copy2(source, run.workspace / 'input.csv')
            self._save(run, 'paused')
            self._start(project, manager, run)
            return {'run_id': run.id, 'status': 'running'}

    def _manager(self, project):
        manager = RunManager(self.adapters, workspace_root=project.runs_dir,
                             gateway_factory=self.gateway_factory)
        manager.events.subscribe(lambda event: ProjectStore().append_runtime_event(project, event))
        registry = ArtifactRegistry(project.root)
        def register(event):
            if event.type == 'artifact.created':
                registry.register(event.run_id, project.runs_dir / event.run_id,
                                  event.payload['path'], created_by=event.node_id,
                                  tool=event.payload.get('tool'))
        manager.events.subscribe(register)
        return manager

    def _save(self, run, status, current=None, error=None):
        summary = {'run_id': run.id, 'status': status, 'workflow_id': run.workflow.id,
                   'workspace': str(run.workspace), 'model_turns': run.model_turns,
                   'used_tokens': run.budget.used_tokens, 'error': error,
                   'nodes': {k: asdict(v) for k, v in run.results.items()}}
        state = dict(summary, workflow=workflow_to_dict(run.workflow), limits=asdict(run.limits),
                     budget={'used_tokens': run.budget.used_tokens,
                             'used_cost_micros': run.budget.used_cost_micros,
                             'cached_tokens': run.budget.cached_tokens},
                     approvals=sorted(run.approvals), provider=run.default_provider.value,
                     current_node=current)
        # Keep control state outside the tool workspace.
        atomic_json(run.workspace.parent / (run.id + '.state.json'), state)
        atomic_json(run.workspace / 'run.json', summary)

    def _restore(self, project, workspace):
        state = json.loads((workspace.parent / (workspace.name + '.state.json')).read_text())
        if state['current_node'] or state['status'] == 'running':
            raise ValueError('Run interrupted during an agent: inspect its artifacts before starting a new run.')
        if state['status'] not in {'paused', 'approval_required'}:
            raise ValueError('only paused or approval-gated runs can resume')
        limits = RunLimits(**state['limits'])
        run = ManagedRun(state['run_id'], parse_workflow(state['workflow']), limits,
                         TokenBudget(limits.max_tokens, limits.max_cost_micros, **state['budget']),
                         workspace, results={k: NodeResult(**dict(v, state=NodeState(v['state'])))
                                             for k, v in state['nodes'].items()},
                         model_turns=state['model_turns'], approvals=set(state['approvals']),
                         default_provider=ProviderName(state['provider']))
        manager = self._manager(project)
        manager.runs[run.id] = run
        return manager, run

    def resume(self, project_path, run_id, approve_node=None):
        with self.lock:
            project, workspace = self._paths(project_path, run_id)
            key = (str(project.root), run_id)
            if key in self.active:
                raise ValueError('run is already executing')
            manager, run = self._restore(project, workspace)
            if approve_node:
                manager.approve(run.id, approve_node)
            self._start(project, manager, run)
            return {'run_id': run.id, 'status': 'running'}

    def pause(self, project_path, run_id):
        with self.lock:
            project, _ = self._paths(project_path, run_id)
            key = (str(project.root), run_id)
            if key not in self.active:
                raise ValueError('run is not executing')
            self.pauses[key].set()
            return {'run_id': run_id, 'status': 'pause_requested'}

    def _start(self, project, manager, run):
        key = (str(project.root), run.id)
        pause = threading.Event()
        self.pauses[key] = pause
        self._save(run, 'running')
        def work():
            try:
                while True:
                    if pause.is_set():
                        self._save(run, 'paused')
                        manager.events.emit('run.paused', run.id)
                        return
                    ready = manager.ready(run.id)
                    if not ready:
                        status = 'failed' if any(r.state == NodeState.FAILED for r in run.results.values()) else 'completed'
                        self._save(run, status)
                        manager.events.emit('run.' + status, run.id)
                        return
                    node = next((n for n in ready if n.harness.approval == 'never' or n.id in run.approvals), None)
                    if node is None:
                        for n in ready:
                            manager.events.emit('agent.awaiting_approval', run.id, node_id=n.id)
                        self._save(run, 'approval_required')
                        manager.events.emit('run.awaiting_approval', run.id, {'nodes': [n.id for n in ready]})
                        return
                    self._save(run, 'running', current=node.id)
                    result = manager.execute(run.id, node.id)
                    self._save(run, 'paused')
                    if result['status'] == 'failed':
                        self._save(run, 'failed', error=result.get('error'))
                        manager.events.emit('run.failed', run.id, {'error': result.get('error')})
                        return
            except Exception as exc:
                self._save(run, 'failed', error=str(exc))
                manager.events.emit('run.failed', run.id, {'error': str(exc)})
            finally:
                with self.lock:
                    self.active.pop(key, None)
                    self.pauses.pop(key, None)
        worker = threading.Thread(target=work, daemon=True, name=run.id)
        self.active[key] = worker
        worker.start()
