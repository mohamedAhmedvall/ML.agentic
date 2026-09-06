from __future__ import annotations

import asyncio
import json
import os
import secrets
from pathlib import Path
from typing import Any

from .artifacts import ArtifactRegistry
from .planning import plan_workflow, workflow_to_dict
from .project_store import ProjectStore
from .run_service import RunService
from .providers import ProviderName
from .runners import ClaudeAdapter, CodexAdapter, CopilotAdapter, OllamaAdapter, ProviderUnavailable


def _adapters() -> dict[ProviderName, Any]:
    return {
        ProviderName.OPENAI_CODEX: CodexAdapter(),
        ProviderName.GITHUB_COPILOT: CopilotAdapter(),
        ProviderName.ANTHROPIC_CLAUDE: ClaudeAdapter(),
        ProviderName.OLLAMA: OllamaAdapter(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if line.strip():
                out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def create_project(name: str, path: str | None = None, projects_root: str = ".ml-agentic/projects") -> dict[str, Any]:
    if not name.strip():
        raise ValueError("project name is required")
    project = ProjectStore(projects_root).create(name.strip(), path)
    return {"id": project.id, "name": project.name, "path": str(project.root)}


def generate_plan_preview(
    project_path: str | Path,
    problem: str,
    dataset_name: str,
    provider: str = "openai_codex",
    model: str = "auto",
    adapters: dict[ProviderName, Any] | None = None,
) -> dict[str, Any]:
    if not problem.strip():
        raise ValueError("problem is required")
    project = ProjectStore().open(project_path)
    dataset = (project.datasets_dir / dataset_name).resolve()
    try:
        dataset.relative_to(project.datasets_dir.resolve())
    except ValueError as exc:
        raise ValueError("dataset must belong to the project") from exc
    if not dataset.is_file():
        raise FileNotFoundError(f"dataset not found in project: {dataset_name}")

    provider_name = ProviderName(provider)
    workflow, usage = plan_workflow(
        problem.strip(),
        "input.csv",
        provider_name,
        model,
        adapters or _adapters(),
    )
    draft = {
        "project_id": project.id,
        "source_dataset": dataset_name,
        "provider": provider_name.value,
        "model": model,
        "workflow": workflow_to_dict(workflow),
        "planner_usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cost_micros": usage.cost_micros,
            "measurement": usage.measurement,
        },
    }
    (project.root / "draft_workflow.json").write_text(
        json.dumps(draft, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    ProjectStore().append_event(
        project,
        "workflow.planned",
        {
            "workflow_id": workflow.id,
            "objective": workflow.objective,
            "nodes": [node.id for node in workflow.nodes],
            "provider": provider_name.value,
            "dataset": dataset_name,
        },
    )
    return draft


def project_snapshot(project_path: str | Path) -> dict[str, Any]:
    project = ProjectStore().open(project_path)
    runs: list[dict[str, Any]] = []
    for directory in sorted(project.runs_dir.iterdir(), reverse=True) if project.runs_dir.exists() else []:
        if not directory.is_dir():
            continue
        summary = directory / "run.json"
        try:
            runs.append(
                _read_json(summary)
                if summary.is_file()
                else {"run_id": directory.name, "status": "running", "workspace": str(directory)}
            )
        except (json.JSONDecodeError, OSError):
            runs.append({"run_id": directory.name, "status": "unknown", "workspace": str(directory)})

    datasets = [
        {"name": path.name, "path": str(path.relative_to(project.root)), "size_bytes": path.stat().st_size}
        for path in sorted(project.datasets_dir.iterdir())
        if path.is_file()
    ] if project.datasets_dir.exists() else []
    draft_file = project.root / "draft_workflow.json"
    draft = _read_json(draft_file) if draft_file.is_file() else None
    return {
        "project": {"id": project.id, "name": project.name, "path": str(project.root)},
        "datasets": datasets,
        "runs": runs,
        "events": _read_events(project.events_file),
        "artifacts": ArtifactRegistry(project.root).list(),
        "draft": draft,
    }


class EventTail:
    """Read complete JSONL records using byte offsets; retain unfinished writes."""

    def __init__(self, path: Path):
        self.path = path
        self.position = 0
        self.identity = None
        if path.is_file():
            stat = path.stat()
            self.identity = (stat.st_dev, stat.st_ino)
            # Start after the last complete record, retaining only a partial tail.
            with path.open("rb") as handle:
                end = stat.st_size
                while end:
                    start = max(0, end - 8192)
                    handle.seek(start)
                    index = handle.read(end - start).rfind(b"\n")
                    if index >= 0:
                        self.position = start + index + 1
                        break
                    end = start

    def read(self) -> list[dict[str, Any]]:
        events = []
        try:
            handle = self.path.open("rb")
        except FileNotFoundError:
            return events
        with handle:
            stat = os.fstat(handle.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if identity != self.identity or stat.st_size < self.position:
                self.position = 0
            self.identity = identity
            handle.seek(self.position)
            while True:
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    break
                self.position = handle.tell()
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(event, dict):
                    events.append(event)
        return events


async def _event_stream(project_path: str):
    project = ProjectStore().open(project_path)
    tail = EventTail(project.events_file)
    yield "event: ready\ndata: {}\n\n"
    while True:
        for event in tail.read():
            yield f"event: runtime\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield ": keepalive\n\n"
        await asyncio.sleep(.75)

def build_app(service=None):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
    except ImportError as exc:
        raise RuntimeError('Install the web extra with: pip install -e ".[web]"') from exc

    app = FastAPI(title="ML.agentic", version="0.4")

    service = service or RunService(_adapters())
    app.state.run_service = service
    token = secrets.token_urlsafe(32)

    @app.middleware("http")
    async def local_only(request, call_next):
        if request.url.hostname not in {"127.0.0.1", "localhost"}:
            return JSONResponse({"detail": "Local host required"}, status_code=403)
        if request.method == "POST":
            if not secrets.compare_digest(request.headers.get("x-ml-agentic-token", ""), token):
                return JSONResponse({"detail": "Reload the dashboard to authorize this action"}, status_code=403)
            try:
                size = int(request.headers.get("content-length", "-1"))
            except ValueError:
                size = -1
            if size < 0 or size > 8_000_000:
                return JSONResponse({"detail": "Maximum request size: 8 MB"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(ProviderUnavailable)
    def provider_unavailable(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    @app.exception_handler(OSError)
    def input_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.post("/api/datasets")
    def upload_dataset(payload: dict[str, Any]):
        project = ProjectStore().open(payload['project_path'])
        name = str(payload['name'])
        if Path(name).name != name or not name.lower().endswith('.csv'):
            raise ValueError('Choose a CSV file name')
        content = str(payload['content']).encode('utf-8')
        if len(content) > 5_000_000:
            raise ValueError('CSV limit: 5 MB')
        with (project.datasets_dir / name).open('xb') as handle:
            handle.write(content)
        ProjectStore().append_event(project, 'dataset.added', {'path': 'datasets/' + name})
        return {'name': name}

    @app.post("/api/runs")
    def launch_run(payload: dict[str, Any]):
        return service.launch(payload['project_path'], int(payload.get('max_tokens', 24000)),
                              int(payload.get('max_model_turns', 12)))

    @app.post("/api/runs/{run_id}/pause")
    def pause_run(run_id: str, payload: dict[str, Any]):
        return service.pause(payload['project_path'], run_id)

    @app.post("/api/runs/{run_id}/resume")
    def resume_run(run_id: str, payload: dict[str, Any]):
        return service.resume(payload['project_path'], run_id, payload.get('approve_node'))

    @app.get("/api/artifacts/{artifact_id}")
    def download_artifact(artifact_id: str, path: str):
        project = ProjectStore().open(path)
        record = ArtifactRegistry(project.root).get(artifact_id)
        file = (project.root / record['path']).resolve()
        if not file.is_relative_to(project.runs_dir.resolve()) or not file.is_file():
            raise ValueError('Artifact is outside the project runs')
        return FileResponse(file, filename=record['name'], media_type='application/octet-stream',
                            headers={'X-Content-Type-Options': 'nosniff'})

    @app.get("/api/project")
    def get_project(path: str = Query(...)):
        try:
            snapshot = project_snapshot(path)
            with service.lock:
                for run in snapshot['runs']:
                    key = (snapshot['project']['path'], run['run_id'])
                    if run.get('status') == 'running' and key not in service.active:
                        run['status'] = 'interrupted'
                        run['error'] = 'Exécution interrompue : vérifier les artefacts avant de relancer.'
            return snapshot
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/projects")
    def new_project(payload: dict[str, Any]):
        try:
            return create_project(
                str(payload.get("name", "")),
                payload.get("path"),
                str(payload.get("projects_root", ".ml-agentic/projects")),
            )
        except (FileExistsError, ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/plan")
    def create_plan(payload: dict[str, Any]):
        try:
            return generate_plan_preview(
                payload["project_path"],
                str(payload.get("problem", "")),
                payload["dataset"],
                str(payload.get("provider", "openai_codex")),
                str(payload.get("model", "auto")),
            )
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/events")
    async def stream_events(path: str = Query(...)):
        try:
            ProjectStore().open(path)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return StreamingResponse(
            _event_stream(path),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return _DASHBOARD_HTML.replace("__CONTROL_TOKEN__", token)

    return app


app = build_app()

_DASHBOARD_HTML = r'''<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="control-token" content="__CONTROL_TOKEN__"><title>ML.agentic — Projets</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#0b101b;color:#edf2fa;font-size:16px;--panel:#121b2a;--border:#2c3950;--muted:#acb9cd;--accent:#8de2c3}*{box-sizing:border-box}body{margin:0}button,input,textarea,select{font:inherit}button,a,input,select,textarea{outline-offset:4px}button{cursor:pointer;border:1px solid var(--border);border-radius:9px;background:#243249;color:inherit;padding:10px 14px}button:hover{background:#324461}button:disabled{opacity:.45;cursor:default}.primary{background:var(--accent);color:#082d23;border-color:var(--accent);font-weight:650}.primary:hover{background:#b0f5da}a{color:#a9d7ff}input,textarea,select{width:100%;padding:11px;border:1px solid #40506a;background:#0d1523;color:inherit;border-radius:8px;margin-top:6px}textarea{min-height:110px;resize:vertical}label{display:block;font-size:14px;font-weight:550;margin:12px 0}header{padding:18px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:16px}h1{font-size:21px;margin:0;letter-spacing:-.5px}h2{font-size:18px;margin:0 0 12px}h3{font-size:16px;margin:0 0 10px}p{line-height:1.55}small,.muted{color:var(--muted);font-size:14px}.shell{display:grid;grid-template-columns:280px minmax(0,1fr);min-height:calc(100vh - 68px)}aside{padding:24px 18px;border-right:1px solid var(--border);background:#0e1623}main{padding:26px;max-width:1500px;width:100%;margin:auto}.card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px;margin-bottom:18px}.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}.workspace{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:22px}.steps{display:flex;gap:12px;margin-bottom:22px;flex-wrap:wrap}.step{border:1px solid var(--border);padding:8px 13px;border-radius:30px;font-size:14px}.step.done{color:var(--accent);border-color:#316757}.actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:12px}.choice{display:block;width:100%;text-align:left;margin:8px 0;overflow-wrap:anywhere}.choice.active{border-color:var(--accent);background:#1a3935}.badge{font-size:13px;padding:3px 8px;background:#293a53;border-radius:6px;display:inline-block;margin:4px 4px 0 0}.success{color:#8de2c3}.failed{color:#ffb3b3}.running{color:#f1d699}.waiting{color:#a8ccff}.empty{padding:25px 10px;color:var(--muted);line-height:1.6}.notice{margin:0 0 20px;padding:14px 18px;border-left:3px solid var(--accent);background:#142d2b;white-space:pre-wrap;overflow-wrap:anywhere}.notice.error{border-color:#ff9f9f;background:#341d26}details{margin-top:16px}summary{cursor:pointer;font-size:14px;color:#bad0ea;padding:6px 0}pre{font-family:ui-monospace,Consolas,monospace;font-size:13px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;max-height:320px;overflow:auto;background:#0a1220;border:1px solid var(--border);border-radius:8px;padding:12px}.node{display:block;width:100%;text-align:left;margin:10px 0;padding:14px}.node.selected{border-color:var(--accent)}.node strong{display:block;margin-bottom:4px}.files a{display:block;padding:12px 0;text-decoration:none;border-bottom:1px solid var(--border)}.files small{display:block;margin-top:5px}.section{margin-top:25px}.budget{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}.budget span{font-size:14px;color:var(--muted)}#projectName{overflow-wrap:anywhere}#agentDetail{min-height:100px}progress{width:100%;accent-color:var(--accent);height:8px}.hidden,[hidden]{display:none!important}@media(max-width:1150px){.workspace{grid-template-columns:1fr}}@media(max-width:760px){.shell{display:block}aside{border-right:0;border-bottom:1px solid var(--border)}main{padding:18px}.row{grid-template-columns:1fr}header{padding:16px}.card{padding:16px}.workspace{gap:0}}@media(prefers-reduced-motion:no-preference){button{transition:background .12s}}
</style></head><body>
<header><h1>ML.agentic <small> / Espace de travail</small></h1><span id="connection" class="muted">Choisis un projet pour commencer</span></header>
<div class="shell"><aside><h2>Projets</h2><form id="createForm"><label for="projectLabel">Nouveau projet</label><input id="projectLabel" required placeholder="Ex. Analyse des ventes" maxlength="120"><button class="primary" style="width:100%;margin-top:10px" type="submit">Créer le projet</button></form>
<details><summary>Ouvrir un dossier existant</summary><form id="openForm"><label for="projectPath">Dossier du projet ML.agentic</label><input id="projectPath" required placeholder="C:\mes-projets\analyse"><button type="submit" style="margin-top:10px">Ouvrir</button></form></details>
<div class="section" id="recentSection" hidden><h3>Récents sur cet appareil</h3><div id="recents"></div></div><div class="section"><h3 id="projectName">Aucun projet ouvert</h3><p id="projectHint" class="muted">Crée un projet pour regrouper ses données, ses exécutions et ses résultats.</p></div>
<div id="runSection" class="section" hidden><h3>Exécutions</h3><button id="showDraft" class="choice">Objectif et plan proposé</button><div id="runs"></div></div></aside>
<main><div id="notice" class="notice" role="status" aria-live="polite" hidden></div><div class="steps"><span id="stepData" class="step">1 · Ajouter des données</span><span id="stepPlan" class="step">2 · Décrire l’objectif</span><span id="stepRun" class="step">3 · Suivre les résultats</span></div>
<div class="workspace"><section><div class="card"><h2>1. Tes données</h2><p class="muted">Importe un CSV. Il sera conservé dans ce projet.</p><label for="upload">Fichier CSV · UTF-8 · 5 Mo maximum</label><input id="upload" type="file" accept=".csv" disabled><div id="datasets" class="muted empty">Les données apparaîtront ici.</div></div>
<form id="planForm" class="card"><h2>2. Quel résultat veux-tu obtenir ?</h2><fieldset id="planningFields" disabled style="border:0;padding:0;margin:0"><label for="dataset">Données à analyser</label><select id="dataset" required><option value="">Importe d’abord un CSV</option></select><label for="problem">Objectif</label><textarea id="problem" required placeholder="Ex. Analyser les ventes par région et produire un rapport avec les principaux constats."></textarea><label for="provider">Provider</label><select id="provider"><option value="openai_codex">Codex · compte ChatGPT</option><option value="github_copilot">GitHub Copilot</option><option value="anthropic_claude">Claude Code</option><option value="ollama">Ollama · local</option></select><p class="muted">Le provider choisi doit être installé et connecté sur ce poste. ML.agentic conserve le contrôle de l’exécution.</p><details><summary>Modèle et limites d’exécution</summary><label for="model">Modèle (« auto » utilise le choix du provider)</label><input id="model" value="auto"><div class="row"><label for="tokens">Budget tokens<input id="tokens" type="number" min="1" max="1000000" value="24000" required></label><label for="turns">Appels au modèle maximum<input id="turns" type="number" min="1" max="1000" value="12" required></label></div><p class="muted">La proposition du plan consomme déjà un appel au provider. Son usage déclaré est compté au lancement.</p></details><div class="actions"><button id="planButton" type="submit" class="primary">Proposer un plan</button><span id="planningStatus" class="muted" aria-live="polite"></span></div></fieldset></form></section>
<section><div class="card"><h2 id="workflowTitle">3. Plan et exécution</h2><p id="objective" class="muted">Une fois l’objectif décrit, les agents proposés et leurs dépendances apparaîtront ici.</p><div id="runSummary"></div><div id="controls" class="actions"></div><div id="workflow" class="empty">Aucun plan pour le moment.</div></div><div class="card"><h2 id="agentTitle">Détail d’un agent</h2><div id="agentDetail" class="muted">Sélectionne un agent pour voir ses outils, ses actions et ses résultats.</div></div><div class="card"><h2>Livrables</h2><div id="artifacts" class="files empty">Les fichiers produits par l’exécution sélectionnée apparaîtront ici.</div></div></section></div></main></div>
<script>
'use strict';
const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={completed:'Terminée',succeeded:'Terminé',running:'En cours',paused:'En pause',pause_requested:'Pause demandée',approval_required:'Validation requise',awaiting_approval:'À approuver',failed:'Échec',skipped:'Non exécuté',pending:'En attente',interrupted:'Interrompue',unknown:'État inconnu'};
const eventLabels={'agent.started':'Agent démarré','agent.completed':'Résultat de l’agent','agent.failed':'Erreur de l’agent','tool.called':'Outil demandé','tool.completed':'Résultat de l’outil','tool.failed':'Erreur de l’outil','model.turn.completed':'Appel au modèle terminé','agent.approved':'Validation enregistrée','agent.awaiting_approval':'Validation requise','artifact.created':'Fichier produit'};
let current=null,data=null,activeRun=null,selected=null,source=null,refreshId=0,busy=false,nodes=[],states={},planning=false;
const statusLabel=s=>labels[s]||s||'En attente';const statusClass=s=>['succeeded','completed'].includes(s)?'success':['failed','interrupted'].includes(s)?'failed':s==='running'?'running':['approval_required','awaiting_approval'].includes(s)?'waiting':'muted';
function notice(message,error=false){$('notice').hidden=!message;$('notice').textContent=message;$('notice').className='notice'+(error?' error':'')}
async function api(url,body){const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-ML-Agentic-Token':document.querySelector('meta[name="control-token"]').content},body:JSON.stringify(body)};let response;try{response=await fetch(url,options)}catch{throw Error('Le serveur ne répond plus. Vérifie que le terminal ML.agentic est ouvert.')}const type=response.headers.get('content-type')||'';const value=type.includes('application/json')?await response.json():{};if(!response.ok)throw Error(typeof value.detail==='string'?value.detail:'Action impossible. Consulte le terminal pour le détail.');return value}
async function action(fn){if(busy)return;busy=true;updateEnabled();try{await fn()}catch(e){notice(e.message,true)}finally{busy=false;updateEnabled()}}
function updateEnabled(){$('planningFields').disabled=!current||!data?.datasets.length||busy;$('upload').disabled=!current||busy;document.querySelectorAll('#createForm button,#openForm button,#controls button').forEach(b=>b.disabled=busy);$('planButton').textContent=planning?'Proposition en cours…':'Proposer un plan'}
function remember(){try{let recent=JSON.parse(localStorage.getItem('ml-agentic-recents')||'[]');recent=recent.filter(x=>x.path!==current);recent.unshift({path:current,name:data.project.name});localStorage.setItem('ml-agentic-recents',JSON.stringify(recent.slice(0,5)));localStorage.setItem('ml-agentic-project',current)}catch{}renderRecents()}
function renderRecents(){let recent=[];try{recent=JSON.parse(localStorage.getItem('ml-agentic-recents')||'[]');if(!Array.isArray(recent))recent=[]}catch{}$('recents').replaceChildren();$('recentSection').hidden=!recent.length;for(const p of recent){const b=document.createElement('button');b.className='choice';b.textContent=p.name;b.title=p.path;b.onclick=()=>action(()=>openProject(p.path));$('recents').appendChild(b)}}
async function openProject(path){const snapshot=await api('/api/project?path='+encodeURIComponent(path));if(source)source.close();current=snapshot.project.path;data=snapshot;activeRun=null;selected=null;$('projectPath').value=current;notice('Projet ouvert : '+data.project.name);remember();render();connect()}
async function refresh(){if(!current)return;const path=current,id=++refreshId;const snapshot=await api('/api/project?path='+encodeURIComponent(path));if(path!==current||id!==refreshId)return;data=snapshot;render()}
function connect(){source=new EventSource('/api/events?path='+encodeURIComponent(current));source.addEventListener('ready',()=>{$('connection').textContent='Suivi connecté';$('connection').className='success';refresh().catch(()=>{})});source.addEventListener('runtime',()=>refresh().catch(()=>{}));source.onerror=()=>{$('connection').textContent='Reconnexion au suivi…';$('connection').className='muted'}}
function ownEvents(){return(data?.events||[]).filter(e=>e.run_id===activeRun)}
function derive(){const events=ownEvents(),start=events.find(e=>e.type==='run.started'&&Array.isArray(e.payload?.nodes));nodes=activeRun?(start?.payload.nodes||[]):(data?.draft?.workflow.nodes||[]).map(n=>({...n,...n.harness}));states={};const map={'agent.started':'running','agent.completed':'succeeded','agent.failed':'failed','agent.skipped':'skipped','agent.awaiting_approval':'awaiting_approval'};for(const e of events)if(e.node_id&&map[e.type])states[e.node_id]=map[e.type];const run=data.runs.find(r=>r.run_id===activeRun);for(const[id,result]of Object.entries(run?.nodes||{}))states[id]=result.state;return {run,start}}
function button(label,handler,primary=false){const b=document.createElement('button');b.textContent=label;b.className=primary?'primary':'';b.onclick=()=>action(handler);return b}
function render(){if(!data)return;const{run,start}=derive();$('projectName').textContent=data.project.name;$('projectHint').textContent=data.datasets.length+' fichier(s) · '+data.runs.length+' exécution(s)';$('runSection').hidden=false;
$('stepData').className='step'+(data.datasets.length?' done':'');$('stepPlan').className='step'+(data.draft?' done':'');$('stepRun').className='step'+(run?.status==='completed'?' done':'');
$('datasets').className='muted';$('datasets').textContent=data.datasets.length?data.datasets.map(d=>d.name+' · '+Math.max(1,Math.round(d.size_bytes/1024))+' Ko').join('\n'):'Aucun fichier importé.';$('datasets').style.whiteSpace='pre-line';
const chosen=$('dataset').value,signature=JSON.stringify(data.datasets.map(d=>d.name));if($('dataset').dataset.signature!==signature){$('dataset').replaceChildren();if(!data.datasets.length)$('dataset').add(new Option('Importe d’abord un CSV',''));for(const d of data.datasets)$('dataset').add(new Option(d.name,d.name));if(data.datasets.some(d=>d.name===chosen))$('dataset').value=chosen;$('dataset').dataset.signature=signature}
$('runs').replaceChildren();for(const [i,r]of data.runs.entries()){const b=button('Exécution '+(i+1)+' · '+statusLabel(r.status),async()=>{activeRun=r.run_id;selected=null;render()});b.title=r.run_id;b.className='choice'+(activeRun===r.run_id?' active':'');$('runs').appendChild(b)}if(!data.runs.length)$('runs').textContent='Aucune exécution lancée.';
$('workflowTitle').textContent=activeRun?'3. Suivi de l’exécution':'3. Vérifier le plan';$('objective').textContent=start?.payload.objective||(!activeRun?data.draft?.workflow.objective:'')||'Propose un plan pour voir les agents et leurs outils.';
$('controls').replaceChildren();$('runSummary').replaceChildren();
if(!activeRun&&data.draft){$('controls').appendChild(button('Valider et lancer',launchRun,true));$('runSummary').textContent=nodes.length+' agents proposés · '+data.draft.provider+' · '+data.draft.model}
if(run){const done=Object.values(run.nodes||{}).filter(n=>n.state==='succeeded').length;$('runSummary').innerHTML=`<p class="${statusClass(run.status)}">${esc(statusLabel(run.status))}</p><progress max="${Math.max(1,nodes.length)}" value="${done}" aria-label="Agents terminés"></progress><div class="budget"><span>${done}/${nodes.length} agents terminés</span><span>${esc(run.used_tokens||0)} tokens déclarés</span><span>${esc(run.model_turns||0)} appels</span></div>${run.error?`<p class="failed">${esc(run.error)}</p>`:''}`;
if(run.status==='running')$('controls').appendChild(button('Pause après cet agent',()=>control('pause')));if(run.status==='paused')$('controls').appendChild(button('Reprendre',()=>control('resume'),true));if(run.status==='approval_required')for(const n of nodes.filter(n=>states[n.id]==='awaiting_approval'))$('controls').appendChild(button('Approuver '+n.role,()=>control('resume',n.id),true))}
$('workflow').className='';$('workflow').replaceChildren();for(const n of nodes){const b=document.createElement('button');b.className='node'+(selected===n.id?' selected':'');b.innerHTML=`<strong>${esc(n.role||n.id)}</strong><span class="${statusClass(states[n.id])}">${esc(activeRun?statusLabel(states[n.id]):'Proposé')}</span><div class="muted">${n.depends_on?.length?'Après : '+esc(n.depends_on.join(', ')):'Sans dépendance'} · ${(n.tools||[]).length} outil(s)</div>`;b.onclick=()=>{selected=n.id;render()};$('workflow').appendChild(b)}if(!nodes.length)$('workflow').innerHTML='<p class="empty">Les agents apparaîtront après la proposition du plan.</p>';
renderAgent();$('artifacts').replaceChildren();const artifacts=data.artifacts.filter(a=>a.run_id===activeRun);for(const a of artifacts){const link=document.createElement('a');link.href='/api/artifacts/'+encodeURIComponent(a.id)+'?path='+encodeURIComponent(current);link.setAttribute('download','');link.textContent='Télécharger '+a.name;const info=document.createElement('small');info.textContent=Math.max(1,Math.round(a.size_bytes/1024))+' Ko · '+(a.created_by||'agent');link.appendChild(info);$('artifacts').appendChild(link)}if(!artifacts.length)$('artifacts').textContent=activeRun?'Aucun livrable produit pour le moment.':'Sélectionne une exécution pour retrouver ses livrables.';updateEnabled()}
function renderAgent(){const n=nodes.find(n=>n.id===selected);$('agentTitle').textContent=n?n.role:'Détail d’un agent';if(!n){$('agentDetail').textContent='Sélectionne un agent pour voir ses outils, ses actions et ses résultats.';return}const events=ownEvents().filter(e=>e.node_id===n.id);$('agentDetail').innerHTML=`<p>${esc(n.provider||'auto')} · ${esc(n.model||'auto')}</p><p>Outils : ${(n.tools||[]).map(t=>`<span class="badge">${esc(t)}</span>`).join('')||'aucun'}</p>${events.length?events.slice().reverse().map(e=>`<details${e.type==='agent.completed'?' open':''}><summary>${esc(eventLabels[e.type]||e.type)}</summary><pre>${esc(JSON.stringify(e.payload||{},null,2))}</pre></details>`).join(''):'<p>Les actions seront visibles après le lancement.</p>'}`}
$('createForm').onsubmit=e=>{e.preventDefault();action(async()=>{const p=await api('/api/projects',{name:$('projectLabel').value.trim()});await openProject(p.path);$('projectLabel').value=''})};$('openForm').onsubmit=e=>{e.preventDefault();action(()=>openProject($('projectPath').value.trim()))};$('showDraft').onclick=()=>{if(busy)return;activeRun=null;selected=null;render()};
$('upload').onchange=()=>action(async()=>{const file=$('upload').files[0];if(!file)return;try{if(file.size>5000000)throw Error('Choisis un CSV de 5 Mo maximum.');await api('/api/datasets',{project_path:current,name:file.name,content:await file.text()});await refresh();$('dataset').value=file.name;notice(file.name+' a été ajouté au projet.')}finally{$('upload').value=''}});
$('planForm').onsubmit=e=>{e.preventDefault();action(async()=>{planning=true;updateEnabled();$('planningStatus').textContent='Le provider prépare les agents. Cela peut prendre un moment.';try{await api('/api/plan',{project_path:current,problem:$('problem').value.trim(),dataset:$('dataset').value,provider:$('provider').value,model:$('model').value.trim()||'auto'});activeRun=null;selected=null;await refresh();notice('Plan prêt. Vérifie les agents et leurs outils, puis valide le lancement.')}finally{planning=false;$('planningStatus').textContent=''}})};
async function launchRun(){if(!$('tokens').checkValidity()||!$('turns').checkValidity())throw Error('Vérifie les limites dans « Modèle et limites d’exécution ».');const r=await api('/api/runs',{project_path:current,max_tokens:Number($('tokens').value),max_model_turns:Number($('turns').value)});activeRun=r.run_id;selected=null;await refresh();notice('Exécution lancée. Sélectionne un agent pour suivre son travail.')}
async function control(action,node){await api('/api/runs/'+encodeURIComponent(activeRun)+'/'+action,{project_path:current,approve_node:node});await refresh();notice(action==='pause'?'Pause demandée : l’agent en cours terminera avant l’arrêt.':'L’exécution reprend.')}
setInterval(()=>{if(current&&!busy)refresh().catch(()=>{$('connection').textContent='Serveur indisponible';$('connection').className='failed'})},2500);
renderRecents();try{const saved=localStorage.getItem('ml-agentic-project');if(saved)action(()=>openProject(saved))}catch{}
</script></body></html>
'''


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Install the web extra with: pip install -e ".[web]"') from exc
    uvicorn.run("agentic_data.web_app:app", host="127.0.0.1", port=8765, reload=False)
