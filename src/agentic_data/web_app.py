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

_DASHBOARD_HTML = r'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ML.agentic</title><meta name="control-token" content="__CONTROL_TOKEN__"><style>
:root{font-family:Inter,system-ui,sans-serif;color:#edf2ff;background:#080d18}*{box-sizing:border-box}body{margin:0}header{height:64px;padding:14px 20px;border-bottom:1px solid #202a42;display:flex;gap:14px;align-items:center;background:#0c1220}h1{font-size:19px;margin:0;white-space:nowrap}input,textarea,select{width:100%;background:#111a2c;border:1px solid #2b3858;color:white;border-radius:9px;padding:9px;margin:5px 0}header input{flex:1;max-width:620px;margin:0}textarea{min-height:90px;resize:vertical}button{background:#eef2ff;border:0;border-radius:9px;padding:10px 14px;font-weight:700;cursor:pointer}.secondary{background:#25314f;color:#e8edff}.layout{display:grid;grid-template-columns:250px minmax(480px,1fr) 330px;height:calc(100vh - 64px)}.side,.detail{padding:15px;border-right:1px solid #202a42;overflow:auto;background:#0c1220}.detail{border-right:0;border-left:1px solid #202a42}.canvas{padding:20px;overflow:auto;background:radial-gradient(#202b43 1px,transparent 1px);background-size:22px 22px}.card,.node{background:#111a2c;border:1px solid #2a3858;border-radius:11px;padding:11px;margin:8px 0}.muted{color:#8f9bb5;font-size:12px}.live,.success{color:#86efac}.failed{color:#fca5a5}.running{color:#fde68a}.waiting{color:#93c5fd}.workflow{min-width:460px;display:flex;flex-direction:column;align-items:center}.level{display:flex;gap:28px;justify-content:center;width:100%;margin:12px 0}.node{width:210px;cursor:pointer;position:relative;transition:.15s}.node:hover,.node.selected{border-color:#8ea8ff;transform:translateY(-1px)}.node .status{font-size:11px;margin-top:7px}.edge{height:24px;width:2px;background:#3d4c70}.pill{display:inline-block;font-size:10px;background:#263454;padding:3px 7px;border-radius:999px;margin:4px 3px 0 0}.event{padding:8px 0;border-bottom:1px solid #202a42}.section{margin-top:18px}.planner{max-width:760px;margin:0 auto 20px;background:#0c1220;border:1px solid #263454;padding:14px;border-radius:13px}.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}@media(max-width:900px){.layout{grid-template-columns:1fr;height:auto}.side,.detail{border:0}.canvas{min-height:500px}}
body{font-size:16px}.muted,.node .status{font-size:14px}.pill{font-size:12px}pre{font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;overflow:auto}a{color:#a8c3ff}button:disabled{opacity:.5;cursor:default}button{margin:4px}.level{flex-wrap:wrap}.workflow{min-width:0}.row{min-width:0}#runs .card{cursor:pointer}#runControls{margin-top:12px}@media(max-width:900px){header{height:auto;flex-wrap:wrap}.layout{display:block}.canvas{padding:12px}.row{grid-template-columns:1fr}}
</style></head><body><header><h1>ML.agentic</h1><input id="path" placeholder="Chemin du projet"><button onclick="openProject()">Ouvrir</button><button class="secondary" onclick="newProject()">Nouveau</button><span id="live" class="muted">offline</span></header><div class="layout"><aside class="side"><b id="projectName">Projet</b><div class="section"><span class="muted">RUNS</span><div id="runs"></div></div><div class="section"><span class="muted">DATASETS</span><div id="datasets"></div></div></aside><main class="canvas"><div class="planner"><b>Créer un workflow</b><textarea id="problem" placeholder="Décris le problème métier / data à résoudre..."></textarea><div class="row"><select id="dataset"></select><select id="provider"><option value="openai_codex">OpenAI Codex</option><option value="anthropic_claude">Claude</option><option value="github_copilot">GitHub Copilot</option><option value="ollama">Ollama</option></select></div><label for="model">Modèle</label><input id="model" value="auto"><div class="row"><label>Budget tokens<input id="tokens" type="number" min="1" max="1000000" value="24000"></label><label>Tours maximum<input id="turns" type="number" min="1" max="1000" value="12"></label></div><label for="upload">Ajouter un CSV (5 Mo)</label><input id="upload" type="file" accept=".csv" onchange="uploadCsv(this)"><button onclick="generatePlan()">Proposer un plan</button> <button id="launch" onclick="launchRun()">Valider et lancer</button><div id="runControls"></div> <span id="planStatus" class="muted"></span></div><div class="muted" id="objective">Ouvre un projet pour afficher son workflow.</div><div id="workflow" class="workflow"></div></main><aside class="detail"><b id="detailTitle">Agent</b><div id="agentDetail" class="muted">Clique sur un agent.</div><div class="section"><b>Activity</b><div id="activity"></div></div><div class="section"><b>Artifacts</b><div id="artifacts"></div></div></aside></div><script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let source,current,data,selected,states={},nodes=[],draftMode=false,activeRun=null;
const apiFetch=(url,opts={})=>fetch(url,{...opts,headers:{...opts.headers,"X-ML-Agentic-Token":document.querySelector('meta[name="control-token"]').content}});
function stateLabel(id){return states[id]||'pending'}function stateClass(s){s=String(s||'unknown');return s==='succeeded'||s==='completed'?'success':s==='failed'?'failed':s==='running'?'running':s.includes('approval')?'waiting':'muted'}
function runEvents(){return (data?.events||[]).filter(e=>e.run_id===activeRun)}
function derive(events){states={};const own=runEvents();const start=own.find(e=>e.type==='run.started'&&Array.isArray(e.payload?.nodes));draftMode=!activeRun;nodes=start?start.payload.nodes:(!activeRun&&data?.draft?data.draft.workflow.nodes.map(n=>({...n,...n.harness})):[]);for(const e of own){if(!e.node_id)continue;const map={'agent.started':'running','agent.completed':'succeeded','agent.failed':'failed','agent.skipped':'skipped','agent.awaiting_approval':'awaiting approval'};if(map[e.type])states[e.node_id]=map[e.type]}const summary=data.runs.find(r=>r.run_id===activeRun);for(const [id,result] of Object.entries(summary?.nodes||{}))states[id]=result.state}

function levels(){const done=new Set(),out=[];let guard=0;while(done.size<nodes.length&&guard++<100){const level=nodes.filter(n=>!done.has(n.id)&&(n.depends_on||[]).every(d=>done.has(d)));if(!level.length)break;out.push(level);level.forEach(n=>done.add(n.id))}return out}
function renderGraph(){const w=document.getElementById('workflow');w.innerHTML='';levels().forEach((lvl,i)=>{if(i)w.insertAdjacentHTML('beforeend','<div class="edge"></div>');const row=document.createElement('div');row.className='level';for(const n of lvl){const s=stateLabel(n.id);const el=document.createElement('div');el.className='node '+(selected===n.id?'selected':'');el.onclick=()=>selectAgent(n.id);el.innerHTML=`<b>${esc(n.role||n.id)}</b><div class="muted">${esc(n.id)}</div><div>${n.provider?`<span class="pill">${esc(n.provider)}</span>`:''}${n.model?`<span class="pill">${esc(n.model)}</span>`:''}</div><div class="status ${stateClass(s)}">● ${esc(s)}</div>`;row.appendChild(el)}w.appendChild(row)})}
function selectAgent(id){selected=id;renderGraph();const n=nodes.find(x=>x.id===id)||{id};const ev=runEvents().filter(e=>e.node_id===id);document.getElementById('detailTitle').textContent=n.role||id;document.getElementById('agentDetail').innerHTML=`<div class="card"><span class="muted">ID</span><br>${esc(id)}<br><br><span class="muted">Provider</span><br>${esc(n.provider||'auto')} / ${esc(n.model||'auto')}<br><br><span class="muted">Dependencies</span><br>${esc((n.depends_on||[]).join(', ')||'—')}<br><br><span class="muted">Tools</span><br>${(n.tools||[]).map(t=>`<span class="pill">${esc(t)}</span>`).join('')||'—'}</div>`;document.getElementById('activity').innerHTML=ev.slice().reverse().map(eventHtml).join('')||'<div class="muted">Aucune activité</div>'}
function eventHtml(e){return `<div class="event"><b>${esc(e.type)}</b><div class="muted">${esc(e.time||'')}</div><pre>${esc(JSON.stringify(e.payload||{},null,2))}</pre></div>`}
function render(){derive(data.events);const start=runEvents().find(e=>e.type==='run.started'&&Array.isArray(e.payload?.nodes));document.getElementById('projectName').textContent=data.project.name;document.getElementById('objective').innerHTML=start?`<b>${esc(start.payload.objective||'Workflow')}</b><div class="muted">${esc(start.run_id)}</div>`:data.draft?`<b>${esc(data.draft.workflow.objective)}</b><div class="waiting">DRAFT · à valider avant exécution</div>`:'Aucun workflow planifié';document.getElementById('runs').innerHTML=data.runs.map(r=>`<div class="card"><b>${esc(r.run_id)}</b><div class="${stateClass(r.status)}">${esc(r.status)}</div></div>`).join('')||'<div class="muted">Aucun</div>';document.getElementById('datasets').innerHTML=data.datasets.map(d=>`<div class="card">${esc(d.name)}</div>`).join('')||'<div class="muted">Aucun</div>';document.getElementById('dataset').innerHTML=data.datasets.map(d=>`<option value="${esc(d.name)}">${esc(d.name)}</option>`).join('');document.getElementById('artifacts').innerHTML=data.artifacts.filter(a=>a.run_id===activeRun).slice().reverse().map(a=>`<div class="card"><a href="/api/artifacts/${encodeURIComponent(a.id)}?path=${encodeURIComponent(current)}" download>${esc(a.name)}</a><div class="muted">${esc(a.category)} · ${a.size_bytes} bytes</div></div>`).join('')||'<div class="muted">Aucun</div>';renderGraph();renderControls();if(selected)selectAgent(selected)}
async function refresh(){const r=await fetch('/api/project?path='+encodeURIComponent(current));const d=await r.json();if(!r.ok)throw Error(d.detail||'Erreur');data=d;render()}
function connect(){if(source)source.close();source=new EventSource('/api/events?path='+encodeURIComponent(current));source.addEventListener('ready',()=>{live.textContent='● live';live.className='live'});source.addEventListener('runtime',()=>refresh());source.onerror=()=>{live.textContent='reconnexion…';live.className='muted'}}
async function openProject(){current=document.getElementById('path').value.trim();activeRun=null;selected=null;if(!current)return;try{await refresh();connect();localStorage.setItem('ml-agentic-project',current)}catch(e){current=null;if(source)source.close();alert(e.message)}}
async function newProject(){const name=prompt('Nom du projet');if(!name)return;const explicit=prompt('Dossier du projet (laisser vide pour automatique)')||null;const r=await apiFetch('/api/projects',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,path:explicit})});const d=await r.json();if(!r.ok){alert(d.detail||'Erreur');return}path.value=d.path;await openProject()}
async function generatePlan(){if(!current)return;const problem=document.getElementById('problem').value.trim(),dataset=document.getElementById('dataset').value,provider=document.getElementById('provider').value;if(!problem||!dataset){alert('Ajoute un problème et un dataset');return}planStatus.textContent='planning…';const r=await apiFetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_path:current,problem,dataset,provider,model:document.getElementById('model').value.trim()||'auto'})});const d=await r.json();if(!r.ok){planStatus.textContent=d.detail||'Provider indisponible';return}planStatus.textContent='Plan prêt';activeRun=null;await refresh()}
function renderControls(){
 document.querySelectorAll('#runs .card').forEach((el,i)=>{el.tabIndex=0;el.setAttribute('role','button');el.onclick=()=>{activeRun=data.runs[i].run_id;selected=null;render()};el.onkeydown=e=>{if(e.key==='Enter')el.click()}});
 document.getElementById('launch').disabled=!data?.draft;
 const r=data.runs.find(r=>r.run_id===activeRun),box=document.getElementById('runControls');box.replaceChildren();
 if(r){const status=document.createElement('p');status.textContent=r.status+(r.error?' · '+r.error:'');box.appendChild(status);
 const add=(label,action,node)=>{const b=document.createElement('button');b.textContent=label;b.onclick=()=>controlRun(action,node);box.appendChild(b)};
 if(r.status==='running')add('Pause après cet agent','pause');
 if(r.status==='paused')add('Reprendre','resume');
 if(r.status==='approval_required')for(const n of nodes.filter(n=>states[n.id]==='awaiting approval'))add('Approuver '+n.role,'resume',n.id);
 }
 const draft=document.createElement('button');draft.textContent='Voir le plan proposé';draft.onclick=()=>{activeRun=null;selected=null;render()};box.appendChild(draft);
}
async function post(url,payload){const r=await apiFetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.detail||'Action impossible');return d}
async function launchRun(){document.getElementById('launch').disabled=true;try{const r=await post('/api/runs',{project_path:current,max_tokens:Number(document.getElementById('tokens').value),max_model_turns:Number(document.getElementById('turns').value)});activeRun=r.run_id;await refresh()}catch(e){alert(e.message)}finally{document.getElementById('launch').disabled=!data?.draft}}
async function controlRun(action,node){try{await post('/api/runs/'+encodeURIComponent(activeRun)+'/'+action,{project_path:current,approve_node:node});await refresh()}catch(e){alert(e.message)}}
async function uploadCsv(input){try{if(!current)throw Error('Ouvre un projet');const file=input.files[0];if(!file)return;if(file.size>5000000)throw Error('Limite : 5 Mo');await post('/api/datasets',{project_path:current,name:file.name,content:await file.text()});await refresh()}catch(e){alert(e.message)}finally{input.value=''}}
setInterval(()=>{if(current)refresh().catch(()=>{})},2000);
const saved=localStorage.getItem('ml-agentic-project');if(saved){path.value=saved;openProject()}
</script></body></html>'''


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Install the web extra with: pip install -e ".[web]"') from exc
    uvicorn.run("agentic_data.web_app:app", host="127.0.0.1", port=8765, reload=False)
