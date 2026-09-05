from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .artifacts import ArtifactRegistry
from .planning import plan_workflow, workflow_to_dict
from .project_store import ProjectStore
from .providers import ProviderName
from .runners import ClaudeAdapter, CodexAdapter, CopilotAdapter, OllamaAdapter


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


async def _event_stream(project_path: str):
    project = ProjectStore().open(project_path)
    event_file = project.events_file
    position = event_file.stat().st_size if event_file.is_file() else 0
    yield "event: ready\ndata: {}\n\n"
    while True:
        if event_file.is_file() and event_file.stat().st_size > position:
            with event_file.open("r", encoding="utf-8") as handle:
                handle.seek(position)
                for line in handle:
                    try:
                        if line.strip():
                            yield f"event: runtime\ndata: {json.dumps(json.loads(line), ensure_ascii=False)}\n\n"
                    except json.JSONDecodeError:
                        pass
                position = handle.tell()
        yield ": keepalive\n\n"
        await asyncio.sleep(0.75)


def build_app():
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import HTMLResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError('Install the web extra with: pip install -e ".[web]"') from exc

    app = FastAPI(title="ML.agentic", version="0.4")

    @app.get("/api/project")
    def get_project(path: str = Query(...)):
        try:
            return project_snapshot(path)
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
        return _DASHBOARD_HTML

    return app


app = build_app()

_DASHBOARD_HTML = r'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ML.agentic</title><style>
:root{font-family:Inter,system-ui,sans-serif;color:#edf2ff;background:#080d18}*{box-sizing:border-box}body{margin:0}header{height:64px;padding:14px 20px;border-bottom:1px solid #202a42;display:flex;gap:14px;align-items:center;background:#0c1220}h1{font-size:19px;margin:0;white-space:nowrap}input,textarea,select{width:100%;background:#111a2c;border:1px solid #2b3858;color:white;border-radius:9px;padding:9px;margin:5px 0}header input{flex:1;max-width:620px;margin:0}textarea{min-height:90px;resize:vertical}button{background:#eef2ff;border:0;border-radius:9px;padding:10px 14px;font-weight:700;cursor:pointer}.secondary{background:#25314f;color:#e8edff}.layout{display:grid;grid-template-columns:250px minmax(480px,1fr) 330px;height:calc(100vh - 64px)}.side,.detail{padding:15px;border-right:1px solid #202a42;overflow:auto;background:#0c1220}.detail{border-right:0;border-left:1px solid #202a42}.canvas{padding:20px;overflow:auto;background:radial-gradient(#202b43 1px,transparent 1px);background-size:22px 22px}.card,.node{background:#111a2c;border:1px solid #2a3858;border-radius:11px;padding:11px;margin:8px 0}.muted{color:#8f9bb5;font-size:12px}.live,.success{color:#86efac}.failed{color:#fca5a5}.running{color:#fde68a}.waiting{color:#93c5fd}.workflow{min-width:460px;display:flex;flex-direction:column;align-items:center}.level{display:flex;gap:28px;justify-content:center;width:100%;margin:12px 0}.node{width:210px;cursor:pointer;position:relative;transition:.15s}.node:hover,.node.selected{border-color:#8ea8ff;transform:translateY(-1px)}.node .status{font-size:11px;margin-top:7px}.edge{height:24px;width:2px;background:#3d4c70}.pill{display:inline-block;font-size:10px;background:#263454;padding:3px 7px;border-radius:999px;margin:4px 3px 0 0}.event{padding:8px 0;border-bottom:1px solid #202a42}.section{margin-top:18px}.planner{max-width:760px;margin:0 auto 20px;background:#0c1220;border:1px solid #263454;padding:14px;border-radius:13px}.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}@media(max-width:900px){.layout{grid-template-columns:1fr;height:auto}.side,.detail{border:0}.canvas{min-height:500px}}
</style></head><body><header><h1>ML.agentic</h1><input id="path" placeholder="Chemin du projet"><button onclick="openProject()">Ouvrir</button><button class="secondary" onclick="newProject()">Nouveau</button><span id="live" class="muted">offline</span></header><div class="layout"><aside class="side"><b id="projectName">Projet</b><div class="section"><span class="muted">RUNS</span><div id="runs"></div></div><div class="section"><span class="muted">DATASETS</span><div id="datasets"></div></div></aside><main class="canvas"><div class="planner"><b>Créer un workflow</b><textarea id="problem" placeholder="Décris le problème métier / data à résoudre..."></textarea><div class="row"><select id="dataset"></select><select id="provider"><option value="openai_codex">OpenAI Codex</option><option value="anthropic_claude">Claude</option><option value="github_copilot">GitHub Copilot</option><option value="ollama">Ollama</option></select></div><button onclick="generatePlan()">Generate plan</button> <span id="planStatus" class="muted"></span></div><div class="muted" id="objective">Ouvre un projet pour afficher son workflow.</div><div id="workflow" class="workflow"></div></main><aside class="detail"><b id="detailTitle">Agent</b><div id="agentDetail" class="muted">Clique sur un agent.</div><div class="section"><b>Activity</b><div id="activity"></div></div><div class="section"><b>Artifacts</b><div id="artifacts"></div></div></aside></div><script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let source,current,data,selected,states={},nodes=[],draftMode=false;
function stateLabel(id){return states[id]||'pending'}function stateClass(s){return s==='succeeded'||s==='completed'?'success':s==='failed'?'failed':s==='running'?'running':s.includes('approval')?'waiting':'muted'}
function derive(events){states={};const start=[...events].reverse().find(e=>e.type==='run.started'&&Array.isArray(e.payload?.nodes));if(start){draftMode=false;nodes=start.payload.nodes.map(n=>typeof n==='string'?{id:n,role:n,depends_on:[]}:n)}else if(data?.draft?.workflow){draftMode=true;nodes=data.draft.workflow.nodes.map(n=>({...n,...n.harness}))}else nodes=[];for(const e of events){if(!e.node_id)continue;if(e.type==='agent.started')states[e.node_id]='running';if(e.type==='agent.completed')states[e.node_id]='succeeded';if(e.type==='agent.failed')states[e.node_id]='failed';if(e.type==='agent.skipped')states[e.node_id]='skipped';if(e.type==='agent.awaiting_approval')states[e.node_id]='awaiting approval'}}
function levels(){const done=new Set(),out=[];let guard=0;while(done.size<nodes.length&&guard++<100){const level=nodes.filter(n=>!done.has(n.id)&&(n.depends_on||[]).every(d=>done.has(d)));if(!level.length)break;out.push(level);level.forEach(n=>done.add(n.id))}return out}
function renderGraph(){const w=document.getElementById('workflow');w.innerHTML='';levels().forEach((lvl,i)=>{if(i)w.insertAdjacentHTML('beforeend','<div class="edge"></div>');const row=document.createElement('div');row.className='level';for(const n of lvl){const s=stateLabel(n.id);const el=document.createElement('div');el.className='node '+(selected===n.id?'selected':'');el.onclick=()=>selectAgent(n.id);el.innerHTML=`<b>${esc(n.role||n.id)}</b><div class="muted">${esc(n.id)}</div><div>${n.provider?`<span class="pill">${esc(n.provider)}</span>`:''}${n.model?`<span class="pill">${esc(n.model)}</span>`:''}</div><div class="status ${stateClass(s)}">● ${esc(s)}</div>`;row.appendChild(el)}w.appendChild(row)})}
function selectAgent(id){selected=id;renderGraph();const n=nodes.find(x=>x.id===id)||{id};const ev=(data?.events||[]).filter(e=>e.node_id===id);document.getElementById('detailTitle').textContent=n.role||id;document.getElementById('agentDetail').innerHTML=`<div class="card"><span class="muted">ID</span><br>${esc(id)}<br><br><span class="muted">Provider</span><br>${esc(n.provider||'auto')} / ${esc(n.model||'auto')}<br><br><span class="muted">Dependencies</span><br>${esc((n.depends_on||[]).join(', ')||'—')}<br><br><span class="muted">Tools</span><br>${(n.tools||[]).map(t=>`<span class="pill">${esc(t)}</span>`).join('')||'—'}</div>`;document.getElementById('activity').innerHTML=ev.slice().reverse().map(eventHtml).join('')||'<div class="muted">Aucune activité</div>'}
function eventHtml(e){return `<div class="event"><b>${esc(e.type)}</b><div class="muted">${esc(e.time||'')}</div></div>`}
function render(){derive(data.events);const start=[...data.events].reverse().find(e=>e.type==='run.started'&&Array.isArray(e.payload?.nodes));document.getElementById('projectName').textContent=data.project.name;document.getElementById('objective').innerHTML=start?`<b>${esc(start.payload.objective||'Workflow')}</b><div class="muted">${esc(start.run_id)}</div>`:data.draft?`<b>${esc(data.draft.workflow.objective)}</b><div class="waiting">DRAFT · à valider avant exécution</div>`:'Aucun workflow planifié';document.getElementById('runs').innerHTML=data.runs.map(r=>`<div class="card"><b>${esc(r.run_id)}</b><div class="${stateClass(r.status)}">${esc(r.status)}</div></div>`).join('')||'<div class="muted">Aucun</div>';document.getElementById('datasets').innerHTML=data.datasets.map(d=>`<div class="card">${esc(d.name)}</div>`).join('')||'<div class="muted">Aucun</div>';document.getElementById('dataset').innerHTML=data.datasets.map(d=>`<option value="${esc(d.name)}">${esc(d.name)}</option>`).join('');document.getElementById('artifacts').innerHTML=data.artifacts.slice().reverse().map(a=>`<div class="card"><b>${esc(a.name)}</b><div class="muted">${esc(a.category)} · ${a.size_bytes} bytes</div></div>`).join('')||'<div class="muted">Aucun</div>';renderGraph();if(selected)selectAgent(selected)}
async function refresh(){const r=await fetch('/api/project?path='+encodeURIComponent(current));const d=await r.json();if(!r.ok){alert(d.detail||'Erreur');return}data=d;render()}
function connect(){if(source)source.close();source=new EventSource('/api/events?path='+encodeURIComponent(current));source.addEventListener('ready',()=>{live.textContent='● live';live.className='live'});source.addEventListener('runtime',()=>refresh());source.onerror=()=>{live.textContent='reconnexion…';live.className='muted'}}
async function openProject(){current=document.getElementById('path').value.trim();if(!current)return;await refresh();connect();localStorage.setItem('ml-agentic-project',current)}
async function newProject(){const name=prompt('Nom du projet');if(!name)return;const explicit=prompt('Dossier du projet (laisser vide pour automatique)')||null;const r=await fetch('/api/projects',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,path:explicit})});const d=await r.json();if(!r.ok){alert(d.detail||'Erreur');return}path.value=d.path;await openProject()}
async function generatePlan(){if(!current)return;const problem=document.getElementById('problem').value.trim(),dataset=document.getElementById('dataset').value,provider=document.getElementById('provider').value;if(!problem||!dataset){alert('Ajoute un problème et un dataset');return}planStatus.textContent='planning…';const r=await fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_path:current,problem,dataset,provider,model:'auto'})});const d=await r.json();if(!r.ok){planStatus.textContent='échec';alert(d.detail||'Erreur');return}planStatus.textContent='draft prêt';await refresh()}
const saved=localStorage.getItem('ml-agentic-project');if(saved){path.value=saved;openProject()}
</script></body></html>'''


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Install the web extra with: pip install -e ".[web]"') from exc
    uvicorn.run("agentic_data.web_app:app", host="127.0.0.1", port=8765, reload=False)
