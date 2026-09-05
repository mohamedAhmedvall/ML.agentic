from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import ArtifactRegistry
from .project_store import ProjectStore


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_snapshot(project_path: str | Path) -> dict[str, Any]:
    store = ProjectStore()
    project = store.open(project_path)

    runs: list[dict[str, Any]] = []
    for run_dir in sorted(project.runs_dir.iterdir(), reverse=True) if project.runs_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        summary_file = run_dir / "run.json"
        if summary_file.is_file():
            try:
                runs.append(_read_json(summary_file))
            except (json.JSONDecodeError, OSError):
                runs.append({"run_id": run_dir.name, "status": "unknown", "workspace": str(run_dir)})
        else:
            runs.append({"run_id": run_dir.name, "status": "running", "workspace": str(run_dir)})

    events: list[dict[str, Any]] = []
    if project.events_file.is_file():
        for line in project.events_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    artifacts = ArtifactRegistry(project.root).list()
    datasets = [
        {
            "name": path.name,
            "path": str(path.relative_to(project.root)),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(project.datasets_dir.iterdir())
        if path.is_file()
    ] if project.datasets_dir.exists() else []

    return {
        "project": {"id": project.id, "name": project.name, "path": str(project.root)},
        "datasets": datasets,
        "runs": runs,
        "events": events,
        "artifacts": artifacts,
    }


def build_app():
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise RuntimeError('Install the web extra with: pip install -e ".[web]"') from exc

    app = FastAPI(title="ML.agentic", version="0.1")

    @app.get("/api/project")
    def get_project(path: str = Query(...)):
        try:
            return project_snapshot(path)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return _DASHBOARD_HTML

    return app


app = build_app()


_DASHBOARD_HTML = r'''<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>ML.agentic</title>
<style>
:root{font-family:Inter,system-ui,sans-serif;color:#e9eef7;background:#0b1020}*{box-sizing:border-box}
body{margin:0;background:#0b1020}header{padding:18px 24px;border-bottom:1px solid #202944;display:flex;gap:16px;align-items:center}
h1{font-size:20px;margin:0}input{flex:1;max-width:760px;background:#121a2d;border:1px solid #2b3658;color:#fff;border-radius:9px;padding:10px 12px}
button{background:#f3f6ff;color:#0b1020;border:0;border-radius:9px;padding:10px 14px;font-weight:700;cursor:pointer}
main{padding:20px;display:grid;grid-template-columns:260px 1fr 320px;gap:16px}.panel{background:#11182a;border:1px solid #222d49;border-radius:14px;padding:16px;min-height:160px}
.card{border:1px solid #2a3553;border-radius:10px;padding:11px;margin:8px 0;background:#151e33}.muted{color:#8f9bb7;font-size:12px}.ok{color:#86efac}.bad{color:#fca5a5}.running{color:#fde68a}
.timeline{max-height:70vh;overflow:auto}.event{padding:9px 0;border-bottom:1px solid #202944}.artifact{display:flex;justify-content:space-between;gap:8px}.pill{font-size:11px;padding:3px 7px;border-radius:999px;background:#263252;color:#c7d2fe}
@media(max-width:950px){main{grid-template-columns:1fr}}
</style>
</head><body>
<header><h1>ML.agentic</h1><input id="projectPath" placeholder="Chemin du projet, ex: .ml-agentic/projects/mon-projet"/><button onclick="loadProject()">Ouvrir</button></header>
<main>
<section class="panel"><h3>Projet</h3><div id="project"></div><h4>Datasets</h4><div id="datasets"></div><h4>Runs</h4><div id="runs"></div></section>
<section class="panel"><h3>Exécution</h3><div id="timeline" class="timeline muted">Ouvre un projet pour voir son activité.</div></section>
<section class="panel"><h3>Artefacts</h3><div id="artifacts"></div></section>
</main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function statusClass(s){return s==='completed'||s==='succeeded'?'ok':s==='failed'?'bad':'running'}
async function loadProject(){const path=document.getElementById('projectPath').value.trim();if(!path)return;const res=await fetch('/api/project?path='+encodeURIComponent(path));const d=await res.json();if(!res.ok){alert(d.detail||'Erreur');return}
document.getElementById('project').innerHTML=`<div class="card"><b>${esc(d.project.name)}</b><div class="muted">${esc(d.project.id)}</div></div>`;
document.getElementById('datasets').innerHTML=d.datasets.map(x=>`<div class="card">${esc(x.name)}<div class="muted">${x.size_bytes} octets</div></div>`).join('')||'<div class="muted">Aucun</div>';
document.getElementById('runs').innerHTML=d.runs.map(r=>`<div class="card"><b>${esc(r.run_id)}</b><div class="${statusClass(r.status)}">${esc(r.status)}</div></div>`).join('')||'<div class="muted">Aucun</div>';
document.getElementById('timeline').innerHTML=d.events.slice().reverse().map(e=>`<div class="event"><b>${esc(e.type)}</b>${e.node_id?` · ${esc(e.node_id)}`:''}<div class="muted">${esc(e.time||'')}</div></div>`).join('')||'<div class="muted">Aucun événement</div>';
document.getElementById('artifacts').innerHTML=d.artifacts.slice().reverse().map(a=>`<div class="card artifact"><div><b>${esc(a.name)}</b><div class="muted">${esc(a.created_by||'runtime')} · ${a.size_bytes} octets</div></div><span class="pill">${esc(a.category)}</span></div>`).join('')||'<div class="muted">Aucun artefact</div>';
localStorage.setItem('ml-agentic-project',path)}
const saved=localStorage.getItem('ml-agentic-project');if(saved){document.getElementById('projectPath').value=saved;loadProject()}
setInterval(()=>{if(document.getElementById('projectPath').value.trim())loadProject()},3000);
</script></body></html>'''


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Install the web extra with: pip install -e ".[web]"') from exc
    uvicorn.run("agentic_data.web_app:app", host="127.0.0.1", port=8765, reload=False)
