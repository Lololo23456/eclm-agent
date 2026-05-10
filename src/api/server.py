"""FastAPI server — expose ECLM Agent via HTTP pour IDE integration.

Usage:
    python -m src.api.server [--host 0.0.0.0] [--port 8765] [--project .]

Continue.dev config (.continue/config.json):
{
  "models": [{
    "title": "ECLM Agent",
    "provider": "openai",
    "model": "eclm",
    "apiBase": "http://localhost:8765/v1",
    "apiKey": "local"
  }],
  "tabAutocompleteModel": {
    "title": "ECLM Complete",
    "provider": "openai",
    "model": "eclm",
    "apiBase": "http://localhost:8765/v1",
    "apiKey": "local"
  }
}
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.models import (
    ChatRequest, ChatResponse,
    CompleteRequest, CompleteResponse,
    GenerateRequest, GenerateResponse,
    HealthResponse,
    ProjectListItem, ProjectRequest, ProjectResponse, TaskStatus,
)
from src.improvement.dpo_collector import DPOCollector
from src.intent.model import IntentExtractor
from src.orchestrator.agent import ECLMAgent
from src.orchestrator.project import ProjectAgent, ProjectSession
from src.shared.config import Config

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ECLM Agent API",
    description="Local AI coding agent — 100% private, no cloud",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State global ──────────────────────────────────────────────────────────────

_config: Config | None = None
_project_root: Path | None = None
_agent: ECLMAgent | None = None
_project_agent: ProjectAgent | None = None
_extractor: IntentExtractor | None = None
_dpo: DPOCollector | None = None

# Sessions en cours d'exécution (session_id -> thread)
_running_sessions: dict[str, threading.Thread] = {}
_session_progress: dict[str, list[str]] = {}  # session_id -> log lines


def _get_agent() -> ECLMAgent:
    assert _agent is not None, "Server not initialized"
    return _agent


def _get_project_agent() -> ProjectAgent:
    assert _project_agent is not None, "Server not initialized"
    return _project_agent


def _get_extractor() -> IntentExtractor:
    assert _extractor is not None, "Server not initialized"
    return _extractor


def _session_to_response(session: ProjectSession) -> ProjectResponse:
    failed = sum(1 for t in session.tasks if t.status == "failed")
    test_score = float(session.test_results.get("score", 0.0)) if session.test_results else None
    return ProjectResponse(
        session_id=session.id,
        brief=session.brief,
        total=session.total,
        done=session.done_count,
        failed=failed,
        output_dir=session.output_dir,
        tasks=[
            TaskStatus(
                index=t.index, label=t.label, status=t.status,
                score=t.verification_score, files_created=t.files_created,
                error=t.error,
            )
            for t in session.tasks
        ],
        test_score=test_score,
        critic_issues=session.critic_issues,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    assert _config is not None and _dpo is not None
    return HealthResponse(
        status="ok",
        ollama_url=_config.ollama_base_url,
        fast_model=_config.fast_model,
        strong_model=_config.strong_model,
        dpo_pairs=_dpo.count(),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Génère du code depuis une commande en français."""
    extractor = _get_extractor()
    agent = _get_agent()

    intent = extractor.extract(req.command)
    if intent.needs_clarification:
        return GenerateResponse(
            success=False, code="", score=0.0,
            message=f"Intention peu claire (confiance {intent.confidence:.0%}). "
                    f"Que voulez-vous faire exactement ?",
        )

    response = agent.run(intent, behavior_tests=req.behavior_tests)
    return GenerateResponse(
        success=response.success,
        code=response.code,
        score=response.score,
        message=response.message,
        written_to=str(response.written_to) if response.written_to else None,
        retries_used=response.retries_used,
    )


@app.post("/project", response_model=ProjectResponse)
def create_project(req: ProjectRequest, background_tasks: BackgroundTasks) -> ProjectResponse:
    """Démarre un nouveau projet. Exécution en arrière-plan."""
    pa = _get_project_agent()
    session = pa.plan(req.brief)
    _session_progress[session.id] = []

    def _run() -> None:
        logs = _session_progress.setdefault(session.id, [])

        def on_start(task: object) -> None:
            from src.orchestrator.project import TaskRecord as TR
            assert isinstance(task, TR)
            logs.append(f"START {task.label}")

        def on_done(task: object, resp: object) -> None:
            from src.orchestrator.project import TaskRecord as TR
            from src.orchestrator.agent import AgentResponse as AR
            assert isinstance(task, TR) and isinstance(resp, AR)
            icon = "✓" if resp.success else "✗"
            logs.append(f"{icon} {task.label} score={resp.score:.2f}")

        def on_critic(issues: object, test_results: object, fix_pass: object = 0) -> None:
            score = test_results.get("score", 0.0) if isinstance(test_results, dict) else 0.0  # type: ignore[union-attr]
            logs.append(f"TESTS score={score:.2f}")

        pa.execute(session, on_task_start=on_start, on_task_done=on_done, on_critic_done=on_critic)
        _running_sessions.pop(session.id, None)

    t = threading.Thread(target=_run, daemon=True)
    _running_sessions[session.id] = t
    t.start()

    return _session_to_response(session)


@app.get("/project/{session_id}", response_model=ProjectResponse)
def get_project(session_id: str) -> ProjectResponse:
    """Retourne l'état d'un projet (polling)."""
    pa = _get_project_agent()
    try:
        session = pa.load(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} introuvable")
    return _session_to_response(session)


@app.get("/project/{session_id}/logs")
def get_project_logs(session_id: str) -> dict[str, object]:
    """Retourne les logs de progression d'un projet."""
    logs = _session_progress.get(session_id, [])
    running = session_id in _running_sessions
    return {"session_id": session_id, "running": running, "logs": logs}


@app.get("/project/{session_id}/stream")
async def stream_project_logs(session_id: str) -> StreamingResponse:
    """Server-Sent Events — stream des logs de progression en temps réel."""
    async def _generate() -> AsyncIterator[str]:
        seen = 0
        while True:
            logs = _session_progress.get(session_id, [])
            for line in logs[seen:]:
                yield f"data: {json.dumps({'log': line})}\n\n"
                seen = len(logs)
            if session_id not in _running_sessions and seen >= len(logs):
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/project/{session_id}/resume", response_model=ProjectResponse)
def resume_project(session_id: str, background_tasks: BackgroundTasks) -> ProjectResponse:
    """Reprend un projet interrompu."""
    pa = _get_project_agent()
    try:
        session = pa.load(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} introuvable")

    if session.done_count == session.total:
        return _session_to_response(session)

    _session_progress.setdefault(session.id, [])

    def _run() -> None:
        pa.execute(session)
        _running_sessions.pop(session.id, None)

    t = threading.Thread(target=_run, daemon=True)
    _running_sessions[session.id] = t
    t.start()
    return _session_to_response(session)


@app.get("/projects", response_model=list[ProjectListItem])
def list_projects() -> list[ProjectListItem]:
    """Liste toutes les sessions de projet."""
    pa = _get_project_agent()
    sessions = pa.list_sessions()
    return [
        ProjectListItem(
            id=str(s["id"]), brief=str(s["brief"]),
            created_at=str(s["created_at"]),
            tasks=int(s["tasks"]), done=int(s["done"]),  # type: ignore[arg-type]
            tech_stack=list(s.get("tech_stack", [])),  # type: ignore[arg-type]
        )
        for s in sessions
    ]


# ── Continue.dev compatible endpoints ─────────────────────────────────────────

@app.post("/v1/completions", response_model=CompleteResponse)
def completions(req: CompleteRequest) -> CompleteResponse:
    """Inline completion compatible Continue.dev."""
    extractor = _get_extractor()
    agent = _get_agent()

    # Construire une intention depuis le contexte de complétion
    prompt = f"Complete this Python code:\n{req.prefix}"
    if req.suffix:
        prompt += f"\n# ... \n{req.suffix}"

    intent = extractor.extract(prompt)
    response = agent.run(intent)

    # Extraire seulement le code ajouté (pas le préfixe déjà présent)
    completion = response.code
    if completion.startswith(req.prefix):
        completion = completion[len(req.prefix):]

    return CompleteResponse(completion=completion.strip())


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(req: ChatRequest) -> ChatResponse:
    """Chat completions compatible OpenAI / Continue.dev."""
    extractor = _get_extractor()
    agent = _get_agent()

    # Prendre le dernier message utilisateur
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="Aucun message utilisateur")

    last_msg = user_msgs[-1].content
    intent = extractor.extract(last_msg)

    if intent.needs_clarification:
        content = (f"Je ne suis pas sûr de comprendre. "
                   f"Pouvez-vous préciser ce que vous voulez faire avec `{intent.target_name}` ?")
    else:
        response = agent.run(intent)
        if response.success:
            content = f"```python\n{response.code}\n```"
            if response.written_to:
                content += f"\n\n✓ Écrit dans `{response.written_to}`"
        else:
            content = (f"Je n'ai pas réussi à générer un code valide (score={response.score:.2f}).\n\n"
                       f"Meilleur candidat :\n```python\n{response.code}\n```")

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    )


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def create_app(project_root: Path | None = None, config: Config | None = None) -> FastAPI:
    """Initialise le serveur avec la config et les agents."""
    global _config, _project_root, _agent, _project_agent, _extractor, _dpo

    _config = config or Config()
    _project_root = project_root or Path(".")
    _agent = ECLMAgent(_config, _project_root)
    _project_agent = ProjectAgent(_config, _project_root)
    _extractor = IntentExtractor(_config)
    _dpo = DPOCollector(_config)
    return app


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ECLM Agent API Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = Config(root_dir=args.project.resolve())
    create_app(project_root=args.project.resolve(), config=config)

    print(f"  ECLM Agent API — http://{args.host}:{args.port}")
    print(f"  Docs          — http://{args.host}:{args.port}/docs")
    print(f"  Health        — http://{args.host}:{args.port}/health")
    print()
    print("  Continue.dev config (.continue/config.json):")
    print(f'  "apiBase": "http://{args.host}:{args.port}/v1"')
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
