"""ProjectAgent — exécute un brief complet de A à Z de façon itérative."""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.orchestrator.agent import AgentResponse, ECLMAgent
from src.shared.config import Config
from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

_STATUS = {"pending", "running", "done", "failed", "skipped"}


@dataclass
class TaskRecord:
    """Une tâche atomique dans un projet."""

    index: int
    action: str
    target_type: str
    target_name: str
    target_file: str
    description: str
    depends_on: list[int] = field(default_factory=list)
    status: str = "pending"
    files_created: list[str] = field(default_factory=list)
    verification_score: float = 0.0
    dpo_collected: int = 0
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def done(self) -> bool:
        return self.status == "done"

    @property
    def label(self) -> str:
        return f"{self.action:8s} {self.target_file}:{self.target_name}"


@dataclass
class ProjectSession:
    """État complet d'un projet — persisté après chaque tâche."""

    id: str
    brief: str
    created_at: str
    tasks: list[TaskRecord]
    estimated_files: list[str] = field(default_factory=list)

    @property
    def done_count(self) -> int:
        return sum(1 for t in self.tasks if t.done)

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def next_pending(self) -> TaskRecord | None:
        return next(
            (t for t in self.tasks if t.status == "pending" and self._deps_done(t)),
            None,
        )

    def _deps_done(self, task: TaskRecord) -> bool:
        done_indices = {t.index for t in self.tasks if t.done}
        return all(d in done_indices for d in task.depends_on)

    @property
    def all_files_created(self) -> list[str]:
        files: list[str] = []
        for t in self.tasks:
            files.extend(t.files_created)
        return list(dict.fromkeys(files))  # dédupliqué, ordre préservé


# ── Prompt de planification ───────────────────────────────────────────────────

_PLAN_PROMPT = """\
Tu es un architecte logiciel Python expert. Tu reçois un brief de projet et tu \
le décomposes en tâches atomiques ORDONNÉES par dépendances.

Règles strictes :
- Chaque tâche = une seule fonction ou classe
- Ordonne par dépendances : ce qui est importé AVANT ce qui l'importe
- Commence par models/types → logique métier → API/routes → tests
- target_file : chemin relatif (ex: "models.py", "src/auth.py")
- action : CREATE | MODIFY | FIX | ADD | TEST
- target_type : function | class
- depends_on : indices des tâches dont cette tâche dépend (liste d'entiers)

Brief : {brief}

Retourne UNIQUEMENT ce JSON (pas d'explication) :
{{
  "tasks": [
    {{
      "index": 0,
      "action": "CREATE",
      "target_type": "class",
      "target_name": "User",
      "target_file": "models.py",
      "description": "User dataclass with id, username, email, hashed_password fields",
      "depends_on": []
    }}
  ],
  "estimated_files": ["models.py", "auth.py"]
}}"""


# ── ProjectAgent ──────────────────────────────────────────────────────────────

class ProjectAgent:
    """Exécute un brief complet de A à Z de façon itérative.

    Stratégie itérative (vs one-shot) :
    - Crée fichier A → ré-indexe RAG → crée fichier B qui voit A
    - Chaque tâche passe par le pipeline complet (C1→C2→C3→FileWriter)
    - Persistance après chaque tâche → toujours resumable
    - En cas d'échec : proposer skip/retry, pas bloquer le projet entier
    """

    def __init__(self, config: Config, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self._agent = ECLMAgent(config, project_root)
        self._sessions_dir = config.data_dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    # ── Planning ──────────────────────────────────────────────────────────────

    def plan(self, brief: str) -> ProjectSession:
        """Décompose un brief en ProjectSession avec tâches ordonnées.

        Args:
            brief: Description du projet en français.

        Returns:
            ProjectSession avec les tâches à exécuter.
        """
        session_id = str(uuid.uuid4())[:12]
        tasks = self._plan_via_ollama(brief) or self._fallback_plan(brief)
        session = ProjectSession(
            id=session_id,
            brief=brief,
            created_at=datetime.now(timezone.utc).isoformat(),
            tasks=tasks,
        )
        self._save(session)
        logger.info("Plan créé: %d tâches, session=%s", len(tasks), session_id)
        return session

    def _plan_via_ollama(self, brief: str) -> list[TaskRecord] | None:
        prompt = _PLAN_PROMPT.format(brief=brief)
        payload = json.dumps({
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "seed": 0},
        }).encode()
        req = urllib.request.Request(
            f"{self.config.ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                raw = str(data.get("response", "")).strip()
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Ollama planner indisponible: %s", exc)
            return None

        return self._parse_plan(raw)

    def _parse_plan(self, raw: str) -> list[TaskRecord] | None:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
            tasks = []
            for t in data.get("tasks", []):
                tasks.append(TaskRecord(
                    index=int(t["index"]),
                    action=str(t.get("action", "CREATE")).upper(),
                    target_type=str(t.get("target_type", "function")),
                    target_name=str(t["target_name"]),
                    target_file=str(t["target_file"]),
                    description=str(t["description"]),
                    depends_on=[int(d) for d in t.get("depends_on", [])],
                ))
            return tasks if tasks else None
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _fallback_plan(self, brief: str) -> list[TaskRecord]:
        """Plan minimal quand Ollama est indisponible."""
        return [TaskRecord(
            index=0,
            action="CREATE",
            target_type="function",
            target_name="main",
            target_file="main.py",
            description=brief,
        )]

    # ── Exécution ─────────────────────────────────────────────────────────────

    def execute(
        self,
        session: ProjectSession,
        on_task_start: object = None,
        on_task_done: object = None,
    ) -> ProjectSession:
        """Exécute toutes les tâches pending de la session de façon itérative.

        Args:
            session: Session à exécuter (peut être partiellement complétée).
            on_task_start: Callback(task) appelé avant chaque tâche.
            on_task_done: Callback(task, response) appelé après chaque tâche.

        Returns:
            Session mise à jour avec les résultats.
        """
        while True:
            task = session.next_pending
            if task is None:
                break
            task.status = "running"
            task.started_at = datetime.now(timezone.utc).isoformat()

            if callable(on_task_start):
                on_task_start(task)  # type: ignore[operator]

            response = self._run_task(task)
            self._update_task(task, response)

            if callable(on_task_done):
                on_task_done(task, response)  # type: ignore[operator]

            # Ré-indexer le RAG après chaque fichier écrit
            if response.written_to:
                self._agent._rag.index_file(response.written_to)

            self._save(session)

        return session

    def _run_task(self, task: TaskRecord) -> AgentResponse:
        intent = IntentJSON(
            action=task.action,
            target_type=task.target_type,
            target_name=task.target_name,
            target_file=task.target_file if task.target_file else None,
            description=task.description,
            confidence=0.92,
        )
        return self._agent.run(intent)

    def _update_task(self, task: TaskRecord, response: AgentResponse) -> None:
        task.verification_score = response.score
        task.completed_at = datetime.now(timezone.utc).isoformat()
        if response.success:
            task.status = "done"
            if response.written_to:
                task.files_created.append(str(response.written_to))
        else:
            task.status = "failed"
            task.error = response.message

    # ── Persistance ───────────────────────────────────────────────────────────

    def _save(self, session: ProjectSession) -> None:
        path = self._sessions_dir / f"{session.id}.json"
        path.write_text(
            json.dumps(asdict(session), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> ProjectSession:
        """Charge une session existante pour la reprendre.

        Args:
            session_id: ID de la session (préfixe suffisant).

        Returns:
            ProjectSession désérialisée.

        Raises:
            FileNotFoundError: Si la session n'existe pas.
        """
        matches = list(self._sessions_dir.glob(f"{session_id}*.json"))
        if not matches:
            raise FileNotFoundError(f"Session introuvable: {session_id}")
        data = json.loads(matches[0].read_text(encoding="utf-8"))
        tasks = [TaskRecord(**t) for t in data.pop("tasks")]
        return ProjectSession(tasks=tasks, **data)

    def list_sessions(self) -> list[dict[str, object]]:
        """Liste les sessions disponibles, triées par date décroissante."""
        sessions = []
        for path in sorted(self._sessions_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append({
                    "id": data["id"],
                    "brief": data["brief"][:60] + "…" if len(data["brief"]) > 60 else data["brief"],
                    "created_at": data["created_at"],
                    "tasks": len(data["tasks"]),
                    "done": sum(1 for t in data["tasks"] if t["status"] == "done"),
                })
            except (KeyError, json.JSONDecodeError):
                continue
        return sessions
