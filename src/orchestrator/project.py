"""ProjectAgent — exécute un brief complet de A à Z de façon itérative."""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.orchestrator.agent import AgentResponse, ECLMAgent
from src.orchestrator.architect import ArchitectAgent
from src.orchestrator.critic import CriticAgent, CriticIssue
from src.orchestrator.dependency_graph import DependencyGraph
from src.shared.config import Config
from src.shared.types import IntentJSON

logger = logging.getLogger(__name__)

_STATUS = {"pending", "running", "done", "failed", "skipped"}
_MAX_FIX_PASSES = 2


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
    complexity: str = "medium"       # low | medium | high — utilisé par ModelRouter
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
        cx = {"low": "·", "medium": "◆", "high": "★"}.get(self.complexity, "◆")
        return f"{self.action:8s} {self.target_file}:{self.target_name} {cx}"


@dataclass
class ProjectSession:
    """État complet d'un projet — persisté après chaque tâche."""

    id: str
    brief: str
    created_at: str
    tasks: list[TaskRecord]
    tech_stack: list[str] = field(default_factory=list)
    estimated_files: list[str] = field(default_factory=list)
    review_gate: str | None = None
    output_dir: str = ""
    critic_issues: list[dict[str, str]] = field(default_factory=list)
    test_results: dict[str, object] = field(default_factory=dict)

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
        return list(dict.fromkeys(files))


class ProjectAgent:
    """Exécute un brief complet de A à Z de façon itérative.

    Stratégie :
    1. ArchitectAgent (32B) décompose le brief en DAG de tâches avec stack + complexité
    2. Review Gate optionnel : demande confirmation avant exécution si décision critique
    3. ECLM Worker exécute chaque tâche avec ModelRouter (fast/strong selon complexité)
    4. Re-indexation RAG après chaque fichier créé → contexte cross-file
    5. Persistance JSON après chaque tâche → reprise possible après crash
    """

    def __init__(self, config: Config, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self._agent = ECLMAgent(config, project_root)
        self._architect = ArchitectAgent(config)
        self._critic = CriticAgent(config)
        self._dep_graph = DependencyGraph()
        self._sessions_dir = config.data_dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    # ── Planning ──────────────────────────────────────────────────────────────

    def plan(self, brief: str) -> ProjectSession:
        """Décompose un brief en ProjectSession via l'ArchitectAgent (32B).

        Args:
            brief: Description du projet en français.

        Returns:
            ProjectSession avec tâches ordonnées, stack tech, et review_gate éventuel.
        """
        session_id = str(uuid.uuid4())[:12]

        # Créer le dossier de sortie dédié à cette session
        output_dir = self.config.data_dir / "projects" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        arch = self._architect.plan(brief)
        tasks = self._tasks_from_arch(arch)

        if not tasks:
            tasks = self._fallback_plan(brief)

        session = ProjectSession(
            id=session_id,
            brief=brief,
            created_at=datetime.now(timezone.utc).isoformat(),
            tasks=tasks,
            tech_stack=arch.get("tech_stack", []),  # type: ignore[arg-type]
            estimated_files=arch.get("folder_structure", []),  # type: ignore[arg-type]
            review_gate=arch.get("review_gate"),  # type: ignore[assignment]
            output_dir=str(output_dir),
        )
        self._save(session)
        logger.info(
            "Plan créé: %d tâches, stack=%s, session=%s",
            len(tasks), session.tech_stack, session_id,
        )
        return session

    def _tasks_from_arch(self, arch: dict[str, object]) -> list[TaskRecord]:
        raw_tasks = arch.get("tasks", [])
        if not isinstance(raw_tasks, list):
            return []
        tasks = []
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            try:
                tasks.append(TaskRecord(
                    index=int(t["index"]),
                    action=str(t.get("action", "CREATE")).upper(),
                    target_type=str(t.get("target_type", "function")),
                    target_name=str(t["target_name"]),
                    target_file=str(t["target_file"]),
                    description=str(t["description"]),
                    depends_on=[int(d) for d in t.get("depends_on", [])],
                    complexity=str(t.get("complexity", "medium")),
                ))
            except (KeyError, ValueError, TypeError):
                continue
        return tasks

    def _fallback_plan(self, brief: str) -> list[TaskRecord]:
        """Plan minimal quand l'ArchitectAgent échoue."""
        return [TaskRecord(
            index=0,
            action="CREATE",
            target_type="function",
            target_name="main",
            target_file="main.py",
            description=brief,
            complexity="high",
        )]

    # ── Exécution ─────────────────────────────────────────────────────────────

    def execute(
        self,
        session: ProjectSession,
        on_task_start: object = None,
        on_task_done: object = None,
        on_critic_done: object = None,
    ) -> ProjectSession:
        """Exécute toutes les tâches pending de la session de façon itérative.

        Args:
            session: Session à exécuter (peut être partiellement complétée).
            on_task_start: Callback(task) appelé avant chaque tâche.
            on_task_done: Callback(task, response) appelé après chaque tâche.
            on_critic_done: Callback(issues, test_results) appelé après la révision finale.

        Returns:
            Session mise à jour avec les résultats.
        """
        out = Path(session.output_dir) if session.output_dir else None

        # Scaffolding initial (uniquement au premier démarrage, pas sur reprise)
        if out and session.done_count == 0:
            self._scaffold_project(session, out)

        self._execute_parallel(session, out, on_task_start, on_task_done)

        if out and session.done_count > 0:
            self._write_readme(session)
            self._post_execute_loop(session, out, on_task_start, on_task_done, on_critic_done)

        return session

    def _scaffold_project(self, session: ProjectSession, out_dir: Path) -> None:
        """Génère les fichiers de scaffolding avant toute tâche."""
        # conftest.py — sys.path pour que pytest trouve les modules
        conftest = out_dir / "conftest.py"
        if not conftest.exists():
            conftest.write_text(
                "import sys\nfrom pathlib import Path\n"
                "sys.path.insert(0, str(Path(__file__).parent))\n",
                encoding="utf-8",
            )
            logger.info("Scaffolding: conftest.py")

        # __init__.py dans chaque dossier package
        stdlib = {"os", "sys", "json", "re", "io", "csv", "math", "time",
                  "datetime", "pathlib", "sqlite3", "logging", "typing",
                  "dataclasses", "collections", "itertools", "functools"}
        for entry in session.estimated_files:
            if not entry.endswith("/"):
                continue
            pkg_dir = out_dir / entry.rstrip("/")
            pkg_dir.mkdir(parents=True, exist_ok=True)
            init = pkg_dir / "__init__.py"
            if not init.exists():
                init.write_text("", encoding="utf-8")

        # pyproject.toml minimal
        toml = out_dir / "pyproject.toml"
        if not toml.exists():
            third_party = [d for d in session.tech_stack if d.lower() not in stdlib]
            deps = "\n".join(f'  "{d}",' for d in third_party)
            toml.write_text(
                '[build-system]\nrequires = ["setuptools"]\n\n'
                '[project]\nname = "generated-project"\nversion = "0.1.0"\n'
                'requires-python = ">=3.11"\n'
                f'dependencies = [\n{deps}\n]\n\n'
                '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
                'pythonpath = ["."]\n',
                encoding="utf-8",
            )
            logger.info("Scaffolding: pyproject.toml")

    def _execute_parallel(
        self,
        session: ProjectSession,
        out: Path | None,
        on_task_start: object,
        on_task_done: object,
    ) -> None:
        """Exécute les tâches en parallèle en respectant le DAG de dépendances."""
        lock = threading.Lock()
        max_workers = self.config.max_parallel_tasks

        def _run(task: TaskRecord) -> tuple[TaskRecord, AgentResponse]:
            return task, self._run_task(task, session.tasks, out)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures: dict[object, TaskRecord] = {}

            while True:
                # Recalculer l'état à chaque itération
                done_indices = {t.index for t in session.tasks if t.done}
                active_files = {t.target_file for t in futures.values()}

                for task in session.tasks:
                    if task.status != "pending":
                        continue
                    if not all(d in done_indices for d in task.depends_on):
                        continue
                    if task.target_file in active_files:
                        continue  # ne pas écrire dans le même fichier en parallèle

                    task.status = "running"
                    task.started_at = datetime.now(timezone.utc).isoformat()
                    if callable(on_task_start):
                        on_task_start(task)  # type: ignore[operator]
                    f = pool.submit(_run, task)
                    futures[f] = task
                    active_files.add(task.target_file)

                if not futures:
                    # Vérifier si des tâches sont bloquées par des dépendances échouées
                    still_pending = [t for t in session.tasks if t.status == "pending"]
                    for task in still_pending:
                        task.status = "skipped"
                        task.error = "Dépendances non résolues (échec amont)"
                    if still_pending:
                        self._save(session)
                    break

                done_fs, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)

                for future in done_fs:
                    task = futures.pop(future)
                    _, response = future.result()
                    self._update_task(task, response)
                    if callable(on_task_done):
                        on_task_done(task, response)  # type: ignore[operator]
                    if response.written_to:
                        self._agent._rag.index_file(response.written_to)
                    with lock:
                        self._save(session)

    def _post_execute_loop(
        self,
        session: ProjectSession,
        out_dir: Path,
        on_task_start: object,
        on_task_done: object,
        on_critic_done: object,
    ) -> None:
        """Boucle tests + Critic + fix : max _MAX_FIX_PASSES tentatives de correction."""
        for fix_pass in range(_MAX_FIX_PASSES + 1):
            test_results = self._run_project_tests(out_dir)
            issues = self._critic.review(out_dir, session.all_files_created)

            if callable(on_critic_done):
                on_critic_done(issues, test_results, fix_pass)  # type: ignore[operator]

            session.critic_issues = [
                {"file": i.file, "issue_type": i.issue_type,
                 "description": i.description, "severity": i.severity}
                for i in issues
            ]
            session.test_results = test_results
            self._save(session)

            # Pas d'erreur → on s'arrête
            score = float(test_results.get("score", 1.0))
            total = int(test_results.get("total", 0))
            errors_only = [i for i in issues if i.severity == "error"]
            if (total == 0 or score >= 1.0) and not errors_only:
                break

            # Dernière passe atteinte → on arrête de corriger
            if fix_pass >= _MAX_FIX_PASSES:
                logger.warning("Max fix passes (%d) atteint — abandon", _MAX_FIX_PASSES)
                break

            failing = self._collect_failing_files(session, test_results, issues, out_dir)
            if not failing:
                break

            logger.info("Fix pass %d/%d — %d fichier(s) à corriger",
                        fix_pass + 1, _MAX_FIX_PASSES, len(failing))
            any_fixed = self._run_fix_pass(
                session, failing, out_dir, on_task_start, on_task_done
            )
            if not any_fixed:
                break

    def _collect_failing_files(
        self,
        session: ProjectSession,
        test_results: dict[str, object],
        issues: list[object],
        out_dir: Path,
    ) -> dict[str, list[str]]:
        """Identifie les fichiers problématiques depuis tests + Critic.

        Returns:
            Dict {rel_path: [error_descriptions]} trié par priorité.
        """
        from src.orchestrator.critic import CriticIssue
        failing: dict[str, list[str]] = {}

        # Failures pytest — "FAILED tests/foo.py::test_bar" ou "ERROR tests/foo.py"
        stdout = str(test_results.get("stdout", ""))
        for line in stdout.splitlines():
            m = re.match(r"(?:FAILED|ERROR)\s+([\w./\\-]+\.py)(?:::|$)", line)
            if m:
                rel = m.group(1).replace("\\", "/")
                failing.setdefault(rel, []).append(line.strip()[:200])

        # Issues Critic (seulement errors — les warnings ne méritent pas un fix)
        for issue in issues:
            if isinstance(issue, CriticIssue) and issue.severity == "error":
                failing.setdefault(issue.file, []).append(issue.description)

        return failing

    def _run_fix_pass(
        self,
        session: ProjectSession,
        failing: dict[str, list[str]],
        out_dir: Path,
        on_task_start: object,
        on_task_done: object,
    ) -> bool:
        """Tente de corriger les fichiers défaillants. Retourne True si ≥1 fix a tourné."""
        any_fixed = False

        for rel_file, errors in failing.items():
            abs_file = out_dir / rel_file
            if not abs_file.exists():
                logger.debug("Fichier introuvable pour fix: %s", rel_file)
                continue

            # Trouver la tâche principale qui a généré ce fichier
            primary = next(
                (t for t in session.tasks
                 if any(rel_file in f or str(abs_file) == f
                        for f in t.files_created)),
                None,
            )
            if primary is None:
                logger.debug("Aucune tâche associée à %s — skip", rel_file)
                continue

            error_block = "\n".join(errors[:6])
            try:
                current = abs_file.read_text(encoding="utf-8")
            except OSError:
                current = ""

            dep_ctx = self._dep_graph.get_context_for_task(primary, session.tasks, out_dir)

            fix_intent = IntentJSON(
                action="CREATE",
                target_type="module",
                target_name=abs_file.stem,
                target_file=rel_file,
                description=(
                    f"Corrige les erreurs suivantes dans {rel_file}:\n\n"
                    f"ERREURS:\n{error_block}\n\n"
                    f"CODE ACTUEL (à corriger):\n{current[:2500]}"
                ),
                confidence=0.95,
            )

            if callable(on_task_start):
                # Fake task pour le callback UI
                class _FixTask:
                    label = f"FIX     {rel_file}"
                    index = primary.index
                on_task_start(_FixTask())  # type: ignore[operator]

            logger.info("Fixing %s (%d erreur(s))", rel_file, len(errors))
            response = self._agent.run(
                fix_intent,
                task_complexity="high",
                target_root=out_dir,
                dependency_context=dep_ctx,
            )

            if callable(on_task_done):
                on_task_done(primary, response)  # type: ignore[operator]

            if response.success and response.written_to:
                self._agent._rag.index_file(response.written_to)
                any_fixed = True
                logger.info("Fixed %s  score=%.2f", rel_file, response.score)
            else:
                logger.warning("Fix échoué pour %s: %s", rel_file, response.message[:100])

        return any_fixed

    def _run_project_tests(self, output_dir: Path) -> dict[str, object]:
        """Lance pytest sur le dossier complet du projet généré."""
        from src.verifier.pipeline import VerificationPipeline
        pipeline = VerificationPipeline(self.config)
        try:
            return pipeline.run_project_tests(output_dir)
        except Exception as exc:
            logger.warning("Project tests failed: %s", exc)
            return {"passed": 0, "failed": 0, "total": 0, "score": 0.0, "stdout": str(exc)}

    def _write_readme(self, session: ProjectSession) -> None:
        """Génère le README.md du projet avec les instructions de lancement."""
        output_dir = Path(session.output_dir)
        readme_path = output_dir / "README.md"
        if readme_path.exists():
            return

        files_created = [
            Path(f).relative_to(output_dir).as_posix()
            for f in session.all_files_created
            if Path(f).is_relative_to(output_dir)
        ]

        guide = self._architect.generate_run_guide(
            brief=session.brief,
            tech_stack=session.tech_stack,
            files_created=files_created,
        )
        readme_path.write_text(guide, encoding="utf-8")
        logger.info("README.md écrit dans %s", readme_path)

    def _run_task(
        self,
        task: TaskRecord,
        all_tasks: list[TaskRecord],
        output_dir: Path | None = None,
    ) -> AgentResponse:
        intent = IntentJSON(
            action=task.action,
            target_type=task.target_type,
            target_name=task.target_name,
            target_file=task.target_file if task.target_file else None,
            description=task.description,
            confidence=0.92,
        )
        dep_ctx = ""
        if output_dir:
            dep_ctx = self._dep_graph.get_context_for_task(task, all_tasks, output_dir)
        return self._agent.run(
            intent,
            task_complexity=task.complexity,
            target_root=output_dir,
            dependency_context=dep_ctx,
        )

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
                brief = data["brief"]
                sessions.append({
                    "id": data["id"],
                    "brief": brief[:60] + "…" if len(brief) > 60 else brief,
                    "created_at": data["created_at"],
                    "tasks": len(data["tasks"]),
                    "done": sum(1 for t in data["tasks"] if t["status"] == "done"),
                    "tech_stack": data.get("tech_stack", []),
                })
            except (KeyError, json.JSONDecodeError):
                continue
        return sessions
