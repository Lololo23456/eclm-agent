"""ECLM Agent — Interface CLI interactive."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.intent.dataset import IntentDataLogger
from src.intent.model import IntentExtractor
from src.orchestrator.agent import AgentResponse, ECLMAgent
from src.orchestrator.project import ProjectAgent, ProjectSession
from src.shared.config import Config
from src.shared.types import IntentJSON

# ── ANSI colors ──────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _RESET


def _banner() -> None:
    print(_c("╔══════════════════════════════════════════════════╗", _BOLD, _CYAN))
    print(_c("║   ECLM Agent — Coding Assistant IA (local)       ║", _BOLD, _CYAN))
    print(_c("╚══════════════════════════════════════════════════╝", _BOLD, _CYAN))
    print(_c("  Tapez votre commande en français.", _DIM))
    print(_c('  Commandes: /index /status /quitter', _DIM))
    print(_c('  Projet:    /project new "..." | /project list | /project resume <id>\n', _DIM))


def _print_intent(intent: IntentJSON) -> None:
    conf_color = _GREEN if intent.confidence >= 0.75 else _YELLOW
    conf_str = _c(f"{intent.confidence:.0%}", conf_color, _BOLD)
    target = f"{intent.target_type}:{intent.target_name}" if intent.target_name else intent.target_type
    print(f"  {_c('[Intent]', _MAGENTA, _BOLD)}  {intent.action} {_c(target, _BOLD)} — confiance {conf_str}")
    if intent.target_file:
        print(f"  {_c('[Fichier]', _DIM)}  {intent.target_file}")
    if intent.constraints:
        print(f"  {_c('[Contraintes]', _DIM)} {', '.join(intent.constraints)}")


def _print_response(response: AgentResponse) -> None:
    if response.success:
        score_str = _c(f"{response.score:.2f}", _GREEN, _BOLD)
        print(f"\n  {_c('✓', _GREEN, _BOLD)} {_c(response.message, _GREEN)}")
        print(f"  {_c('Score:', _DIM)} {score_str}  {_c(f'({response.retries_used+1} essai(s))', _DIM)}")
        if response.written_to:
            print(f"  {_c('Écrit dans:', _DIM)} {_c(str(response.written_to), _CYAN)}")
        print(f"\n{_c('─' * 52, _DIM)}")
        print(response.code)
        print(_c("─" * 52, _DIM))
    else:
        print(f"\n  {_c('✗', _RED, _BOLD)} {_c(response.message, _YELLOW)}")
        if response.code:
            print(f"\n{_c('Meilleur candidat (non validé):', _DIM)}")
            print(_c("─" * 52, _DIM))
            print(response.code)
            print(_c("─" * 52, _DIM))


def _print_project_summary(session: ProjectSession) -> None:
    done = session.done_count
    total = session.total
    failed = sum(1 for t in session.tasks if t.status == "failed")
    bar_filled = int(done / total * 20) if total > 0 else 0
    bar = _c("█" * bar_filled, _GREEN) + _c("░" * (20 - bar_filled), _DIM)
    print(f"\n  {_c('Session:', _DIM)} {session.id}")
    print(f"  {_c('Brief:', _DIM)} {session.brief[:60]}")
    print(f"  {bar} {done}/{total} tâches")
    if failed:
        print(f"  {_c(f'{failed} échouée(s)', _RED)}")
    files = session.all_files_created
    if files:
        print(f"  {_c('Fichiers créés:', _DIM)} {', '.join(files)}")
    print()


def _run_project(
    brief: str,
    project_agent: ProjectAgent,
    project_root: Path,
) -> None:
    print(_c(f"\n  Planification du projet...", _DIM))
    session = project_agent.plan(brief)
    print(f"  {_c('✓', _GREEN)} {session.total} tâches planifiées (session: {session.id})")
    print()

    def on_start(task: object) -> None:  # type: ignore[type-arg]
        from src.orchestrator.project import TaskRecord as TR
        assert isinstance(task, TR)
        print(f"  {_c('[' + str(task.index + 1) + '/' + str(session.total) + ']', _DIM, _BOLD)} {task.label}")

    def on_done(task: object, resp: object) -> None:  # type: ignore[type-arg]
        from src.orchestrator.project import TaskRecord as TR
        from src.orchestrator.agent import AgentResponse as AR
        assert isinstance(task, TR) and isinstance(resp, AR)
        if resp.success:
            score_str = _c(f"{resp.score:.2f}", _GREEN)
            written = f" → {_c(str(resp.written_to), _CYAN)}" if resp.written_to else ""
            print(f"      {_c('✓', _GREEN)} score={score_str}{written}")
        else:
            print(f"      {_c('✗', _RED)} {resp.message[:80]}")

    project_agent.execute(session, on_task_start=on_start, on_task_done=on_done)
    _print_project_summary(session)


def _handle_special(
    command: str,
    agent: ECLMAgent,
    project_agent: ProjectAgent,
    config: Config,
    project_root: Path,
) -> bool:
    """Traite les commandes spéciales. Retourne True si traité."""
    cmd = command.strip()
    cmd_lower = cmd.lower()

    if cmd_lower in ("/quitter", "/quit", "/exit", "exit", "quitter"):
        print(_c("\nÀ bientôt.", _DIM))
        sys.exit(0)

    if cmd_lower == "/index":
        print(_c("  Indexation du codebase...", _DIM))
        n = agent.index_project()
        print(f"  {_c('✓', _GREEN)} {n} chunks indexés")
        return True

    if cmd_lower == "/status":
        logger_data = IntentDataLogger(config)
        n_examples = logger_data.count()
        sessions = project_agent.list_sessions()
        print(f"  {_c('Dataset intent:', _DIM)} {n_examples} exemples")
        print(f"  {_c('Modèle C2:', _DIM)} Ollama ({config.ollama_model})")
        print(f"  {_c('Projet:', _DIM)} {config.root_dir}")
        print(f"  {_c('Sessions:', _DIM)} {len(sessions)} sauvegardée(s)")
        return True

    # ── /project commands ────────────────────────────────────────────────────
    if cmd_lower.startswith("/project"):
        parts = cmd.split(None, 2)
        sub = parts[1].lower() if len(parts) > 1 else ""

        if sub == "new" and len(parts) >= 3:
            brief = parts[2].strip('"\'')
            _run_project(brief, project_agent, project_root)
            return True

        if sub == "list":
            sessions = project_agent.list_sessions()
            if not sessions:
                print(_c("  Aucune session sauvegardée.", _DIM))
            for s in sessions:
                done_str = _c(f"{s['done']}/{s['tasks']}", _GREEN if s['done'] == s['tasks'] else _YELLOW)
                print(f"  {_c(str(s['id']), _BOLD)}  {done_str}  {s['brief']}")
            return True

        if sub == "resume" and len(parts) >= 3:
            session_id = parts[2].strip()
            try:
                session = project_agent.load(session_id)
                remaining = sum(1 for t in session.tasks if t.status == "pending")
                if remaining == 0:
                    print(_c(f"  Session {session_id} déjà complète.", _GREEN))
                    return True
                print(f"  {_c('Reprise:', _DIM)} {remaining} tâche(s) restante(s)")

                def on_start(task: object) -> None:  # type: ignore[type-arg]
                    from src.orchestrator.project import TaskRecord as TR
                    assert isinstance(task, TR)
                    print(f"  {_c('[' + str(task.index + 1) + '/' + str(session.total) + ']', _DIM, _BOLD)} {task.label}")

                def on_done(task: object, resp: object) -> None:  # type: ignore[type-arg]
                    from src.orchestrator.project import TaskRecord as TR
                    from src.orchestrator.agent import AgentResponse as AR
                    assert isinstance(task, TR) and isinstance(resp, AR)
                    icon = _c('✓', _GREEN) if resp.success else _c('✗', _RED)
                    print(f"      {icon} score={resp.score:.2f}")

                project_agent.execute(session, on_task_start=on_start, on_task_done=on_done)
                _print_project_summary(session)
            except FileNotFoundError:
                print(_c(f"  Session introuvable: {session_id}", _RED))
            return True

        print(_c('  Usage: /project new "brief" | /project list | /project resume <id>', _YELLOW))
        return True

    if cmd_lower.startswith("/"):
        print(_c(f"  Commande inconnue: {cmd_lower}", _YELLOW))
        return True

    return False


def run_repl(project_root: Path, config: Config) -> None:
    """Boucle interactive principale."""
    extractor = IntentExtractor(config)
    data_logger = IntentDataLogger(config)
    agent = ECLMAgent(config, project_root)
    project_agent = ProjectAgent(config, project_root)

    _banner()

    while True:
        try:
            raw = input(_c("→ ", _CYAN, _BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print(_c("\nÀ bientôt.", _DIM))
            break

        if not raw:
            continue

        if _handle_special(raw, agent, project_agent, config, project_root):
            continue

        # ── C0 : extraction d'intention ─────────────────────────────────────
        print(_c("  Analyse de l'intention...", _DIM), end="\r")
        intent = extractor.extract(raw)
        _print_intent(intent)

        if intent.needs_clarification:
            question = agent._clarification_question(intent)
            print(f"\n  {_c('?', _YELLOW, _BOLD)} {question}\n")
            data_logger.log(raw, intent, validated=False)
            continue

        # ── Tests comportement optionnels ────────────────────────────────────
        behavior_tests: list[str] = []

        # ── Pipeline complet ─────────────────────────────────────────────────
        print(_c("  Génération en cours...   ", _DIM), end="\r")
        response = agent.run(intent, behavior_tests=behavior_tests)
        _print_response(response)

        data_logger.log(raw, intent, validated=response.success)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ECLM Agent — Assistant de codage IA local")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("."),
        help="Racine du projet cible (défaut: répertoire courant)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = Config(root_dir=args.project.resolve())
    run_repl(project_root=args.project.resolve(), config=config)


if __name__ == "__main__":
    main()
