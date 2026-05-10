"""Benchmark de l'agent sur un ensemble de tâches de référence.

Usage:
    python scripts/run_benchmark.py [--suite quick|full] [--output data/benchmarks/]

Le benchmark mesure :
- Pass@1 : taux de succès au premier essai
- Pass@3 : taux de succès dans les 3 essais autorisés
- Score moyen composite
- Temps moyen par tâche

Les résultats sont sauvegardés en JSONL pour suivre la progression mensuelle.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# On remonte d'un niveau pour accéder aux src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.project import ProjectAgent
from src.shared.config import Config


# ── Suite de benchmarks ────────────────────────────────────────────────────────

QUICK_SUITE = [
    {
        "id": "bm_add_fn",
        "brief": 'Crée une fonction Python "add(a: int, b: int) -> int" qui retourne a + b avec un docstring.',
        "expect_files": 1,
        "expect_score_min": 0.85,
    },
    {
        "id": "bm_dataclass",
        "brief": 'Crée un dataclass Python "Point" avec x: float et y: float, une méthode distance(other: Point) -> float.',
        "expect_files": 1,
        "expect_score_min": 0.80,
    },
    {
        "id": "bm_cli_greet",
        "brief": 'Crée un CLI Python avec argparse qui prend un argument --name et affiche "Hello, <name>!" avec gestion de l\'erreur si name est vide.',
        "expect_files": 1,
        "expect_score_min": 0.75,
    },
]

FULL_SUITE = QUICK_SUITE + [
    {
        "id": "bm_todo_cli",
        "brief": "Crée un CLI todo list en Python avec SQLite : commandes add, list, done, delete. Inclure les tests pytest.",
        "expect_files": 3,
        "expect_score_min": 0.70,
    },
    {
        "id": "bm_rest_api",
        "brief": "Crée une API REST minimaliste en Python avec FastAPI : GET /items, POST /items, DELETE /items/{id}. Données en mémoire. Inclure les tests.",
        "expect_files": 3,
        "expect_score_min": 0.65,
    },
    {
        "id": "bm_csv_processor",
        "brief": "Crée un module Python qui lit un CSV, filtre les lignes selon une condition, et écrit le résultat dans un nouveau CSV. Inclure les tests.",
        "expect_files": 2,
        "expect_score_min": 0.70,
    },
]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_benchmark(suite: list[dict[str, object]], config: Config) -> dict[str, object]:
    results = []
    total_start = time.time()

    for task in suite:
        task_id = str(task["id"])
        brief = str(task["brief"])
        expect_score_min = float(task.get("expect_score_min", 0.75))  # type: ignore[arg-type]

        print(f"\n  [{task_id}] {brief[:70]}...")
        task_start = time.time()

        project_agent = ProjectAgent(config, Path("."))
        session = project_agent.plan(brief)

        task_scores: list[float] = []
        all_passed = True

        def on_task_done(t: object, r: object) -> None:
            from src.orchestrator.project import TaskRecord as TR
            from src.orchestrator.agent import AgentResponse as AR
            assert isinstance(t, TR) and isinstance(r, AR)
            task_scores.append(r.score)
            icon = "✓" if r.success else "✗"
            print(f"      {icon} {t.label[:50]}  score={r.score:.2f}")

        project_agent.execute(session, on_task_done=on_task_done)

        elapsed = time.time() - task_start
        avg_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
        passed = session.done_count
        total = session.total
        pass_rate = passed / total if total > 0 else 0.0

        ok = pass_rate >= 0.8 and avg_score >= expect_score_min
        print(f"  {'✓' if ok else '✗'} {task_id}: {passed}/{total} tâches  "
              f"avg_score={avg_score:.2f}  {elapsed:.0f}s")

        results.append({
            "id": task_id,
            "ok": ok,
            "pass_rate": pass_rate,
            "avg_score": avg_score,
            "tasks_done": passed,
            "tasks_total": total,
            "elapsed_s": round(elapsed, 1),
            "session_id": session.id,
        })

    total_elapsed = time.time() - total_start
    ok_count = sum(1 for r in results if r["ok"])
    avg_score_all = sum(r["avg_score"] for r in results) / len(results) if results else 0.0  # type: ignore[index]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_fast": config.fast_model,
        "model_strong": config.strong_model,
        "tasks_total": len(results),
        "tasks_ok": ok_count,
        "pass_rate": ok_count / len(results) if results else 0.0,
        "avg_score": round(avg_score_all, 3),
        "total_elapsed_s": round(total_elapsed, 1),
        "results": results,
    }

    print(f"\n{'='*52}")
    print(f"  Benchmark : {ok_count}/{len(results)} tâches OK  "
          f"avg_score={avg_score_all:.2f}  total={total_elapsed:.0f}s")
    print(f"{'='*52}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ECLM Agent")
    parser.add_argument("--suite", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = Config()
    suite = QUICK_SUITE if args.suite == "quick" else FULL_SUITE

    print(f"  Benchmark '{args.suite}' — {len(suite)} tâche(s)")
    print(f"  Fast model: {config.fast_model}")
    print(f"  Strong model: {config.strong_model}")

    summary = run_benchmark(suite, config)

    out_dir = args.output or config.benchmarks_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"benchmark_{args.suite}_{ts}.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Résultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
