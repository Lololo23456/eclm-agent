# /benchmark

Lance le benchmark privé pour évaluer les performances de l'agent.

## Usage
```
/benchmark run              ← lance les tâches du benchmark
/benchmark compare          ← compare avec le dernier run
/benchmark add "<tâche>"    ← ajoute une tâche au benchmark
```

## Ce que tu dois faire

### /benchmark run
1. Vérifier que `data/benchmarks/benchmark_tasks.jsonl` existe (≥ 10 tâches)
2. Pour chaque tâche :
   ```python
   from src.intent.model import IntentExtractor
   from src.orchestrator.agent import ECLMAgent
   from src.shared.config import Config
   from pathlib import Path

   cfg = Config()
   extractor = IntentExtractor(cfg)
   agent = ECLMAgent(cfg, Path("."))

   intent = extractor.extract(task["command"])
   response = agent.run(intent, behavior_tests=task.get("tests", []))
   ```
3. Calculer et afficher :
   ```
   Benchmark 2026-05-09
   ────────────────────────────────────────
   Pass rate 1er essai    ?/? (objectif ≥ 70%)
   Pass rate final        ?/? (objectif ≥ 92%)
   Score composite moyen  ? (objectif ≥ 0.80)
   Latence moyenne        ?s (objectif < 15s)
   ────────────────────────────────────────
   ```
4. Sauvegarder dans `data/benchmarks/results_<timestamp>.json`

### /benchmark compare
Comparer les 2 derniers fichiers de résultats et indiquer : AMÉLIORATION / RÉGRESSION / STABLE.
Ne JAMAIS déployer un modèle si régression détectée.

### /benchmark add "<tâche>"
Format :
```json
{
  "id": "uuid",
  "command": "crée une fonction...",
  "expected_action": "CREATE",
  "expected_target": "function_name",
  "tests": ["def test_...(): assert ..."],
  "difficulty": 1
}
```
Ajouter systématiquement une tâche pour chaque bug trouvé en production.

## Règles
- JAMAIS déployer un nouveau modèle si benchmark_score régresse
- Benchmark = tâches réelles du projet, pas synthétiques
- Lancer `/benchmark run` avant et après chaque `/train dpo`
