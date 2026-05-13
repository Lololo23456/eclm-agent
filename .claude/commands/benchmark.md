# /benchmark

Lance le benchmark privé pour mesurer la qualité de l'agent.

## Usage
```
/benchmark               ← quick (3 projets, ~15-30 min sur 7B)
/benchmark quick         ← idem
/benchmark full          ← suite complète (~1h sur 7B)
/benchmark compare       ← compare modèle courant vs baseline sauvegardée
/benchmark add "<tâche>" ← ajoute une tâche au benchmark
```

## Ce que tu dois faire

### /benchmark (quick ou sans argument)
```bash
python scripts/run_benchmark.py --mode quick 2>&1 | tee /tmp/bench_output.txt
```

Afficher le résultat structuré :
```
Benchmark quick — 3 projets
─────────────────────────────────────────────────
  bm_add_fn       ✓ 2/2   score=0.95   614s
  bm_dataclass    ✓ 2/2   score=0.97   337s
  bm_cli_greet    ✓ 4/4   score=0.96   819s
─────────────────────────────────────────────────
  Total : 3/3 projets OK   avg=0.96   Pass@1=100%
  Baseline (2026-05-10)  : avg=0.96
  Delta : ±0.00  → stable
```

### /benchmark full
```bash
python scripts/run_benchmark.py --mode full
```

### /benchmark compare
Compare le dernier JSON dans `data/benchmarks/` vs l'avant-dernier.
```bash
ls -t data/benchmarks/benchmark_*.json | head -2
# puis comparer avg_score et pass_rate
```
- RÉGRESSION si avg_score < baseline − 0.02
- Ne JAMAIS déployer un modèle fine-tuné en régression

### /benchmark add "<tâche>"
Ajouter une tâche au fichier `scripts/run_benchmark.py` (tableau `_BENCHMARKS`). Format :
```python
BenchmarkSpec(
    id="bm_<nom>",
    brief="<description complète>",
    expected_files=["src/...", "tests/..."],
    difficulty=1,  # 1-3
)
```
Ajouter une tâche pour chaque bug trouvé en production.

## Interprétation des scores

| avg_score | Statut |
|-----------|--------|
| ≥ 0.95 | Excellent |
| 0.90–0.94 | Bon |
| 0.80–0.89 | Acceptable |
| < 0.80 | À corriger avant déploiement |

## Baseline de référence
- Fichier : `data/benchmarks/benchmark_quick_20260510_1255.json`
- Score : avg=0.96 / Pass@1=100%
- Contexte : M3 Air, 7B seulement

## Règles
- JAMAIS déployer un modèle sans benchmark
- Toujours sauvegarder le JSON résultat dans `data/benchmarks/`
- Lancer `/benchmark` avant ET après chaque `/train dpo` ou `/train eclm`
