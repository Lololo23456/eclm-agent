# /train

Lance l'entraînement d'un composant spécifique.

## Usage
```
/train dpo             ← DPO mensuel (nécessite ≥ 100 paires)
/train eclm            ← GRPO fine-tune du Code Worker (nécessite ≥ 500 exemples curriculum)
/train intent          ← CamemBERT Intent Extractor (nécessite ≥ 500 exemples labelisés)
/train planner         ← Flan-T5 AST Planner (nécessite ≥ 500 paires)
/train testgen         ← Flan-T5 TestGenerator (nécessite ≥ 500 paires)
```

## Prérequis — vérifier AVANT de lancer

```bash
# Paires DPO disponibles
ls data/dpo_pairs/*.jsonl | xargs wc -l

# Curriculum ECLM
ls data/training/eclm/*.jsonl | xargs wc -l

# Dataset intent
wc -l data/training/intent/intent_bootstrap.jsonl 2>/dev/null || echo "manquant"

# GPU VRAM disponible
nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "Pas de GPU NVIDIA"
```

## /train dpo — Le plus prioritaire

Lance le DPO sur les paires collectées automatiquement en prod.

```bash
# Vérifier le nombre de paires
python -c "
from src.improvement.dpo_collector import DPOCollector
from src.shared.config import Config
c = DPOCollector(Config())
print(f'Paires disponibles: {c.count()}')
print(f'Seuil pour DPO: 100 (stable: 500)')
"

# Lancer si ≥ 100 paires
python scripts/monthly_finetune.py \
  --pairs data/dpo_pairs/ \
  --output models/eclm_dpo/

# Benchmark obligatoire après
python scripts/run_benchmark.py --mode quick
```

Déployer uniquement si benchmark_score ≥ baseline (0.96).

## /train eclm — GRPO avec execution reward

Entraîne le Code Worker via Reinforcement Learning from Execution (RLE).

```bash
# Vérifier le curriculum (besoin ≥ 500 exemples)
ls data/training/eclm/curriculum_*.jsonl

# Générer le curriculum si manquant
python scripts/build_curriculum.py --source-dir ~/path/to/python/repos

# Lancer le GRPO
python -m src.eclm.train \
  --curriculum data/training/eclm/ \
  --output models/eclm/ \
  --base-model qwen2.5-coder:7b
# Exporte automatiquement GGUF + Modelfile pour Ollama
```

## /train intent — CamemBERT (optionnel, bas priorité)

> ⚠️ Seulement utile si le 7B Ollama manque de précision sur l'intent extraction.
> L'Ollama 7B fait déjà du bon travail zero-shot.

```bash
# Générer les exemples (si < 500)
python scripts/bootstrap_dataset.py --backend ollama --n 2000
# ou avec API Claude (~10€, plus rapide) :
# ANTHROPIC_API_KEY=sk-... python scripts/bootstrap_dataset.py --backend claude --n 2000

# Entraîner quand ≥ 500 exemples
python -m src.intent.train \
  --data data/training/intent/ \
  --output models/intent/
```

## /train planner — Flan-T5 (optionnel, bas priorité)

```bash
# Générer les exemples via bootstrap
python scripts/bootstrap_dataset.py --backend ollama --n 2000

# Entraîner
python src/planner/train.py \
  --data data/training/planner/ \
  --output models/planner/
```

## /train testgen — Flan-T5 (optionnel)

```bash
python src/verifier/test_generator/train.py \
  --data data/training/test_generator/ \
  --output models/testgen/
```

## Ordre de priorité

```
1. /train dpo      ← impact immédiat, dès 100 paires
2. /train eclm     ← nécessite curriculum (build_curriculum.py)
3. /train intent   ← optionnel, 7B Ollama suffisant
4. /train planner  ← optionnel, règles + 7B Ollama suffisants
5. /train testgen  ← optionnel
```

## Règles absolues
- JAMAIS déployer sans `/benchmark` d'abord
- JAMAIS entraîner ECLM sur données non vérifiées par le Verifier
- Toujours backup du modèle précédent avant d'écraser
- Si GPU < 8GB VRAM → utiliser CPU (lent, possible) ou louer un GPU cloud
