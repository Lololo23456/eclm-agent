# ECLM Agent — Agent de Codage IA Privé

## Vision

Construire un agent de codage IA **local, auto-améliorant, sans limite de token**.

Principe fondamental : **l'exécution est la vérité**. Le score de vérification (tests qui passent,
mypy, ruff) est le seul signal qui compte — pas next-token prediction.

**Architecture V2 — Hybride adaptatif** :
- **Tier local** : Ollama 7B (toujours disponible, gratuit, rapide)
- **Tier cloud** : Claude API (on-demand, pour les tâches complexes / ambiguës)
- Le routeur choisit automatiquement selon complexité et confiance.

---

## Architecture — Pipeline Hybride

```
[Utilisateur — français]
        ↓
[Router] — complexité + confiance → choix du tier
        ↓                                    ↓
[Tier Local — Ollama 7B]          [Tier Cloud — Claude API]
   • Intent extraction                • Architecture complexe
   • Code generation (worker)         • Ambiguïté haute
   • Test generation                  • Critique cross-fichiers
   • Simple planning
        ↓
[Contexte] — ChromaDB RAG + DependencyGraph
        ↓
[AST Planner] → ASTOperationPlan (LLM + règles déterministes)
        ↓
[Code Worker × k candidats] — Ollama (beam_k adaptatif)
        ↓
[C3] Multi-layer Verifier — 6 couches
        ↓ meilleur candidat
[IDE — Continue.dev / CLI REPL]
        ↓
[C4] Self-Improvement — DPO mensuel automatique
```

### Tier Local — Ollama (toujours disponible)
- **Modèle** : `qwen2.5-coder:7b` (4.7 GB, ~10 tokens/s sur CPU, ~60 t/s sur GPU récent)
- **Rôle** : intent extraction, génération code, tests, planning simple
- **GPU requis** : ≥ 6 GB VRAM (ou CPU avec ≥ 16 GB RAM)
- **Coût** : 0€, pas de limite de token
- Toutes les requêtes Ollama ont un `timeout=60s` max

### Tier Cloud — Claude API (optionnel)
- **Modèle** : `claude-haiku-4-5` (rapide, ~0.001€/req) ou `claude-sonnet-4-6` (meilleur)
- **Rôle** : tâches architecturales, critique globale, ambiguïté haute (confidence < 0.6)
- **Activation** : `ANTHROPIC_API_KEY` dans `.env`
- **Sans clé** : dégradation gracieuse → Ollama 7B pour tout
- **Jamais** : pour la génération de code en boucle (coût)

### Adaptation GPU automatique
```python
# src/shared/config.py — détection VRAM au démarrage
VRAM_8GB   → qwen2.5-coder:7b  (Q4_K_M)
VRAM_12GB  → qwen2.5-coder:14b (Q4_K_M)
VRAM_24GB  → qwen2.5-coder:32b (Q4_K_M)
CPU_ONLY   → qwen2.5-coder:7b  (lent mais fonctionnel)
```

---

## Composants

### Intent Extraction (C0)
- **Implémentation actuelle** : Ollama 7B (zero-shot, JSON structuré)
- **Futur optionnel** : fine-tune CamemBERT si ≥ 500 exemples français collectés
- **Règle** : si confiance < 0.6 → Claude API ; si toujours < 0.75 → UNE question

**Structure IntentJSON** :
```json
{
  "action": "MODIFY|CREATE|DELETE|REFACTOR|FIX|ADD|RENAME|EXPLAIN|CONVERT|TEST|OPTIMIZE|EXTRACT|MERGE|SPLIT",
  "target_type": "function|class|file|module|endpoint|test",
  "target_name": "string",
  "target_file": "string|null",
  "description": "string — description normalisée en anglais",
  "constraints": ["string"],
  "confidence": 0.0-1.0
}
```

### AST Planner (C1)
- **Implémentation actuelle** : règles déterministes + Ollama 7B pour cas complexes
- **Futur optionnel** : fine-tune Flan-T5 si ≥ 500 paires (intent, plan) collectées
- **Output** : `ASTOperationPlan` — liste ordonnée d'`ASTOperation`

**Types d'ASTOperation** :
```
ADD_PARAM · MODIFY_BODY · REMOVE_PARAM · ADD_RETURN_TYPE
RENAME_SYMBOL · ADD_IMPORT · CREATE_FUNCTION · CREATE_CLASS
ADD_METHOD · DELETE_NODE · UPDATE_CALL_SITES · ADD_DECORATOR
EXTRACT_FUNCTION · CREATE_MODULE
```

### Code Worker (C2)
- **Implémentation** : Ollama 7B (qwen2.5-coder) via `ECLMCore`
- **Beam search adaptatif** : k=1 (CREATE simple) → k=5 (REFACTOR complexe)
- **Représentation** : AST-first (génère du code via opérations AST, pas texte brut)
- **Signal d'entraînement** : `reward = tests_pass_rate` (GRPO, pas next-token)

### Multi-layer Verifier (C3)
- **6 couches** (dans l'ordre) :
  1. Syntax check — `ast.parse()` (~0ms)
  2. Type check — `mypy` depuis `/tmp` (pas de pyproject.toml projet)
  3. Lint — `ruff check` (<1s)
  4. Tests behaviour — fournis par l'utilisateur
  5. Tests impl — `TestGenerator` Ollama (isolé, jamais les candidats C2)
  6. Property tests — `hypothesis`
- **Score composite** : `0.4*tests_b + 0.3*tests_i + 0.2*property + 0.1*lint`
- **Sandbox** : Docker si disponible, sinon `LocalSandbox` (subprocess isolé)
- **Retry** : si score < 0.8 → réinjection erreur → max 3 passes

### Self-Improvement (C4)
- **DPO** : paires (rejected, chosen) depuis échecs→succès et self-play
- **GRPO** : unsloth + trl.GRPOTrainer, reward = exécution, export GGUF
- **Cycle** : mensuel — collect → fine-tune → benchmark → deploy si score ↑
- **Adversarial** : `AdversarialLoop` génère à 4 températures, meilleur = chosen

### Primitive Library
- **ChromaDB** local (`data/primitives/`)
- Top-3 par similarité cosine à chaque requête
- Ajout via `/add-primitive` uniquement (validation complète)

### GraphRAG / DependencyGraph
- **Mode REPL** : ChromaDB RAG incrémental (indexé après chaque fichier)
- **Mode projet** : DependencyGraph seulement (extract_public_api par fichier)
- Chunking AST-aware (Tree-sitter), jamais tokenization arbitraire

---

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| LLM serving | Ollama (local) |
| Modèle code | qwen2.5-coder:7b (ou 14b/32b selon VRAM) |
| LLM cloud | Claude API (optionnel) |
| AST parsing | `ast` stdlib + Tree-sitter |
| Vector store | ChromaDB ≥ 0.4 |
| Fine-tuning | unsloth + trl (GRPO/DPO) |
| Type check | mypy --strict |
| Lint | ruff |
| Property tests | hypothesis |
| Sandbox | Docker ou LocalSandbox |
| IDE | Continue.dev + FastAPI server |
| Framework ML | PyTorch ≥ 2.2 + HuggingFace |

---

## Structure des Fichiers

```
eclm-agent/
├── CLAUDE.md
├── .claude/
│   ├── commands/           ← Slash commands (skills)
│   │   ├── project.md      /project new/status/resume/plan
│   │   ├── verify.md       /verify [fichier]
│   │   ├── train.md        /train intent|eclm|planner|dpo
│   │   ├── benchmark.md    /benchmark quick|full
│   │   ├── scaffold.md     /scaffold component|test|script
│   │   ├── dpo-collect.md  /dpo-collect
│   │   ├── add-primitive.md /add-primitive
│   │   └── debug.md        /debug [fichier] — analyse les erreurs
│   └── memory/
│
├── src/
│   ├── shared/             ← types.py + config.py (VRAM detection)
│   ├── intent/             ← C0: Ollama (+ CamemBERT optionnel)
│   ├── planner/            ← C1: règles + Ollama
│   ├── eclm/               ← C2: ECLMCore + beam_search + GRPO train
│   ├── verifier/           ← C3: pipeline + sandbox + scorer + testgen
│   ├── library/            ← Primitive Library (ChromaDB)
│   ├── improvement/        ← C4: dpo_collector + adversarial + finetune
│   ├── orchestrator/       ← agent + project + architect + router + rag
│   └── api/                ← FastAPI server (Continue.dev compatible)
│
├── data/
│   ├── primitives/         ← ChromaDB (gitignored)
│   ├── training/           ← intent/ planner/ eclm/ test_generator/
│   ├── dpo_pairs/          ← JSONL auto-généré (gitignored)
│   ├── benchmarks/         ← résultats benchmark JSON
│   └── sessions/           ← sessions projet JSON (gitignored)
│
├── tests/                  ← test_*.py (pytest)
├── scripts/                ← bootstrap_dataset.py, run_benchmark.py, ...
├── docker/sandbox/         ← Dockerfile sandbox Python
├── main.py                 ← CLI REPL interactif
└── requirements.txt
```

---

## Conventions de Code

### Python
- Python **3.11+**
- Type hints partout — `mypy --strict` doit passer
- `@dataclass(frozen=True)` pour les types immutables
- Pas de classes si une fonction suffit
- `ruff` pour le formatage (jamais `black` ni `isort`)

### Imports
```python
# stdlib → third-party → local
import ast
from pathlib import Path
from dataclasses import dataclass

import torch
import chromadb

from src.shared.types import IntentJSON
from src.shared.config import Config
```

### Tests
- `pytest` uniquement
- Chaque fonction publique a ≥ 1 test
- Fixtures dans `tests/conftest.py`
- Hypothesis pour les fonctions pures

---

## Règles Critiques

1. **L'exécution est la vérité** — jamais déployer sans score verifier ≥ 0.8
2. **Isoler TestGenerator** — jamais voir les candidats C2 avant de générer les tests
3. **Pas d'exécution hors sandbox** — tout passe par `src/verifier/sandbox.py`
4. **Pas de path hardcodé** — tout dans `src/shared/config.py`
5. **Claude API = cerveau, pas ouvrier** — jamais en boucle de génération code
6. **Une seule question max** — si intent confidence < 0.75, poser UNE question
7. **Jamais entraîner sur données non vérifiées** — curriculum pré-filtré par C3
8. **Dégradation gracieuse** — sans Docker → LocalSandbox ; sans API key → Ollama tout

---

## Variables d'Environnement (.env)

```bash
# Optionnel — tier cloud
ANTHROPIC_API_KEY=sk-...

# Paths (defaults dans config.py)
ECLM_DATA_DIR=./data
ECLM_MODELS_DIR=./models

# Modèles (auto-détecté selon VRAM si non défini)
ECLM_FAST_MODEL=qwen2.5-coder:7b
ECLM_STRONG_MODEL=qwen2.5-coder:7b   # ou 14b/32b selon GPU
ECLM_USE_CLAUDE_API=false             # true si ANTHROPIC_API_KEY présente

# Hyperparamètres
ECLM_BEAM_WIDTH=5
ECLM_MAX_RETRIES=3
ECLM_CONFIDENCE_THRESHOLD=0.75
ECLM_MIN_VERIFICATION_SCORE=0.8
```

---

## Métriques de Succès

| Métrique | Objectif actuel | Long terme |
|----------|----------------|------------|
| Tests passing | 171/171 ✅ | maintenir |
| Pass@1 benchmark | 100% (7B, M3 Air) | ≥ 70% |
| Score moyen | 0.96 | ≥ 0.92 |
| Latence simple (7B) | ~30s (M3 Air CPU) | < 15s (GPU) |
| Paires DPO | 4 | ≥ 100 pour finetune |

## Roadmap

### Maintenant (Mac M3 Air + vieux GPU)
- [x] Pipeline complet opérationnel
- [x] Mode projet A→Z
- [x] Self-improvement (DPO + GRPO + adversarial)
- [ ] Accumuler paires DPO en usage réel (objectif : 100+)
- [ ] `python scripts/bootstrap_dataset.py --backend ollama --n 2000`

### Quand GPU performant disponible (≥ 12GB VRAM)
- [ ] Tirer qwen2.5-coder:14b ou 32b
- [ ] Premier GRPO fine-tune (besoin ≥ 500 exemples curriculum)
- [ ] Docker sandbox complet

### Optionnel / si usage intensif
- [ ] Fine-tune CamemBERT intent (si ≥ 500 exemples labelisés)
- [ ] Fine-tune Flan-T5 planner (si ≥ 500 paires)
