# ECLM Agent — Agent de Codage IA Privé

## Vision

Construire un agent de codage IA **100% local, gratuit, sans limite de token, auto-améliorant**.
Ce n'est PAS un LLM classique. C'est un **Execution-Guided Compositional Language Model** (ECLM).

Principe fondamental : **l'exécution est la vérité**. Le signal d'entraînement n'est pas "quel token vient après"
mais "est-ce que le code fonctionne ?". La vérification remplace la taille du modèle.

---

## Architecture — 5 Composants Spécialisés

```
[Utilisateur — français]
        ↓
[C0] CamemBERT Intent Extractor  ←──── GraphRAG Codebase Context
        ↓ intent JSON
[C1] AST Planner                 ←──── Primitive Library (ChromaDB)
        ↓ plan d'opérations AST
[C2] ECLM Core (×5 candidats)   ←──── Primitive Library
        ↓ AST-diffs candidats
[C3] Multi-layer Verifier
        ↓ meilleur candidat validé
[IDE — Continue.dev]
        ↓ toute interaction
[C4] Self-Improvement Loop ──────────► re-fine-tune mensuel
```

### C0 — CamemBERT Intent Extractor
- **Rôle** : Français → Vecteur d'intention structuré (JSON)
- **Modèle base** : `camembert-base` (110M params, pré-entraîné sur français)
- **Fine-tuning** : ~2000 exemples de commandes coding français → intent JSON
- **Durée entraînement** : ~2h sur GTX 1080+
- **Output** : `IntentJSON` avec action, cible, contrainte, confiance
- **Règle** : si confiance < 0.75 → poser UNE seule question ciblée, jamais plus
- **Ne fait PAS** : générer du code, comprendre la syntaxe, raisonner

**Structure IntentJSON** :
```json
{
  "action": "MODIFY|CREATE|DELETE|REFACTOR|FIX|ADD|RENAME|EXPLAIN|CONVERT|TEST",
  "target_type": "function|class|file|module|endpoint|test",
  "target_name": "string",
  "target_file": "string|null",
  "description": "string — description normalisée en anglais",
  "constraints": ["string"],
  "confidence": 0.0-1.0
}
```

**Actions supportées (exhaustif)** :
`MODIFY · CREATE · DELETE · REFACTOR · FIX · ADD · RENAME · EXPLAIN · CONVERT · TEST · OPTIMIZE · EXTRACT · MERGE · SPLIT`

---

### GraphRAG Codebase Context (module, pas un modèle)
- **Rôle** : Fournir le sous-graphe AST pertinent à chaque requête
- **Chunking** : Tree-sitter (AST-aware, pas tokenization arbitraire)
- **Stockage** : ChromaDB local (`data/codebase_index/`)
- **Dépendances** : graphe de dépendances entre modules (qui importe quoi)
- **Mise à jour** : incrémentale à chaque fichier sauvegardé
- **Règle** : si `target_file` est null dans IntentJSON → requête sémantique pour trouver le fichier

---

### C1 — AST Planner
- **Rôle** : IntentJSON + contexte AST → liste ordonnée d'opérations AST atomiques
- **Modèle** : Transformer seq2seq ~200M params, entraîné sur paires (intent, plan)
- **Output** : `ASTOperationPlan` — liste ordonnée de `ASTOperation`
- **Ne génère PAS de code** — seulement des opérations structurées

**Types d'ASTOperation** :
```python
ADD_PARAM(target, param_name, param_type, default_value, position)
MODIFY_BODY(target, description)  # description normalisée pour l'ECLM
REMOVE_PARAM(target, param_name)
ADD_RETURN_TYPE(target, type)
RENAME_SYMBOL(old_name, new_name, scope)
ADD_IMPORT(module, symbol)
CREATE_FUNCTION(name, params, return_type, docstring)
CREATE_CLASS(name, bases, docstring)
ADD_METHOD(class_name, method_spec)
DELETE_NODE(target)
UPDATE_CALL_SITES(old_signature, new_signature)
ADD_DECORATOR(target, decorator)
EXTRACT_FUNCTION(source_target, new_name, lines)
```

---

### C2 — ECLM Core (le cœur)
- **Rôle** : Exécuter chaque ASTOperation et générer le code résultant
- **Modèle** : ~500M params, entraîné sur EXÉCUTION (pas next-token prediction)
- **Input** : une ASTOperation + sous-arbre AST actuel + primitives pertinentes
- **Output** : 5 candidats de code (beam search sur opérations AST valides)
- **Représentation** : opère sur AST via `ast` (Python) ou Tree-sitter — JAMAIS sur texte brut
- **Syntaxe invalide** : impossible par construction (AST uniquement)

**Signal d'entraînement** :
```
reward = 1.0  si tous les tests passent
reward = 0.5  si tests partiels (trace d'erreur fournie comme contexte)
reward = 0.0  si erreur de syntax/runtime avant même les tests
```

**Contrainte critique** : ne jamais utiliser next-token prediction comme objectif principal.
L'objectif est `maximize(execution_reward)`.

---

### C3 — Multi-layer Verifier
- **Rôle** : Scorer les 5 candidats ECLM, retourner le meilleur ou déclencher retry
- **Layers dans l'ordre** :
  1. **Syntax check** : `ast.parse()` — élimine instantanément (~0ms)
  2. **Type check** : `mypy --strict` — < 1s
  3. **Linting** : `ruff check` — < 1s
  4. **Tests comportement** : fournis par l'utilisateur — vérité de référence
  5. **Tests implémentation** : générés par TestGenerator (modèle 150M séparé)
  6. **Property tests** : Hypothesis — edge cases automatiques
- **Docker sandbox** : toute exécution dans container isolé (`docker/sandbox/`)
- **Score composite** : `0.4*tests_behavior + 0.3*tests_impl + 0.2*property + 0.1*lint_score`
- **Retry** : si score < 0.8 → self-reflection (erreur réinjectée) → max 3 retries

**TestGenerator** (modèle séparé ~150M params) :
- Totalement ISOLÉ de l'ECLM pendant génération
- Entraîné sur paires (code → tests)
- Ne voit PAS les candidats ECLM avant génération des tests
- Raison : éviter le cercle vicieux "code faux + test qui valide l'erreur"

---

### Primitive Library
- **Rôle** : Mémoire de fonctions atomiques vérifiées et testées
- **Stockage** : ChromaDB local (`data/primitives/`)
- **Structure de chaque primitive** :
```json
{
  "id": "uuid",
  "code": "string",
  "tests": ["string"],
  "intent_vector": [float],
  "language": "python",
  "domain": "parsing|http|auth|data|io|...",
  "verified_at": "ISO8601",
  "score": 0.0-1.0,
  "usage_count": int
}
```
- **Ajout** : uniquement via `/add-primitive` — validation complète obligatoire
- **Récupération** : top-3 par similarité cosine avec intent vector
- **Mise à jour** : re-vérification hebdomadaire automatique

---

### C4 — Self-Improvement Loop
- **Rôle** : Améliorer l'ECLM et le Planner en continu sans intervention humaine
- **Sources de données DPO** :
  1. Solution qui échoue puis réussit après retry → paire (rejected, chosen)
  2. Correction manuelle de l'utilisateur → paire (rejected, chosen)
  3. Self-play adversarial : Generator vs Critic
- **Stockage** : `data/dpo_pairs/` — format JSONL standard
- **Cycle** : mensuel — collecter → DPO fine-tune → benchmark → déployer si ≥ benchmark
- **Benchmark privé** : `data/benchmarks/` — 50-100 tâches réelles du projet

---

## Stack Technique

| Composant | Technologie | Version |
|-----------|-------------|---------|
| Modèle intent | `camembert-base` | HuggingFace |
| Framework ML | PyTorch | ≥ 2.2 |
| Training | HuggingFace `transformers` + `trl` | latest |
| Fine-tuning | QLoRA via `peft` | latest |
| AST parsing | `tree-sitter` + `ast` (stdlib) | — |
| Vector store | `chromadb` | ≥ 0.4 |
| Type checking | `mypy` | strict |
| Linting | `ruff` | latest |
| Property tests | `hypothesis` | latest |
| Sandbox | Docker + `python:3.11-slim` | — |
| IDE plugin | Continue.dev | latest |
| LLM serving | Ollama | latest |

---

## Structure des Fichiers

```
eclm-agent/
├── CLAUDE.md                          ← CE FICHIER
├── .claude/
│   ├── commands/                      ← Slash commands Claude Code
│   │   ├── add-primitive.md
│   │   ├── verify.md
│   │   ├── train.md
│   │   ├── benchmark.md
│   │   ├── scaffold.md
│   │   └── dpo-collect.md
│   └── memory/                        ← Mémoire persistante
│       ├── architecture.md
│       ├── decisions.md
│       └── progress.md
│
├── src/
│   ├── intent/                        ← C0: CamemBERT NLU
│   │   ├── __init__.py
│   │   ├── model.py                   # IntentExtractor class
│   │   ├── dataset.py                 # IntentDataset
│   │   ├── train.py                   # Script fine-tuning
│   │   └── inference.py               # IntentJSON output
│   │
│   ├── planner/                       ← C1: AST Planner
│   │   ├── __init__.py
│   │   ├── model.py                   # ASTPlanner class
│   │   ├── operations.py              # ASTOperation types (dataclasses)
│   │   ├── dataset.py
│   │   └── train.py
│   │
│   ├── eclm/                          ← C2: ECLM Core
│   │   ├── __init__.py
│   │   ├── model.py                   # ECLMCore class
│   │   ├── ast_ops.py                 # AST manipulation (Tree-sitter)
│   │   ├── beam_search.py             # Beam search sur AST ops
│   │   ├── dataset.py                 # Curriculum dataset
│   │   └── train.py                   # RL training loop
│   │
│   ├── verifier/                      ← C3: Verifier
│   │   ├── __init__.py
│   │   ├── pipeline.py                # VerificationPipeline class
│   │   ├── sandbox.py                 # Docker sandbox wrapper
│   │   ├── scorer.py                  # Score composite
│   │   └── test_generator/
│   │       ├── model.py               # TestGenerator (150M, ISOLÉ)
│   │       └── train.py
│   │
│   ├── library/                       ← Primitive Library
│   │   ├── __init__.py
│   │   ├── store.py                   # ChromaDB wrapper
│   │   ├── primitive.py               # Primitive dataclass
│   │   └── retrieval.py               # Semantic search
│   │
│   ├── improvement/                   ← C4: Self-improvement
│   │   ├── __init__.py
│   │   ├── dpo_collector.py           # Collecte paires DPO
│   │   ├── adversarial.py             # Self-play Generator vs Critic
│   │   └── finetune.py                # DPO fine-tune runner
│   │
│   ├── orchestrator/                  ← Chef d'orchestre
│   │   ├── __init__.py
│   │   ├── agent.py                   # ECLMAgent — point d'entrée principal
│   │   ├── rag.py                     # GraphRAG + ChromaDB
│   │   └── context.py                 # Gestion du contexte codebase
│   │
│   └── shared/
│       ├── types.py                   # Types partagés (IntentJSON, ASTOperation, etc.)
│       ├── config.py                  # Config centrale (paths, hyperparams)
│       └── logging.py                 # Logger structuré
│
├── data/
│   ├── primitives/                    ← ChromaDB primitive library
│   ├── training/
│   │   ├── intent/                    # intent_train.jsonl, intent_val.jsonl
│   │   ├── planner/                   # planner_train.jsonl
│   │   ├── eclm/                      # curriculum_*.jsonl (sorted by complexity)
│   │   └── test_generator/            # testgen_train.jsonl
│   ├── benchmarks/                    # benchmark_tasks.jsonl
│   └── dpo_pairs/                     # dpo_*.jsonl (auto-généré en prod)
│
├── tests/
│   ├── test_intent.py
│   ├── test_planner.py
│   ├── test_eclm.py
│   ├── test_verifier.py
│   └── test_library.py
│
├── docker/
│   ├── sandbox/
│   │   └── Dockerfile                 # python:3.11-slim, no network
│   └── compose.yml
│
├── scripts/
│   ├── bootstrap_dataset.py           # Génère dataset via Claude API (one-time)
│   ├── build_curriculum.py            # Trie The Stack par complexité
│   ├── run_benchmark.py               # Lance le benchmark privé
│   └── monthly_finetune.py            # Pipeline DPO mensuel
│
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Conventions de Code

### Python
- Python **3.11+** obligatoire
- Type hints **partout** — mypy strict doit passer sans erreur
- Dataclasses pour tous les types de données (`@dataclass(frozen=True)` pour les immutables)
- Pas de classes si une fonction suffit
- Docstrings Google style sur toutes les fonctions publiques
- `ruff` pour le formatage — jamais de `black` ni `isort` séparément

### Nommage
- Classes : `PascalCase`
- Fonctions/variables : `snake_case`
- Constantes : `UPPER_SNAKE_CASE`
- Fichiers : `snake_case.py`
- Types dans `shared/types.py` — jamais redéfinis ailleurs

### Imports
```python
# Ordre : stdlib → third-party → local
import ast
import json
from pathlib import Path
from dataclasses import dataclass

import torch
import chromadb
from transformers import AutoTokenizer

from src.shared.types import IntentJSON, ASTOperation
from src.shared.config import Config
```

### Tests
- `pytest` uniquement
- Chaque fonction publique a au moins un test
- Tests dans `tests/test_<module>.py`
- Fixtures dans `tests/conftest.py`
- Hypothesis pour les fonctions pures (property-based)

---

## Règles Critiques — JAMAIS Violer

1. **JAMAIS** utiliser next-token prediction comme objectif principal pour l'ECLM
   → L'objectif est toujours `maximize(execution_reward)`

2. **JAMAIS** générer du texte brut dans l'ECLM
   → Toujours des opérations AST ou des diffs AST

3. **JAMAIS** laisser le TestGenerator voir les candidats ECLM avant de générer les tests
   → Isolation totale pour éviter le cercle vicieux

4. **JAMAIS** exécuter du code généré en dehors du Docker sandbox
   → Toute exécution passe par `src/verifier/sandbox.py`

5. **JAMAIS** ajouter une primitive sans validation complète (tous les tests verts)
   → Passer par `/add-primitive` qui enforce la vérification

6. **JAMAIS** hardcoder des paths — tout dans `src/shared/config.py`

7. **JAMAIS** entraîner l'ECLM sur du code non vérifié
   → Le curriculum doit être pré-filtré par le Verifier

8. **JAMAIS** poser plus d'une question à l'utilisateur pour clarifier une intention
   → Si confiance < 0.75, formuler UNE question maximalement informative

---

## Interfaces Entre Composants

```python
# C0 → C1
intent: IntentJSON = intent_extractor.extract("modifie la fonction login...")
context: ASTContext = rag.get_context(intent)

# C1 → C2
plan: ASTOperationPlan = planner.plan(intent, context)

# C2 → C3 (pour chaque opération du plan)
candidates: list[ASTCandidate] = eclm.generate(operation, context, k=5)

# C3 → sortie
result: VerificationResult = verifier.verify(candidates, behavioral_tests)
if result.best_score < 0.8:
    # self-reflection loop
    candidates = eclm.generate(operation, context, error=result.error, k=5)
```

---

## Workflow de Développement

1. Implémenter d'abord `src/shared/types.py` — toutes les interfaces en premier
2. Implémenter chaque composant de façon **indépendante** avec ses tests unitaires
3. Intégrer via `src/orchestrator/agent.py` seulement quand les composants passent leurs tests
4. Valider sur le benchmark privé avant tout merge

**Ordre d'implémentation recommandé** :
1. `shared/types.py` + `shared/config.py`
2. `verifier/sandbox.py` (Docker) — le fondement de tout
3. `library/store.py` (ChromaDB)
4. `intent/` (CamemBERT)
5. `eclm/ast_ops.py` (manipulation AST)
6. `eclm/model.py` + `eclm/train.py`
7. `planner/`
8. `verifier/pipeline.py` (complet)
9. `orchestrator/agent.py`
10. `improvement/`

---

## Variables d'Environnement (.env)

```bash
# Optionnel — seulement pour le bootstrap one-time
ANTHROPIC_API_KEY=sk-...

# Paths (defaults dans config.py)
ECLM_DATA_DIR=./data
ECLM_MODELS_DIR=./models
ECLM_DOCKER_IMAGE=eclm-sandbox:latest

# Hyperparamètres (overridables)
ECLM_BEAM_WIDTH=5
ECLM_MAX_RETRIES=3
ECLM_CONFIDENCE_THRESHOLD=0.75
ECLM_MIN_VERIFICATION_SCORE=0.8
```

---

## Métriques de Succès

| Métrique | Objectif |
|----------|---------|
| Intent extraction accuracy | ≥ 95% sur benchmark |
| Verification pass rate (first try) | ≥ 70% |
| Verification pass rate (after retry) | ≥ 92% |
| Latence totale (RTX 4090) | < 15s pour une opération simple |
| Taille totale des modèles | < 4 GB (tous composants) |
| Score benchmark privé | Améliore chaque mois |
