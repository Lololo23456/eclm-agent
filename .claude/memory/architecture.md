# Architecture — Mémoire Persistante

## Identité du projet
Nom : ECLM Agent (Execution-Guided Compositional Language Model)
Objectif : Agent de codage privé, gratuit, local, auto-améliorant
Paradigme : PAS un LLM classique — apprentissage par exécution, pas par next-token

## Les 5 composants et leurs tailles

| Composant | Modèle base | Params | Rôle |
|-----------|------------|--------|------|
| C0 IntentExtractor | camembert-base | 110M | Français → IntentJSON |
| C1 ASTPlanner | Transformer seq2seq | 200M | IntentJSON → ASTOperationPlan |
| C2 ECLMCore | Custom RL model | 500M | ASTOperation → 5 candidats |
| C3 TestGenerator | Petit encoder-decoder | 150M | Code → Tests (ISOLÉ) |
| GraphRAG | Pas un modèle | — | Codebase indexing |
| PrimitiveLibrary | Pas un modèle | — | ChromaDB retrieval |
| C4 SelfImprovement | DPO on C1+C2 | — | Amélioration continue |

**Total paramètres : ~960M ≈ 1B**

## Flux de données principal
```
Français → IntentJSON → ASTOperationPlan → [ASTOperation × N] → [5 candidats] → VerificationResult → Code
```

## Interface centrale : IntentJSON
```python
@dataclass(frozen=True)
class IntentJSON:
    action: str           # MODIFY|CREATE|DELETE|REFACTOR|FIX|ADD|RENAME|EXPLAIN|CONVERT|TEST|OPTIMIZE|EXTRACT|MERGE|SPLIT
    target_type: str      # function|class|file|module|endpoint|test
    target_name: str
    target_file: str | None
    description: str      # normalisé en anglais
    constraints: list[str]
    confidence: float     # 0.0-1.0, seuil = 0.75
```

## Interface centrale : ASTOperation
```python
@dataclass(frozen=True)
class ASTOperation:
    op_type: str          # ADD_PARAM|MODIFY_BODY|REMOVE_PARAM|ADD_RETURN_TYPE|RENAME_SYMBOL|...
    target: str           # "module::class::function"
    params: dict[str, Any]
```

## Principe de vérification (score composite)
```
score = 0.4 × tests_behavior + 0.3 × tests_impl + 0.2 × property_tests + 0.1 × lint_score
seuil = 0.8 → retry si en dessous (max 3)
```

## Ordre d'implémentation validé
1. shared/types.py + shared/config.py
2. verifier/sandbox.py (Docker)
3. library/store.py (ChromaDB)
4. intent/ (CamemBERT)
5. eclm/ast_ops.py
6. eclm/model.py + train.py
7. planner/
8. verifier/pipeline.py (complet)
9. orchestrator/agent.py
10. improvement/
