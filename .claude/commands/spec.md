# /spec

Enrichit ou corrige la spec d'une tâche dans un plan existant.

## Usage
```
/spec data/plans/mon_projet.json 3       ← enrichit la tâche index 3
/spec data/plans/mon_projet.json User    ← enrichit la tâche nommée "User"
/spec data/plans/mon_projet.json         ← liste toutes les tâches et leurs lacunes
```

## Ce que tu dois faire

### 1. Charger le plan
```python
import json
from pathlib import Path

args = "$ARGUMENTS".split()
plan_path = Path(args[0])
plan = json.loads(plan_path.read_text())
```

### 2. Identifier la tâche cible
- Si 2ème argument est un entier → `plan["tasks"][index]`
- Si 2ème argument est un string → chercher par `target_name`
- Si pas de 2ème argument → afficher toutes les tâches avec score de complétude

**Score de complétude par tâche** :
```
✓ signature présente et typée
✓ imports listés
✓ contraintes formulées
✓ tests concrets (pas de pass/TODO)
✓ public_api défini
```

### 3. Enrichir la spec

Pour la tâche cible, enrichir :
- `spec.signature` → signature complète avec tous les types
- `spec.constraints` → règles impératives précises
- `spec.imports` → liste exhaustive
- `tests` → 3 tests concrets minimum (nominal + edge + erreur)
- `public_api` → ce que les autres modules importent

### 4. Sauvegarder les modifications
```python
plan["tasks"][task_index] = enriched_task
plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
print(f"Tâche {task_index} enrichie dans {plan_path}")
```

### 5. Afficher le diff des changements
Montrer avant/après pour chaque champ modifié.

## Règles
- Ne jamais changer `index`, `depends_on`, `action`, `target_file` sans confirmation
- Les tests enrichis doivent rester cohérents avec la signature
- Signaler si une tâche a des dépendances manquantes (dépend de X mais X n'est pas dans le plan)
