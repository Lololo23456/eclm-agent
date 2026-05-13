# /plan

Génère un dossier de projet complet (plan.json) que le pipeline d'agents locaux exécutera.

## Usage
```
/plan "<brief en français>"
/plan                         ← brief dans le contexte de la conversation
```

## Ce que tu dois faire

### 1. Analyser le brief

Si `$ARGUMENTS` est fourni, c'est le brief. Sinon, utilise le brief de la conversation.

Explorer le codebase si nécessaire :
```bash
find . -name "*.py" -not -path "*/.*" | head -30
```

### 2. Produire le plan JSON

**Réfléchis profondément** avant d'écrire. Le plan doit être assez précis pour qu'un modèle 7B
implémente chaque tâche sans faire de choix d'architecture.

Le plan doit contenir pour chaque tâche :
- **signature exacte** (nom, paramètres typés, return type)
- **imports nécessaires** listés explicitement
- **contraintes** formulées comme des règles impératives
- **2-3 tests pytest** concrets (entrée → sortie attendue)
- **public_api** : ce que les autres tâches importent de celle-ci

Format JSON (respecter EXACTEMENT cette structure) :
```json
{
  "id": "<uuid4>",
  "name": "<slug-sans-espaces>",
  "brief": "<brief original>",
  "created_at": "<ISO8601>",
  "created_by": "claude-code",
  "stack": {
    "language": "python",
    "frameworks": ["fastapi"],
    "dependencies": ["fastapi", "pydantic"],
    "test_framework": "pytest",
    "python_version": "3.11"
  },
  "architecture": {
    "summary": "Description en 2-3 phrases de l'architecture",
    "files": {
      "src/models.py": "rôle exact du fichier",
      "src/api.py": "rôle exact du fichier",
      "tests/test_api.py": "tests d'intégration"
    },
    "key_decisions": [
      "SQLite plutôt que PostgreSQL car projet solo sans concurrence"
    ]
  },
  "tasks": [
    {
      "index": 0,
      "action": "CREATE",
      "target_type": "class",
      "target_name": "NomExact",
      "target_file": "src/models.py",
      "depends_on": [],
      "complexity": "low",
      "spec": {
        "description": "Description précise : ce que fait cette classe/fonction, invariants, cas limites",
        "signature": "@dataclass(frozen=True)\nclass NomExact:",
        "fields_or_params": ["id: int", "email: str", "created_at: datetime"],
        "return_type": null,
        "constraints": [
          "Lever ValueError si email ne contient pas '@'",
          "id doit être > 0"
        ],
        "imports": [
          "from dataclasses import dataclass",
          "from datetime import datetime",
          "import re"
        ],
        "example_usage": "u = NomExact(id=1, email='a@b.com', created_at=datetime.now())"
      },
      "tests": [
        "def test_nomexact_nominal():\n    u = NomExact(id=1, email='a@b.com', created_at=datetime.now())\n    assert u.email == 'a@b.com'",
        "def test_nomexact_invalid_email():\n    import pytest\n    with pytest.raises(ValueError):\n        NomExact(id=1, email='invalid', created_at=datetime.now())"
      ],
      "public_api": "NomExact(id: int, email: str, created_at: datetime)"
    }
  ]
}
```

### 3. Règles critiques pour les tâches

- **Ordre** : types/models → config → logique métier → API/CLI → tests
- **Atomicité** : une tâche = une seule fonction ou classe (jamais un fichier entier)
- **Tests concrets** : pas de `# TODO` — des assertions réelles avec des valeurs concrètes
- **Signatures complètes** : tous les types annotés, y compris return type
- **Imports exhaustifs** : lister TOUS les imports dont la tâche a besoin
- **Tests modules** : les tests unitaires sont une seule tâche par fichier de test (target_type="module")
- **depends_on** : indices entiers des tâches dont celle-ci dépend (jamais noms)

### 4. Sauvegarder et afficher

```python
import json, uuid
from datetime import datetime
from pathlib import Path

plan["id"] = str(uuid.uuid4())
plan["created_at"] = datetime.now().isoformat()

output_dir = Path("data/plans")
output_dir.mkdir(parents=True, exist_ok=True)
slug = plan["name"]
path = output_dir / f"{slug}.json"
path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
print(f"Plan sauvegardé : {path}")
```

Afficher un résumé :
```
Plan généré : <name>
─────────────────────────────────────────────────
  Stack    : python 3.11 · fastapi · pydantic
  Fichiers : src/models.py · src/api.py · tests/
  Tâches   : 8 (low:4 · medium:3 · high:1)
  Dépend   : models → api → tests
─────────────────────────────────────────────────
  Fichier  : data/plans/<name>.json

Pour exécuter :
  python main.py --from-plan data/plans/<name>.json
```

## Règles absolues
- Ne JAMAIS générer de tâche sans signature et tests
- Ne JAMAIS écrire de tests avec `pass` ou `# TODO`
- Les tests doivent tester le comportement, pas l'implémentation
- Si le brief est ambigu sur un choix architectural critique → poser UNE question avant de générer
- La qualité du plan détermine la qualité du code produit par les agents 7B
