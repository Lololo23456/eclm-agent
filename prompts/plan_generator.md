# Prompt universel — Générateur de plan ECLM

**Fonctionne avec** : Claude, GPT-4, Gemini, Mistral, Ollama local, n'importe quel LLM.

**Usage** : Copier le bloc ci-dessous, remplacer `{BRIEF}` par ton projet, envoyer.

---

## PROMPT À COPIER-COLLER

```
Tu es un architecte logiciel Python expert. Tu génères des plans de projet ultra-précis
au format JSON pour des agents de code automatisés (modèles 7B locaux).

PROJET : {BRIEF}

RÈGLES CRITIQUES (lire avant de générer) :
1. Chaque tâche = UNE SEULE fonction ou classe (jamais un fichier entier)
2. Signature exacte avec tous les types annotés (PEP 604, no Optional)
3. Tests concrets : assertions réelles avec des valeurs (jamais `pass` ou `# TODO`)
4. Imports EXHAUSTIFS : lister TOUS les imports dont la tâche a besoin
5. Ordre : models/types → store/db → logique → api/cli → tests
6. depends_on : indices entiers des tâches parentes (jamais des noms)
7. Pour les fichiers de test : target_type="module", tests=[] (le code contient les tests)

Retourne UNIQUEMENT le JSON suivant, sans markdown, sans explication :

{
  "name": "<slug-sans-espaces>",
  "brief": "<brief original>",
  "stack": {
    "language": "python",
    "frameworks": ["<framework si nécessaire, sinon []>"],
    "dependencies": ["<dep1>", "<dep2>"],
    "test_framework": "pytest",
    "python_version": "3.11"
  },
  "architecture": {
    "summary": "<description de l'architecture en 2-3 phrases>",
    "files": {
      "src/models.py": "<rôle exact>",
      "src/app.py": "<rôle exact>",
      "tests/test_app.py": "<rôle exact>"
    },
    "key_decisions": [
      "<décision architecturale importante et pourquoi>"
    ]
  },
  "tasks": [
    {
      "index": 0,
      "action": "CREATE",
      "target_type": "class",
      "target_name": "NomExactDeLaClasse",
      "target_file": "src/models.py",
      "depends_on": [],
      "complexity": "low",
      "spec": {
        "description": "<description précise : ce que fait cette classe, ses invariants, ses cas limites>",
        "signature": "@dataclass(frozen=True)\nclass NomExact:  # ou def foo(a: int) -> str:",
        "fields_or_params": ["champ1: type", "champ2: type = valeur_defaut"],
        "return_type": null,
        "constraints": [
          "Lever ValueError si <condition>",
          "<règle impérative>"
        ],
        "imports": [
          "from dataclasses import dataclass",
          "from datetime import datetime"
        ],
        "example_usage": "obj = NomExact(champ1=valeur)  # résultat attendu"
      },
      "tests": [
        "def test_nomexact_nominal():\n    from src.models import NomExact\n    obj = NomExact(champ1=valeur)\n    assert obj.champ1 == valeur",
        "def test_nomexact_invalid():\n    import pytest\n    from src.models import NomExact\n    with pytest.raises(ValueError):\n        NomExact(champ1=valeur_invalide)"
      ],
      "public_api": "NomExact(champ1: type, champ2: type = defaut)"
    }
  ]
}
```

---

## Exemples de briefs

```
"crée une API REST pour gérer des utilisateurs (CRUD, SQLite)"
"crée une CLI todo list avec click et persistance JSON"
"crée un scraper Python qui extrait les prix depuis une URL"
"crée un système de cache LRU thread-safe"
"crée un parseur de fichiers CSV avec validation des colonnes"
```

---

## Après avoir obtenu le JSON

1. Sauvegarder dans `data/plans/nom_projet.json`
2. Optionnel : affiner avec `/spec data/plans/nom_projet.json <index>`
3. Lancer le pipeline :
   ```bash
   python main.py --from-plan data/plans/nom_projet.json
   ```

---

## Via Ollama local (zéro token)

```bash
python scripts/generate_plan.py "ton brief en français"
# → génère data/plans/<nom>.json automatiquement
```

## Via API REST (serveur Linux)

```bash
curl -X POST http://ton-serveur:8765/v1/pipeline/plan \
  -H "Content-Type: application/json" \
  -d '{"brief": "crée une API REST simple"}'
```
