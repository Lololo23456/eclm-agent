# /add-primitive

Ajoute une nouvelle primitive vérifiée dans la Primitive Library (ChromaDB).

## Usage
```
/add-primitive <domaine>    ← domaine : parsing | http | auth | data | io | math | str | ...
```

## Ce que tu dois faire

1. **Demander le code** à ajouter (si pas déjà dans le contexte)

2. **Vérification obligatoire** (dans l'ordre, tout doit passer) :
   ```bash
   # Syntax
   python -c "import ast; ast.parse(open('tmp_primitive.py').read())"

   # Types
   mypy --ignore-missing-imports tmp_primitive.py

   # Lint
   ruff check tmp_primitive.py

   # Tests fournis par l'utilisateur
   python -m pytest tmp_test_primitive.py -q
   ```
   Si un layer échoue → STOP, corriger avant d'ajouter.

3. **Créer la Primitive** et l'ajouter via `PrimitiveStore.add()` :
   ```python
   from src.library.primitive import Primitive
   from src.library.store import PrimitiveStore
   from src.shared.config import Config

   p = Primitive(
       code=code,
       tests=tests,
       domain="<domaine>",
       description="<description en anglais>",
       score=1.0,
   )
   store = PrimitiveStore(Config())
   store.add(p, embedding=intent_embedding)
   ```

4. **Confirmer** : afficher l'ID et le score de la primitive ajoutée.

## Règles critiques
- JAMAIS ajouter une primitive avec score < 0.8
- JAMAIS ajouter sans tests (au moins un test comportement)
- JAMAIS ajouter du code non vérifié par le verifier
- La description doit être en anglais (c'est le query language du retriever)
