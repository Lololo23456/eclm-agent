# /verify

Lance la pipeline de vérification complète sur le code spécifié.

## Usage
```
/verify [fichier_ou_fonction]
/verify          ← vérifie les fichiers modifiés depuis le dernier commit
/verify src/eclm/ast_ops.py
/verify calculate_discount
```

## Ce que tu dois faire

1. **Identifier les cibles** :
   - Si `$ARGUMENTS` fourni → vérifier ce fichier/fonction spécifique
   - Sinon → `git diff --name-only HEAD` pour trouver les fichiers modifiés

2. **Layer 1 — Syntax** (< 10ms) :
   ```bash
   python -c "import ast; ast.parse(open('$FILE').read()); print('OK')"
   ```

3. **Layer 2 — Types** (< 5s) :
   ```bash
   mypy --ignore-missing-imports --no-error-summary $FILE
   ```
   ⚠️ mypy sans `--strict` — le code généré par Ollama n'a pas toujours des annotations complètes.
   Passer à `--strict` uniquement sur `src/shared/` et `src/verifier/`.

4. **Layer 3 — Lint** (< 2s) :
   ```bash
   ruff check $FILE
   ```

5. **Layer 4 — Tests unitaires** :
   ```bash
   # Sans Docker (mode dégradé) :
   python -m pytest tests/ -x -q --tb=short

   # Avec Docker sandbox (mode complet) :
   docker run --rm --network none \
     -v $(pwd):/workspace:ro \
     eclm-sandbox:latest \
     pytest tests/ -x -q --tb=short
   ```

6. **Layer 5 — Property tests** (si Hypothesis disponible) :
   ```bash
   python -m pytest tests/ -k "hypothesis" --hypothesis-seed=0 -q
   ```

7. **Rapport final** :
   ```
   Résultat verification src/eclm/ast_ops.py
   ─────────────────────────────────────────
   ✓ Syntax      OK
   ✓ Types       OK (mypy)
   ✓ Lint        0 violations (ruff)
   ✓ Tests       23/23 passés
   ✓ Property    12/12 passés
   ─────────────────────────────────────────
   Score : 1.00  ← prêt à merger
   ```
   - Si score < 0.8 → lister les échecs avec contexte exact et proposer corrections

## Règles
- Signaler mypy strict uniquement sur les modules `shared/` et `verifier/`
- Si Docker absent → indiquer clairement "mode dégradé (4/6 couches)"
- Ne JAMAIS marquer comme validé si syntax échoue
