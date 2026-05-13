# /debug

Analyse une erreur ou un fichier et propose un diagnostic + correction.

## Usage
```
/debug                         ← analyse les dernières erreurs pytest
/debug src/eclm/model.py       ← analyse un fichier spécifique
/debug "TypeError: ..."        ← analyse un message d'erreur direct
```

## Ce que tu dois faire

### 1. Collecter le contexte

Si `$ARGUMENTS` est un chemin de fichier :
```bash
python -m pytest tests/ -x --tb=long -q 2>&1 | tail -50
mypy src/ --ignore-missing-imports --no-error-summary 2>&1 | head -30
```

Si `$ARGUMENTS` est un message d'erreur → chercher la source :
```bash
grep -rn "mot_cle_erreur" src/ tests/ --include="*.py" | head -20
```

Si pas d'arguments → lancer les tests et capturer les échecs :
```bash
python -m pytest tests/ --tb=short -q 2>&1 | tail -40
```

### 2. Identifier la cause racine

Pour chaque erreur :
- **Type d'erreur** : ImportError / AttributeError / TypeError / AssertionError / etc.
- **Fichier et ligne** : lire le stack trace complet
- **Contexte** : lire les lignes autour de l'erreur dans le fichier
- **Cause probable** : interface changée ? Import manquant ? Type incorrect ?

### 3. Proposer une correction

Format de réponse :
```
Erreur identifiée : <type> dans <fichier>:<ligne>
Cause : <explication concise>

Fix :
```python
# Avant
<code problématique>

# Après  
<code corrigé>
```

Risques : <effets secondaires potentiels>
```

### 4. Vérifier après correction

```bash
python -m pytest tests/test_<module>.py -x -q --tb=short
```

## Règles
- Toujours lire le stack trace complet avant de conclure
- Ne jamais modifier plus de code que nécessaire pour le fix
- Signaler si l'erreur vient d'une dépendance externe (pas dans src/)
- Si l'erreur est dans un test : vérifier si c'est le test ou le code qui est faux
