# /project

Lance le mode projet — l'agent décompose un brief complet en tâches ordonnées et les exécute de A à Z.

## Usage
```
/project new "<brief en français>"    ← démarre un nouveau projet
/project status                        ← état du projet en cours
/project resume                        ← reprend là où on s'est arrêté
/project plan "<brief>"                ← affiche le plan sans exécuter
```

## Ce que tu dois faire

### /project new "<brief>"

1. **Appeler ProjectPlanner.plan(brief)** pour obtenir la liste ordonnée de tâches :
   - Chaque tâche = (action, target_type, target_name, target_file, description)
   - Estimer la complexité totale et le nombre de fichiers
   - Identifier les dépendances entre tâches (A avant B car B importe A)

2. **Afficher le plan complet** avant d'exécuter :
   ```
   Plan détecté — 8 tâches, ~3 fichiers :
   [1] CREATE  models.py         → class User
   [2] CREATE  models.py         → class Product
   [3] CREATE  database.py       → function connect_db
   [4] CREATE  auth.py           → function login
   [5] CREATE  auth.py           → function register
   [6] CREATE  routes.py         → function api_users
   [7] CREATE  main.py           → function main
   [8] CREATE  tests/test_api.py → function test_login
   Continuer ? [O/n]
   ```

3. **Exécuter chaque tâche séquentiellement** :
   - Pour chaque tâche : intent → C1 → C2 → C3 → FileWriter
   - Ré-indexer le RAG après chaque fichier créé (les tâches suivantes voient ce qui vient d'être fait)
   - Sauvegarder l'état dans `data/sessions/<session_id>.json` après chaque tâche
   - Afficher la progression : `[3/8] CREATE database.py:connect_db ✓ (score=0.92)`

4. **En cas d'échec** (score < 0.8 après max_retries) :
   - Sauvegarder l'état (session resumable)
   - Afficher l'erreur et proposer : `[r]etry / [s]kip / [e]dit manuellement / [a]rrêter`

5. **Résumé final** :
   ```
   Projet terminé — 7/8 tâches ✓
   Fichiers créés : models.py, database.py, auth.py, routes.py, main.py
   Fichiers échoués : tests/test_api.py (score=0.61)
   Paires DPO collectées : 2
   ```

### /project resume
1. Lister les sessions dans `data/sessions/` triées par date
2. Charger la session la plus récente (ou demander laquelle)
3. Afficher les tâches déjà faites (✓) et reprendre à la première non-faite
4. Ré-indexer les fichiers déjà créés dans le RAG avant de reprendre

### /project plan "<brief>"
- Même chose que `new` mais s'arrête après l'affichage du plan
- Utile pour valider avant d'exécuter

## Format session JSON (data/sessions/)
```json
{
  "id": "uuid",
  "brief": "crée une API REST...",
  "created_at": "ISO8601",
  "tasks": [
    {
      "index": 0,
      "action": "CREATE",
      "target_name": "User",
      "target_file": "models.py",
      "status": "done",
      "score": 0.95,
      "written_to": "models.py"
    }
  ]
}
```

## Règles
- TOUJOURS afficher le plan et demander confirmation avant d'exécuter
- TOUJOURS sauvegarder l'état après chaque tâche (session resumable)
- TOUJOURS ré-indexer le RAG après chaque fichier écrit (contexte cohérent)
- Ne JAMAIS écraser un fichier existant sans avertissement explicite
- Si une tâche échoue 3 fois → proposer skip, pas bloquer le projet entier
