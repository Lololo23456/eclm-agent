# Décisions d'Architecture — Pourquoi ces choix

## CamemBERT pour l'interface française
**Décision** : utiliser camembert-base (pas un LLM généraliste)
**Pourquoi** : déjà pré-entraîné sur des millions de pages françaises → le français est résolu.
Fine-tuning sur 2000 exemples de commandes coding → domaine très étroit → très efficace.
**Alternative rejetée** : utiliser un LLM généraliste → trop gros, trop cher, overkill pour un classificateur.

## Opérations AST plutôt que génération de tokens
**Décision** : l'ECLM génère des diffs AST, pas du texte brut
**Pourquoi** : syntaxe invalide impossible par construction. Espace de recherche 100× plus petit.
Le beam search sur AST ops est déterministe et vérifiable.
**Alternative rejetée** : next-token prediction → hallucination, syntaxe invalide, espace de recherche énorme.

## Exécution comme signal d'entraînement
**Décision** : reward = résultat d'exécution réelle, pas imitation de code humain
**Pourquoi** : vérité non ambiguë, pas besoin de labels humains, impossible à falsifier.
Du code bugué sur GitHub ne contamine pas l'entraînement.
**Alternative rejetée** : supervised learning sur GitHub → signal bruité, code bugué = mauvais signal.

## TestGenerator isolé de l'ECLM
**Décision** : deux modèles séparés, aucune communication pendant génération
**Pourquoi** : éviter le cercle vicieux "code faux + test qui valide l'erreur".
Le TestGenerator génère ses tests AVANT de voir les candidats ECLM.
**Alternative rejetée** : même modèle pour code + tests → cercle vicieux garanti.

## ChromaDB local pour tout
**Décision** : ChromaDB pour Primitive Library ET codebase indexing
**Pourquoi** : 100% local, zéro donnée externe, requêtes < 50ms, simple à opérer.
**Alternative rejetée** : Pinecone, Weaviate → cloud = données qui quittent la machine.

## Docker sandbox pour toute exécution
**Décision** : aucun code généré n'est exécuté en dehors d'un container
**Pourquoi** : sécurité absolue. Le modèle peut générer n'importe quoi → ça ne touche pas la machine hôte.
**Alternative rejetée** : subprocess direct → risque de sécurité réel.

## DPO mensuel plutôt que continu
**Décision** : collecter les paires DPO en continu, re-fine-tune une fois par mois
**Pourquoi** : stability. Re-fine-tune trop fréquent → instabilité du modèle, régressions silencieuses.
Le benchmark privé avant chaque déploiement garantit zéro régression.
**Alternative rejetée** : online learning continu → instable, risque de catastrophic forgetting.

## Curriculum progressif pour l'ECLM
**Décision** : entraîner d'abord sur primitives simples, puis combiner vers complexe
**Pourquoi** : convergence 3-5× plus rapide qu'un entraînement aléatoire (démontré en recherche).
Évite que le modèle apprenne des patterns incorrects sur des exemples trop difficiles.
**Alternative rejetée** : entraînement aléatoire → lent, instable, plateaux fréquents.

## Bootstrap via Claude API (one-time)
**Décision** : utiliser Claude API une seule fois pour générer les premiers exemples
**Pourquoi** : ~10€ pour 500 exemples parfaits. Infiniment moins cher que labellisation manuelle.
Après ce bootstrap, le système est totalement autonome.
**Alternative rejetée** : scraper GitHub aléatoirement → signal bruité, code non vérifié.

## LocalSandbox par défaut (Docker optionnel)
**Décision** : `prefer_local_sandbox=True` par défaut — Docker activé via env var uniquement
**Pourquoi** : Docker coûte cher en batterie et en latence sur M3 Air (spinning up containers).
Le code généré par l'agent est de confiance (pas du code utilisateur arbitraire) → LocalSandbox suffisant.
Docker reste disponible pour les environnements CI/prod où l'isolation est critique.
**Alternative rejetée** : Docker obligatoire → trop lourd pour le dev quotidien sur portable.

## Adaptive beam width selon la complexité de l'opération
**Décision** : k adaptatif — k=1 déterministe, k=2 ops légères, k=3 défaut, k=5 ops lourdes
**Pourquoi** : générer 5 candidats pour un ADD_DOCSTRING (déterministe) est du gaspillage pur.
Sur M3 Air (mémoire partagée CPU/GPU), chaque appel Ollama coûte ~2-4s. Économiser 3 appels
sur des ops triviales = 6-12s de latence en moins.
**Alternative rejetée** : beam_width fixe k=5 → inutilement lent sur les ops simples.

## Exécution itérative de projet (pas one-shot)
**Décision** : créer chaque fichier séquentiellement avec re-indexation RAG entre chaque tâche
**Pourquoi** : quand l'agent crée `models.py` puis `auth.py`, `auth.py` doit voir `models.py` dans
son contexte RAG. One-shot (tout générer d'abord, écrire ensuite) perd ce contexte inter-tâches.
L'approche itérative garantit que chaque fichier a accès au code déjà écrit.
Bonus : reprise possible après crash (session persistée JSON après chaque tâche complétée).
**Alternative rejetée** : génération one-shot → contexte RAG incomplet, pas resumable.
