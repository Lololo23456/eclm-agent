# Progression du Projet

## Statut global : 🟢 Phase 9 complète — Mode Projet opérationnel

**Date dernière mise à jour** : 2026-05-09
**Tests** : 86/86 ✅

---

## Phases

### Phase 0 — Setup ✅
- [x] Architecture définie dans CLAUDE.md
- [x] Slash commands créés
- [x] Mémoire initialisée
- [x] Structure de dossiers créée

### Phase 1 — Fondations ✅
- [x] `src/shared/types.py` — IntentJSON, ASTOperation, ASTOperationPlan, ASTCandidate, VerificationResult, Primitive, DPOPair
- [x] `src/shared/config.py` — Config centrale + Config.for_testing() + ollama_base_url/model
- [x] `docker/sandbox/Dockerfile` — python:3.11-slim, no network, user non-root

### Phase 2 — Composants de base ✅
- [x] `src/library/primitive.py` — Primitive dataclass + sérialisation ChromaDB
- [x] `src/library/store.py` — PrimitiveStore (add/search/delete/update_usage/list_by_domain)
- [x] `src/library/retrieval.py` — PrimitiveRetriever (embedding cosine, top-k)
- [x] `src/verifier/sandbox.py` — DockerSandbox (--network=none, timeout, SandboxResult)
- [x] `src/verifier/scorer.py` — SyntaxChecker (ast.parse), TypeChecker (mypy), LintScorer (ruff)
- [x] `src/verifier/pipeline.py` — VerificationPipeline multi-couches + _parse_pytest_score
- [x] `src/verifier/test_generator/model.py` — TestGenerator stub (isolé, lazy load)
- [x] Tests : test_library.py (13), test_verifier.py (16)

### Phase 3 — Intent Extractor (C0) ✅
- [x] `src/intent/model.py` — IntentExtractor via Ollama (→ CamemBERT quand 500 ex.)
- [x] `src/intent/dataset.py` — IntentDataLogger (log auto JSONL pour CamemBERT)
- [x] `src/intent/train.py` — fine-tuning CamemBERT (stub, déclenche à 500 exemples)
- [x] `scripts/bootstrap_dataset.py` — génère 2000 paires via Claude API (one-time)

### Phase 4 — ECLM Core (C2) ✅
- [x] `src/eclm/ast_ops.py` — 8 transformations AST déterministes sans LLM
  - ADD_PARAM, REMOVE_PARAM, ADD_RETURN_TYPE, RENAME_SYMBOL
  - ADD_IMPORT, DELETE_NODE, ADD_DECORATOR, ADD_DOCSTRING
- [x] `src/eclm/model.py` — ECLMCore (déterministe d'abord, Ollama si LLMRequired)
- [x] `src/eclm/beam_search.py` — filtre syntaxe + re-rank lint
- [x] `src/eclm/dataset.py` — CurriculumDataset (trié par complexité)
- [x] `src/eclm/train.py` — RL training loop stub (objectif: maximize execution_reward)
- [x] Tests : test_eclm.py (23)

### Phase 5 — Planner + Verifier (C1 + C3) ✅
- [x] `src/planner/model.py` — ASTPlanner (règles directes + Ollama pour cas complexes)
- [x] `src/planner/dataset.py` — PlannerExample + load_planner_dataset
- [x] `src/planner/train.py` — stub (seq2seq ~200M)
- [x] Tests : test_planner.py (8)

### Phase 6 — Orchestrateur ✅
- [x] `src/orchestrator/context.py` — tree-sitter AST-aware chunking + build_context + get_target_code
- [x] `src/orchestrator/rag.py` — CodebaseIndex ChromaDB incrémental
- [x] `src/orchestrator/agent.py` — ECLMAgent complet (C1→C2→beam→C3→DPO)
- [x] `main.py` — CLI REPL interactif (couleurs ANSI, /index /status /quitter)

### Phase 7 — Self-Improvement (C4) ✅
- [x] `src/improvement/dpo_collector.py` — RunRecord + DPOCollector (auto failure→success)
- [x] `src/improvement/finetune.py` — DPO runner (déclenche à 100 paires)
- [x] `src/improvement/adversarial.py` — self-play Generator vs Critic (stub)

### Phase 8 — Runtime & IDE 🟡 EN COURS
- [x] **Ollama installé** via brew (v0.23.2, service démarré au login)
- [x] **Modèle qwen2.5-coder:7b tiré** (4.7 GB, ID: dae161e27b0e)
- [x] **Pipeline end-to-end validé** — CREATE score=1.00 ✓
- [x] **Bugs corrigés** : strip markdown Ollama, mypy non-strict, op déterministe sur code vide
- [x] **LocalSandbox** implémenté (default M3 Air) + auto_sandbox() factory
- [x] **Adaptive beam width** (k=1 déterministe → k=5 ops lourdes)
- [ ] **Docker installé** (pour sandbox complet — verifier 6/6 couches)
- [ ] **Image Docker construite** : `docker build -t eclm-sandbox:latest docker/sandbox/`
- [ ] Config Continue.dev (`.continue/config.json`)
- [ ] Test end-to-end avec VS Code
- [ ] Tuning latence (objectif < 15s)

### Phase 9 — Mode Projet (A→Z) ✅ COMPLÈTE
Permet de décrire un projet complet en français → l'agent crée tous les fichiers itérativement.
- [x] `src/orchestrator/project.py` — TaskRecord + ProjectSession + ProjectAgent
  - Planification via Ollama (décompose brief → N tâches ordonnées avec dépendances)
  - Exécution itérative : create A → verify → re-index RAG → create B (voit A)
  - Persistance JSON après chaque tâche → resumable après crash
  - `load(session_id)` + `list_sessions()` pour reprendre
- [x] `data/sessions/` — répertoire créé automatiquement au premier `/project new`
- [x] CLI mode projet dans `main.py` (`/project new "..."` + `/project list` + `/project resume`)
- [x] Tests : test_project.py (11 tests)

---

## Décisions prises (→ voir decisions.md pour le pourquoi)
- ✅ Ollama + qwen2.5-coder:7b comme C2 stand-in (→ ECLMCore entraîné plus tard)
- ✅ `ast.unparse()` Python 3.11+ pour les ops déterministes (pas tree-sitter pour écriture)
- ✅ ASTPlanner hybride : règles directes + Ollama (pas seq2seq pour l'instant)
- ✅ Import circulaire résolu : orchestrator/__init__.py minimal
- ✅ LocalSandbox par défaut (Docker optionnel) — M3 Air performance
- ✅ Adaptive beam width — économise batterie sur ops simples
- ✅ Exécution itérative de projet avec re-indexation RAG inter-tâches

## Métriques actuelles
| Métrique | Valeur |
|----------|--------|
| Tests passing | 86/86 |
| Fichiers src/ | 32+ |
| Composants complets | 9/9 (code) |
| Runtime prêt | Ollama ✅ — Docker ❌ |
| Pipeline validé | CREATE score=1.00 ✓ |
| Mode projet | ✅ (TaskRecord, session JSON, iteratif) |
| Paires DPO | 0 (accumulation en prod) |
| Exemples intent | 0 (accumulation en prod) |

## Roadmap (révisée post-review #2 — 2026-05-09)

### Infrastructure — Serveur RTX 4090 🔲 PARALLÈLE (toi)
- [ ] Ubuntu 24.04 LTS sur PC Desktop (dual boot ou disque dédié)
- [ ] Ollama en service systemd + Qwen2.5-Coder 32B (Q4_K_M)
- [ ] OpenWebUI (interface web + API REST)
- [ ] Tailscale (accès distant sécurisé depuis MacBook)
- [ ] Docker pour sandbox complet (6/6 couches verifier)
- [ ] Monitoring : nvtop + Netdata

### Phase 9b — Model Router 🔲 NEXT ★★★★★ (3-4 jours)
Levier le plus fort sur la qualité actuelle du système.
- [ ] `src/orchestrator/router.py` — ModelRouter (fast vs strong)
  - Fast path : 7B ou 14B — vérification, ops simples, lint
  - Strong path : 32B — planning, architecture, génération complexe
- [ ] Intégrer dans ECLMAgent et ProjectAgent
- [ ] Config : `ECLM_FAST_MODEL`, `ECLM_STRONG_MODEL`, `ECLM_STRONG_URL`
- [ ] Tests : test_router.py

### Phase 10 — Architect Agent (Multi-agent) 🔲 ★★★★★ (1-2 semaines)
Remplace le planning Ollama 7B par un vrai agent architecte 32B.
- [ ] `src/orchestrator/architect.py` — ArchitectAgent
  - Brief → architecture complète (stack, structure dossiers, choix tech)
  - DAG de tâches avec dépendances et estimations de complexité
  - Review Gates : demande confirmation avant décisions irréversibles
- [ ] `src/orchestrator/critic.py` — CriticAgent
  - Analyse les échecs, propose des corrections ciblées
  - Injecte le feedback dans la boucle de retry
- [ ] Intégrer dans ProjectAgent (remplace `_plan_via_ollama`)
- [ ] Tests : test_architect.py, test_critic.py

### Phase 11 — Dependency Graph + Multi-fichiers 🔲 ★★★★ (4-5 jours)
- [ ] `src/orchestrator/dependency_graph.py` — graphe imports entre modules
- [ ] UPDATE_CALL_SITES effectif (renommer une fonction met à jour les appelants)
- [ ] GraphRAG cross-file : summaries de sessions + relations entre classes
- [ ] RAG comprend les relations entre fichiers différents

### Phase 12 — Self-Improvement Actif 🔲 ★★★★ (ongoing)
- [ ] Collecte trajectoires par projet (succès + échecs) → DPO automatique
- [ ] Fine-tune tous les 5-10 projets sur données réelles
- [ ] `scripts/run_benchmark.py` — 20-30 tâches réelles croissantes
- [ ] Benchmark mensuel : pass rate ≥ 70% first-try, ≥ 92% after retry

### Phase 13 — Intégration IDE 🔲 ★★★ (1 semaine)
- [ ] `src/server/api.py` — HTTP server FastAPI (pour Continue.dev / custom)
- [ ] `.continue/config.json` — config Continue.dev
- [ ] Contexte IDE (fichier ouvert, sélection courante)

## Prochaine action immédiate
👉 Choix : (A) commandes serveur Ubuntu/4090 | (B) code ModelRouter | (C) plan semaine par semaine
