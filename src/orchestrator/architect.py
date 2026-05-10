"""ArchitectAgent — planification de projet via le modèle fort (32B)."""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

from src.orchestrator.router import ModelRouter
from src.shared.config import Config

logger = logging.getLogger(__name__)

_ARCHITECT_PROMPT_PREFIX = """\
Tu es un architecte logiciel senior Python avec 15 ans d'expérience. \
Tu reçois un brief de projet et tu produis un plan d'architecture complet et exhaustif.

Brief : """

_ARCHITECT_PROMPT_SUFFIX = """

Ton plan doit :
1. Choisir la stack technique optimale pour ce brief (frameworks, libs Python)
2. Définir une structure de fichiers claire et maintenable
3. Décomposer le projet en tâches ATOMIQUES ordonnées (une tâche = une seule fonction ou classe)
4. Modéliser les dépendances entre tâches sous forme de DAG
5. Évaluer la complexité de chaque tâche

Règles strictes :
- Ordonne par dépendances : models/types → config → logique métier → API/CLI → tests
- depends_on : liste des INDEX des tâches dont cette tâche dépend (entiers)
- complexity : "low" (trivial/déterministe), "medium" (génération standard), "high" (algo complexe)
- target_file : chemin relatif cohérent avec folder_structure (ex: "src/models/user.py")
- action : CREATE | MODIFY | ADD | TEST
- target_type : function | class | module
- Sois EXHAUSTIF : inclure TOUTES les fonctions/classes nécessaires pour un projet fonctionnel
- FICHIERS DE TEST : une seule tâche par fichier test (target_type="module", target_name="NomDuModule"),
  qui génère le fichier de test COMPLET avec tous ses cas de test — jamais une tâche par fonction de test
- Si un choix architectural est critique et irréversible (type de DB, stratégie auth, \
protocole de communication), formule une question courte dans "review_gate" (sinon null)

Retourne UNIQUEMENT ce JSON valide, sans commentaires, sans markdown :
{
  "tech_stack": ["lib1", "lib2"],
  "folder_structure": ["src/", "tests/"],
  "review_gate": null,
  "tasks": [
    {
      "index": 0,
      "action": "CREATE",
      "target_type": "class",
      "target_name": "NomExact",
      "target_file": "src/models.py",
      "description": "Description précise et complète de cette classe/fonction",
      "depends_on": [],
      "complexity": "low"
    }
  ]
}"""


class ArchitectAgent:
    """Produit un plan d'architecture complet à partir d'un brief en français.

    Utilise le modèle fort (32B) pour garantir un DAG de tâches cohérent,
    avec choix de stack, structure de fichiers, et Review Gates sur les
    décisions critiques irréversibles.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._router = ModelRouter(config)

    def plan(self, brief: str) -> dict[str, object]:
        """Génère un plan d'architecture complet.

        Args:
            brief: Description du projet en français.

        Returns:
            Dict avec keys: tasks, tech_stack, folder_structure, review_gate.
            Retourne None sur les champs optionnels si Ollama indisponible.
        """
        raw = self._call_strong_model(brief)
        if raw:
            parsed = self._parse_response(raw)
            if parsed:
                logger.info(
                    "ArchitectAgent: %d tâches, stack=%s, review_gate=%s",
                    len(parsed.get("tasks", [])),
                    parsed.get("tech_stack", []),
                    parsed.get("review_gate"),
                )
                return parsed

        logger.warning("ArchitectAgent: parsing échoué — plan vide retourné")
        return {"tasks": [], "tech_stack": [], "folder_structure": [], "review_gate": None}

    def _call_strong_model(self, brief: str) -> str:
        model = self._router.for_planning()
        prompt = _ARCHITECT_PROMPT_PREFIX + brief + _ARCHITECT_PROMPT_SUFFIX
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "seed": 42},
        }).encode()

        req = urllib.request.Request(
            f"{self.config.ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                return str(data.get("response", "")).strip()
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("ArchitectAgent Ollama error: %s", exc)
            return ""

    def generate_run_guide(
        self,
        brief: str,
        tech_stack: list[str],
        files_created: list[str],
    ) -> str:
        """Génère les instructions pour lancer le projet (README.md).

        Args:
            brief: Brief original du projet.
            tech_stack: Librairies utilisées.
            files_created: Fichiers générés.

        Returns:
            Markdown avec instructions d'install, config et lancement.
        """
        model = self._router.for_planning()
        files_list = "\n".join(f"- {f}" for f in files_created)
        stack_str = ", ".join(tech_stack) if tech_stack else "Python stdlib"
        prompt = (
            "Tu es un expert Python. Un projet vient d'être généré automatiquement.\n\n"
            f"Brief : {brief}\n"
            f"Stack : {stack_str}\n"
            f"Fichiers créés :\n{files_list}\n\n"
            "Génère un README.md complet et pratique. Règles strictes :\n"
            "- Ne liste dans 'pip install' QUE les libs tierces (pas stdlib : os, json, sqlite3, etc.)\n"
            "- Les exemples d'utilisation doivent correspondre EXACTEMENT au type de projet "
            "(CLI → commandes shell, API → curl, lib → import Python)\n"
            "- Déduis le point d'entrée correct depuis les fichiers listés\n"
            "- Si le projet est un CLI, donne des exemples de commandes CLI, PAS de curl\n"
            "- Si le projet est une API REST, donne des exemples curl, PAS de commandes CLI\n\n"
            "Structure :\n"
            "1. Titre + description courte (2-3 lignes)\n"
            "2. Prérequis (version Python)\n"
            "3. Installation (pip install uniquement les libs tierces)\n"
            "4. Configuration (.env si nécessaire)\n"
            "5. Lancement (commandes exactes adaptées au type de projet)\n"
            "6. Tests : pytest <fichier_test>\n"
            "7. Exemples concrets adaptés au projet\n\n"
            "Réponds directement en markdown, sans introduction ni commentaire."
        )
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "seed": 1},
        }).encode()
        req = urllib.request.Request(
            f"{self.config.ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return str(data.get("response", "")).strip()
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Run guide generation failed: %s", exc)
            return f"# {brief}\n\n## Installation\n```bash\npip install {stack_str}\n```\n"

    def _parse_response(self, raw: str) -> dict[str, object] | None:
        # Cherche le premier objet JSON complet
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        raw_tasks = data.get("tasks", [])
        if not isinstance(raw_tasks, list) or not raw_tasks:
            return None

        tasks = []
        for t in raw_tasks:
            try:
                tasks.append({
                    "index": int(t["index"]),
                    "action": str(t.get("action", "CREATE")).upper(),
                    "target_type": str(t.get("target_type", "function")),
                    "target_name": str(t["target_name"]),
                    "target_file": str(t["target_file"]),
                    "description": str(t["description"]),
                    "depends_on": [int(d) for d in t.get("depends_on", [])],
                    "complexity": str(t.get("complexity", "medium")),
                })
            except (KeyError, ValueError, TypeError):
                continue

        if not tasks:
            return None

        return {
            "tasks": tasks,
            "tech_stack": [str(s) for s in data.get("tech_stack", [])],
            "folder_structure": [str(f) for f in data.get("folder_structure", [])],
            "review_gate": data.get("review_gate") or None,
        }
