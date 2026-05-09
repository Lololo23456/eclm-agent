"""Génère le dataset d'intention via l'API Claude (one-time bootstrap)."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

# Ce script nécessite ANTHROPIC_API_KEY dans l'environnement.
# Lance une seule fois pour générer ~2000 paires (commande FR → IntentJSON).

_COMMANDS_FR = [
    "modifie la fonction {func} pour ajouter la gestion des erreurs",
    "ajoute un paramètre timeout à la fonction {func}",
    "renomme la fonction {func} en {func}_v2",
    "crée une classe {cls} avec les méthodes __init__ et __repr__",
    "supprime la fonction {func} du module",
    "ajoute le type de retour à la fonction {func}",
    "refactore la fonction {func} pour la rendre plus lisible",
    "corrige le bug dans la fonction {func}",
    "ajoute des tests pour la fonction {func}",
    "optimise la fonction {func} pour réduire la complexité",
    "extrait la logique de {func} dans une fonction séparée",
    "ajoute un décorateur @staticmethod à la méthode {func}",
    "convertis la fonction {func} en méthode de classe {cls}",
    "ajoute une docstring à la fonction {func}",
    "ajoute un import de {module} au fichier",
]

_FUNCS = ["parse", "validate", "process", "handle", "compute", "fetch", "send", "load", "save", "format"]
_CLASSES = ["Parser", "Validator", "Processor", "Handler", "Client", "Server", "Manager"]
_MODULES = ["json", "pathlib", "logging", "datetime", "typing", "dataclasses"]

_SYSTEM = """Tu génères des paires (commande en français, IntentJSON) pour entraîner un modèle NLU.
Retourne UNIQUEMENT un JSON valide correspondant à cette commande.
Actions valides: MODIFY CREATE DELETE REFACTOR FIX ADD RENAME EXPLAIN CONVERT TEST OPTIMIZE EXTRACT MERGE SPLIT
Target types: function class file module endpoint test
Format:
{"action":"...","target_type":"...","target_name":"...","target_file":null,"description":"...","constraints":[],"confidence":0.95}"""


def _random_command() -> str:
    template = random.choice(_COMMANDS_FR)
    func = random.choice(_FUNCS)
    cls = random.choice(_CLASSES)
    module = random.choice(_MODULES)
    return template.format(func=func, cls=cls, module=module)


def generate(n: int = 2000, output_dir: Path = Path("data/training/intent")) -> None:
    """Génère n paires (commande → IntentJSON) via l'API Claude.

    Args:
        n: Nombre d'exemples à générer.
        output_dir: Répertoire de sortie.
    """
    import anthropic  # pip install anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY manquante")

    client = anthropic.Anthropic(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "intent_bootstrap.jsonl"

    print(f"Génération de {n} exemples → {out_path}")
    count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            command = _random_command()
            try:
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=256,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": command}],
                )
                raw = msg.content[0].text.strip()
                intent_data = json.loads(raw)
                record = {"command": command, "intent": intent_data, "validated": False}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"  {count}/{n} exemples générés")
                time.sleep(0.05)  # rate limit
            except Exception as exc:
                print(f"  Erreur ligne {i}: {exc}")

    print(f"Terminé: {count} exemples dans {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("data/training/intent"))
    args = parser.parse_args()
    generate(n=args.n, output_dir=args.output)
