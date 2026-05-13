"""Génère un plan.json via Ollama local — zéro token Claude requis.

Usage:
    python scripts/generate_plan.py "crée une API REST pour des utilisateurs"
    python scripts/generate_plan.py "crée une CLI todo list" --model qwen2.5-coder:7b
    python scripts/generate_plan.py "brief" --output data/plans/mon_plan.json
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_SYSTEM = """\
Tu es un architecte logiciel Python expert. Tu génères des plans de projet ultra-précis
au format JSON pour des agents de code automatisés (modèles 7B).

RÈGLES CRITIQUES :
- Chaque tâche = UNE SEULE fonction ou classe (jamais un fichier entier)
- Signature exacte avec tous les types annotés
- Tests concrets avec assertions réelles (jamais de pass ou TODO)
- Imports EXHAUSTIFS listés explicitement
- Ordre : models/types → store/db → logic → api/cli → tests
- depends_on : indices entiers uniquement
Retourne UNIQUEMENT du JSON valide, sans markdown, sans explication."""

_PROMPT_TEMPLATE = """\
Brief du projet : {brief}

Génère un plan JSON complet et précis. Format EXACT :

{{
  "name": "<slug-sans-espaces>",
  "brief": "{brief}",
  "stack": {{
    "language": "python",
    "frameworks": ["<framework>"],
    "dependencies": ["<dep1>", "<dep2>"],
    "test_framework": "pytest",
    "python_version": "3.11"
  }},
  "architecture": {{
    "summary": "<description 2-3 phrases>",
    "files": {{
      "src/models.py": "<rôle>",
      "src/app.py": "<rôle>",
      "tests/test_app.py": "<rôle>"
    }},
    "key_decisions": ["<décision architecturale>"]
  }},
  "tasks": [
    {{
      "index": 0,
      "action": "CREATE",
      "target_type": "class",
      "target_name": "NomExact",
      "target_file": "src/models.py",
      "depends_on": [],
      "complexity": "low",
      "spec": {{
        "description": "<description précise du comportement>",
        "signature": "<signature Python complète>",
        "fields_or_params": ["param1: type", "param2: type = default"],
        "return_type": "<type ou null>",
        "constraints": ["<règle impérative 1>", "<règle impérative 2>"],
        "imports": ["from x import y", "import z"],
        "example_usage": "<code Python d'exemple>"
      }},
      "tests": [
        "def test_nomexact_nominal():\\n    <import>\\n    <assert concret>",
        "def test_nomexact_invalid():\\n    import pytest\\n    with pytest.raises(<Error>):\\n        <code>"
      ],
      "public_api": "<signature exportée>"
    }}
  ]
}}

Génère toutes les tâches nécessaires pour un projet fonctionnel et testé. JSON uniquement :"""


def _call_ollama(brief: str, ollama_url: str, model: str) -> str:
    import urllib.request

    prompt = _PROMPT_TEMPLATE.format(brief=brief)
    payload = json.dumps({
        "model": model,
        "prompt": f"{_SYSTEM}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 4096},
    }).encode()

    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return str(json.loads(resp.read())["response"]).strip()


def _extract_json(raw: str) -> dict:  # type: ignore[type-arg]
    """Extrait le JSON d'une réponse LLM (avec ou sans markdown)."""
    # Retirer les balises markdown
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    # Trouver le premier { équilibré
    start = raw.find("{")
    if start == -1:
        raise ValueError("Aucun JSON trouvé dans la réponse")
    depth = 0
    end = start
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return dict(json.loads(raw[start:end]))  # type: ignore[arg-type]


def generate_plan(
    brief: str,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen2.5-coder:7b",
    output_path: Path | None = None,
) -> Path:
    """Génère un plan.json depuis un brief via Ollama local."""
    import requests  # type: ignore[import-untyped]

    # Vérifier Ollama
    try:
        requests.get(f"{ollama_url}/api/tags", timeout=5).raise_for_status()
    except Exception:
        raise ConnectionError(f"Ollama inaccessible sur {ollama_url} — lancez: ollama serve")

    print(f"Génération du plan via {model}...")
    raw = _call_ollama(brief, ollama_url, model)

    plan = _extract_json(raw)
    plan["id"] = str(uuid.uuid4())
    plan["created_at"] = datetime.now().isoformat()
    plan["created_by"] = f"ollama/{model}"
    plan.setdefault("brief", brief)

    # Valider structure minimale
    if "tasks" not in plan or not plan["tasks"]:
        raise ValueError("Plan invalide : aucune tâche générée")

    # Sauvegarder
    if output_path is None:
        name = plan.get("name", "projet").replace(" ", "_")
        output_path = Path("data/plans") / f"{name}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2))

    n = len(plan["tasks"])
    files = set(t.get("target_file", "") for t in plan["tasks"])
    print(f"\nPlan généré : {plan.get('name', '?')}")
    print("─" * 51)
    print(f"  Modèle   : {model}")
    print(f"  Tâches   : {n}")
    print(f"  Fichiers : {' · '.join(sorted(files))}")
    print("─" * 51)
    print(f"  Fichier  : {output_path}")
    print()
    print(f"Pour exécuter :")
    print(f"  python main.py --from-plan {output_path}")

    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Génère un plan.json via Ollama local")
    parser.add_argument("brief", help="Brief du projet en français")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    generate_plan(args.brief, args.ollama_url, args.model, args.output)


if __name__ == "__main__":
    main()
