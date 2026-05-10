"""Génère le dataset d'intention pour CamemBERT (one-time bootstrap).

Deux backends disponibles :
  --backend claude  → API Anthropic (claude-haiku, ~10€ pour 2000 exemples)
  --backend ollama  → modèle local Ollama (gratuit, ~2-3h sur CPU/GPU)

Usage:
    # Avec Ollama (gratuit) :
    python scripts/bootstrap_dataset.py --backend ollama --n 2000

    # Avec Claude API (rapide, ~10€) :
    ANTHROPIC_API_KEY=sk-... python scripts/bootstrap_dataset.py --backend claude --n 2000
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    "crée une fonction {func} qui valide les entrées utilisateur",
    "merge les fonctions {func} et {func}_helper en une seule",
    "explique ce que fait la fonction {func}",
    "convertis la fonction {func} en version asynchrone",
    "ajoute la gestion des exceptions à la classe {cls}",
    "crée un endpoint REST pour {func}",
    "ajoute une propriété calculée à la classe {cls}",
    "optimise la boucle dans {func} pour les grandes listes",
    "divise le module {module} en deux fichiers séparés",
    "fixe le bug de type dans la fonction {func}",
]

_FUNCS = [
    "parse", "validate", "process", "handle", "compute", "fetch",
    "send", "load", "save", "format", "convert", "transform",
    "filter", "sort", "merge", "split", "connect", "disconnect",
    "authenticate", "authorize", "register", "login", "logout",
]
_CLASSES = [
    "Parser", "Validator", "Processor", "Handler", "Client", "Server",
    "Manager", "Controller", "Repository", "Service", "Factory",
]
_MODULES = [
    "json", "pathlib", "logging", "datetime", "typing", "dataclasses",
    "asyncio", "requests", "sqlite3", "csv", "hashlib",
]

_SYSTEM = (
    "Tu génères des paires (commande en français, IntentJSON) pour entraîner un modèle NLU.\n"
    "Retourne UNIQUEMENT un objet JSON valide sans explication.\n"
    "Actions valides: MODIFY CREATE DELETE REFACTOR FIX ADD RENAME EXPLAIN CONVERT TEST OPTIMIZE EXTRACT MERGE SPLIT\n"
    "Target types: function class file module endpoint test\n"
    'Format exact: {"action":"...","target_type":"...","target_name":"...","target_file":null,'
    '"description":"...","constraints":[],"confidence":0.95}'
)


def _random_command() -> str:
    template = random.choice(_COMMANDS_FR)
    func = random.choice(_FUNCS)
    cls = random.choice(_CLASSES)
    module = random.choice(_MODULES)
    return template.format(func=func, cls=cls, module=module)


# ── Backend Claude API ────────────────────────────────────────────────────────

def _generate_one_claude(client: object, command: str) -> dict[str, object] | None:
    import anthropic  # type: ignore[import-untyped]
    assert isinstance(client, anthropic.Anthropic)
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_SYSTEM,
            messages=[{"role": "user", "content": command}],
        )
        raw = str(msg.content[0].text).strip()
        intent_data: dict[str, object] = json.loads(raw)
        return {"command": command, "intent": intent_data, "validated": False}
    except Exception as exc:
        print(f"  Erreur Claude: {exc}")
        return None


def generate_claude(n: int, output_path: Path) -> int:
    try:
        import anthropic  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY manquante.\n"
            "Utilisez --backend ollama pour générer sans clé API."
        )

    client = anthropic.Anthropic(api_key=api_key)
    count = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for i in range(n):
            command = _random_command()
            record = _generate_one_claude(client, command)
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            if count % 100 == 0 and count > 0:
                print(f"  {count}/{n} exemples (claude)")
            time.sleep(0.05)
    return count


# ── Backend Ollama (gratuit) ──────────────────────────────────────────────────

def _generate_one_ollama(ollama_url: str, model: str, command: str) -> dict[str, object] | None:
    import requests  # type: ignore[import-untyped]

    prompt = (
        f"{_SYSTEM}\n\n"
        f"Commande française: {command}\n\n"
        f"Réponds UNIQUEMENT avec le JSON, rien d'autre:"
    )
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3, "num_predict": 200}},
            timeout=30,
        )
        resp.raise_for_status()
        raw = str(resp.json().get("response", "")).strip()
        # Extraire le JSON si entouré de backticks
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        intent_data: dict[str, object] = json.loads(raw)
        return {"command": command, "intent": intent_data, "validated": False}
    except Exception as exc:
        return None


def generate_ollama(
    n: int,
    output_path: Path,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen2.5-coder:7b",
) -> int:
    import requests  # type: ignore[import-untyped]

    # Vérifier qu'Ollama est accessible
    try:
        requests.get(f"{ollama_url}/api/tags", timeout=5).raise_for_status()
    except Exception:
        raise ConnectionError(
            f"Ollama inaccessible sur {ollama_url}.\n"
            f"Lancez: ollama serve"
        )

    count = 0
    errors = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for i in range(n):
            command = _random_command()
            record = _generate_one_ollama(ollama_url, model, command)
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                errors = 0
            else:
                errors += 1
                if errors >= 10:
                    print(f"  10 erreurs consécutives — vérifiez qu'Ollama répond correctement")
                    errors = 0
            if (i + 1) % 100 == 0:
                print(f"  {count}/{n} exemples valides générés (ollama)")
    return count


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap du dataset d'intentions ECLM")
    parser.add_argument("--n", type=int, default=2000, help="Nombre d'exemples à générer")
    parser.add_argument("--output", type=Path, default=Path("data/training/intent"))
    parser.add_argument(
        "--backend", choices=["ollama", "claude"], default="ollama",
        help="Backend LLM (ollama=gratuit|claude=rapide ~10€)",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5-coder:7b",
                        help="Modèle Ollama à utiliser (backend=ollama)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "intent_bootstrap.jsonl"

    # Reprendre là où on s'est arrêté
    existing = sum(1 for _ in open(output_path, encoding="utf-8")) if output_path.exists() else 0
    remaining = args.n - existing
    if remaining <= 0:
        print(f"Déjà {existing} exemples — dataset complet.")
        return

    print(f"Backend: {args.backend}")
    print(f"Objectif: {args.n} exemples ({existing} déjà générés, {remaining} restants)")
    print(f"Sortie: {output_path}")
    print()

    if args.backend == "claude":
        count = generate_claude(remaining, output_path)
    else:
        print(f"Modèle Ollama: {args.model} sur {args.ollama_url}")
        print("Astuce: le backend ollama est ~10× plus lent que Claude mais 100% gratuit.")
        print()
        count = generate_ollama(remaining, output_path, args.ollama_url, args.model)

    total = existing + count
    print(f"\nTerminé: {count} nouveaux exemples → {total} total dans {output_path}")

    if total >= 500:
        print(f"\n✓ Seuil atteint — lancer l'entraînement:")
        print(f"  python -m src.intent.train")
    else:
        print(f"\n⚠ {500 - total} exemples manquants pour déclencher l'entraînement.")
        print(f"  Relancez: python scripts/bootstrap_dataset.py --n {500 - total} --backend ollama")


if __name__ == "__main__":
    main()
