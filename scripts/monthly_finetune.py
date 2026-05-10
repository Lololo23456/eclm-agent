"""Pipeline de fine-tuning mensuel — à lancer sur le serveur RTX 4090.

Usage:
    python scripts/monthly_finetune.py [--dry-run] [--min-pairs 50]

Étapes :
1. Compte les paires DPO disponibles
2. Lance DPO fine-tuning QLoRA (unsloth + trl)
3. Exporte le modèle fusionné en GGUF
4. Affiche les instructions pour l'importer dans Ollama
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.improvement.dpo_collector import DPOCollector
from src.improvement.finetune import MIN_PAIRS_FOR_FINETUNE, load_dpo_pairs, run_finetune
from src.shared.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning mensuel ECLM")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les stats sans lancer le fine-tuning")
    parser.add_argument("--min-pairs", type=int, default=MIN_PAIRS_FOR_FINETUNE)
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
                        help="Modèle HuggingFace de base à fine-tuner")
    args = parser.parse_args()

    config = Config()
    pairs = load_dpo_pairs(config)
    n = len(pairs)

    print(f"  Paires DPO disponibles : {n}")
    print(f"  Minimum requis         : {args.min_pairs}")

    if n == 0:
        print("\n  Aucune paire DPO trouvée.")
        print("  Utilisez l'agent pour générer des projets — les paires se collectent automatiquement.")
        sys.exit(0)

    # Distribution des scores
    chosen_scores = [float(p.get("chosen_score", 0)) for p in pairs]
    rejected_scores = [float(p.get("rejected_score", 0)) for p in pairs]
    sources = {}
    for p in pairs:
        src = str(p.get("source", "unknown"))
        sources[src] = sources.get(src, 0) + 1

    avg_chosen = sum(chosen_scores) / n
    avg_rejected = sum(rejected_scores) / n
    print(f"  Score moyen chosen     : {avg_chosen:.3f}")
    print(f"  Score moyen rejected   : {avg_rejected:.3f}")
    print(f"  Sources                : {sources}")

    if n < args.min_pairs:
        print(f"\n  Pas assez de paires ({n} < {args.min_pairs}). Fine-tuning reporté.")
        sys.exit(0)

    if args.dry_run:
        print("\n  [dry-run] Fine-tuning non lancé.")
        print(f"  Pour lancer : python {__file__} --base-model {args.base_model}")
        sys.exit(0)

    print(f"\n  Lancement du fine-tuning DPO sur {n} paires...")
    print(f"  Modèle de base : {args.base_model}")
    print(f"  Output         : {config.models_dir / 'eclm' / 'dpo_lora'}")
    print()

    try:
        run_finetune(config, base_model=args.base_model)
        gguf_path = config.models_dir / "eclm" / "eclm_dpo.gguf"
        print("\n" + "="*52)
        print("  Fine-tuning terminé avec succès !")
        print()
        print("  Pour utiliser le modèle dans Ollama :")
        print(f"  1. Créer un Modelfile :")
        print(f"       FROM {gguf_path}")
        print(f"       PARAMETER temperature 0.2")
        print(f"  2. Importer :")
        print(f"       ollama create eclm-dpo -f Modelfile")
        print(f"  3. Configurer dans .env :")
        print(f"       ECLM_FAST_MODEL=eclm-dpo")
        print(f"       ECLM_STRONG_MODEL=qwen2.5-coder:32b")
        print("="*52)
    except (ValueError, RuntimeError) as exc:
        print(f"\n  Erreur: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
