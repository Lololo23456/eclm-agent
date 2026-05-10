"""DPO fine-tuning mensuel de l'ECLMCore via QLoRA + unsloth."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from src.shared.config import Config

logger = logging.getLogger(__name__)

MIN_PAIRS_FOR_FINETUNE = 50


def load_dpo_pairs(config: Config) -> list[dict[str, str]]:
    """Charge toutes les paires DPO disponibles.

    Returns:
        Liste de dicts {prompt, chosen, rejected, chosen_score, rejected_score}.
    """
    pairs: list[dict[str, str]] = []
    for path in sorted(config.dpo_pairs_dir.glob("dpo_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return pairs


def _pairs_to_chatml(pairs: list[dict[str, str]]) -> list[dict[str, object]]:
    """Convertit les paires DPO au format ChatML attendu par trl.DPOTrainer."""
    dataset = []
    for p in pairs:
        prompt = p.get("prompt", "")
        chosen = p.get("chosen", "")
        rejected = p.get("rejected", "")
        if not (prompt and chosen and rejected):
            continue
        # Format DPO : prompt → (chosen, rejected) comme messages ChatML
        dataset.append({
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    return dataset


def _check_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            logger.info("GPU détecté: %s", result.stdout.strip())
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def run_finetune(config: Config, base_model: str = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit") -> None:
    """Lance le DPO fine-tuning QLoRA sur le GPU local.

    Nécessite : pip install unsloth trl datasets
    Optimisé pour RTX 4090 (24 GB VRAM).

    Args:
        config: Configuration du projet.
        base_model: Modèle HuggingFace à fine-tuner (format unsloth bnb-4bit).

    Raises:
        ValueError: Si pas assez de paires DPO.
        RuntimeError: Si GPU absent ou dépendances manquantes.
    """
    pairs = load_dpo_pairs(config)
    n = len(pairs)

    if n < MIN_PAIRS_FOR_FINETUNE:
        raise ValueError(
            f"Seulement {n} paires DPO — minimum {MIN_PAIRS_FOR_FINETUNE} requis. "
            f"Continuez à utiliser l'agent pour accumuler du signal."
        )

    if not _check_gpu():
        raise RuntimeError(
            "Aucun GPU NVIDIA détecté. Le fine-tuning nécessite le serveur RTX 4090."
        )

    try:
        from unsloth import FastLanguageModel
        import torch
        from trl import DPOTrainer, DPOConfig
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            f"Dépendances manquantes: {exc}\n"
            "Sur le serveur: pip install unsloth trl datasets"
        ) from exc

    output_dir = config.models_dir / "eclm" / "dpo_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement du modèle %s ...", base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,  # auto-détection (bfloat16 sur RTX 4090)
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset_list = _pairs_to_chatml(pairs)
    logger.info("%d paires DPO converties en format ChatML", len(dataset_list))
    dataset = Dataset.from_list(dataset_list)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        beta=0.1,                   # DPO temperature — plus bas = plus conservateur
        max_length=1024,
        max_prompt_length=512,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,             # unsloth gère la référence avec PEFT
        tokenizer=tokenizer,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    logger.info("Démarrage DPO fine-tuning sur %d exemples...", len(split["train"]))
    trainer.train()

    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info("Adapteur LoRA sauvegardé dans %s", output_dir / "final")

    # Exporter en GGUF pour Ollama
    _export_to_gguf(output_dir / "final", config.models_dir / "eclm" / "eclm_dpo.gguf")


def _export_to_gguf(lora_dir: Path, gguf_path: Path) -> None:
    """Fusionne LoRA + base et exporte en GGUF pour Ollama.

    Nécessite llama.cpp installé sur le serveur.
    """
    try:
        import torch
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(str(lora_dir))
        merged_dir = lora_dir.parent / "merged"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        logger.info("Modèle fusionné dans %s", merged_dir)
    except Exception as exc:
        logger.warning("Export GGUF skipped (unsloth merge failed): %s", exc)
        return

    # llama.cpp convert
    convert_script = Path(os.getenv("LLAMA_CPP_DIR", "/opt/llama.cpp")) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        logger.warning(
            "llama.cpp introuvable à %s — exporte manuellement avec :\n"
            "  python convert_hf_to_gguf.py %s --outfile %s --outtype q4_k_m",
            convert_script, lora_dir.parent / "merged", gguf_path,
        )
        return

    cmd = [
        sys.executable, str(convert_script),
        str(lora_dir.parent / "merged"),
        "--outfile", str(gguf_path),
        "--outtype", "q4_k_m",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("GGUF exporté : %s", gguf_path)
        logger.info(
            "Pour l'utiliser dans Ollama :\n"
            "  ollama create eclm-dpo -f <Modelfile avec FROM %s>",
            gguf_path,
        )
    else:
        logger.error("Conversion GGUF échouée: %s", result.stderr[:500])


if __name__ == "__main__":
    run_finetune(Config())
