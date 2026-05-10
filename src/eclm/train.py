"""Entraînement GRPO de l'ECLMCore — objectif maximize(execution_reward)."""
from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.eclm.dataset import load_curriculum, load_from_dpo_pairs
from src.shared.config import Config

logger = logging.getLogger(__name__)

MIN_EXAMPLES = 500
BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LORA_R = 16
LORA_ALPHA = 16
MAX_SEQ_LENGTH = 2048
NUM_GENERATIONS = 4  # GRPO: N completions par prompt pour le reward comparatif


# ── Reward function ────────────────────────────────────────────────────────────

def _check_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _run_ruff(code: str, tmpdir: Path) -> float:
    """Retourne 1.0 si ruff passe, 0.5 si warnings, 0.0 si erreurs bloquantes."""
    src = tmpdir / "solution.py"
    src.write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["ruff", "check", "--select=E,F", "--quiet", str(src)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return 1.0
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    return 0.5 if len(lines) <= 3 else 0.0


def _parse_pytest_score(stdout: str) -> float:
    """Extrait le ratio de tests passés depuis la sortie pytest."""
    for line in reversed(stdout.splitlines()):
        if "passed" in line:
            parts = line.split()
            try:
                passed_idx = parts.index("passed") - 1
                passed = int(parts[passed_idx])
                failed = 0
                if "failed" in parts:
                    failed = int(parts[parts.index("failed") - 1])
                total = passed + failed
                return passed / total if total > 0 else 0.5
            except (ValueError, IndexError):
                pass
        if "no tests ran" in line:
            return 0.5
    return 0.0


def compute_reward(code: str, tests: list[str], timeout: int = 10) -> float:
    """Évalue le code généré via exécution réelle.

    Signal d'entraînement :
    - 0.0  si erreur de syntaxe ou crash avant les tests
    - 0.3  si syntaxe ok + ruff ok mais aucun test disponible
    - 0.5  si tests partiellement passés
    - 1.0  si tous les tests passent

    Args:
        code: Code Python généré par le modèle.
        tests: Corps de fonctions pytest à exécuter.
        timeout: Timeout en secondes pour l'exécution.

    Returns:
        Score entre 0.0 et 1.0.
    """
    if not _check_syntax(code):
        return 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ruff_score = _run_ruff(code, tmp)

        if not tests:
            # Vérifier que le module s'importe sans erreur
            result = subprocess.run(
                [sys.executable, "-c",
                 f"import sys; sys.path.insert(0, '{tmpdir}'); import solution"],
                capture_output=True, text=True, timeout=timeout,
            )
            base = 0.3 if result.returncode == 0 else 0.1
            return min(1.0, base + 0.1 * ruff_score)

        # Écrire solution + tests
        (tmp / "solution.py").write_text(code, encoding="utf-8")
        test_bodies = "\n\n".join(tests)
        test_file = (
            f"import sys\nsys.path.insert(0, '{tmpdir}')\n"
            f"from solution import *\nimport pytest\n\n{test_bodies}\n"
        )
        (tmp / "test_run.py").write_text(test_file, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "test_run.py", "-q", "--tb=no"],
                capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return 0.1

        if result.returncode == 0:
            return 1.0
        score = _parse_pytest_score(result.stdout)
        # Partial reward: 0.5 * ratio_passé + bonus ruff
        return min(0.95, 0.5 * score + 0.05 * ruff_score)


# ── Prompt formatting ──────────────────────────────────────────────────────────

def _build_prompt(current_code: str, op_type: str, target: str, description: str) -> str:
    ctx = f"Current code:\n```python\n{current_code}\n```\n\n" if current_code.strip() else ""
    return (
        f"You are an expert Python developer. Apply the following transformation.\n\n"
        f"Operation: {op_type} on `{target}`\n"
        f"Description: {description}\n\n"
        f"{ctx}"
        f"Output ONLY the complete resulting Python code, no explanation:\n```python\n"
    )


def _extract_code(completion: str) -> str:
    """Extrait le bloc de code Python d'une complétion (retire les backticks)."""
    if "```python" in completion:
        return completion.split("```python")[-1].split("```")[0]
    if "```" in completion:
        return completion.split("```")[1].split("```")[0]
    return completion


# ── Main training entry point ──────────────────────────────────────────────────

def train(
    config: Config,
    max_complexity: int = 5,
    base_model: str = BASE_MODEL,
    include_dpo: bool = True,
) -> None:
    """Lance l'entraînement GRPO de l'ECLMCore par curriculum.

    Args:
        config: Configuration du projet.
        max_complexity: Complexité max du curriculum à cette étape.
        base_model: Identifiant HuggingFace du modèle de base.
        include_dpo: Ajouter les paires DPO au curriculum.

    Raises:
        ValueError: Si pas assez d'exemples de curriculum.
        FileNotFoundError: Si le curriculum est introuvable.
        ImportError: Si unsloth/trl non installés.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore[import-untyped]
        from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-untyped]
        from datasets import Dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Installez les dépendances GPU: pip install unsloth trl datasets"
        ) from exc

    # ── Charger le dataset ────────────────────────────────────────────────────
    examples = load_curriculum(config.data_dir, max_complexity=max_complexity)
    if include_dpo and config.dpo_pairs_dir.exists():
        dpo_examples = load_from_dpo_pairs(config.dpo_pairs_dir)
        examples.extend(dpo_examples)
        logger.info("+ %d exemples DPO ajoutés au curriculum", len(dpo_examples))

    if len(examples) < MIN_EXAMPLES:
        raise ValueError(
            f"Seulement {len(examples)} exemples — minimum {MIN_EXAMPLES} requis. "
            f"Lancez: python scripts/build_curriculum.py"
        )

    # Indexer les tests par position pour la reward function
    tests_index: list[list[str]] = []
    prompt_rows: list[dict[str, str]] = []

    for ex in examples:
        description = str(ex.operation.params.get("description", ex.operation.target))
        prompt = _build_prompt(
            ex.current_code, ex.operation.op_type, ex.operation.target, description
        )
        prompt_rows.append({"prompt": prompt})
        tests_index.append(ex.tests)

    dataset: Any = Dataset.from_list(prompt_rows)

    # ── Reward function (appelée par GRPOTrainer) ─────────────────────────────
    def reward_fn(completions: list[str], **kwargs: Any) -> list[float]:
        indices = kwargs.get("indices", list(range(len(completions))))
        rewards = []
        for i, completion in enumerate(completions):
            code = _extract_code(completion).strip()
            dataset_idx = indices[i] if i < len(indices) else 0
            tests = tests_index[dataset_idx % len(tests_index)]
            try:
                r = compute_reward(code, tests)
            except Exception as exc:
                logger.debug("Reward error: %s", exc)
                r = 0.0
            rewards.append(r)
        return rewards

    # ── Charger le modèle de base avec QLoRA ─────────────────────────────────
    logger.info("Chargement du modèle: %s", base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,   # bfloat16 auto-détecté sur RTX 4090
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Configuration GRPO ────────────────────────────────────────────────────
    output_dir = config.eclm_model_dir / "grpo_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=1024,
        max_completion_length=1024,
        temperature=0.9,
        beta=0.01,              # KL penalty — garder proche du modèle de base
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",       # pas de wandb par défaut
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info(
        "Entraînement GRPO — %d exemples, %d epochs, %d générations/prompt",
        len(examples), grpo_args.num_train_epochs, NUM_GENERATIONS,
    )
    trainer.train()

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    final_dir = config.eclm_model_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Modèle sauvegardé: %s", final_dir)

    _export_gguf(model, tokenizer, config)


def _export_gguf(model: Any, tokenizer: Any, config: Config) -> None:
    """Exporte le modèle fine-tuné en GGUF Q4_K_M pour Ollama."""
    gguf_dir = config.eclm_model_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    gguf_name = str(gguf_dir / "eclm-grpo")

    try:
        model.save_pretrained_gguf(gguf_name, tokenizer, quantization_method="q4_k_m")
        gguf_path = Path(gguf_name + ".gguf")
        logger.info("GGUF exporté: %s", gguf_path)
        _create_modelfile(config, gguf_path)
    except Exception as exc:
        logger.warning("Export GGUF échoué (%s) — modèle HF disponible dans final/", exc)


def _create_modelfile(config: Config, gguf_path: Path) -> None:
    """Crée un Modelfile Ollama pour le déploiement local."""
    modelfile = config.eclm_model_dir / "Modelfile"
    modelfile.write_text(
        f"FROM {gguf_path}\n\n"
        f'SYSTEM "You are ECLM, an expert Python code transformation assistant. '
        f'Apply code operations precisely and output only valid Python code."\n\n'
        f"PARAMETER temperature 0.2\n"
        f"PARAMETER top_p 0.9\n"
        f"PARAMETER num_ctx 4096\n",
        encoding="utf-8",
    )
    logger.info("Modelfile: %s", modelfile)
    logger.info("Déployer avec: ollama create eclm-grpo -f %s", modelfile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraîne l'ECLM avec GRPO (execution reward)")
    parser.add_argument("--max-complexity", type=int, default=5, help="Complexité max du curriculum")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Modèle HuggingFace de base")
    parser.add_argument("--no-dpo", action="store_true", help="Ne pas inclure les paires DPO")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train(
        Config(),
        max_complexity=args.max_complexity,
        base_model=args.base_model,
        include_dpo=not args.no_dpo,
    )
