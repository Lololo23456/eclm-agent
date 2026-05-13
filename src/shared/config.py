"""Configuration centrale du projet ECLM."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def _detect_vram_gb() -> int:
    """Retourne la VRAM disponible en GB (0 si pas de GPU NVIDIA détecté)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).decode().strip().split("\n")[0]
        return int(out) // 1024
    except Exception:
        return 0


def _auto_model(vram_gb: int) -> str:
    """Choisit le meilleur modèle Ollama selon la VRAM disponible."""
    if vram_gb >= 20:
        return "qwen2.5-coder:32b"
    if vram_gb >= 10:
        return "qwen2.5-coder:14b"
    return "qwen2.5-coder:7b"


@dataclass
class Config:
    """Config centrale — toutes les constantes du projet."""

    # Paths
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("ECLM_DATA_DIR", "data")))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("ECLM_MODELS_DIR", "models")))

    # Hyperparamètres Verifier
    beam_width: int = int(os.getenv("ECLM_BEAM_WIDTH", "5"))
    max_retries: int = int(os.getenv("ECLM_MAX_RETRIES", "3"))
    confidence_threshold: float = float(os.getenv("ECLM_CONFIDENCE_THRESHOLD", "0.75"))
    min_verification_score: float = float(os.getenv("ECLM_MIN_VERIFICATION_SCORE", "0.8"))

    # Ollama — serveur principal
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

    # Model Router — fast (vérification, ops simples) vs strong (planning, génération complexe)
    # Si non défini en env, auto-détecte selon VRAM au démarrage
    fast_model: str = os.getenv("ECLM_FAST_MODEL", "qwen2.5-coder:7b")
    strong_model: str = field(default="")  # résolu dans __post_init__

    # Claude API — tier cloud optionnel
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_fast_model: str = "claude-haiku-4-5-20251001"
    claude_strong_model: str = "claude-sonnet-4-6"
    use_claude_api: bool = False  # résolu dans __post_init__

    # Performance
    prefer_local_sandbox: bool = os.getenv("ECLM_LOCAL_SANDBOX", "true").lower() == "true"
    adaptive_beam_width: bool = os.getenv("ECLM_ADAPTIVE_BEAM", "true").lower() == "true"
    max_parallel_tasks: int = int(os.getenv("ECLM_MAX_PARALLEL_TASKS", "3"))

    # Docker
    docker_image: str = os.getenv("ECLM_DOCKER_IMAGE", "eclm-sandbox:latest")
    docker_timeout_seconds: int = 30
    docker_memory_limit: str = "512m"

    # ChromaDB
    chroma_primitives_collection: str = "primitives"
    chroma_codebase_collection: str = "codebase"

    # DPO
    dpo_min_pairs_for_finetune: int = 100
    dpo_dedup_similarity_threshold: float = 0.9

    def __post_init__(self) -> None:
        # Résoudre les paths relatifs depuis root_dir
        if not self.data_dir.is_absolute():
            self.data_dir = self.root_dir / self.data_dir
        if not self.models_dir.is_absolute():
            self.models_dir = self.root_dir / self.models_dir

        # Auto-détecter le modèle strong selon la VRAM si non défini
        if not self.strong_model:
            env_strong = os.getenv("ECLM_STRONG_MODEL", "")
            if env_strong:
                self.strong_model = env_strong
            else:
                vram = _detect_vram_gb()
                self.strong_model = _auto_model(vram)

        # Activer Claude API si clé présente et non désactivée explicitement
        if self.anthropic_api_key and os.getenv("ECLM_USE_CLAUDE_API", "auto") != "false":
            self.use_claude_api = True

    @classmethod
    def for_testing(cls) -> Config:
        """Config pour les tests — utilise des dossiers temporaires."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        return cls(
            data_dir=tmp / "data",
            models_dir=tmp / "models",
            docker_timeout_seconds=10,
        )

    @property
    def primitives_dir(self) -> Path:
        return self.data_dir / "primitives"

    @property
    def dpo_pairs_dir(self) -> Path:
        return self.data_dir / "dpo_pairs"

    @property
    def benchmarks_dir(self) -> Path:
        return self.data_dir / "benchmarks"

    @property
    def intent_model_dir(self) -> Path:
        return self.models_dir / "intent"

    @property
    def eclm_model_dir(self) -> Path:
        return self.models_dir / "eclm"

    @property
    def planner_model_dir(self) -> Path:
        return self.models_dir / "planner"

    @property
    def testgen_model_dir(self) -> Path:
        return self.models_dir / "testgen"
