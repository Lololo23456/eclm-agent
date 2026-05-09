"""Configuration centrale du projet ECLM."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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

    # Ollama (C2 stand-in jusqu'à l'entraînement de l'ECLMCore)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

    # Performance — M3 Air optimisations
    prefer_local_sandbox: bool = os.getenv("ECLM_LOCAL_SANDBOX", "true").lower() == "true"
    adaptive_beam_width: bool = os.getenv("ECLM_ADAPTIVE_BEAM", "true").lower() == "true"

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
