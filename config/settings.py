"""Configuration and settings module for GANForge."""
from __future__ import annotations

import os
from pathlib import Path

# Base Project Root
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if available
ENV_PATH: Path = BASE_DIR / ".env"
try:
    from dotenv import load_dotenv  # type: ignore
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
    else:
        load_dotenv()
except ImportError:
    # Fallback to os.environ when python-dotenv is not yet installed
    pass

# Data Directories
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Outputs Directories
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
SAMPLES_DIR: Path = OUTPUTS_DIR / "samples"
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
REPORTS_DIR: Path = BASE_DIR / "reports"

# 50-Epoch Colab Artifact Directories
FINAL_RUN_DIR: Path = BASE_DIR / "final_run_50_epochs"
FINAL_RUN_METRICS: Path = FINAL_RUN_DIR / "metrics" / "training_history.csv"
FINAL_RUN_SAMPLES: Path = FINAL_RUN_DIR / "samples"

# Ensure all critical directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLES_DIR,
    CHECKPOINTS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Image & Preprocessing Constants
IMAGE_SIZE: tuple[int, int] = (64, 64)
CHANNELS: int = 3
TARGET_DATASET_SIZE: int = 300
TRAIN_VAL_SPLIT_RATIO: float = 0.8
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# DCGAN Architecture Hyperparameters
LATENT_DIM: int = int(os.getenv("DEFAULT_LATENT_DIM", "100"))
DEFAULT_BATCH_SIZE: int = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))
DEFAULT_EPOCHS: int = int(os.getenv("DEFAULT_EPOCHS", "50"))
LEARNING_RATE_GEN: float = 0.0002
LEARNING_RATE_DISC: float = 0.0002
BETA_1: float = 0.5
BETA_2: float = 0.999

# Evaluation & BLIP Settings
BLIP_MODEL_ID: str = "Salesforce/blip-image-captioning-base"
FID_IMAGE_SIZE: tuple[int, int] = (299, 299)  # InceptionV3 input size

# Gemini API Configuration for AI Auto-Report Generation (Unit U2)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Groq LLM Configuration (Legacy/Fallback)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_FALLBACK_MODEL: str = "llama-3.3-70b-versatile"

# Responsible AI & Dataset Metadata Defaults
DEFAULT_DATASET_LICENSE: str = "No separate dataset licence declared"
DEFAULT_PROVENANCE: str = "Algorithmic Procedural Rangoli Generator (data/rangoli_generator.py)"
