"""Callbacks, metric logging, and sample grid generation for DCGAN training.

Handles:
- Accurate per-epoch loss tracking and duration logging
- Export of training logs in CSV and JSON formats
- Fixed-latent noise sample generation across milestone epochs (1, 10, 25, 50)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config.settings import LATENT_DIM, METRICS_DIR, RANDOM_SEED, SAMPLES_DIR
from data.dataset_loader import denormalize_images

logger = logging.getLogger("GANForge.Callbacks")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TrainingLogger:
    """Records, aggregates, and exports epoch-wise training metrics in CSV and JSON."""

    def __init__(self, metrics_dir: Path | str = METRICS_DIR) -> None:
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []

    def log_epoch(
        self,
        epoch: int,
        generator_loss: float,
        discriminator_loss: float,
        disc_real_loss: float,
        disc_fake_loss: float,
        epoch_duration_sec: float,
        batches_processed: int,
        images_processed: int,
        fid_score: Optional[float] = None,
    ) -> dict[str, Any]:
        """Logs the actual metrics computed during a single training epoch.

        Args:
            epoch: 1-based epoch index.
            generator_loss: Mean Generator loss for the epoch.
            discriminator_loss: Mean Discriminator loss for the epoch.
            disc_real_loss: Mean Discriminator real-classification loss.
            disc_fake_loss: Mean Discriminator fake-classification loss.
            epoch_duration_sec: Duration of epoch training in seconds.
            batches_processed: Number of batches in the epoch.
            images_processed: Number of real images passed through the epoch.
            fid_score: Optional Fréchet Inception Distance score for the epoch.

        Returns:
            Dictionary record of the logged metrics.
        """
        record = {
            "epoch": epoch,
            "generator_loss": round(float(generator_loss), 6),
            "discriminator_loss": round(float(discriminator_loss), 6),
            "disc_real_loss": round(float(disc_real_loss), 6),
            "disc_fake_loss": round(float(disc_fake_loss), 6),
            "fid_score": round(float(fid_score), 4) if fid_score is not None else None,
            "epoch_duration_sec": round(float(epoch_duration_sec), 3),
            "batches_processed": int(batches_processed),
            "images_processed": int(images_processed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(record)
        return record

    def save_metrics(
        self,
        csv_filename: str = "training_history.csv",
        json_filename: str = "training_history.json",
    ) -> tuple[Path, Path]:
        """Persists the complete training history to both CSV and JSON formats.

        Args:
            csv_filename: Target CSV filename.
            json_filename: Target JSON filename.

        Returns:
            Tuple of (csv_path, json_path).
        """
        if not self.history:
            logger.warning("No metrics to save: training history is empty.")
            return self.metrics_dir / csv_filename, self.metrics_dir / json_filename

        csv_path = self.metrics_dir / csv_filename
        json_path = self.metrics_dir / json_filename

        # Save to CSV using pandas
        df = pd.DataFrame(self.history)
        df.to_csv(csv_path, index=False)

        # Save to JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Saved metric history to CSV:  {csv_path}")
        logger.info(f"Saved metric history to JSON: {json_path}")
        return csv_path, json_path

    @classmethod
    def load_metrics(cls, json_path: Path | str) -> list[dict[str, Any]]:
        """Loads metric records from a saved JSON history file."""
        src = Path(json_path)
        if not src.exists():
            raise FileNotFoundError(f"Metric file not found: {src}")
        with open(src, "r", encoding="utf-8") as f:
            return json.load(f)


class SampleGridCallback:
    """Generates visual sample grids across milestone epochs using a fixed latent noise vector."""

    def __init__(
        self,
        generator: tf.keras.Model,
        output_dir: Path | str = SAMPLES_DIR,
        num_samples: int = 16,
        latent_dim: int = LATENT_DIM,
        seed: int = RANDOM_SEED,
        milestone_epochs: Sequence[int] = (1, 10, 25, 50),
    ) -> None:
        """Initializes the sample generator callback.

        Args:
            generator: The DCGAN Generator model.
            output_dir: Directory where sample grids will be saved.
            num_samples: Total images in the grid (e.g. 16 for 4x4).
            latent_dim: Dimensionality of latent noise vector (default 100).
            seed: Reproducible seed for fixed noise vector.
            milestone_epochs: Epoch milestones at which grids must be exported.
        """
        self.generator = generator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.milestone_epochs = set(milestone_epochs)

        # Fixed latent vectors: ensures visual consistency across all training epochs
        tf.random.set_seed(seed)
        self.fixed_noise = tf.random.normal([num_samples, latent_dim], seed=seed)

    def is_milestone(self, epoch: int, is_final: bool = False) -> bool:
        """Checks if the given epoch is a requested milestone or final epoch."""
        return (epoch in self.milestone_epochs) or is_final

    def generate_and_save(
        self,
        epoch: int,
        is_final: bool = False,
        grid_shape: tuple[int, int] = (4, 4),
    ) -> Optional[Path]:
        """Generates synthetic images from fixed noise and saves a high-res grid.

        Args:
            epoch: Current training epoch number.
            is_final: Flag indicating if this is the final epoch of training.
            grid_shape: (rows, cols) of the visualization grid (default 4x4).

        Returns:
            Path to saved PNG file if saved, or None if not a milestone.
        """
        if not self.is_milestone(epoch, is_final):
            return None

        # Generator forward pass in inference mode
        raw_preds = self.generator(self.fixed_noise, training=False).numpy()

        # Denormalize from [-1.0, 1.0] back to [0, 255] uint8
        display_images = denormalize_images(raw_preds)

        rows, cols = grid_shape
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        axes = np.array(axes).reshape(rows, cols)

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            if idx < len(display_images):
                ax.imshow(display_images[idx])
                ax.set_title(f"Sample #{idx + 1}", fontsize=7)
            ax.axis("off")

        suffix = " (Final)" if is_final and epoch not in self.milestone_epochs else ""
        plt.suptitle(f"GANForge - Generated Samples at Epoch {epoch}{suffix}", fontsize=12, y=0.98)
        plt.tight_layout()

        dest = self.output_dir / f"epoch_{epoch:03d}_grid.png"
        plt.savefig(dest, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Exported generated sample grid for Epoch {epoch} to: {dest}")
        return dest
