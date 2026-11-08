"""Checkpoint management for DCGAN models and optimizers.

Provides robust saving, restoring, and resuming capabilities for Generator,
Discriminator, and their respective Adam optimizer states using TensorFlow 2 CheckpointManager.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import tensorflow as tf

from config.settings import CHECKPOINTS_DIR

logger = logging.getLogger("GANForge.Checkpoint")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class GANCheckpointManager:
    """Manages TensorFlow checkpoints for Generator, Discriminator, and Adam optimizers."""

    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        generator_optimizer: tf.keras.optimizers.Optimizer,
        discriminator_optimizer: tf.keras.optimizers.Optimizer,
        checkpoint_dir: Path | str = CHECKPOINTS_DIR,
        max_to_keep: int = 5,
    ) -> None:
        """Initializes the checkpoint manager.

        Args:
            generator: The DCGAN Generator model.
            discriminator: The DCGAN Discriminator model.
            generator_optimizer: Adam optimizer for the Generator.
            discriminator_optimizer: Adam optimizer for the Discriminator.
            checkpoint_dir: Directory where checkpoints are stored.
            max_to_keep: Maximum number of recent checkpoints to retain.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.gen_opt = generator_optimizer
        self.disc_opt = discriminator_optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Core TensorFlow Checkpoint bundling models and optimizers
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.gen_opt,
            discriminator_optimizer=self.disc_opt,
        )

        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=str(self.checkpoint_dir),
            max_to_keep=max_to_keep,
            checkpoint_name="dcgan_ckpt",
        )

        self.metadata_file = self.checkpoint_dir / "checkpoint_meta.json"

    def save(
        self,
        epoch: int,
        metrics: Optional[dict[str, float]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> str:
        """Saves a checkpoint for the current epoch and updates metadata.

        Args:
            epoch: The completed training epoch number (1-based).
            metrics: Optional dictionary of current metrics (e.g. losses).
            config: Optional training configuration dictionary.

        Returns:
            String path to the saved checkpoint file.
        """
        save_path = self.manager.save(checkpoint_number=epoch)
        if save_path is None:
            raise RuntimeError(f"Failed to save checkpoint at epoch {epoch}")

        meta = {
            "latest_epoch": epoch,
            "checkpoint_path": save_path,
            "metrics": metrics or {},
            "config": config or {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved checkpoint for epoch {epoch} at: {save_path}")
        return save_path

    def restore_latest(self) -> int:
        """Restores model and optimizer states from the latest available checkpoint.

        Returns:
            The restored epoch number (int). Returns 0 if no checkpoint is found.
        """
        latest_ckpt = self.manager.latest_checkpoint
        if not latest_ckpt:
            logger.info("No existing checkpoint found. Initializing from scratch.")
            return 0

        logger.info(f"Restoring checkpoint from: {latest_ckpt}")
        status = self.checkpoint.restore(latest_ckpt)
        status.expect_partial()  # Tolerate non-critical auxiliary optimizer variables

        restored_epoch = 0
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                restored_epoch = int(meta.get("latest_epoch", 0))
            except Exception as e:
                logger.warning(f"Could not parse checkpoint metadata file: {e}")

        # Fallback: extract epoch number from checkpoint suffix if metadata missing
        if restored_epoch == 0:
            try:
                suffix = latest_ckpt.split("-")[-1]
                restored_epoch = int(suffix)
            except ValueError:
                restored_epoch = 0

        logger.info(f"Successfully restored checkpoint from epoch {restored_epoch}.")
        return restored_epoch

    def export_weights(self, export_dir: Optional[Path | str] = None, prefix: str = "final") -> dict[str, Path]:
        """Exports standalone Keras .weights.h5 files for direct inference.

        Args:
            export_dir: Destination directory (defaults to checkpoint_dir).
            prefix: Prefix for exported filenames.

        Returns:
            Dictionary with paths to exported weights.
        """
        dest_dir = Path(export_dir) if export_dir else self.checkpoint_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        gen_weights_path = dest_dir / f"{prefix}_generator.weights.h5"
        disc_weights_path = dest_dir / f"{prefix}_discriminator.weights.h5"

        self.generator.save_weights(str(gen_weights_path))
        self.discriminator.save_weights(str(disc_weights_path))

        logger.info(f"Exported Generator weights to: {gen_weights_path}")
        logger.info(f"Exported Discriminator weights to: {disc_weights_path}")

        return {
            "generator_weights": gen_weights_path,
            "discriminator_weights": disc_weights_path,
        }
