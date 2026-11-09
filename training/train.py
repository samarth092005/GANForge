"""DCGAN Custom Training Pipeline using TensorFlow 2 / Keras tf.GradientTape.

Executes adversarial training:
- Loads real preprocessed images from train_images.npy
- Independent GradientTape updates for Generator and Discriminator
- GPU/CPU hardware detection with dynamic memory growth
- Milestones sample grid exports at Epochs 1, 10, 25, 50
- Real metric logging in CSV and JSON formats
- Checkpoint management with resume capability
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from config.settings import (
    BETA_1,
    BETA_2,
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    LATENT_DIM,
    LEARNING_RATE_DISC,
    LEARNING_RATE_GEN,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SAMPLES_DIR,
)
from evaluation.fid import FIDEvaluator
from models.discriminator import build_discriminator
from models.generator import build_generator
from models.losses import discriminator_loss, generator_loss, get_optimizers
from training.callbacks import SampleGridCallback, TrainingLogger
from training.checkpoint import GANCheckpointManager

logger = logging.getLogger("GANForge.Train")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def configure_hardware() -> str:
    """Detects available TensorFlow GPU accelerators and configures memory growth.

    Returns:
        Hardware description string (e.g. 'GPU: NVIDIA GeForce RTX 3060' or 'CPU').
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device_names = [gpu.name for gpu in gpus]
            msg = f"Accelerated hardware detected: {', '.join(device_names)} with dynamic memory growth."
            logger.info(msg)
            return msg
        except RuntimeError as e:
            logger.warning(f"Error setting GPU memory growth: {e}")
            return f"GPU detected: {gpus}"
    else:
        msg = "No compatible GPU detected. Executing on CPU (Intel / AMD host)."
        logger.info(msg)
        return msg


def load_dataset(
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[tf.data.Dataset, int]:
    """Loads train_images.npy and constructs an optimized tf.data.Dataset pipeline.

    Args:
        processed_dir: Path to directory containing train_images.npy.
        batch_size: Batch size for training (default 32).
        seed: Random seed for shuffling.

    Returns:
        Tuple of (tf.data.Dataset, total_images_count).
    """
    train_path = Path(processed_dir) / "train_images.npy"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset array not found at: {train_path}\n"
            "Please run the dataset preparation pipeline first:\n"
            "    python -m data.prepare_rangoli_dataset --count 300"
        )

    # Load normalized array [N, 64, 64, 3] in [-1.0, 1.0]
    train_images = np.load(train_path).astype(np.float32)
    total_images = len(train_images)

    dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(buffer_size=total_images, seed=seed)
        .batch(batch_size, drop_remainder=False)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    logger.info(f"Loaded {total_images} training images from: {train_path} [Batch Size={batch_size}]")
    return dataset, total_images


def train_dcgan(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    latent_dim: int = LATENT_DIM,
    learning_rate_gen: float = LEARNING_RATE_GEN,
    learning_rate_disc: float = LEARNING_RATE_DISC,
    beta_1: float = BETA_1,
    beta_2: float = BETA_2,
    label_smoothing: float = 0.1,
    resume: bool = False,
    data_dir: Path | str = PROCESSED_DATA_DIR,
    val_data_path: Optional[Path | str] = None,
    checkpoints_dir: Path | str = CHECKPOINTS_DIR,
    samples_dir: Path | str = SAMPLES_DIR,
    metrics_dir: Path | str = METRICS_DIR,
    milestone_epochs: Sequence[int] = (1, 10, 25, 50),
    seed: int = RANDOM_SEED,
    checkpoint_frequency: int = 1,
    enable_fid: bool = True,
    fid_evaluator: Optional[FIDEvaluator] = None,
    verbose: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model, list[dict[str, Any]]]:
    """Executes the full DCGAN adversarial training loop.

    Args:
        epochs: Target number of epochs to train (default 50).
        batch_size: Mini-batch size (default 32).
        latent_dim: Dimension of latent vector z (default 100).
        learning_rate_gen: Adam learning rate for Generator (default 0.0002).
        learning_rate_disc: Adam learning rate for Discriminator (default 0.0002).
        beta_1: Adam beta1 momentum decay (default 0.5).
        beta_2: Adam beta2 moment decay (default 0.999).
        label_smoothing: One-sided label smoothing for real images (default 0.1).
        resume: Whether to resume training from the latest checkpoint.
        data_dir: Directory containing processed train_images.npy.
        val_data_path: Optional explicit path to val_images.npy.
        checkpoints_dir: Directory for storing model checkpoints.
        samples_dir: Directory for exporting visual sample grids.
        metrics_dir: Directory for saving training history logs.
        milestone_epochs: Epoch milestones for sample grid generation.
        seed: Reproducible random seed.
        checkpoint_frequency: Save checkpoint every N epochs (default 1).
        enable_fid: Whether to evaluate Fréchet Inception Distance (FID) after each epoch.
        fid_evaluator: Optional pre-configured FIDEvaluator instance.
        verbose: Whether to print per-epoch progress logs.

    Returns:
        Tuple of (trained_generator, trained_discriminator, training_history).
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # 1. Hardware Initialization
    hw_info = configure_hardware()

    # 2. Dataset Loading (Train Set)
    dataset, total_images = load_dataset(data_dir, batch_size=batch_size, seed=seed)

    # 3. Model Architecture Instantiation
    generator = build_generator(latent_dim=latent_dim)
    discriminator = build_discriminator()

    # 4. Optimizers
    gen_opt, disc_opt = get_optimizers(
        learning_rate_gen=learning_rate_gen,
        learning_rate_disc=learning_rate_disc,
        beta_1=beta_1,
        beta_2=beta_2,
    )

    # 5. Checkpoint & Callback Managers
    ckpt_mgr = GANCheckpointManager(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=gen_opt,
        discriminator_optimizer=disc_opt,
        checkpoint_dir=checkpoints_dir,
    )

    sample_cb = SampleGridCallback(
        generator=generator,
        output_dir=samples_dir,
        num_samples=16,
        latent_dim=latent_dim,
        seed=seed,
        milestone_epochs=milestone_epochs,
    )

    logger_cb = TrainingLogger(metrics_dir=metrics_dir)

    # 6. FID Evaluation Precomputation (InceptionV3 initialized once, real validation stats precomputed)
    if enable_fid and fid_evaluator is None:
        vpath = Path(val_data_path) if val_data_path else Path(data_dir) / "val_images.npy"
        if vpath.exists():
            try:
                logger.info(f"Initializing FID evaluation using validation set: {vpath}")
                fid_evaluator = FIDEvaluator(
                    val_data_path=vpath,
                    num_samples=60,
                    latent_dim=latent_dim,
                    seed=seed,
                )
            except Exception as e:
                logger.warning(f"Could not initialize FID evaluator ({e}). Training will continue without FID.")
                fid_evaluator = None
        else:
            logger.warning(
                f"Validation dataset not found at '{vpath}'. FID evaluation will be skipped.\n"
                "To enable FID, ensure val_images.npy exists in data_dir."
            )
            fid_evaluator = None

    # 7. Checkpoint Restoration (if requested)
    start_epoch = 0
    if resume:
        start_epoch = ckpt_mgr.restore_latest()
        if start_epoch > 0:
            logger.info(f"Resuming training from epoch {start_epoch} up to target {epochs}.")
            # Load existing history if available and filter to avoid duplicate epoch entries
            existing_json = Path(metrics_dir) / "training_history.json"
            if existing_json.exists():
                try:
                    raw_history = TrainingLogger.load_metrics(existing_json)
                    # Strictly retain only epochs up to start_epoch (prevents duplication upon resume)
                    logger_cb.history = [
                        rec for rec in raw_history if int(rec.get("epoch", 0)) <= start_epoch
                    ]
                    logger.info(
                        f"Loaded {len(logger_cb.history)} historical epoch metric records up to Epoch {start_epoch}."
                    )
                except Exception as e:
                    logger.warning(f"Could not load previous metric history: {e}")

    if start_epoch >= epochs:
        logger.info(f"Model is already trained for {start_epoch} epochs (>= target {epochs}).")
        return generator, discriminator, logger_cb.history

    # 8. Compiled TensorFlow Training Step with tf.GradientTape
    @tf.function
    def train_step(real_images_batch: tf.Tensor, current_batch_size: tf.Tensor):
        # Generate random latent vectors z ~ N(0, 1)
        noise = tf.random.normal([current_batch_size, latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Forward pass: Generator synthesizes images
            generated_images = generator(noise, training=True)

            # Forward pass: Discriminator scores real and generated images
            real_logits = discriminator(real_images_batch, training=True)
            fake_logits = discriminator(generated_images, training=True)

            # Compute adversarial losses
            loss_g = generator_loss(fake_logits)
            loss_d, loss_d_real, loss_d_fake = discriminator_loss(
                real_logits, fake_logits, label_smoothing=label_smoothing
            )

        # Backpropagation: Compute separate gradients
        gradients_g = gen_tape.gradient(loss_g, generator.trainable_variables)
        gradients_d = disc_tape.gradient(loss_d, discriminator.trainable_variables)

        # Optimizer Step: Independent parameter updates
        gen_opt.apply_gradients(zip(gradients_g, generator.trainable_variables))
        disc_opt.apply_gradients(zip(gradients_d, discriminator.trainable_variables))

        return loss_g, loss_d, loss_d_real, loss_d_fake

    # 9. Training Execution Loop
    config_dict = {
        "epochs": epochs,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "lr_gen": learning_rate_gen,
        "lr_disc": learning_rate_disc,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "label_smoothing": label_smoothing,
        "checkpoint_frequency": checkpoint_frequency,
        "seed": seed,
        "hardware": hw_info,
    }

    print("\n" + "=" * 70)
    print(f"  GANForge DCGAN Training Engine: Epochs {start_epoch + 1} -> {epochs}")
    print(f"  Target Samples: {total_images} | Batch Size: {batch_size} | Latent Dim: {latent_dim}")
    print("=" * 70)

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start_time = time.perf_counter()
        batch_gen_losses: list[float] = []
        batch_disc_losses: list[float] = []
        batch_real_losses: list[float] = []
        batch_fake_losses: list[float] = []
        num_batches = 0

        for real_batch in dataset:
            curr_b_size = tf.shape(real_batch)[0]
            l_g, l_d, l_d_real, l_d_fake = train_step(real_batch, curr_b_size)

            # Guard against NaN / Inf divergence
            if tf.math.is_nan(l_g) or tf.math.is_inf(l_g):
                raise RuntimeError(f"Divergence detected: Generator loss became NaN/Inf at epoch {epoch}!")
            if tf.math.is_nan(l_d) or tf.math.is_inf(l_d):
                raise RuntimeError(f"Divergence detected: Discriminator loss became NaN/Inf at epoch {epoch}!")

            batch_gen_losses.append(float(l_g.numpy()))
            batch_disc_losses.append(float(l_d.numpy()))
            batch_real_losses.append(float(l_d_real.numpy()))
            batch_fake_losses.append(float(l_d_fake.numpy()))
            num_batches += 1

        epoch_duration = time.perf_counter() - epoch_start_time

        # Epoch-averaged metrics
        avg_loss_g = float(np.mean(batch_gen_losses))
        avg_loss_d = float(np.mean(batch_disc_losses))
        avg_loss_d_real = float(np.mean(batch_real_losses))
        avg_loss_d_fake = float(np.mean(batch_fake_losses))

        # Evaluate FID after every epoch using fixed 60 latent samples
        current_fid: Optional[float] = None
        if fid_evaluator is not None:
            try:
                fid_start = time.perf_counter()
                current_fid = fid_evaluator.evaluate(generator)
                fid_duration = time.perf_counter() - fid_start
                logger.info(
                    f"Epoch {epoch} FID Score: {current_fid:.4f} (eval duration: {fid_duration:.2f}s)"
                )
            except Exception as e:
                logger.warning(f"Error evaluating FID at epoch {epoch}: {e}")
                current_fid = None

        # Record metrics in logger callback
        rec = logger_cb.log_epoch(
            epoch=epoch,
            generator_loss=avg_loss_g,
            discriminator_loss=avg_loss_d,
            disc_real_loss=avg_loss_d_real,
            disc_fake_loss=avg_loss_d_fake,
            epoch_duration_sec=epoch_duration,
            batches_processed=num_batches,
            images_processed=total_images,
            fid_score=current_fid,
        )

        # Save metrics after every epoch to both CSV and JSON
        logger_cb.save_metrics()

        if verbose:
            fid_str = f" | FID: {current_fid:7.4f}" if current_fid is not None else ""
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] "
                f"Duration: {epoch_duration:5.2f}s | "
                f"Loss G: {avg_loss_g:6.4f} | "
                f"Loss D: {avg_loss_d:6.4f} (Real: {avg_loss_d_real:6.4f}, Fake: {avg_loss_d_fake:6.4f})"
                f"{fid_str}"
            )

        # Generate sample grid at milestone epochs or final epoch
        is_final_epoch = (epoch == epochs)
        sample_cb.generate_and_save(epoch=epoch, is_final=is_final_epoch)

        # Periodic checkpointing (configurable frequency, defaulting to 1 = every epoch)
        if (epoch % checkpoint_frequency == 0) or is_final_epoch:
            ckpt_mgr.save(epoch=epoch, metrics=rec, config=config_dict)

    # 10. Final Persistence & Weight Export
    logger_cb.save_metrics()
    ckpt_mgr.export_weights(export_dir=checkpoints_dir, prefix=f"epoch_{epochs:03d}")

    print("=" * 70)
    print(f"[+] Training successfully completed up to Epoch {epochs}!")
    print(f"[+] Metrics saved in : {metrics_dir}")
    print(f"[+] Samples saved in : {samples_dir}")
    print(f"[+] Weights saved in : {checkpoints_dir}")
    print("=" * 70 + "\n")

    return generator, discriminator, logger_cb.history


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for DCGAN training."""
    parser = argparse.ArgumentParser(description="GANForge DCGAN Training Engine.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Target epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr-gen", type=float, default=LEARNING_RATE_GEN, help="Generator LR")
    parser.add_argument("--lr-disc", type=float, default=LEARNING_RATE_DISC, help="Discriminator LR")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--data-dir", type=str, default=str(PROCESSED_DATA_DIR), help="Processed dataset folder")
    parser.add_argument("--val-data-path", type=str, default=None, help="Explicit path to val_images.npy")
    parser.add_argument("--output-dir", type=str, default=str(CHECKPOINTS_DIR.parent), help="Outputs base folder")
    parser.add_argument("--checkpoint-frequency", type=int, default=1, help="Save checkpoint every N epochs (default: 1)")
    parser.add_argument("--no-fid", action="store_true", help="Disable FID evaluation")
    args = parser.parse_args(argv)

    output_root = Path(args.output_dir)
    checkpoints_dir = output_root / "checkpoints"
    samples_dir = output_root / "samples"
    metrics_dir = output_root / "metrics"

    train_dcgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate_gen=args.lr_gen,
        learning_rate_disc=args.lr_disc,
        resume=args.resume,
        seed=args.seed,
        data_dir=args.data_dir,
        val_data_path=args.val_data_path,
        checkpoints_dir=checkpoints_dir,
        samples_dir=samples_dir,
        metrics_dir=metrics_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        enable_fid=not args.no_fid,
    )


if __name__ == "__main__":
    main()
