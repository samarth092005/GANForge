"""Unit and integration tests for the DCGAN training pipeline, checkpointing, and callbacks.

Verifies:
- Training step gradient backpropagation
- Weight updates in Generator and Discriminator
- Checkpoint persistence and restoration
- Resume functionality
- Milestone sample grid generation
- Loss logging in CSV and JSON formats
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from config.settings import CHANNELS, IMAGE_SIZE, LATENT_DIM
from evaluation.fid import FIDEvaluator
from models.discriminator import build_discriminator
from models.generator import build_generator
from models.losses import discriminator_loss, generator_loss, get_optimizers
from tests.test_fid_evaluation import MockInceptionV3
from training.callbacks import SampleGridCallback, TrainingLogger
from training.checkpoint import GANCheckpointManager
from training.train import main, train_dcgan


class TestTrainingPipeline(unittest.TestCase):
    """Test suite for DCGAN training loop, callbacks, and checkpointing."""

    def setUp(self) -> None:
        """Create temporary sandbox directory for test artifacts."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="ganforge_train_test_"))
        self.checkpoints_dir = self.test_dir / "checkpoints"
        self.samples_dir = self.test_dir / "samples"
        self.metrics_dir = self.test_dir / "metrics"
        self.data_dir = self.test_dir / "data"

        for d in [self.checkpoints_dir, self.samples_dir, self.metrics_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.batch_size = 4
        self.latent_dim = LATENT_DIM

    def tearDown(self) -> None:
        """Clean up temporary test sandbox."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_dataset(self, count: int = 8) -> Path:
        """Helper to create small synthetic train_images.npy."""
        synthetic_images = np.random.uniform(
            -1.0, 1.0, size=(count, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)
        ).astype(np.float32)
        save_path = self.data_dir / "train_images.npy"
        np.save(save_path, synthetic_images)
        return save_path

    def _create_synthetic_val_dataset(self, count: int = 6) -> Path:
        """Helper to create small synthetic val_images.npy."""
        synthetic_val = np.random.uniform(
            -1.0, 1.0, size=(count, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)
        ).astype(np.float32)
        save_path = self.data_dir / "val_images.npy"
        np.save(save_path, synthetic_val)
        return save_path

    # ---------------------------------------------------------
    # 1. Training Step & Weight Updates
    # ---------------------------------------------------------
    def test_training_step_and_weight_updates(self) -> None:
        """Verify forward pass, loss finiteness, and actual parameter updates."""
        generator = build_generator(latent_dim=self.latent_dim)
        discriminator = build_discriminator()
        gen_opt, disc_opt = get_optimizers(learning_rate_gen=0.001, learning_rate_disc=0.001)

        real_batch = tf.random.uniform(
            [self.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS],
            minval=-1.0,
            maxval=1.0,
        )

        # Snapshot weights before update
        gen_weights_before = [tf.identity(w) for w in generator.trainable_variables]
        disc_weights_before = [tf.identity(w) for w in discriminator.trainable_variables]

        noise = tf.random.normal([self.batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = generator(noise, training=True)
            real_logits = discriminator(real_batch, training=True)
            fake_logits = discriminator(fake_images, training=True)

            loss_g = generator_loss(fake_logits)
            loss_d, loss_d_real, loss_d_fake = discriminator_loss(real_logits, fake_logits)

        # Verify finite scalar losses
        self.assertTrue(tf.math.is_finite(loss_g))
        self.assertTrue(tf.math.is_finite(loss_d))

        # Compute & apply gradients
        gen_grads = gen_tape.gradient(loss_g, generator.trainable_variables)
        disc_grads = disc_tape.gradient(loss_d, discriminator.trainable_variables)

        gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))
        disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        # Verify weights actually changed
        gen_changed = any(
            not np.allclose(w_before.numpy(), w_after.numpy())
            for w_before, w_after in zip(gen_weights_before, generator.trainable_variables)
        )
        disc_changed = any(
            not np.allclose(w_before.numpy(), w_after.numpy())
            for w_before, w_after in zip(disc_weights_before, discriminator.trainable_variables)
        )

        self.assertTrue(gen_changed, "Generator weights did not update after training step!")
        self.assertTrue(disc_changed, "Discriminator weights did not update after training step!")

    # ---------------------------------------------------------
    # 2. Checkpoint Save, Restore & Resume
    # ---------------------------------------------------------
    def test_checkpoint_save_and_restore(self) -> None:
        """Verify checkpoint saves weights and restores exact parameter values."""
        gen = build_generator(latent_dim=self.latent_dim)
        disc = build_discriminator()
        gen_opt, disc_opt = get_optimizers()

        mgr = GANCheckpointManager(
            generator=gen,
            discriminator=disc,
            generator_optimizer=gen_opt,
            discriminator_optimizer=disc_opt,
            checkpoint_dir=self.checkpoints_dir,
        )

        # Save checkpoint at epoch 5
        save_path = mgr.save(epoch=5, metrics={"gen_loss": 0.5, "disc_loss": 0.8})
        self.assertTrue(Path(save_path + ".index").exists() or Path(save_path).exists())
        self.assertTrue(mgr.metadata_file.exists())

        # Instantiate fresh models
        fresh_gen = build_generator(latent_dim=self.latent_dim)
        fresh_disc = build_discriminator()
        fresh_gen_opt, fresh_disc_opt = get_optimizers()

        fresh_mgr = GANCheckpointManager(
            generator=fresh_gen,
            discriminator=fresh_disc,
            generator_optimizer=fresh_gen_opt,
            discriminator_optimizer=fresh_disc_opt,
            checkpoint_dir=self.checkpoints_dir,
        )

        restored_epoch = fresh_mgr.restore_latest()
        self.assertEqual(restored_epoch, 5)

        # Check weight equality
        for orig_w, restored_w in zip(gen.trainable_variables, fresh_gen.trainable_variables):
            np.testing.assert_allclose(orig_w.numpy(), restored_w.numpy(), rtol=1e-5, atol=1e-5)

    def test_weight_export_files(self) -> None:
        """Verify export_weights produces standalone .weights.h5 files."""
        gen = build_generator(latent_dim=self.latent_dim)
        disc = build_discriminator()
        gen_opt, disc_opt = get_optimizers()

        mgr = GANCheckpointManager(gen, disc, gen_opt, disc_opt, self.checkpoints_dir)
        paths = mgr.export_weights(self.checkpoints_dir, prefix="unit_test")

        self.assertTrue(paths["generator_weights"].exists())
        self.assertTrue(paths["discriminator_weights"].exists())

    # ---------------------------------------------------------
    # 3. Callbacks: Logger & Sample Generator
    # ---------------------------------------------------------
    def test_training_logger_csv_and_json(self) -> None:
        """Verify TrainingLogger correctly records records and saves valid CSV/JSON."""
        logger_cb = TrainingLogger(metrics_dir=self.metrics_dir)

        logger_cb.log_epoch(
            epoch=1,
            generator_loss=1.234,
            discriminator_loss=0.876,
            disc_real_loss=0.450,
            disc_fake_loss=0.426,
            epoch_duration_sec=2.5,
            batches_processed=8,
            images_processed=240,
            fid_score=45.6789,
        )

        csv_p, json_p = logger_cb.save_metrics()
        self.assertTrue(csv_p.exists())
        self.assertTrue(json_p.exists())

        loaded_json = TrainingLogger.load_metrics(json_p)
        self.assertEqual(len(loaded_json), 1)
        self.assertEqual(loaded_json[0]["epoch"], 1)
        self.assertAlmostEqual(loaded_json[0]["generator_loss"], 1.234)
        self.assertAlmostEqual(loaded_json[0]["fid_score"], 45.6789)

    def test_sample_grid_callback_milestones(self) -> None:
        """Verify SampleGridCallback exports PNG grid on milestone epochs."""
        gen = build_generator(latent_dim=self.latent_dim)
        sample_cb = SampleGridCallback(
            generator=gen,
            output_dir=self.samples_dir,
            num_samples=4,
            latent_dim=self.latent_dim,
            seed=42,
            milestone_epochs=(1, 5),
        )

        # Non-milestone epoch should return None
        out_2 = sample_cb.generate_and_save(epoch=2, is_final=False, grid_shape=(2, 2))
        self.assertIsNone(out_2)

        # Milestone epoch 1 should produce a PNG file
        out_1 = sample_cb.generate_and_save(epoch=1, is_final=False, grid_shape=(2, 2))
        self.assertIsNotNone(out_1)
        self.assertTrue(out_1.exists())
        self.assertEqual(out_1.suffix, ".png")

        # Final epoch should generate even if not in milestone_epochs
        out_final = sample_cb.generate_and_save(epoch=3, is_final=True, grid_shape=(2, 2))
        self.assertIsNotNone(out_final)
        self.assertTrue(out_final.exists())

    # ---------------------------------------------------------
    # 4. End-to-End Mini Training Loop Execution
    # ---------------------------------------------------------
    def test_mini_training_loop_run(self) -> None:
        """Execute a 2-epoch mini training loop on synthetic data."""
        self._create_synthetic_dataset(count=8)

        gen, disc, history = train_dcgan(
            epochs=2,
            batch_size=4,
            latent_dim=self.latent_dim,
            resume=False,
            data_dir=self.data_dir,
            checkpoints_dir=self.checkpoints_dir,
            samples_dir=self.samples_dir,
            metrics_dir=self.metrics_dir,
            milestone_epochs=(1, 2),
            seed=42,
            checkpoint_frequency=1,
            verbose=False,
        )

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["epoch"], 1)
        self.assertEqual(history[1]["epoch"], 2)

        # Confirm artifacts were created
        self.assertTrue((self.metrics_dir / "training_history.csv").exists())
        self.assertTrue((self.metrics_dir / "training_history.json").exists())
        self.assertTrue((self.samples_dir / "epoch_001_grid.png").exists())
        self.assertTrue((self.samples_dir / "epoch_002_grid.png").exists())
        self.assertTrue((self.checkpoints_dir / "checkpoint_meta.json").exists())

    def test_resume_training_and_metric_deduplication(self) -> None:
        """Verify resume continues to target epoch and appends metrics without duplicates."""
        self._create_synthetic_dataset(count=8)

        # 1. Run 1 epoch initially
        gen1, disc1, hist1 = train_dcgan(
            epochs=1,
            batch_size=4,
            latent_dim=self.latent_dim,
            resume=False,
            data_dir=self.data_dir,
            checkpoints_dir=self.checkpoints_dir,
            samples_dir=self.samples_dir,
            metrics_dir=self.metrics_dir,
            checkpoint_frequency=1,
            verbose=False,
        )
        self.assertEqual(len(hist1), 1)
        self.assertEqual(hist1[0]["epoch"], 1)

        # 2. Resume to epoch 2
        gen2, disc2, hist2 = train_dcgan(
            epochs=2,
            batch_size=4,
            latent_dim=self.latent_dim,
            resume=True,
            data_dir=self.data_dir,
            checkpoints_dir=self.checkpoints_dir,
            samples_dir=self.samples_dir,
            metrics_dir=self.metrics_dir,
            checkpoint_frequency=1,
            verbose=False,
        )
        self.assertEqual(len(hist2), 2)
        epochs_logged = [h["epoch"] for h in hist2]
        self.assertEqual(
            epochs_logged,
            [1, 2],
            "Metric history contains duplicated or missing epochs upon resume!",
        )

    def test_fixed_sample_noise_consistency(self) -> None:
        """Verify fixed sample noise remains perfectly identical across distinct callback instances."""
        gen = build_generator(latent_dim=self.latent_dim)
        cb1 = SampleGridCallback(gen, output_dir=self.samples_dir, num_samples=16, seed=42)
        cb2 = SampleGridCallback(gen, output_dir=self.samples_dir, num_samples=16, seed=42)

        np.testing.assert_array_equal(cb1.fixed_noise.numpy(), cb2.fixed_noise.numpy())

    def test_training_pipeline_with_fid_evaluation(self) -> None:
        """Verify FID evaluation is executed after each epoch, recorded in history, and saved in metadata."""
        import pandas as pd
        self._create_synthetic_dataset(count=8)
        self._create_synthetic_val_dataset(count=6)

        # Mock FIDEvaluator to avoid downloading InceptionV3 weights during unit test
        mock_model = MockInceptionV3(feature_dim=64)
        evaluator = FIDEvaluator(
            val_data_path=self.data_dir / "val_images.npy",
            model=mock_model,
            num_samples=6,
            latent_dim=self.latent_dim,
            seed=42,
        )

        gen, disc, history = train_dcgan(
            epochs=2,
            batch_size=4,
            latent_dim=self.latent_dim,
            resume=False,
            data_dir=self.data_dir,
            checkpoints_dir=self.checkpoints_dir,
            samples_dir=self.samples_dir,
            metrics_dir=self.metrics_dir,
            milestone_epochs=(1, 2),
            seed=42,
            checkpoint_frequency=1,
            enable_fid=True,
            fid_evaluator=evaluator,
            verbose=False,
        )

        self.assertEqual(len(history), 2)
        # FID must be recorded in history for each epoch
        for epoch_rec in history:
            self.assertIn("fid_score", epoch_rec)
            self.assertIsNotNone(epoch_rec["fid_score"])
            self.assertIsInstance(epoch_rec["fid_score"], float)
            self.assertGreaterEqual(epoch_rec["fid_score"], 0.0)

        # Confirm CSV contains fid_score
        csv_path = self.metrics_dir / "training_history.csv"
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        self.assertIn("fid_score", df.columns)
        self.assertEqual(len(df), 2)
        self.assertFalse(df["fid_score"].isnull().any())

        # Confirm JSON contains fid_score
        json_path = self.metrics_dir / "training_history.json"
        self.assertTrue(json_path.exists())
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        self.assertEqual(len(json_data), 2)
        self.assertIn("fid_score", json_data[0])
        self.assertIsNotNone(json_data[0]["fid_score"])

        # Confirm Checkpoint metadata includes FID
        meta_path = self.checkpoints_dir / "checkpoint_meta.json"
        self.assertTrue(meta_path.exists())
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        self.assertIn("fid_score", meta_data.get("metrics", {}))
        self.assertIsNotNone(meta_data["metrics"]["fid_score"])

    def test_resume_training_preserves_fid_history(self) -> None:
        """Verify resuming training preserves earlier FID history without duplicates or data loss."""
        self._create_synthetic_dataset(count=8)
        self._create_synthetic_val_dataset(count=6)

        mock_model = MockInceptionV3(feature_dim=64)
        evaluator1 = FIDEvaluator(
            val_data_path=self.data_dir / "val_images.npy",
            model=mock_model,
            num_samples=6,
            latent_dim=self.latent_dim,
            seed=42,
        )

        # 1. Run 1 epoch
        train_dcgan(
            epochs=1,
            batch_size=4,
            latent_dim=self.latent_dim,
            resume=False,
            data_dir=self.data_dir,
            checkpoints_dir=self.checkpoints_dir,
            samples_dir=self.samples_dir,
            metrics_dir=self.metrics_dir,
            checkpoint_frequency=1,
            enable_fid=True,
            fid_evaluator=evaluator1,
            verbose=False,
        )

        # 2. Resume to epoch 2
        evaluator2 = FIDEvaluator(
            val_data_path=self.data_dir / "val_images.npy",
            model=mock_model,
            num_samples=6,
            latent_dim=self.latent_dim,
            seed=42,
        )

        gen, disc, hist = train_dcgan(
            epochs=2,
            batch_size=4,
            latent_dim=self.latent_dim,
            resume=True,
            data_dir=self.data_dir,
            checkpoints_dir=self.checkpoints_dir,
            samples_dir=self.samples_dir,
            metrics_dir=self.metrics_dir,
            checkpoint_frequency=1,
            enable_fid=True,
            fid_evaluator=evaluator2,
            verbose=False,
        )

        self.assertEqual(len(hist), 2)
        self.assertEqual([h["epoch"] for h in hist], [1, 2])
        self.assertIsNotNone(hist[0]["fid_score"])
        self.assertIsNotNone(hist[1]["fid_score"])

    def test_custom_output_dir_isolation_and_resume(self) -> None:
        """Verify custom output_dir receives all artifacts and default directories remain untouched."""
        from config.settings import CHECKPOINTS_DIR, METRICS_DIR, SAMPLES_DIR

        self._create_synthetic_dataset(count=8)
        self._create_synthetic_val_dataset(count=6)

        custom_root = self.test_dir / "custom_outputs"
        custom_ckpt = custom_root / "checkpoints"
        custom_samples = custom_root / "samples"
        custom_metrics = custom_root / "metrics"

        # Record file counts in default output directories prior to training
        default_ckpts_before = set(CHECKPOINTS_DIR.glob("*"))
        default_samples_before = set(SAMPLES_DIR.glob("*"))
        default_metrics_before = set(METRICS_DIR.glob("*"))

        mock_model = MockInceptionV3(feature_dim=64)
        evaluator1 = FIDEvaluator(
            val_data_path=self.data_dir / "val_images.npy",
            model=mock_model,
            num_samples=6,
            latent_dim=self.latent_dim,
            seed=42,
        )

        # 1. Run Epoch 1 with custom output directories
        train_dcgan(
            epochs=1,
            batch_size=4,
            latent_dim=self.latent_dim,
            resume=False,
            data_dir=self.data_dir,
            checkpoints_dir=custom_ckpt,
            samples_dir=custom_samples,
            metrics_dir=custom_metrics,
            checkpoint_frequency=1,
            enable_fid=True,
            fid_evaluator=evaluator1,
            verbose=False,
        )

        # Verify all artifacts exist in custom_root
        self.assertTrue((custom_ckpt / "checkpoint_meta.json").exists())
        self.assertTrue((custom_samples / "epoch_001_grid.png").exists())
        self.assertTrue((custom_metrics / "training_history.csv").exists())
        self.assertTrue((custom_metrics / "training_history.json").exists())

        # Verify default directories remain completely untouched
        self.assertEqual(set(CHECKPOINTS_DIR.glob("*")), default_ckpts_before)
        self.assertEqual(set(SAMPLES_DIR.glob("*")), default_samples_before)
        self.assertEqual(set(METRICS_DIR.glob("*")), default_metrics_before)

        # 2. Resume to Epoch 2 using the same custom directory
        evaluator2 = FIDEvaluator(
            val_data_path=self.data_dir / "val_images.npy",
            model=mock_model,
            num_samples=6,
            latent_dim=self.latent_dim,
            seed=42,
        )

        gen, disc, hist = train_dcgan(
            epochs=2,
            batch_size=4,
            latent_dim=self.latent_dim,
            resume=True,
            data_dir=self.data_dir,
            checkpoints_dir=custom_ckpt,
            samples_dir=custom_samples,
            metrics_dir=custom_metrics,
            checkpoint_frequency=1,
            enable_fid=True,
            fid_evaluator=evaluator2,
            verbose=False,
        )

        self.assertEqual(len(hist), 2)
        self.assertEqual([h["epoch"] for h in hist], [1, 2])
        meta = json.loads((custom_ckpt / "checkpoint_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["latest_epoch"], 2)

        # Default directories still untouched
        self.assertEqual(set(CHECKPOINTS_DIR.glob("*")), default_ckpts_before)
        self.assertEqual(set(SAMPLES_DIR.glob("*")), default_samples_before)
        self.assertEqual(set(METRICS_DIR.glob("*")), default_metrics_before)

    def test_cli_output_dir_derivation(self) -> None:
        """Verify CLI correctly derives checkpoints_dir, samples_dir, and metrics_dir from --output-dir."""
        from unittest.mock import patch

        custom_root = self.test_dir / "cli_test_out"
        cli_argv = [
            "--epochs", "1",
            "--batch-size", "4",
            "--data-dir", str(self.data_dir),
            "--output-dir", str(custom_root),
            "--no-fid",
        ]

        with patch("training.train.train_dcgan") as mock_train:
            main(cli_argv)
            mock_train.assert_called_once()
            call_kwargs = mock_train.call_args[1]

            self.assertEqual(call_kwargs["checkpoints_dir"], custom_root / "checkpoints")
            self.assertEqual(call_kwargs["samples_dir"], custom_root / "samples")
            self.assertEqual(call_kwargs["metrics_dir"], custom_root / "metrics")
            self.assertEqual(call_kwargs["epochs"], 1)
            self.assertEqual(call_kwargs["enable_fid"], False)


if __name__ == "__main__":
    unittest.main()
