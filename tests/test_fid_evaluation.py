"""Unit tests for Fréchet Inception Distance (FID) evaluation and preprocessing.

Verifies:
- Preprocessing and resizing of generated and real images to InceptionV3 format (299x299, [-1, 1])
- Defensive validation of malformed and invalid inputs (NaNs, Infs, bad shapes, bad channels)
- Feature extraction with mock InceptionV3 architecture (no internet weights downloaded)
- Activation statistics calculation (mean vector, covariance matrix, sample size guards)
- Fréchet distance mathematical formulation and numerical covariance stabilization
- FIDEvaluator precomputation and epoch-wise generator evaluation
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import tensorflow as tf

from config.settings import FID_IMAGE_SIZE, LATENT_DIM
from evaluation.fid import (
    APPROXIMATE_SAMPLE_NOTE,
    INCEPTION_FEATURE_DIM,
    FIDEvaluator,
    calculate_activation_statistics,
    calculate_frechet_distance,
    extract_features,
    preprocess_images_for_inception,
)
from models.generator import build_generator


class MockInceptionV3(tf.keras.Model):
    """Deterministic mock feature extractor reproducing InceptionV3 output shape (None, 2048)."""

    def __init__(self, feature_dim: int = INCEPTION_FEATURE_DIM) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.gap = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        pooled = self.gap(inputs)  # [B, 3]
        repeats = (self.feature_dim // 3) + 1
        tiled = tf.tile(pooled, [1, repeats])[:, : self.feature_dim]
        # Introduce a deterministic non-zero bias for full-rank activation
        indices = tf.range(self.feature_dim, dtype=tf.float32) * 0.001
        return tiled + indices


class TestFIDEvaluation(unittest.TestCase):
    """Test suite covering FID mathematical formulas, preprocessing, and evaluator."""

    def setUp(self) -> None:
        np.random.seed(42)
        tf.random.set_seed(42)
        self.mock_model = MockInceptionV3()

    # -------------------------------------------------------------------------
    # 1. Preprocessing & Input Validation Tests
    # -------------------------------------------------------------------------
    def test_preprocess_valid_4d_images(self) -> None:
        """Verify 4D image array in [-1, 1] is resized to (299, 299) and kept in range."""
        raw = np.random.uniform(-1.0, 1.0, size=(4, 64, 64, 3)).astype(np.float32)
        out = preprocess_images_for_inception(raw)

        self.assertEqual(out.shape, (4, 299, 299, 3))
        self.assertEqual(out.dtype, tf.float32)
        self.assertGreaterEqual(float(tf.reduce_min(out)), -1.0)
        self.assertLessEqual(float(tf.reduce_max(out)), 1.0)

    def test_preprocess_single_3d_image(self) -> None:
        """Verify 3D single image (H, W, 3) is automatically expanded to batch dimension 1."""
        raw = np.random.uniform(-1.0, 1.0, size=(64, 64, 3)).astype(np.float32)
        out = preprocess_images_for_inception(raw)

        self.assertEqual(out.shape, (1, 299, 299, 3))

    def test_preprocess_uint8_scaling(self) -> None:
        """Verify uint8 [0, 255] images are scaled to [-1.0, 1.0]."""
        raw = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        raw[1] = 255  # second image all white
        out = preprocess_images_for_inception(raw)

        self.assertEqual(out.shape, (2, 299, 299, 3))
        # Image 0 was 0 -> -1.0
        self.assertAlmostEqual(float(tf.reduce_min(out[0])), -1.0, places=4)
        # Image 1 was 255 -> 1.0
        self.assertAlmostEqual(float(tf.reduce_max(out[1])), 1.0, places=4)

    def test_preprocess_float_unit_range(self) -> None:
        """Verify float [0.0, 1.0] images are transformed to [-1.0, 1.0]."""
        raw = np.array([[[[0.0, 0.5, 1.0]]]], dtype=np.float32)
        out = preprocess_images_for_inception(raw)

        self.assertEqual(out.shape, (1, 299, 299, 3))
        self.assertAlmostEqual(float(tf.reduce_min(out)), -1.0, places=4)
        self.assertAlmostEqual(float(tf.reduce_max(out)), 1.0, places=4)

    def test_preprocess_invalid_inputs_raise(self) -> None:
        """Verify malformed or corrupted inputs correctly raise appropriate exceptions."""
        # 1. None input
        with self.assertRaises(TypeError):
            preprocess_images_for_inception(None)

        # 2. Non-tensor / non-array input
        with self.assertRaises(TypeError):
            preprocess_images_for_inception("not_an_image")

        # 3. Empty tensor (0 samples)
        with self.assertRaises(ValueError):
            preprocess_images_for_inception(np.empty((0, 64, 64, 3), dtype=np.float32))

        # 4. Invalid rank (2D or 5D)
        with self.assertRaises(ValueError):
            preprocess_images_for_inception(np.zeros((64, 64), dtype=np.float32))
        with self.assertRaises(ValueError):
            preprocess_images_for_inception(np.zeros((1, 2, 64, 64, 3), dtype=np.float32))

        # 5. Wrong channels (grayscale 1-channel or RGBA 4-channel)
        with self.assertRaises(ValueError):
            preprocess_images_for_inception(np.zeros((2, 64, 64, 1), dtype=np.float32))
        with self.assertRaises(ValueError):
            preprocess_images_for_inception(np.zeros((2, 64, 64, 4), dtype=np.float32))

        # 6. NaN or Inf values
        with self.assertRaises(ValueError):
            nan_arr = np.zeros((2, 64, 64, 3), dtype=np.float32)
            nan_arr[0, 0, 0, 0] = np.nan
            preprocess_images_for_inception(nan_arr)

        with self.assertRaises(ValueError):
            inf_arr = np.zeros((2, 64, 64, 3), dtype=np.float32)
            inf_arr[0, 0, 0, 0] = np.inf
            preprocess_images_for_inception(inf_arr)

    # -------------------------------------------------------------------------
    # 2. Activation Statistics Tests
    # -------------------------------------------------------------------------
    def test_calculate_activation_statistics_valid(self) -> None:
        """Verify mean vector and covariance matrix calculation."""
        features = np.random.randn(20, 16).astype(np.float64)
        mu, sigma = calculate_activation_statistics(features)

        self.assertEqual(mu.shape, (16,))
        self.assertEqual(sigma.shape, (16, 16))
        np.testing.assert_allclose(mu, np.mean(features, axis=0))
        np.testing.assert_allclose(sigma, np.cov(features, rowvar=False))

    def test_calculate_activation_statistics_guards(self) -> None:
        """Verify invalid feature matrices are guarded against."""
        # 1. Fewer than 2 samples cannot compute covariance
        with self.assertRaises(ValueError):
            calculate_activation_statistics(np.random.randn(1, 16))

        # 2. 1D features
        with self.assertRaises(ValueError):
            calculate_activation_statistics(np.random.randn(16))

        # 3. NaNs in features
        nan_feat = np.random.randn(10, 16)
        nan_feat[0, 0] = np.nan
        with self.assertRaises(ValueError):
            calculate_activation_statistics(nan_feat)

    # -------------------------------------------------------------------------
    # 3. Fréchet Distance & Numerical Stabilization Tests
    # -------------------------------------------------------------------------
    def test_frechet_distance_identical_distributions(self) -> None:
        """FID between identical Gaussian distributions must be exactly 0.0."""
        mu = np.random.randn(32)
        sigma = np.diag(np.random.uniform(0.5, 2.0, size=32))

        fid = calculate_frechet_distance(mu, sigma, mu, sigma)
        self.assertAlmostEqual(fid, 0.0, places=5)

    def test_frechet_distance_known_mean_offset(self) -> None:
        """FID with equal covariances and shifted mean should equal ||mu1 - mu2||^2."""
        mu1 = np.zeros(10)
        mu2 = np.ones(10) * 2.0  # diff norm squared = 10 * 4 = 40.0
        sigma = np.eye(10)

        fid = calculate_frechet_distance(mu1, sigma, mu2, sigma)
        self.assertAlmostEqual(fid, 40.0, places=4)

    def test_frechet_distance_singular_covariance_stabilization(self) -> None:
        """Verify singular / rank-deficient covariance matrices are stabilized without crashing."""
        dim = 20
        # Rank-1 covariance matrices (singular)
        v1 = np.random.randn(dim, 1)
        sigma1 = v1.dot(v1.T)
        v2 = np.random.randn(dim, 1)
        sigma2 = v2.dot(v2.T)
        mu1 = np.random.randn(dim)
        mu2 = np.random.randn(dim)

        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
        self.assertTrue(np.isfinite(fid))
        self.assertGreaterEqual(fid, 0.0)

    def test_frechet_distance_complex_result_handling(self) -> None:
        """Verify imaginary components from numerical imprecision are safely stripped."""
        dim = 10
        mu = np.zeros(dim)
        # Asymmetric construct that could create slight imaginary roots
        sigma1 = np.diag(np.linspace(1e-6, 1.0, dim))
        sigma2 = np.diag(np.linspace(1.0, 1e-6, dim))

        fid = calculate_frechet_distance(mu, sigma1, mu, sigma2)
        self.assertIsInstance(fid, float)
        self.assertGreaterEqual(fid, 0.0)

    def test_frechet_distance_invalid_inputs_raise(self) -> None:
        """Verify mismatched shapes and non-finite values in distance calculation raise ValueError."""
        mu1 = np.zeros(5)
        mu2 = np.zeros(6)  # mismatched length
        sigma1 = np.eye(5)
        sigma2 = np.eye(5)

        with self.assertRaises(ValueError):
            calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        # Non-square covariance
        with self.assertRaises(ValueError):
            calculate_frechet_distance(mu1, np.ones((5, 4)), mu1, sigma1)

        # NaN in covariance
        bad_cov = np.eye(5)
        bad_cov[0, 0] = np.nan
        with self.assertRaises(ValueError):
            calculate_frechet_distance(mu1, bad_cov, mu1, sigma1)

    # -------------------------------------------------------------------------
    # 4. Feature Extraction & FIDEvaluator Integration Tests
    # -------------------------------------------------------------------------
    def test_extract_features_with_mock_model(self) -> None:
        """Verify feature extraction produces (N, 2048) array with mock model."""
        images = np.random.uniform(-1.0, 1.0, size=(6, 64, 64, 3)).astype(np.float32)
        features = extract_features(images, model=self.mock_model, batch_size=4)

        self.assertEqual(features.shape, (6, INCEPTION_FEATURE_DIM))
        self.assertTrue(np.isfinite(features).all())

    def test_fid_evaluator_lifecycle(self) -> None:
        """Test FIDEvaluator precomputation, fixed noise, and generator evaluation."""
        # 60 synthetic validation images matching project specification
        val_images = np.random.uniform(-1.0, 1.0, size=(60, 64, 64, 3)).astype(np.float32)

        evaluator = FIDEvaluator(
            val_images=val_images,
            model=self.mock_model,
            num_samples=60,
            latent_dim=LATENT_DIM,
            seed=42,
        )

        # 1. Verify precomputation occurred
        self.assertEqual(evaluator.mu_real.shape, (INCEPTION_FEATURE_DIM,))
        self.assertEqual(evaluator.sigma_real.shape, (INCEPTION_FEATURE_DIM, INCEPTION_FEATURE_DIM))
        self.assertEqual(evaluator.fixed_eval_noise.shape, (60, LATENT_DIM))

        # 2. Evaluate Generator
        generator = build_generator(latent_dim=LATENT_DIM)
        fid_score = evaluator.evaluate(generator)

        self.assertIsInstance(fid_score, float)
        self.assertTrue(np.isfinite(fid_score))
        self.assertGreaterEqual(fid_score, 0.0)

        # 3. Evaluate image batch directly
        fake_images = np.random.uniform(-1.0, 1.0, size=(60, 64, 64, 3)).astype(np.float32)
        fid_images = evaluator.evaluate_images(fake_images)
        self.assertIsInstance(fid_images, float)
        self.assertGreaterEqual(fid_images, 0.0)

    def test_fid_evaluator_missing_file_raises(self) -> None:
        """Verify non-existent val_images path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            FIDEvaluator(
                val_data_path="non_existent_val_images_path.npy",
                model=self.mock_model,
            )

    def test_fid_evaluator_empty_val_images_raises(self) -> None:
        """Verify empty validation images array raises ValueError."""
        with self.assertRaises(ValueError):
            FIDEvaluator(
                val_images=np.empty((0, 64, 64, 3), dtype=np.float32),
                model=self.mock_model,
            )


if __name__ == "__main__":
    unittest.main()
