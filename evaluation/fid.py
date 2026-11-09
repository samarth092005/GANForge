"""Fréchet Inception Distance (FID) evaluation module for GANForge.

Implements quantitative evaluation of generated image distributions against real validation
data using InceptionV3 pooled feature representations (2048-D).

Theoretical Background:
    The Fréchet Inception Distance (Heusel et al., 2017) measures the Wasserstein-2 distance
    between two multivariate Gaussian distributions fitted to features extracted from a deep
    convolutional network (InceptionV3):
        FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * (Sigma_r * Sigma_g)^(1/2))

Sample Size & Approximation Note:
    Evaluating FID with 60 validation samples (the 20% validation split of the 300-image
    Rangoli dataset) provides a consistent, relative convergence benchmark for monitoring
    DCGAN training progress across epochs. Absolute publication-grade FID benchmarks
    traditionally utilize 10,000 to 50,000 samples. Consequently, the scores reported here
    are approximate distribution indicators optimized for training evaluation.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg
import tensorflow as tf

from config.settings import (
    FID_IMAGE_SIZE,
    LATENT_DIM,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)

logger = logging.getLogger("GANForge.FID")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Module Constants
INCEPTION_INPUT_SHAPE: tuple[int, int, int] = (299, 299, 3)
INCEPTION_FEATURE_DIM: int = 2048
DEFAULT_FID_EPS: float = 1e-6
APPROXIMATE_SAMPLE_NOTE: str = (
    "FID evaluation is computed using 60 validation samples. This provides an "
    "approximate relative benchmark indicator for tracking training progress, "
    "whereas standard absolute FID typically requires 10k-50k samples."
)


def get_inception_feature_extractor(
    weights: Optional[str] = "imagenet",
    input_shape: tuple[int, int, int] = INCEPTION_INPUT_SHAPE,
) -> tf.keras.Model:
    """Instantiates a frozen InceptionV3 model configured for 2048-D feature extraction.

    Args:
        weights: Weights to load ('imagenet' for pre-trained weights, None for uninitialized).
        input_shape: Input image dimensions (defaults to (299, 299, 3)).

    Returns:
        Frozen tf.keras.Model producing 2048-dimensional pooled output features.
    """
    logger.info(f"Loading InceptionV3 feature extractor (weights={weights})...")
    model = tf.keras.applications.InceptionV3(
        include_top=False,
        pooling="avg",
        weights=weights,
        input_shape=input_shape,
    )
    model.trainable = False
    return model


def preprocess_images_for_inception(images: Any) -> tf.Tensor:
    """Validates, resizes, and scales images to [-1.0, 1.0] for InceptionV3 inference.

    - Validates image dimensions, channels, and numeric finiteness.
    - Resizes to (299, 299) using bilinear interpolation.
    - Accurately normalizes pixels to [-1.0, 1.0] matching InceptionV3 specifications.

    Args:
        images: Input images as numpy.ndarray or tf.Tensor. Accepts shapes (N, H, W, 3)
                or (H, W, 3). Pixel values may be uint8 [0, 255], float [0, 1],
                float [0, 255], or float [-1, 1].

    Returns:
        tf.Tensor with shape (N, 299, 299, 3), dtype tf.float32, with values in [-1.0, 1.0].

    Raises:
        TypeError: If input is None or not convertible to a tensor.
        ValueError: If tensor has invalid dimensions, channel count != 3, is empty,
                    or contains NaN or Inf values.
    """
    if images is None:
        raise TypeError("Input images cannot be None.")

    if not isinstance(images, (np.ndarray, tf.Tensor, list, tuple)):
        raise TypeError(f"Expected numpy ndarray or tf.Tensor, got {type(images)}.")

    tensor = tf.convert_to_tensor(images)

    # Check rank / dimensions
    ndims = tensor.shape.ndims
    if ndims == 3:
        tensor = tf.expand_dims(tensor, axis=0)
    elif ndims != 4:
        raise ValueError(f"Expected 3D (H, W, C) or 4D (N, H, W, C) image tensor, got shape {tensor.shape}.")

    # Check sample count
    batch_count = tensor.shape[0]
    if batch_count is not None and batch_count == 0:
        raise ValueError("Image tensor cannot be empty (0 samples).")

    # Check channels (must be RGB 3 channels)
    channels = tensor.shape[-1]
    if channels != 3:
        raise ValueError(f"Expected 3 color channels (RGB), got shape {tensor.shape}.")

    # Determine input type & normalize to [-1.0, 1.0]
    orig_dtype = getattr(images, "dtype", None)
    is_integer_dtype = (
        orig_dtype in (np.uint8, np.int16, np.int32, tf.uint8, tf.int32)
        or (isinstance(orig_dtype, np.dtype) and np.issubdtype(orig_dtype, np.integer))
        or tensor.dtype in (tf.uint8, tf.int16, tf.int32, tf.int64)
    )

    tensor = tf.cast(tensor, tf.float32)

    # Validate numerical finiteness
    if not tf.reduce_all(tf.math.is_finite(tensor)):
        raise ValueError("Input images contain non-finite (NaN or Inf) values.")

    if is_integer_dtype:
        # Standard uint8 [0, 255] -> [-1, 1]
        tensor = (tensor / 127.5) - 1.0
    else:
        min_val = float(tf.reduce_min(tensor))
        max_val = float(tf.reduce_max(tensor))

        if max_val > 1.0:
            # Float in [0, 255]
            tensor = (tensor / 127.5) - 1.0
        elif min_val >= 0.0 and max_val <= 1.0:
            # Float in [0.0, 1.0]
            tensor = (tensor * 2.0) - 1.0
        else:
            # Already in [-1.0, 1.0] (e.g. GANForge standard output or train/val arrays)
            tensor = tf.clip_by_value(tensor, -1.0, 1.0)

    # Resize to InceptionV3 required resolution (299, 299)
    current_h, current_w = tensor.shape[1], tensor.shape[2]
    if (current_h, current_w) != FID_IMAGE_SIZE:
        tensor = tf.image.resize(tensor, FID_IMAGE_SIZE, method="bilinear")
        tensor = tf.clip_by_value(tensor, -1.0, 1.0)

    return tensor


def extract_features(
    images: Any,
    model: tf.keras.Model,
    batch_size: int = 32,
) -> np.ndarray:
    """Extracts 2048-dimensional pooled features from images using InceptionV3.

    Args:
        images: Raw or preprocessed image batch (ndarray or Tensor).
        model: InceptionV3 feature extractor model.
        batch_size: Mini-batch size for inference to conserve memory.

    Returns:
        numpy ndarray of shape (N, 2048) containing activation vectors.

    Raises:
        ValueError: If extracted features contain non-finite numbers.
    """
    preprocessed = preprocess_images_for_inception(images)
    total_images = int(tf.shape(preprocessed)[0])

    feature_batches: list[np.ndarray] = []
    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        batch = preprocessed[start_idx:end_idx]
        features = model(batch, training=False)
        feature_batches.append(features.numpy())

    all_features = np.concatenate(feature_batches, axis=0)

    if not np.isfinite(all_features).all():
        raise ValueError("Extracted feature representations contain NaN or Inf values.")

    return all_features


def calculate_activation_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes the mean vector and covariance matrix of feature activations.

    Args:
        features: 2D numpy array of shape (N, D) containing activation vectors.

    Returns:
        Tuple of (mu, sigma):
            mu: Mean vector of shape (D,).
            sigma: Sample covariance matrix of shape (D, D).

    Raises:
        ValueError: If features is not 2-dimensional, has fewer than 2 samples,
                    or contains non-finite numbers.
    """
    if not isinstance(features, np.ndarray):
        features = np.asarray(features)

    if features.ndim != 2:
        raise ValueError(
            f"Activation features must be a 2D array of shape (N, D), got shape {features.shape}."
        )

    num_samples = features.shape[0]
    if num_samples < 2:
        raise ValueError(
            f"At least 2 samples required to compute sample covariance, got {num_samples}."
        )

    if not np.isfinite(features).all():
        raise ValueError("Activation features contain NaN or Inf values.")

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = DEFAULT_FID_EPS,
) -> float:
    """Calculates the Fréchet Inception Distance between two multivariate Gaussians.

    Implements:
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))

    Includes numerical stabilization:
    - Automatically adds diagonal regularization (eps * I) if matrices are singular
      or if direct matrix square root yields non-finite values.
    - Strips imaginary round-off components from matrix square root.
    - Clamps near-zero negative values caused by floating-point imprecision.

    Args:
        mu1: Mean vector of first distribution (shape [D,]).
        sigma1: Covariance matrix of first distribution (shape [D, D]).
        mu2: Mean vector of second distribution (shape [D,]).
        sigma2: Covariance matrix of second distribution (shape [D, D]).
        eps: Small numerical stabilization offset added to covariance diagonal if needed.

    Returns:
        Calculated Fréchet Inception Distance as a non-negative float.

    Raises:
        ValueError: If input dimensions are inconsistent or contain non-finite values.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vectors have mismatched shapes: {mu1.shape} vs {mu2.shape}.")
    if sigma1.shape != sigma2.shape:
        raise ValueError(
            f"Covariance matrices have mismatched shapes: {sigma1.shape} vs {sigma2.shape}."
        )
    if sigma1.shape[0] != sigma1.shape[1]:
        raise ValueError(f"Covariance matrix 1 must be square, got {sigma1.shape}.")
    if sigma2.shape[0] != sigma2.shape[1]:
        raise ValueError(f"Covariance matrix 2 must be square, got {sigma2.shape}.")
    if mu1.shape[0] != sigma1.shape[0]:
        raise ValueError(
            f"Mean length ({mu1.shape[0]}) does not match covariance dimension ({sigma1.shape[0]})."
        )
    if not np.isfinite(mu1).all() or not np.isfinite(mu2).all():
        raise ValueError("Mean vectors contain NaN or Inf values.")
    if not np.isfinite(sigma1).all() or not np.isfinite(sigma2).all():
        raise ValueError("Covariance matrices contain NaN or Inf values.")

    diff = mu1 - mu2

    # Product of covariance matrices
    cov_prod = sigma1.dot(sigma2)

    # Compute matrix square root with stabilization fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.linalg.LinAlgWarning)
        try:
            covmean = scipy.linalg.sqrtm(cov_prod)
            if isinstance(covmean, tuple):
                covmean = covmean[0]
        except Exception as err:
            logger.warning(f"Direct matrix square root failed ({err}); applying eps-stabilization.")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            if isinstance(covmean, tuple):
                covmean = covmean[0]

    # Check for NaN / Inf in covmean and stabilize if needed
    if not np.isfinite(covmean).all():
        logger.warning("Matrix square root produced non-finite values; adding eps to diagonal.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]

    # Numerical precision error can produce slight imaginary components
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            max_imag = float(np.max(np.abs(covmean.imag)))
            logger.warning(f"Noticeable imaginary component in sqrtm ({max_imag:.4e}); discarding imaginary part.")
        covmean = covmean.real

    tr_covmean = float(np.trace(covmean))

    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)

    # Floating point precision guard: clamp small negative round-off to 0.0
    if fid < 0.0:
        if fid > -1e-4:
            fid = 0.0
        else:
            logger.warning(f"Negative FID value ({fid:.6f}) clamped to 0.0.")
            fid = 0.0

    return float(fid)


class FIDEvaluator:
    """Manages precomputation and per-epoch evaluation of Fréchet Inception Distance.

    Optimized Evaluation Design:
    1. InceptionV3 is instantiated only once upon evaluator construction.
    2. Activation statistics (mu_real, sigma_real) for the 60 validation images are
       precomputed ONCE before training starts, eliminating redundant feature extraction.
    3. Strictly references real VALIDATION images (val_images.npy), never training data.
    4. Evaluates a reproducible fixed set of 60 latent samples across epochs.
    5. Safely handles covariance matrix stabilization and numerical round-off.

    Note on Sample Size:
        Evaluating FID with 60 validation samples provides an approximate benchmark indicator.
        Standard FID typically uses 10,000 to 50,000 samples for absolute publication rankings.
        The 60-sample evaluation is optimized for fast tracking of relative convergence dynamics
        against the fixed 60-image validation split.
    """

    def __init__(
        self,
        val_images: Optional[np.ndarray] = None,
        val_data_path: Optional[Path | str] = None,
        model: Optional[tf.keras.Model] = None,
        num_samples: int = 60,
        latent_dim: int = LATENT_DIM,
        seed: int = RANDOM_SEED,
        batch_size: int = 32,
        eps: float = DEFAULT_FID_EPS,
    ) -> None:
        """Initializes the FID evaluator, precomputes validation statistics, and prepares fixed noise.

        Args:
            val_images: Optional pre-loaded validation image array.
            val_data_path: Path to val_images.npy (defaults to PROCESSED_DATA_DIR / val_images.npy).
            model: Optional pre-constructed InceptionV3 model (for unit testing or custom backbones).
            num_samples: Number of evaluation samples to generate (default 60 matching validation set).
            latent_dim: Dimensionality of latent vector z (default 100).
            seed: Reproducible seed for fixed evaluation noise.
            batch_size: Batch size for feature extraction inference.
            eps: Covariance stabilization offset.
        """
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.seed = seed
        self.batch_size = batch_size
        self.eps = eps

        # 1. Load Real Validation Reference
        if val_images is not None:
            self.val_images = val_images
        else:
            path = Path(val_data_path) if val_data_path else PROCESSED_DATA_DIR / "val_images.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Validation images array not found at: {path}\n"
                    "FID evaluation requires validation images as the real reference.\n"
                    "Please run: python -m data.prepare_rangoli_dataset"
                )
            self.val_images = np.load(path)

        if len(self.val_images) == 0:
            raise ValueError("Validation images array is empty; cannot evaluate FID.")

        logger.info(
            f"FIDEvaluator initialized with {len(self.val_images)} validation reference images "
            f"[num_samples={self.num_samples}, seed={self.seed}]."
        )
        logger.info(APPROXIMATE_SAMPLE_NOTE)

        # 2. Initialize InceptionV3 Model Once
        if model is not None:
            self.model = model
            logger.info("FIDEvaluator using provided feature extractor model.")
        else:
            self.model = get_inception_feature_extractor(weights="imagenet")

        # 3. Precompute Real Validation Features & Statistics Once
        logger.info("Precomputing real validation activation statistics...")
        self.real_features = extract_features(
            self.val_images, model=self.model, batch_size=self.batch_size
        )
        self.mu_real, self.sigma_real = calculate_activation_statistics(self.real_features)
        logger.info("Real validation activation statistics precomputed successfully.")

        # 4. Generate Fixed Evaluation Noise (60 samples)
        tf.random.set_seed(self.seed)
        self.fixed_eval_noise = tf.random.normal([self.num_samples, self.latent_dim], seed=self.seed)

    def evaluate(self, generator: tf.keras.Model) -> float:
        """Evaluates the Generator model on the fixed latent samples against precomputed real stats.

        Args:
            generator: The DCGAN Generator model.

        Returns:
            Calculated Fréchet Inception Distance score as a float.
        """
        # Synthesize fake images in inference mode
        fake_images = generator(self.fixed_eval_noise, training=False)
        return self.evaluate_images(fake_images)

    def evaluate_images(self, fake_images: Any) -> float:
        """Evaluates a batch of generated images against precomputed real validation stats.

        Args:
            fake_images: Array or Tensor of generated images.

        Returns:
            Calculated Fréchet Inception Distance score as a float.
        """
        fake_features = extract_features(
            fake_images, model=self.model, batch_size=self.batch_size
        )
        mu_fake, sigma_fake = calculate_activation_statistics(fake_features)
        fid = calculate_frechet_distance(
            self.mu_real, self.sigma_real, mu_fake, sigma_fake, eps=self.eps
        )
        return float(fid)
