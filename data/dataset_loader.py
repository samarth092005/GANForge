"""Dataset loading, validation, preprocessing, and splitting pipeline for GANForge.

Supports JPG, JPEG, and PNG image formats, validates file integrity,
converts to RGB, resizes to target resolution (64x64), normalizes to [-1, 1],
and splits into train/validation sets with reproducible seeding.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config.settings import (
    CHANNELS,
    IMAGE_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SAMPLES_DIR,
    TARGET_DATASET_SIZE,
    TRAIN_VAL_SPLIT_RATIO,
)
from data.metadata import DatasetMetadata

# Configure module logger
logger = logging.getLogger("GANForge.DatasetLoader")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass
class DatasetStats:
    """Stores quantitative statistics and metrics about processed dataset."""

    total_scanned: int
    valid_count: int
    rejected_count: int
    rejection_reasons: dict[str, int]
    train_count: int
    val_count: int
    image_shape: tuple[int, int, int]
    pixel_min: float
    pixel_max: float
    pixel_mean: float
    pixel_std: float
    split_ratio: float
    random_seed: int
    processed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary representation."""
        return asdict(self)

    def save(self, filepath: Path | str) -> Path:
        """Save statistics to JSON file."""
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return dest

    @classmethod
    def load(cls, filepath: Path | str) -> DatasetStats:
        """Load statistics from JSON file."""
        src = Path(filepath)
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            total_scanned=data["total_scanned"],
            valid_count=data["valid_count"],
            rejected_count=data["rejected_count"],
            rejection_reasons=data["rejection_reasons"],
            train_count=data["train_count"],
            val_count=data["val_count"],
            image_shape=tuple(data["image_shape"]),  # type: ignore
            pixel_min=float(data["pixel_min"]),
            pixel_max=float(data["pixel_max"]),
            pixel_mean=float(data["pixel_mean"]),
            pixel_std=float(data["pixel_std"]),
            split_ratio=float(data["split_ratio"]),
            random_seed=int(data["random_seed"]),
            processed_at=data.get("processed_at", ""),
        )


def denormalize_images(images: np.ndarray) -> np.ndarray:
    """Converts images from normalized [-1, 1] range back to [0, 255] uint8.

    Args:
        images: Array of normalized image floats with values in [-1, 1].

    Returns:
        Array of images with values in [0, 255] as np.uint8.
    """
    denorm = ((images + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return denorm


def normalize_image(image_arr: np.ndarray) -> np.ndarray:
    """Normalizes uint8 [0, 255] image array to float32 [-1, 1].

    Args:
        image_arr: Array of uint8 pixel values.

    Returns:
        Array of float32 values scaled to [-1.0, 1.0].
    """
    return (image_arr.astype(np.float32) / 127.5) - 1.0


def validate_and_load_image(
    file_path: Path | str,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Validates an image file, checks integrity, converts to RGB, and resizes.

    Args:
        file_path: Path to the target image file.
        target_size: Target (width, height) tuple for resizing.

    Returns:
        Tuple of (processed_numpy_array, rejection_reason).
        If valid, array is returned and rejection_reason is None.
        If invalid, array is None and rejection_reason describes the failure.
    """
    path = Path(file_path)

    # 1. Check extension
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return None, f"unsupported_extension_{path.suffix.lower()}"

    # 2. Check file existence and non-zero size
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None, "empty_or_missing_file"
    except OSError as err:
        return None, f"os_error_{err.__class__.__name__}"

    # 3. Open and verify image structure integrity
    try:
        with Image.open(path) as img:
            # Force verification of internal data streams
            img.verify()

        # Reopen after verify() because verify() closes/invalidates the image object
        with Image.open(path) as img:
            # Handle EXIF rotation if present
            img = ImageOps.exif_transpose(img)

            # Convert to RGB (handles RGBA, Palette, Grayscale)
            if img.mode != "RGB":
                if img.mode in ("RGBA", "LA") or (
                    img.mode == "P" and "transparency" in img.info
                ):
                    # Composite transparent images over a solid white background
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    alpha_src = img.convert("RGBA")
                    bg.paste(alpha_src, mask=alpha_src.split()[-1])
                    img = bg
                else:
                    img = img.convert("RGB")

            # Resize to target 64x64 using high-quality anti-aliasing
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            arr = np.array(img, dtype=np.uint8)

            # Validate final shape
            expected_shape = (target_size[1], target_size[0], 3)
            if arr.shape != expected_shape:
                return None, f"unexpected_shape_{arr.shape}"

            return arr, None

    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as err:
        return None, f"corrupted_{err.__class__.__name__}"


def create_preview_grid(
    images: np.ndarray,
    output_path: Path | str,
    grid_size: tuple[int, int] = (4, 4),
    title: str = "GANForge - Processed Image Preview (64x64, Normalized)",
) -> Path:
    """Generates and saves a grid visualization of processed images.

    Args:
        images: Array of images in [-1, 1] or [0, 255].
        output_path: Target path where image will be saved.
        grid_size: (rows, cols) of the preview grid.
        title: Title of the visualization.

    Returns:
        Path to the saved grid image.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = grid_size
    num_to_show = min(len(images), rows * cols)

    if num_to_show == 0:
        raise ValueError("Cannot create preview grid from empty image array.")

    # Determine if denormalization is needed
    if np.issubdtype(images.dtype, np.floating) and images.min() < 0:
        display_images = denormalize_images(images[:num_to_show])
    else:
        display_images = images[:num_to_show].astype(np.uint8)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx < num_to_show:
            ax.imshow(display_images[idx])
            ax.set_title(f"#{idx + 1}", fontsize=8)
        ax.axis("off")

    plt.suptitle(title, fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved preview grid to: {dest}")
    return dest


def split_dataset(
    images: np.ndarray,
    train_ratio: float = TRAIN_VAL_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits an image array into training and validation subsets reproducibly.

    Args:
        images: Numpy array of images [N, H, W, C].
        train_ratio: Proportion of images allocated to training (default 0.8).
        seed: Random seed for shuffling.

    Returns:
        Tuple of (train_images, val_images).
    """
    total = len(images)
    if total < 2:
        raise ValueError(
            f"Dataset contains {total} images. Minimum 2 images required to perform a train/val split."
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    train_count = int(np.floor(total * train_ratio))
    # Ensure at least 1 validation sample if total >= 2
    if train_count == total:
        train_count = total - 1
    if train_count == 0:
        train_count = 1

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    train_set = images[train_indices]
    val_set = images[val_indices]

    logger.info(
        f"Split {total} images -> Train: {len(train_set)} ({train_ratio*100:.1f}%), "
        f"Val: {len(val_set)} ({(1 - train_ratio)*100:.1f}%) [Seed={seed}]"
    )
    return train_set, val_set


class DatasetPipeline:
    """End-to-end dataset preparation pipeline."""

    def __init__(
        self,
        raw_dir: Path | str = RAW_DATA_DIR,
        processed_dir: Path | str = PROCESSED_DATA_DIR,
        target_size: tuple[int, int] = IMAGE_SIZE,
        train_ratio: float = TRAIN_VAL_SPLIT_RATIO,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.random_seed = random_seed

        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        metadata: Optional[DatasetMetadata] = None,
        save_preview: bool = True,
        preview_path: Optional[Path | str] = None,
    ) -> tuple[np.ndarray, np.ndarray, DatasetStats]:
        """Executes the dataset ingestion, validation, normalization, and splitting.

        Args:
            metadata: DatasetMetadata object recording licensing and consent.
            save_preview: Whether to save a preview grid of processed images.
            preview_path: Optional explicit path for the preview grid.

        Returns:
            Tuple of (train_images, val_images, DatasetStats).
        """
        # 1. Enforce responsible AI consent verification
        if metadata is not None:
            metadata.validate_consent()
            metadata_dest = self.processed_dir / "dataset_metadata.json"
            metadata.save(metadata_dest)
            logger.info(f"Verified and recorded consent metadata at: {metadata_dest}")
        else:
            logger.warning(
                "No DatasetMetadata supplied. Ensure consent and licensing are recorded before model deployment."
            )

        # 2. Check raw directory
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"Raw dataset directory does not exist: {self.raw_dir}"
            )

        # 3. Scan files
        all_candidates = [
            p
            for p in self.raw_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ]
        total_scanned = len(all_candidates)
        logger.info(f"Scanning {total_scanned} files in: {self.raw_dir}")

        if total_scanned == 0:
            raise ValueError(
                f"No files found in {self.raw_dir}. Please place images (.jpg, .jpeg, .png) in this folder."
            )

        valid_arrays: list[np.ndarray] = []
        rejection_reasons: dict[str, int] = {}

        for file_path in all_candidates:
            arr, reason = validate_and_load_image(file_path, self.target_size)
            if reason is not None:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                logger.debug(f"Rejected '{file_path.name}': {reason}")
            else:
                valid_arrays.append(arr)  # type: ignore

        valid_count = len(valid_arrays)
        rejected_count = sum(rejection_reasons.values())

        logger.info(
            f"Validation complete: {valid_count} accepted, {rejected_count} rejected "
            f"out of {total_scanned} total scanned."
        )

        if valid_count == 0:
            raise ValueError(
                f"All {total_scanned} scanned files were invalid or rejected. "
                f"Reasons: {rejection_reasons}"
            )

        # 4. Stack and Normalize to [-1.0, 1.0]
        raw_stack = np.stack(valid_arrays, axis=0)  # [N, 64, 64, 3], uint8
        normalized_data = normalize_image(raw_stack)  # [N, 64, 64, 3], float32

        # 5. Split into Train and Validation
        train_data, val_data = split_dataset(
            normalized_data,
            train_ratio=self.train_ratio,
            seed=self.random_seed,
        )

        # 6. Compute statistics
        stats = DatasetStats(
            total_scanned=total_scanned,
            valid_count=valid_count,
            rejected_count=rejected_count,
            rejection_reasons=rejection_reasons,
            train_count=len(train_data),
            val_count=len(val_data),
            image_shape=(self.target_size[1], self.target_size[0], CHANNELS),
            pixel_min=float(normalized_data.min()),
            pixel_max=float(normalized_data.max()),
            pixel_mean=float(normalized_data.mean()),
            pixel_std=float(normalized_data.std()),
            split_ratio=self.train_ratio,
            random_seed=self.random_seed,
        )

        # 7. Persist outputs
        train_path = self.processed_dir / "train_images.npy"
        val_path = self.processed_dir / "val_images.npy"
        stats_path = self.processed_dir / "dataset_stats.json"

        np.save(train_path, train_data)
        np.save(val_path, val_data)
        stats.save(stats_path)

        logger.info(f"Saved processed train images to: {train_path} {train_data.shape}")
        logger.info(f"Saved processed val images to: {val_path} {val_data.shape}")
        logger.info(f"Saved dataset statistics to: {stats_path}")

        # 8. Save preview grid
        if save_preview:
            grid_out = (
                Path(preview_path)
                if preview_path
                else SAMPLES_DIR / "dataset_preview_grid.png"
            )
            create_preview_grid(
                train_data,
                output_path=grid_out,
                grid_size=(4, 4),
                title=f"GANForge Dataset Preview (Train Set, N={len(train_data)})",
            )

        return train_data, val_data, stats


def load_processed_data(
    processed_dir: Path | str = PROCESSED_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, DatasetStats]:
    """Loads pre-processed numpy arrays and stats from the processed directory.

    Args:
        processed_dir: Directory containing train_images.npy and val_images.npy.

    Returns:
        Tuple of (train_images, val_images, stats).
    """
    p_dir = Path(processed_dir)
    train_path = p_dir / "train_images.npy"
    val_path = p_dir / "val_images.npy"
    stats_path = p_dir / "dataset_stats.json"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Processed dataset arrays not found in {p_dir}. "
            "Please run DatasetPipeline.run() first."
        )

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    stats = DatasetStats.load(stats_path) if stats_path.exists() else None  # type: ignore

    return train_data, val_data, stats  # type: ignore
