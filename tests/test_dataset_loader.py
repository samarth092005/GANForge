"""Unit and integration tests for GANForge dataset pipeline and metadata tracking."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data.dataset_loader import (
    DatasetPipeline,
    DatasetStats,
    create_preview_grid,
    denormalize_images,
    load_processed_data,
    normalize_image,
    split_dataset,
    validate_and_load_image,
)
from data.metadata import DatasetMetadata


class TestDatasetPipeline(unittest.TestCase):
    """Test suite verifying validation, preprocessing, consent metadata, and splitting."""

    def setUp(self) -> None:
        """Create a temporary sandbox directory for test artifacts."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="ganforge_test_"))
        self.raw_dir = self.test_dir / "raw"
        self.processed_dir = self.test_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up temporary test artifacts."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_sample_image(
        self,
        filename: str,
        size: tuple[int, int] = (100, 100),
        mode: str = "RGB",
        color: tuple = (255, 0, 0),
    ) -> Path:
        """Helper to create synthetic test images."""
        filepath = self.raw_dir / filename
        img = Image.new(mode, size, color)
        img.save(filepath)
        return filepath

    # ---------------------------------------------------------
    # 1. Normalization & Denormalization Tests
    # ---------------------------------------------------------
    def test_normalization_range_and_dtype(self) -> None:
        """Verify uint8 [0, 255] maps precisely to float32 [-1.0, 1.0]."""
        raw_pixels = np.array([0, 127, 255], dtype=np.uint8).reshape((1, 1, 3))
        normalized = normalize_image(raw_pixels)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized[0, 0, 0]), -1.0, places=4)
        self.assertAlmostEqual(float(normalized[0, 0, 2]), 1.0, places=4)
        self.assertTrue(normalized.min() >= -1.0)
        self.assertTrue(normalized.max() <= 1.0)

    def test_denormalization_reversibility(self) -> None:
        """Verify normalized float [-1.0, 1.0] reversibly recovers uint8 [0, 255]."""
        original = np.array([[[[0, 128, 255]]]], dtype=np.uint8)
        normalized = normalize_image(original)
        recovered = denormalize_images(normalized)

        self.assertEqual(recovered.dtype, np.uint8)
        self.assertEqual(int(recovered[0, 0, 0, 0]), 0)
        self.assertEqual(int(recovered[0, 0, 0, 2]), 255)
        # 128 maps to (128/127.5)-1 = 0.00392, and ((0.00392+1)*127.5) = 128
        self.assertEqual(int(recovered[0, 0, 0, 1]), 128)

    # ---------------------------------------------------------
    # 2. Image Loading, RGB Conversion & Resizing Tests
    # ---------------------------------------------------------
    def test_valid_rgb_image_loading_and_resizing(self) -> None:
        """Verify a standard RGB JPEG is resized to (64, 64, 3)."""
        file_path = self._create_sample_image("valid.jpg", size=(120, 90), mode="RGB")
        arr, reason = validate_and_load_image(file_path, target_size=(64, 64))

        self.assertIsNone(reason)
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (64, 64, 3))
        self.assertEqual(arr.dtype, np.uint8)

    def test_rgba_to_rgb_conversion(self) -> None:
        """Verify RGBA PNG with transparency is converted to 3-channel RGB."""
        file_path = self._create_sample_image(
            "transparent.png", size=(80, 80), mode="RGBA", color=(0, 255, 0, 128)
        )
        arr, reason = validate_and_load_image(file_path, target_size=(64, 64))

        self.assertIsNone(reason)
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (64, 64, 3))

    def test_grayscale_to_rgb_conversion(self) -> None:
        """Verify grayscale 'L' mode image is converted to 3-channel RGB."""
        file_path = self._create_sample_image(
            "gray.png", size=(75, 75), mode="L", color=200
        )
        arr, reason = validate_and_load_image(file_path, target_size=(64, 64))

        self.assertIsNone(reason)
        self.assertIsNotNone(arr)
        self.assertEqual(arr.shape, (64, 64, 3))
        # Channels should match for grayscale converted to RGB
        self.assertEqual(arr[0, 0, 0], arr[0, 0, 1])
        self.assertEqual(arr[0, 0, 1], arr[0, 0, 2])

    # ---------------------------------------------------------
    # 3. Corrupted & Invalid File Handling
    # ---------------------------------------------------------
    def test_rejection_of_empty_zero_byte_file(self) -> None:
        """Verify 0-byte file is caught and rejected."""
        empty_path = self.raw_dir / "empty.jpg"
        empty_path.touch()
        arr, reason = validate_and_load_image(empty_path)

        self.assertIsNone(arr)
        self.assertEqual(reason, "empty_or_missing_file")

    def test_rejection_of_corrupt_non_image_bytes(self) -> None:
        """Verify file with .png extension containing garbage text is rejected."""
        fake_png = self.raw_dir / "fake.png"
        fake_png.write_text("This is not a real PNG image content.")
        arr, reason = validate_and_load_image(fake_png)

        self.assertIsNone(arr)
        self.assertTrue("corrupted" in str(reason))

    def test_rejection_of_unsupported_extension(self) -> None:
        """Verify unsupported formats (e.g. .txt, .pdf) are rejected."""
        text_file = self.raw_dir / "notes.txt"
        text_file.write_text("dataset notes")
        arr, reason = validate_and_load_image(text_file)

        self.assertIsNone(arr)
        self.assertEqual(reason, "unsupported_extension_.txt")

    # ---------------------------------------------------------
    # 4. Train / Validation Split Reproducibility
    # ---------------------------------------------------------
    def test_split_dataset_ratio_and_seed(self) -> None:
        """Verify 80/20 train-val split counts and seed reproducibility."""
        dummy_data = np.zeros((10, 64, 64, 3), dtype=np.float32)
        train_a, val_a = split_dataset(dummy_data, train_ratio=0.8, seed=42)
        train_b, val_b = split_dataset(dummy_data, train_ratio=0.8, seed=42)

        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(val_a), 2)
        # Check seed reproducibility
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)

    def test_split_dataset_small_sample_edge_case(self) -> None:
        """Verify split on minimal 2-sample dataset allocates 1 train, 1 val."""
        dummy_data = np.zeros((2, 64, 64, 3), dtype=np.float32)
        train_set, val_set = split_dataset(dummy_data, train_ratio=0.8, seed=42)

        self.assertEqual(len(train_set), 1)
        self.assertEqual(len(val_set), 1)

    # ---------------------------------------------------------
    # 5. Consent and Licence Metadata Tracking
    # ---------------------------------------------------------
    def test_metadata_consent_validation_enforcement(self) -> None:
        """Verify ValueError is raised if consent is not confirmed."""
        unconsented = DatasetMetadata(
            dataset_name="Test Rangoli",
            source="Personal Camera",
            license="CC-BY 4.0",
            consent_confirmed=False,  # NOT confirmed
        )
        with self.assertRaises(ValueError) as ctx:
            unconsented.validate_consent()
        self.assertIn("Consent confirmation is required", str(ctx.exception))

    def test_metadata_save_and_load(self) -> None:
        """Verify valid metadata is correctly saved and loaded as JSON."""
        meta = DatasetMetadata(
            dataset_name="Rangoli Traditional Benchmark",
            source="Algorithmic Generator / Wikimedia CC",
            license="Creative Commons Attribution 4.0",
            consent_confirmed=True,
            source_url="https://example.org/rangoli",
            consent_notes="Verified public domain / algorithmic creation.",
            total_images_declared=300,
        )
        meta_file = self.processed_dir / "dataset_metadata.json"
        meta.save(meta_file)

        self.assertTrue(meta_file.exists())
        loaded = DatasetMetadata.load(meta_file)
        self.assertEqual(loaded.dataset_name, meta.dataset_name)
        self.assertEqual(loaded.license, meta.license)
        self.assertTrue(loaded.consent_confirmed)

    # ---------------------------------------------------------
    # 6. End-to-End Pipeline Execution
    # ---------------------------------------------------------
    def test_end_to_end_pipeline_execution(self) -> None:
        """Run full pipeline on mixed folder of valid, corrupt, and unsupported files."""
        # 1. Create 8 valid images
        for i in range(8):
            ext = "png" if i % 2 == 0 else "jpg"
            color = (i * 25, 100, 200)
            self._create_sample_image(
                f"valid_{i:02d}.{ext}", size=(80, 80), color=color
            )

        # 2. Add 1 corrupt file and 1 unsupported file
        (self.raw_dir / "corrupted.jpg").write_bytes(b"garbage bytes")
        (self.raw_dir / "readme.txt").write_text("irrelevant text")

        meta = DatasetMetadata(
            dataset_name="Test Suite Dataset",
            source="Synthetic Unit Test",
            license="CC0",
            consent_confirmed=True,
        )

        pipeline = DatasetPipeline(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            target_size=(64, 64),
            train_ratio=0.75,
            random_seed=123,
        )

        train_imgs, val_imgs, stats = pipeline.run(
            metadata=meta,
            save_preview=True,
            preview_path=self.processed_dir / "test_preview.png",
        )

        # 3. Assert stats
        self.assertEqual(stats.total_scanned, 10)
        self.assertEqual(stats.valid_count, 8)
        self.assertEqual(stats.rejected_count, 2)
        self.assertEqual(stats.train_count, 6)  # 75% of 8
        self.assertEqual(stats.val_count, 2)  # 25% of 8
        self.assertEqual(train_imgs.shape, (6, 64, 64, 3))
        self.assertEqual(val_imgs.shape, (2, 64, 64, 3))
        self.assertTrue(stats.pixel_min >= -1.0)
        self.assertTrue(stats.pixel_max <= 1.0)

        # 4. Assert saved files
        self.assertTrue((self.processed_dir / "train_images.npy").exists())
        self.assertTrue((self.processed_dir / "val_images.npy").exists())
        self.assertTrue((self.processed_dir / "dataset_stats.json").exists())
        self.assertTrue((self.processed_dir / "dataset_metadata.json").exists())
        self.assertTrue((self.processed_dir / "test_preview.png").exists())

        # 5. Test loader function
        t_loaded, v_loaded, s_loaded = load_processed_data(self.processed_dir)
        self.assertEqual(t_loaded.shape, train_imgs.shape)
        self.assertEqual(v_loaded.shape, val_imgs.shape)
        self.assertEqual(s_loaded.valid_count, 8)


if __name__ == "__main__":
    unittest.main()
