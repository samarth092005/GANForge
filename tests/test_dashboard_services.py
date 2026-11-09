"""Unit tests for GANForge Streamlit Dashboard services and report generation.

Verifies:
- Metrics CSV loading and validation
- Missing metrics and corrupt CSV handling
- Best FID calculation and epoch association
- BLIP input validation (types, channels, empty buffers, paths)
- Gemini missing API key handling and validation
- Report prompt and fallback deterministic generation
- Dashboard startup resilience without training artifacts
"""
from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image

from app import (
    find_sample_grid_path,
    find_training_history_csv,
    load_dataset_metadata,
    load_dataset_stats,
    load_training_history,
)
from reports.report_generator import (
    extract_metrics_summary,
    generate_fallback_report,
    prepare_report_prompt,
)
from services.blip_service import BlipCaptionService, get_blip_service
from services.gemini_service import GeminiReportService


class TestDashboardServices(unittest.TestCase):
    """Test suite covering dashboard data resolution, services, and reporting."""

    def setUp(self) -> None:
        """Create temporary test directory sandbox."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="ganforge_dashboard_test_"))
        self.metrics_dir = self.test_dir / "metrics"
        self.samples_dir = self.test_dir / "samples"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up temporary test artifacts."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_sample_csv(self, filename: str = "training_history.csv") -> Path:
        """Create a synthetic training_history.csv for testing."""
        csv_path = self.metrics_dir / filename
        data = {
            "epoch": [1, 10, 25, 50],
            "generator_loss": [1.85, 2.45, 3.10, 3.82],
            "discriminator_loss": [1.15, 0.72, 0.65, 0.58],
            "disc_real_loss": [0.65, 0.42, 0.38, 0.32],
            "disc_fake_loss": [0.50, 0.30, 0.27, 0.26],
            "fid_score": [480.25, 420.50, 395.10, 361.1206],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    # ==========================================================================
    # 1. Metrics CSV Loading & Missing Metrics
    # ==========================================================================

    def test_metrics_csv_loading_valid(self) -> None:
        """Verify that a valid training_history.csv is parsed and sorted correctly."""
        csv_p = self._create_sample_csv()
        df = load_training_history(csv_p)

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 4)
        self.assertListEqual(list(df["epoch"]), [1, 10, 25, 50])
        self.assertIn("generator_loss", df.columns)
        self.assertIn("discriminator_loss", df.columns)
        self.assertIn("fid_score", df.columns)

    def test_missing_metrics_handling(self) -> None:
        """Verify that None, missing files, or corrupt CSVs return None gracefully."""
        self.assertIsNone(load_training_history(None))
        self.assertIsNone(load_training_history(self.test_dir / "non_existent.csv"))

        # Corrupted CSV (missing required columns)
        bad_csv = self.metrics_dir / "bad_history.csv"
        bad_csv.write_text("some_col,another_col\n1,2\n", encoding="utf-8")
        self.assertIsNone(load_training_history(bad_csv))

    # ==========================================================================
    # 2. Best FID Calculation
    # ==========================================================================

    def test_best_fid_calculation(self) -> None:
        """Verify minimum FID and corresponding epoch are precisely identified."""
        csv_p = self._create_sample_csv()
        df = pd.read_csv(csv_p)
        summary = extract_metrics_summary(df)

        self.assertTrue(summary["has_fid"])
        self.assertAlmostEqual(summary["final_fid"], 361.1206, places=4)
        self.assertAlmostEqual(summary["best_fid"], 361.1206, places=4)
        self.assertEqual(summary["best_fid_epoch"], 50)
        self.assertEqual(summary["total_epochs"], 4)

    def test_metrics_summary_without_fid(self) -> None:
        """Verify summary generation when fid_score column is absent or all NaN."""
        df = pd.DataFrame({
            "epoch": [1, 2],
            "generator_loss": [1.5, 2.0],
            "discriminator_loss": [1.0, 0.8],
        })
        summary = extract_metrics_summary(df)
        self.assertFalse(summary["has_fid"])
        self.assertIsNone(summary["best_fid"])
        self.assertIsNone(summary["best_fid_epoch"])

    # ==========================================================================
    # 3. BLIP Input Validation
    # ==========================================================================

    def test_blip_input_validation(self) -> None:
        """Verify BLIP image validation rejects invalid inputs and accepts valid images."""
        service = get_blip_service()

        # Invalid: None
        with self.assertRaises(TypeError):
            service.validate_and_load_image(None)

        # Invalid: unsupported type
        with self.assertRaises(TypeError):
            service.validate_and_load_image(12345)  # type: ignore

        # Invalid: empty byte string
        with self.assertRaises(ValueError):
            service.validate_and_load_image(b"")

        # Invalid: non-existent file
        with self.assertRaises(FileNotFoundError):
            service.validate_and_load_image(self.test_dir / "does_not_exist.png")

        # Invalid: NumPy array with bad channels
        bad_arr = np.zeros((32, 32, 7), dtype=np.uint8)
        with self.assertRaises(ValueError):
            service.validate_and_load_image(bad_arr)

        # Valid: PIL Image
        pil_img = Image.new("RGB", (64, 64), color=(255, 128, 0))
        res_pil = service.validate_and_load_image(pil_img)
        self.assertEqual(res_pil.size, (64, 64))
        self.assertEqual(res_pil.mode, "RGB")

        # Valid: NumPy array in [-1.0, 1.0] (GANForge standard)
        norm_arr = np.zeros((64, 64, 3), dtype=np.float32)
        res_arr = service.validate_and_load_image(norm_arr)
        self.assertEqual(res_arr.size, (64, 64))
        self.assertEqual(res_arr.mode, "RGB")

        # Valid: Saved Image Path
        img_path = self.samples_dir / "test_sample.png"
        pil_img.save(img_path)
        res_path = service.validate_and_load_image(img_path)
        self.assertEqual(res_path.size, (64, 64))

    # ==========================================================================
    # 4. Gemini Missing API Key & Input Validation
    # ==========================================================================

    def test_gemini_missing_api_key_handling(self) -> None:
        """Verify GeminiReportService raises clear ValueError if API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                GeminiReportService(api_key="")
            self.assertIn("Gemini API key is required", str(ctx.exception))

    def test_report_generation_input_validation(self) -> None:
        """Verify input validation on metrics DataFrame and empty prompts."""
        # Test validate_metrics_data
        with self.assertRaises(TypeError):
            GeminiReportService.validate_metrics_data(None)  # type: ignore

        with self.assertRaises(ValueError):
            GeminiReportService.validate_metrics_data(pd.DataFrame())

        with self.assertRaises(ValueError):
            GeminiReportService.validate_metrics_data(pd.DataFrame({"epoch": [1]}))

        # Test generate_report with empty prompt
        svc = GeminiReportService(api_key="mock_test_key")
        with self.assertRaises(ValueError):
            svc.generate_report("")

    # ==========================================================================
    # 5. Gemini Mocked Generation & Prompt Construction
    # ==========================================================================

    def test_prepare_report_prompt_structure(self) -> None:
        """Verify that the generated prompt includes critical caveats and metrics."""
        csv_p = self._create_sample_csv()
        df = pd.read_csv(csv_p)
        prompt = prepare_report_prompt(
            metrics_df=df,
            dataset_metadata={"dataset_name": "Test Rangoli", "license": "MIT"},
            dataset_stats={"train_count": 240, "val_count": 60, "valid_count": 300},
        )

        self.assertIn("P40", prompt)
        self.assertIn("60 real validation images", prompt)
        self.assertIn("approximate relative tracking indicator", prompt)
        self.assertIn("361.1206", prompt)
        self.assertIn("DO NOT claim CC-BY 4.0", prompt)
        self.assertIn("blurry", prompt)

    @patch("google.genai.Client")
    def test_gemini_mocked_generation(self, mock_client_cls: MagicMock) -> None:
        """Verify Gemini service sends prompt via SDK and extracts response.text."""
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        mock_response = MagicMock()
        mock_response.text = "# Academic Report\n\nGenerated content from Gemini."
        mock_instance.models.generate_content.return_value = mock_response

        svc = GeminiReportService(api_key="mock_key", model_name="gemini-2.5-flash")
        result = svc.generate_report("Generate technical report.")

        self.assertIn("# Academic Report", result)
        mock_instance.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash",
            contents="Generate technical report.",
        )

    # ==========================================================================
    # 6. Dashboard Startup Without Training Artifacts & Fallback Report
    # ==========================================================================

    def test_dashboard_startup_without_training_artifacts(self) -> None:
        """Verify dashboard data resolution executes safely when artifacts are missing."""
        empty_sandbox = self.test_dir / "empty_root"
        empty_sandbox.mkdir(parents=True, exist_ok=True)

        # File lookups return None without crashing
        self.assertIsNone(find_training_history_csv(str(empty_sandbox)))
        self.assertIsNone(find_sample_grid_path(50, str(empty_sandbox)))

        # Metadata loaders return non-empty default fallbacks
        meta = load_dataset_metadata()
        self.assertIsInstance(meta, dict)
        self.assertIn("dataset_name", meta)

        stats = load_dataset_stats()
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["valid_count"], 300)

    def test_deterministic_fallback_report_compilation(self) -> None:
        """Verify offline fallback report produces complete, publication-grade Markdown."""
        csv_p = self._create_sample_csv()
        df = pd.read_csv(csv_p)
        report = generate_fallback_report(
            metrics_df=df,
            dataset_metadata={"dataset_name": "Rangoli Benchmark", "license": "MIT"},
            dataset_stats={"train_count": 240, "val_count": 60, "valid_count": 300},
            sample_captions={"Epoch 50 Grid": "A circular geometric design with colorful patterns."},
        )

        self.assertIn("# GANForge: Custom DCGAN Fine-Tuning and Automated Technical Report", report)
        self.assertIn("Academic Project ID: P40", report)
        self.assertIn("361.1206", report)
        self.assertIn("approximate internal tracking indicator", report)
        self.assertIn("blurry", report)
        self.assertIn("Responsible AI", report)
        self.assertIn("does not claim Creative Commons Attribution (CC-BY 4.0)", report)


if __name__ == "__main__":
    unittest.main()
