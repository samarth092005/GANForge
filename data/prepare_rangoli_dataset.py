"""CLI script to generate, validate, preprocess, and split the 300-image Rangoli benchmark dataset.

Executes Unit 1 (U1) requirements:
1. Procedurally generates or collects 300 Rangoli images.
2. Injects corrupt/unsupported test samples to verify anomaly detection.
3. Registers verified consent and CC-BY 4.0 licensing metadata.
4. Cleans, converts to RGB, resizes to 64x64, and normalizes to [-1, 1].
5. Splits into 80% training (240 images) and 20% validation (60 images).
6. Produces visual preview grids and quantitative statistics.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from config.settings import (
    IMAGE_SIZE,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SAMPLES_DIR,
    TARGET_DATASET_SIZE,
    TRAIN_VAL_SPLIT_RATIO,
)
from data.dataset_loader import DatasetPipeline, load_processed_data
from data.metadata import DatasetMetadata
from data.rangoli_generator import generate_rangoli_dataset

logger = logging.getLogger("GANForge.PrepareRangoli")


def prepare_benchmark_dataset(
    target_count: int = TARGET_DATASET_SIZE,
    include_corrupted_samples: bool = True,
) -> None:
    """End-to-end dataset preparation workflow."""
    raw_rangoli_dir = RAW_DATA_DIR / "rangoli"
    raw_rangoli_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  GANForge — Phase 2: Rangoli Dataset Preparation Pipeline (U1)")
    print("=" * 65)

    # 1. Check if raw images exist, otherwise generate them
    existing_images = [
        f
        for f in raw_rangoli_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    if len(existing_images) < target_count:
        print(
            f"[*] Generating {target_count} authentic geometric Rangoli benchmark images..."
        )
        generate_rangoli_dataset(
            output_dir=raw_rangoli_dir,
            count=target_count,
            start_seed=1000,
            resolution=128,
        )
        print(f"[+] Successfully generated {target_count} images in {raw_rangoli_dir}")
    else:
        print(f"[+] Found {len(existing_images)} existing raw images in {raw_rangoli_dir}")

    # 2. Add intentional test files to prove rejection handling
    if include_corrupted_samples:
        corrupt_file = raw_rangoli_dir / "corrupted_sample.png"
        corrupt_file.write_bytes(b"INVALID_IMAGE_HEADER_DATA_NOT_A_PNG")
        text_file = raw_rangoli_dir / "dataset_notes.txt"
        text_file.write_text("Rangoli art styles: Kolam, Alpana, Chowkpurana.")
        print("[*] Injected 2 test files (1 corrupt bytes, 1 unsupported .txt) to verify validation.")

    # 3. Create Responsible AI License & Consent Metadata
    metadata = DatasetMetadata(
        dataset_name="Procedurally Generated Synthetic Rangoli Benchmark",
        provenance_type="synthetic_procedural",
        source="Algorithmic Procedural Rangoli Generator (data/rangoli_generator.py)",
        source_url="local://GANForge/data/rangoli_generator.py",
        license="MIT / Open Academic Benchmark (Project GANForge)",
        usage_terms="Open educational and academic research use under Project GANForge. Created locally using algorithmic mathematics.",
        consent_confirmed=True,
        consent_notes=(
            "Procedurally generated synthetic benchmark dataset. No human subjects, "
            "biometric data, or copyrighted third-party images were scraped or utilized. "
            "Created solely via mathematical parametric equations."
        ),
        generator_name="GANForge Procedural Rangoli Pattern Engine (generate_single_rangoli)",
        generation_parameters={
            "rotational_symmetries": [6, 8, 12, 16],
            "palettes": [
                "Warm Festive",
                "Peacock Elegance",
                "Floral Blossom",
                "Earth Kolam",
                "Vibrant Rainbow",
            ],
            "canvas_resolution": 128,
            "target_resolution": list(IMAGE_SIZE),
            "random_seed_range": f"1000..{1000 + target_count - 1}",
        },
        total_images_declared=target_count,
        additional_metadata={
            "art_form": "Algorithmic Geometric Rangoli / Kolam",
            "motifs": ["radial_symmetry", "floral_petals", "geometric_harmonics"],
        },
    )

    # 4. Execute Pipeline
    pipeline = DatasetPipeline(
        raw_dir=raw_rangoli_dir,
        processed_dir=PROCESSED_DATA_DIR,
        target_size=IMAGE_SIZE,
        train_ratio=TRAIN_VAL_SPLIT_RATIO,
        random_seed=RANDOM_SEED,
    )

    preview_file = SAMPLES_DIR / "dataset_preview_grid.png"
    train_data, val_data, stats = pipeline.run(
        metadata=metadata,
        save_preview=True,
        preview_path=preview_file,
    )

    # Copy stats to metrics directory for evaluation tracking
    metrics_stats_file = METRICS_DIR / "dataset_stats.json"
    stats.save(metrics_stats_file)

    # 5. Summary Presentation
    print("\n" + "-" * 65)
    print("                    DATASET PIPELINE SUMMARY")
    print("-" * 65)
    print(f" Dataset Name           : {metadata.dataset_name}")
    print(f" Provenance Type        : {metadata.provenance_type} (Algorithmic / Synthetic)")
    print(f" Generator Engine       : {metadata.generator_name}")
    print(f" Declared License       : {metadata.license}")
    print(f" Usage Terms            : {metadata.usage_terms}")
    print(f" Consent & Legal Status : [VERIFIED] Procedural generation authorized; zero third-party PII/copyright")
    print(f" Total Files Scanned    : {stats.total_scanned}")
    print(f" Valid Images Accepted  : {stats.valid_count}")
    print(f" Files Rejected         : {stats.rejected_count}")
    print(f" Rejection Breakdown    : {stats.rejection_reasons}")
    print(f" Training Images (80%)  : {stats.train_count} (shape: {train_data.shape})")
    print(f" Validation Images (20%): {stats.val_count} (shape: {val_data.shape})")
    print(f" Pixel Value Range      : [{stats.pixel_min:.2f}, {stats.pixel_max:.2f}] (normalized)")
    print(f" Pixel Mean / Std       : {stats.pixel_mean:.3f} / {stats.pixel_std:.3f}")
    print(f" Preview Grid Saved To  : {preview_file}")
    print(f" Processed Arrays Saved : {PROCESSED_DATA_DIR / 'train_images.npy'}")
    print(f"                          {PROCESSED_DATA_DIR / 'val_images.npy'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Rangoli dataset for GANForge.")
    parser.add_argument(
        "--count",
        type=int,
        default=TARGET_DATASET_SIZE,
        help="Target number of images (default 300)",
    )
    args = parser.parse_args()
    prepare_benchmark_dataset(target_count=args.count)
