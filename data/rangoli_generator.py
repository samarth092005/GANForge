"""Procedural Rangoli pattern generator for GANForge demonstration and benchmarking.

Generates diverse traditional Indian geometric Rangoli patterns utilizing:
- Rotational radial symmetries (4, 6, 8, 12-fold)
- Concentric circular harmonics & petal lobes
- Harmonious traditional color palettes (turmeric yellow, vermilion red, peacock green, indigo blue)
- Kolam dot-grid structures

All generated images are released under CC-BY 4.0 with explicit algorithmic provenance.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TRADITIONAL_PALETTES = [
    # Warm Festive: Vermilion, Turmeric, Marigold, Deep Red
    ["#D32F2F", "#F57C00", "#FFD54F", "#B71C1C", "#FFF9C4"],
    # Peacock Elegance: Royal Blue, Cyan, Emerald, Gold
    ["#1A237E", "#00838F", "#00E676", "#FFD700", "#E8EAF6"],
    # Floral Blossom: Magenta, Rose, Coral, Soft Amber
    ["#880E4F", "#E91E63", "#FF80AB", "#FFA726", "#FCE4EC"],
    # Earth Kolam: Terracotta, Rice Chalk White, Ochre, Sand
    ["#BF360C", "#FFFFFF", "#FFB300", "#8D6E63", "#3E2723"],
    # Vibrant Rainbow Festive
    ["#C2185B", "#7B1FA2", "#512DA8", "#303F9F", "#00796B", "#FBC02D"],
]


def generate_single_rangoli(
    seed: int,
    resolution: int = 256,
) -> Image.Image:
    """Procedurally generates a single artistic Rangoli pattern.

    Args:
        seed: Random seed for pattern generation.
        resolution: Canvas resolution (resolution x resolution).

    Returns:
        PIL Image in RGB format.
    """
    rng = np.random.default_rng(seed)

    # Choose symmetry order (4, 6, 8, 12, or 16)
    symmetry = int(rng.choice([6, 8, 12, 16]))
    palette = TRADITIONAL_PALETTES[int(rng.integers(0, len(TRADITIONAL_PALETTES)))]
    bg_color = palette[-1]
    fg_colors = palette[:-1]

    fig, ax = plt.subplots(
        figsize=(resolution / 100.0, resolution / 100.0), dpi=100
    )
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Parametric angle sweep
    theta = np.linspace(0, 2 * np.pi, 1200)

    # Generate concentric layers
    num_layers = int(rng.integers(3, 7))
    for layer_idx in range(num_layers):
        color = fg_colors[layer_idx % len(fg_colors)]
        freq = symmetry * int(rng.choice([1, 2]))
        base_radius = 0.15 + (layer_idx * 0.13)
        amplitude = 0.04 + rng.uniform(0.01, 0.05)

        # Petal modulation
        r = base_radius + amplitude * np.sin(freq * theta + rng.uniform(0, np.pi))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        alpha = float(rng.uniform(0.7, 0.95))
        lw = float(rng.uniform(1.2, 2.5))

        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)

        # Occasional fill
        if rng.random() > 0.5:
            ax.fill(x, y, color=color, alpha=0.15)

    # Add radial rays / dots
    num_dots = symmetry * int(rng.choice([1, 2]))
    dot_radius = 0.2 + (num_layers * 0.1)
    dot_angles = np.linspace(0, 2 * np.pi, num_dots, endpoint=False)
    dot_color = fg_colors[0]

    for d_theta in dot_angles:
        dx = dot_radius * np.cos(d_theta)
        dy = dot_radius * np.sin(d_theta)
        ax.plot(
            dx,
            dy,
            marker="o",
            markersize=rng.uniform(3, 6),
            color=dot_color,
            alpha=0.85,
        )

    # Center motif
    center_color = fg_colors[int(rng.integers(0, len(fg_colors)))]
    center_radius = float(rng.uniform(0.06, 0.12))
    center_circle = plt.Circle(
        (0, 0), center_radius, color=center_color, alpha=0.9
    )
    ax.add_patch(center_circle)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    # Convert canvas to PIL Image
    rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
    pil_img = Image.fromarray(rgba_buffer).convert("RGB")
    plt.close(fig)

    return pil_img


def generate_rangoli_dataset(
    output_dir: Path | str,
    count: int = 300,
    start_seed: int = 1000,
    resolution: int = 128,
) -> list[Path]:
    """Generates a complete benchmark dataset of Rangoli images.

    Args:
        output_dir: Destination folder for images.
        count: Number of images to generate (default 300).
        start_seed: Initial random seed.
        resolution: Image dimension.

    Returns:
        List of generated image paths.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    for i in range(count):
        seed = start_seed + i
        img = generate_single_rangoli(seed=seed, resolution=resolution)
        # Alternate between png and jpg to test multiple formats
        ext = "png" if (i % 2 == 0) else "jpg"
        file_path = dest / f"rangoli_{i+1:04d}.{ext}"
        if ext == "jpg":
            img.save(file_path, "JPEG", quality=95)
        else:
            img.save(file_path, "PNG")
        generated_paths.append(file_path)

    return generated_paths
