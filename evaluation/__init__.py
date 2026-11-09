"""Evaluation module for GANForge.

Provides quantitative evaluation metrics including Fréchet Inception Distance (FID).
"""
from evaluation.fid import (
    FIDEvaluator,
    calculate_activation_statistics,
    calculate_frechet_distance,
    extract_features,
    get_inception_feature_extractor,
    preprocess_images_for_inception,
)

__all__ = [
    "FIDEvaluator",
    "calculate_activation_statistics",
    "calculate_frechet_distance",
    "extract_features",
    "get_inception_feature_extractor",
    "preprocess_images_for_inception",
]
