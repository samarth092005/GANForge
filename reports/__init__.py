"""Reports compilation and generation package for GANForge."""
from __future__ import annotations

from reports.report_generator import (
    extract_metrics_summary,
    generate_fallback_report,
    prepare_report_prompt,
)

__all__ = [
    "extract_metrics_summary",
    "prepare_report_prompt",
    "generate_fallback_report",
]
