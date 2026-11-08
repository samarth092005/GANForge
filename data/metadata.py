"""Dataset metadata, consent recording, and provenance tracking for GANForge.

Ensures responsible AI standards by verifying image consent, dataset licensing,
and provenance documentation before training data can be approved.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class DatasetMetadata:
    """Stores provenance, consent, and licensing information for an uploaded dataset."""

    dataset_name: str
    source: str
    license: str
    consent_confirmed: bool
    source_url: str = ""
    consent_notes: str = ""
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_images_declared: int = 0
    additional_metadata: dict[str, Any] = field(default_factory=dict)

    def validate_consent(self) -> None:
        """Validates that explicit consent has been confirmed and license is declared.

        Raises:
            ValueError: If consent is not confirmed or mandatory fields are missing.
        """
        if not self.dataset_name or not self.dataset_name.strip():
            raise ValueError("Dataset name cannot be empty.")
        if not self.source or not self.source.strip():
            raise ValueError("Dataset source must be documented.")
        if not self.license or not self.license.strip():
            raise ValueError("Dataset license information must be explicitly declared.")
        if not self.consent_confirmed:
            raise ValueError(
                "Responsible AI Violation: Consent confirmation is required. "
                "You must verify that you possess the necessary rights or permission "
                "to process and fine-tune models on these images."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)

    def save(self, filepath: Path | str) -> Path:
        """Save metadata as a formatted JSON file.

        Args:
            filepath: Destination JSON file path.

        Returns:
            Path to the saved JSON file.
        """
        self.validate_consent()
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return dest

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetMetadata:
        """Create a DatasetMetadata instance from a dictionary."""
        return cls(
            dataset_name=data.get("dataset_name", "Unknown Dataset"),
            source=data.get("source", "Unknown Source"),
            license=data.get("license", "Unspecified"),
            consent_confirmed=bool(data.get("consent_confirmed", False)),
            source_url=data.get("source_url", ""),
            consent_notes=data.get("consent_notes", ""),
            registered_at=data.get(
                "registered_at", datetime.now(timezone.utc).isoformat()
            ),
            total_images_declared=int(data.get("total_images_declared", 0)),
            additional_metadata=data.get("additional_metadata", {}),
        )

    @classmethod
    def load(cls, filepath: Path | str) -> DatasetMetadata:
        """Load metadata from a JSON file.

        Args:
            filepath: Path to the JSON metadata file.

        Returns:
            DatasetMetadata instance.
        """
        src = Path(filepath)
        if not src.exists():
            raise FileNotFoundError(f"Metadata file not found: {src}")
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
