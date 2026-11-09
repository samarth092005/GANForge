
"""Google Gemini AI integration service for GANForge (Unit U2)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

# Load project-root .env before importing configuration.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from config.settings import GEMINI_MODEL  # noqa: E402

logger = logging.getLogger("GANForge.GeminiService")


class GeminiReportService:
    """Communicates with Google's Gemini API using the official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = self._resolve_api_key(api_key)
        self.model_name = model_name or GEMINI_MODEL
        self._client: Any = None

    @staticmethod
    def _resolve_api_key(explicit_key: Optional[str] = None) -> str:
        """Resolve API key from explicit input, environment, or Streamlit secrets."""

        if explicit_key and explicit_key.strip():
            return explicit_key.strip()

        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key:
            return env_key

        try:
            import streamlit as st

            secret_key = st.secrets.get("GEMINI_API_KEY", "")
            if secret_key:
                return str(secret_key).strip()
        except Exception:
            pass

        raise ValueError(
            "Gemini API key not found. Add GEMINI_API_KEY to the project .env "
            "or configure Streamlit secrets."
        )

    @property
    def client(self) -> Any:
        """Lazily initialize the official Gemini client."""

        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "The official 'google-genai' package is required. "
                    "Install it using: pip install google-genai"
                ) from exc

            self._client = genai.Client(api_key=self.api_key)

        return self._client

    @staticmethod
    def validate_metrics_data(metrics_df: pd.DataFrame) -> None:
        """Validate the training history DataFrame."""

        if metrics_df is None or not isinstance(metrics_df, pd.DataFrame):
            raise TypeError(
                "Training metrics must be provided as a non-None pandas DataFrame."
            )

        if metrics_df.empty:
            raise ValueError("Training metrics DataFrame is empty (0 epochs recorded).")

        required_cols = {
            "epoch",
            "generator_loss",
            "discriminator_loss",
        }

        missing = required_cols - set(metrics_df.columns)

        if missing:
            raise ValueError(
                f"Training metrics DataFrame is missing required columns: "
                f"{sorted(missing)}"
            )

    def generate_report(self, prompt: str) -> str:
        """Generate a Markdown technical report using Gemini."""

        if not prompt or not prompt.strip():
            raise ValueError("Report generation prompt cannot be empty.")

        logger.info(
            "Submitting report generation request to Gemini model: '%s'...",
            self.model_name,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            if not response or not getattr(response, "text", None):
                raise RuntimeError(
                    "Gemini API returned an empty response or invalid payload."
                )

            report_text = response.text.strip()

            logger.info("Successfully received report from Gemini.")
            return report_text

        except Exception as exc:
            logger.exception("Gemini report generation failed.")
            raise RuntimeError(
                f"Gemini API generation failed for model '{self.model_name}'. "
                "Check the API configuration and service logs."
            ) from exc
