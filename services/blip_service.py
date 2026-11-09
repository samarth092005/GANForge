
"""BLIP Image Captioning service for GANForge (Unit U3)."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from config.settings import BLIP_MODEL_ID

logger = logging.getLogger("GANForge.BLIPService")

_GLOBAL_BLIP_SERVICE: Optional["BlipCaptionService"] = None


class BlipCaptionService:
    """Manages lazy-loading and inference for the BLIP image captioning model."""

    def __init__(self, model_id: str = BLIP_MODEL_ID) -> None:
        self.model_id = model_id
        self._processor: Any = None
        self._model: Any = None
        self._device: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load_model(self, device: Optional[str] = None) -> None:
        """Load BLIP model lazily."""

        if self.is_loaded:
            return

        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise ImportError(
                "BLIP requires torch and transformers."
            ) from exc

        self._device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            "Loading BLIP model '%s' on %s",
            self.model_id,
            self._device,
        )

        try:
            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_id
            )
            self._model.to(self._device)
            self._model.eval()

            logger.info("BLIP model loaded successfully.")

        except Exception as exc:
            self._processor = None
            self._model = None
            self._device = None
            logger.exception("BLIP model loading failed.")
            raise RuntimeError("Unable to load BLIP model.") from exc

    @staticmethod
    def validate_and_load_image(
        image_input: Union[Image.Image, Path, str, bytes, np.ndarray]
    ) -> Image.Image:
        """Convert supported image inputs into RGB PIL images."""

        if image_input is None:
            raise TypeError("Image input cannot be None.")

        if isinstance(image_input, Image.Image):
            if image_input.width == 0 or image_input.height == 0:
                raise ValueError("Image has zero dimensions.")
            return image_input.convert("RGB")

        if isinstance(image_input, (Path, str)):
            path = Path(image_input)

            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")

            try:
                with Image.open(path) as img:
                    img.load()
                    return img.convert("RGB")
            except Exception as exc:
                raise ValueError("Unable to read image file.") from exc

        if isinstance(image_input, (bytes, bytearray)):
            if not image_input:
                raise ValueError("Image byte stream is empty.")

            try:
                with Image.open(io.BytesIO(image_input)) as img:
                    img.load()
                    return img.convert("RGB")
            except Exception as exc:
                raise ValueError("Unable to decode image bytes.") from exc

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("NumPy image array is empty.")

            if image_input.ndim == 2:
                arr = image_input

                if arr.dtype != np.uint8:
                    if arr.min() >= 0 and arr.max() <= 1:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

                return Image.fromarray(arr).convert("RGB")

            elif image_input.ndim == 3:
                _, _, channels = image_input.shape

                if channels not in (1, 3, 4):
                    raise ValueError(
                        f"Unsupported image channel count: {channels}"
                    )

                arr = image_input

                if arr.dtype != np.uint8:
                    arr = arr.astype(np.float32)

                    if arr.min() >= -1.05 and arr.max() <= 1.05 and arr.min() < 0:
                        arr = (arr + 1.0) * 127.5
                    elif arr.min() >= 0 and arr.max() <= 1.05:
                        arr = arr * 255.0

                    arr = np.clip(arr, 0, 255).astype(np.uint8)

                return Image.fromarray(arr).convert("RGB")

            else:
                raise ValueError(
                    f"Expected 2D or 3D NumPy array, got {image_input.ndim}D."
                )

        raise TypeError(
            f"Unsupported image input type: {type(image_input).__name__}"
        )

    def generate_caption(
        self,
        image_input: Union[Image.Image, Path, str, bytes, np.ndarray],
        prompt: Optional[str] = None,
        max_new_tokens: int = 40,
    ) -> str:
        """Generate an image caption using BLIP."""

        pil_img = self.validate_and_load_image(image_input)

        if not self.is_loaded:
            self.load_model()

        import torch

        try:
            if prompt:
                inputs = self._processor(
                    images=pil_img,
                    text=prompt,
                    return_tensors="pt",
                )
            else:
                inputs = self._processor(
                    images=pil_img,
                    return_tensors="pt",
                )

            inputs = {
                key: value.to(self._device)
                for key, value in inputs.items()
            }

            self._model.eval()

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            caption = self._processor.decode(
                output_ids[0],
                skip_special_tokens=True,
            ).strip()

            if not caption:
                raise RuntimeError("BLIP returned an empty caption.")

            return caption[0].upper() + caption[1:]

        except Exception as exc:
            logger.exception("BLIP caption generation failed.")
            raise RuntimeError(
                "BLIP could not generate a caption. Check inference logs."
            ) from exc


def get_blip_service(
    model_id: str = BLIP_MODEL_ID,
) -> BlipCaptionService:
    """Return the cached singleton BLIP service."""

    global _GLOBAL_BLIP_SERVICE

    if (
        _GLOBAL_BLIP_SERVICE is None
        or _GLOBAL_BLIP_SERVICE.model_id != model_id
    ):
        _GLOBAL_BLIP_SERVICE = BlipCaptionService(model_id=model_id)

    return _GLOBAL_BLIP_SERVICE
