"""Donut OCR service."""

import logging
from io import BytesIO

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OCRService:
    """Loads and runs the Donut OCR model."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor: DonutProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None

    def _load_once(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        logger.info("Loading Donut model on %s", self.device)
        self._processor = DonutProcessor.from_pretrained(self.settings.donut_model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.settings.donut_model_id
        )
        self._model.to(self.device)
        self._model.eval()

    def extract_text(self, image_bytes: bytes) -> str:
        """Extract text from image using Donut OCR."""
        self._load_once()
        assert self._processor is not None
        assert self._model is not None

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        encoding = self._processor(images=image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(self.device)

        with torch.no_grad():
            generated = self._model.generate(
                pixel_values,
                decoder_start_token_id=self._processor.tokenizer.convert_tokens_to_ids(
                    "<s_ocr>"
                ),
                max_length=self.settings.max_generate_tokens,
            )

        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()
