"""Donut OCR service."""

import logging
import time
from io import BytesIO

import torch
from PIL import Image, ImageOps
from transformers import DonutProcessor, VisionEncoderDecoderModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class OCRService:
    """Loads and runs the Donut OCR model."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if self.settings.device_preference == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif self.settings.device_preference == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor: DonutProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None

    def _load_once(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        logger.info("Loading Donut model on %s", self.device)
        if not self.settings.donut_model_path.exists():
            raise FileNotFoundError(
                f"Local Donut model path not found: {self.settings.donut_model_path}"
            )
        self._processor = DonutProcessor.from_pretrained(
            self.settings.donut_model_path,
            local_files_only=self.settings.local_files_only,
        )
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.settings.donut_model_path,
            local_files_only=self.settings.local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()

    def extract_text(self, image_bytes: bytes) -> str:
        """Extract text from image using Donut OCR."""
        req_start = time.perf_counter()
        self._load_once()
        assert self._processor is not None
        assert self._model is not None

        t_decode = time.perf_counter()
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        decode_ms = (time.perf_counter() - t_decode) * 1000

        t_pre = time.perf_counter()
        encoding = self._processor(images=image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(self.device)
        pre_ms = (time.perf_counter() - t_pre) * 1000

        t_gen = time.perf_counter()
        with torch.inference_mode():
            generated = self._model.generate(
                pixel_values,
                decoder_start_token_id=self._processor.tokenizer.convert_tokens_to_ids(
                    "<s_ocr>"
                ),
                max_length=self.settings.max_generate_tokens,
                num_beams=1,
                early_stopping=True,
            )
        gen_ms = (time.perf_counter() - t_gen) * 1000

        t_post = time.perf_counter()
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        decode_text_ms = (time.perf_counter() - t_post) * 1000

        if self.settings.enable_latency_logging:
            total_ms = (time.perf_counter() - req_start) * 1000
            logger.info(
                "ocr_latency_ms decode=%.1f preprocess=%.1f generate=%.1f decode_text=%.1f total=%.1f",
                decode_ms,
                pre_ms,
                gen_ms,
                decode_text_ms,
                total_ms,
            )
        return text.strip()
