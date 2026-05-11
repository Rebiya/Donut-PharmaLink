"""LLM service using OpenRouter with schema enforcement."""

import json
import logging
import time
from typing import List

import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.models.schemas import PrescriptionResponse

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a medical prescription structuring assistant.

You receive:
- raw OCR text
- normalized drug names

STRICT RULES:
1. NEVER invent or modify drug names
2. ONLY use normalized_drugs list
3. Extract dosage only if explicitly present
4. Output MUST be valid JSON only

OUTPUT:
{
  "confidence_score": float,
  "description": string,
  "drugs": [
    {"name": string, "dosage": string}
  ]
}

RULES:
- description is mandatory
- drugs optional
- dosage optional
- confidence_score must be between 0 and 1

Edge cases:
- unclear text -> low confidence
- no drugs -> omit drugs field
"""


class LLMService:
    """Wrap OpenRouter chat API with schema validation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        headers = {"Content-Type": "application/json"}
        if self.settings.openrouter_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openrouter_api_key}"
        self.client = httpx.Client(
            timeout=httpx.Timeout(20.0, connect=5.0),
            headers=headers,
        )

    def build_payload(self, raw_text: str, normalized_drugs: List[str]) -> str:
        """Construct user payload for model."""
        payload = {
            "raw_ocr_text": raw_text,
            "normalized_drugs": normalized_drugs,
        }
        return json.dumps(payload, ensure_ascii=True)

    def structure_prescription(
        self, raw_text: str, normalized_drugs: List[str]
    ) -> PrescriptionResponse:
        """Request schema-compliant structured JSON and validate it."""
        schema = PrescriptionResponse.model_json_schema()
        t_start = time.perf_counter()
        try:
            if not self.settings.openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter calls")
            payload = {
                "model": self.settings.openrouter_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self.build_payload(raw_text, normalized_drugs),
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "prescription_response",
                        "strict": True,
                        "schema": schema,
                    },
                },
                "temperature": 0,
            }
            content = ""
            for _ in range(2):
                response = self.client.post(
                    self.settings.openrouter_base_url,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                try:
                    parsed = PrescriptionResponse.model_validate_json(content)
                    if self.settings.enable_latency_logging:
                        logger.info(
                            "openrouter_latency_ms total=%.1f",
                            (time.perf_counter() - t_start) * 1000,
                        )
                    return parsed
                except ValidationError:
                    continue

            logger.debug("LLM raw response (invalid schema): %s", content)
            parsed = PrescriptionResponse.model_validate_json(content)
            return parsed
        except ValidationError as exc:
            logger.warning("Invalid LLM JSON, using fallback response: %s", exc)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning("LLM unavailable, using fallback response: %s", exc)

        fallback_drugs = [{"name": name} for name in normalized_drugs] or None
        return PrescriptionResponse(
            confidence_score=0.25 if raw_text.strip() else 0.05,
            description=(
                "Structured output generated with fallback mode because local "
                "LLM response was unavailable or invalid."
            ),
            drugs=fallback_drugs,
        )
