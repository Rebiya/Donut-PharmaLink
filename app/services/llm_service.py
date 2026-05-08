"""LLM service using local Ollama with schema enforcement."""

import json
import logging
from typing import List

from ollama import Client
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
    """Wrap Ollama chat API with schema validation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = Client(host=self.settings.ollama_host)

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
        try:
            response = self.client.chat(
                model=self.settings.ollama_model,
                format=schema,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self.build_payload(raw_text, normalized_drugs),
                    },
                ],
            )
            content = response["message"]["content"]
            logger.debug("LLM raw response: %s", content)
            parsed = PrescriptionResponse.model_validate_json(content)
            return parsed
        except ValidationError as exc:
            logger.warning("Invalid LLM JSON, using fallback response: %s", exc)
        except Exception as exc:  # pragma: no cover - depends on local ollama runtime
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
