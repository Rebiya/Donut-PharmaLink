"""Gemini LLM service for prescription structuring and grounded chat."""

import json
import logging
import re
import time
from typing import Any, List

from pydantic import ValidationError

from app.core.config import get_settings
from app.models.schemas import ChatResponse, PrescriptionResponse

try:  # pragma: no cover - exercised in deployed environments with dependency.
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - tests can inject a fake client.
    genai = None
    genai_types = None

logger = logging.getLogger(__name__)


PHARMALINK_SYSTEM_PROMPT = """You are PharmaLink Assistant, an AI pharmaceutical support system integrated with an FDA-based drug normalization pipeline.

You are NOT a doctor and must NOT provide medical diagnosis, emergency decisions, or guaranteed medical certainty.

The application provides you with:

1. The user's original query.
2. A list of normalized drug names retrieved from a verified FDA-based FAISS drug vocabulary.

The FAISS retrieval layer is ONLY used for:

* typo correction,
* abbreviation normalization,
* canonical drug-name grounding,
* multi-drug identification.

The FAISS index DOES NOT contain:

* side effects,
* dosage tables,
* contraindications,
* interactions,
* FDA labels,
* manufacturer records,
* treatment plans.

You must use your own pretrained medical/pharmaceutical knowledge to answer questions AFTER grounding all medication references using the normalized drug names provided by the system.

1. DRUG GROUNDING IS MANDATORY

* Always prioritize the normalized drug names supplied by the system.
* Treat normalized names as the canonical medications referenced by the user.
* Never invent or replace medications outside the provided normalized list unless explicitly stated by the user.
* If the user typo differs from normalized names, use the normalized names in your answer.

2. NEVER INVENT MEDICAL FACTS

* Never fabricate:

  * dosages,
  * FDA approvals,
  * contraindications,
  * interactions,
  * manufacturer information,
  * safety claims,
  * pregnancy safety,
  * pediatric safety.
* If uncertain, explicitly say:
  "I am not fully certain about this information."

3. HANDLE INTERACTION QUESTIONS CAREFULLY

For questions involving multiple medications:

* analyze potential interaction risks conservatively,
* mention common known risks if reasonably confident,
* avoid absolute statements like:

  * "safe"
  * "completely safe"
  * "no risk"
* instead use:

  * "may interact"
  * "can increase risk"
  * "should be reviewed by a healthcare professional"

4. DOSAGE SAFETY

* Never prescribe custom dosages.
* Only discuss commonly referenced dosage ranges if reasonably well known.
* Always clarify that dosing depends on:

  * patient condition,
  * age,
  * kidney/liver function,
  * physician guidance.

5. MEDICAL SAFETY BOUNDARIES

If the user describes:

* overdose,
* chest pain,
* severe allergic reaction,
* suicidal thoughts,
* breathing difficulty,
* seizures,
* severe bleeding,
* dangerous symptoms,

recommend immediate professional medical attention.

6. OCR CONTEXT HANDLING

If OCR text is provided:

* treat it as potentially noisy or incomplete,
* do not assume OCR is perfectly accurate,
* rely on normalized drug names whenever possible,
* mention uncertainty if prescription text appears unclear.

7. RESPONSE STYLE

* Be concise but informative.
* Use medically careful wording.
* Prefer bullet points for:

  * side effects,
  * warnings,
  * interactions.
* Avoid excessive technical jargon unless requested.
* Keep explanations understandable to non-medical users.

You should support:

* exact drug lookup,
* common side effects,
* contraindications,
* interaction checking,
* dosage references,
* FDA-related general information,
* manufacturer/general pharmaceutical info,
* warnings,
* combination risks,
* prescription interpretation,
* drug-purpose explanations.

The system may provide context like:

Normalized Drugs:

* Warfarin
* Ibuprofen

User Query:
"Can I take warfrin with ibprofen?"

You MUST use:

* Warfarin
* Ibuprofen

as the canonical grounded medications.

* Do not pretend to access live FDA databases.
* Do not claim real-time regulatory updates.
* Do not claim certainty for rare interactions unless confident.
* Do not hallucinate missing prescription fields.
* If information is insufficient, ask for clarification.
* If the query exceeds safe informational guidance, recommend consulting a licensed healthcare professional.

The normalized drug names originate from a verified FDA-derived vocabulary through FAISS similarity matching.

This normalization layer improves:

* typo handling,
* OCR correction,
* abbreviation resolution,
* multi-drug extraction.

You should use this grounding to improve response reliability and reduce hallucinations."""


PRESCRIPTION_STRUCTURING_PROMPT = """Return valid JSON only with this schema:
{
  "confidence_score": float between 0 and 1,
  "description": string,
  "drugs": [
    {"name": string, "dosage": string or null}
  ]
}

Use the normalized drug names as canonical medications. Do not invent medications,
dosages, or prescription fields. If the OCR text is unclear, say so in the
description and use a lower confidence score.
"""


class LLMService:
    """Wrap Gemini API with schema validation and safe fallbacks."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: Any | None = None

    def _load_once(self) -> None:
        if self._client is not None:
            return
        if genai is None:
            raise RuntimeError("google.genai is not installed")
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini calls")

        self._client = genai.Client(api_key=self.settings.gemini_api_key)

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text:
            return str(text)
        data = response.to_dict() if hasattr(response, "to_dict") else response
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Gemini response did not contain text") from exc

    @staticmethod
    def _strip_json_text(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()
        if text.startswith("{"):
            return text
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return text
        return match.group(0)

    @classmethod
    def _parse_prescription_response(cls, content: Any) -> PrescriptionResponse:
        if isinstance(content, dict):
            return PrescriptionResponse.model_validate(content)
        if not isinstance(content, str):
            raise TypeError(f"Unsupported Gemini content type: {type(content).__name__}")
        return PrescriptionResponse.model_validate_json(cls._strip_json_text(content))

    @staticmethod
    def _gemini_error_message(exc: Exception) -> str:
        status = getattr(exc, "status_code", None)
        if status is None:
            return str(exc)
        return f"status={status} {exc}"

    def _generate(self, prompt: str, *, json_mode: bool = False) -> str:
        self._load_once()
        assert self._client is not None
        assert genai_types is not None

        generation_config: dict[str, Any] = {
            "system_instruction": PHARMALINK_SYSTEM_PROMPT,
            "temperature": 0,
            "max_output_tokens": self.settings.gemini_max_output_tokens,
        }
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        response = self._client.models.generate_content(
            model=self.settings.gemini_model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                http_options=genai_types.HttpOptions(
                    timeout=int(self.settings.gemini_timeout * 1000)
                ),
                **generation_config,
            ),
        )
        return self._extract_text(response)

    def build_payload(self, raw_text: str, normalized_drugs: List[str]) -> str:
        """Construct prescription structuring payload."""
        payload = {
            "raw_ocr_text": raw_text,
            "normalized_drugs": normalized_drugs,
        }
        return json.dumps(payload, ensure_ascii=True)

    def build_chat_prompt(self, message: str, normalized_drugs: List[str]) -> str:
        """Construct grounded chatbot prompt."""
        normalized = "\n".join(f"* {drug}" for drug in normalized_drugs) or "* None"
        return (
            f"Normalized Drugs:\n{normalized}\n\n"
            f'User Query:\n"{message}"\n\n'
            "Answer the user using the normalized drug names as the canonical "
            "grounding. Return a concise pharmaceutical support response."
        )

    def structure_prescription(
        self, raw_text: str, normalized_drugs: List[str]
    ) -> PrescriptionResponse:
        """Request schema-compliant prescription JSON and validate it."""
        t_start = time.perf_counter()
        prompt = (
            f"{PRESCRIPTION_STRUCTURING_PROMPT}\n\n"
            f"Prescription context:\n{self.build_payload(raw_text, normalized_drugs)}"
        )
        content = ""
        try:
            for attempt in range(2):
                content = self._generate(prompt, json_mode=True)
                try:
                    parsed = self._parse_prescription_response(content)
                    if self.settings.enable_latency_logging:
                        logger.info(
                            "gemini_structuring_latency_ms attempts=%d total=%.1f",
                            attempt + 1,
                            (time.perf_counter() - t_start) * 1000,
                        )
                    return parsed
                except (ValidationError, ValueError, TypeError):
                    if attempt == 0:
                        prompt = (
                            f"{prompt}\n\nPrevious response was invalid JSON. "
                            "Retry with only valid JSON matching the schema."
                        )
                        continue
                    raise
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning(
                "Gemini structuring unavailable, using fallback: %s",
                self._gemini_error_message(exc),
            )
            if content:
                logger.debug("Invalid Gemini structuring response: %s", content)

        fallback_drugs = [{"name": name} for name in normalized_drugs] or None
        return PrescriptionResponse(
            confidence_score=0.25 if raw_text.strip() else 0.05,
            description=(
                "Structured output generated with fallback mode because the "
                "Gemini response was unavailable or invalid."
            ),
            drugs=fallback_drugs,
        )

    def answer_chat(self, message: str, normalized_drugs: List[str]) -> ChatResponse:
        """Generate a grounded pharmaceutical chatbot answer."""
        prompt = self.build_chat_prompt(message, normalized_drugs)

        try:
            response = self._generate(prompt, json_mode=False).strip()

        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning(
                "Gemini chat unavailable, using fallback: %s",
                self._gemini_error_message(exc),
            )

            response = (
                "I could not reach the pharmaceutical assistant right now. "
                "Please review medication questions with a licensed healthcare "
                "professional, especially for interactions, dosing, or urgent symptoms."
            )

        warnings = [
            "This is informational support, not medical diagnosis.",
            "Consult a licensed healthcare professional for personal medical advice.",
        ]

        return ChatResponse(
            user_query=message,
            normalized_drugs=normalized_drugs,
            response=response,
            warnings=warnings,
        )
