"""API routes."""

import logging

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from app.models.schemas import PredictResponse
from app.services.extraction_service import ExtractionService
from app.services.llm_service import LLMService
from app.services.normalization_service import NormalizationService
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

router = APIRouter()

ocr_service = OCRService()
extraction_service = ExtractionService()
normalization_service = NormalizationService()
llm_service = LLMService()


@router.get("/")
async def health() -> dict[str, str]:
    """Basic health endpoint for service checks."""
    return {"status": "ok", "service": "donut-pharmalink"}


@router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Return empty favicon to avoid noisy 404 logs."""
    return Response(status_code=204)


@router.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)) -> PredictResponse:
    """Run OCR, drug normalization, and LLM structuring."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        raw_text = ocr_service.extract_text(image_bytes)
        candidates = extraction_service.extract_candidates(raw_text)
        normalized_drugs = normalization_service.normalize(candidates)
        result = llm_service.structure_prescription(raw_text, normalized_drugs)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction pipeline failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        raw_text=raw_text,
        candidate_tokens=candidates,
        normalized_drugs=normalized_drugs,
        result=result,
    )
