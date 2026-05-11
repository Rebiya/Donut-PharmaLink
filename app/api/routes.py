"""API routes."""

import logging
import time

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from app.core.config import get_settings
from app.models.schemas import ChatRequest, ChatResponse, PredictResponse
from app.services.extraction_service import ExtractionService
from app.services.llm_service import LLMService
from app.services.normalization_service import NormalizationService
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)
settings = get_settings()

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

    t_total = time.perf_counter()
    try:
        t = time.perf_counter()
        raw_text = ocr_service.extract_text(image_bytes)
        ocr_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        candidates = extraction_service.extract_candidates(raw_text)
        extract_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        normalized_drugs = normalization_service.normalize(candidates)
        norm_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        result = llm_service.structure_prescription(raw_text, normalized_drugs)
        llm_ms = (time.perf_counter() - t) * 1000
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction pipeline failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    if settings.enable_latency_logging:
        logger.info(
            "predict_latency_ms ocr=%.1f extract=%.1f normalize=%.1f llm=%.1f total=%.1f",
            ocr_ms,
            extract_ms,
            norm_ms,
            llm_ms,
            (time.perf_counter() - t_total) * 1000,
        )

    return PredictResponse(
        raw_text=raw_text,
        candidate_tokens=candidates,
        normalized_drugs=normalized_drugs,
        result=result,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer pharmaceutical questions using FAISS-grounded drug names."""
    if not settings.enable_chat_endpoint:
        raise HTTPException(status_code=404, detail="Chat endpoint is disabled.")

    t_total = time.perf_counter()
    try:
        t = time.perf_counter()
        candidates = extraction_service.extract_candidates(request.message)
        extract_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        normalized_drugs = normalization_service.normalize(candidates)
        norm_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        result = llm_service.answer_chat(request.message, normalized_drugs)
        llm_ms = (time.perf_counter() - t) * 1000
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat pipeline failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    if settings.enable_latency_logging:
        logger.info(
            "chat_latency_ms extract=%.1f normalize=%.1f llm=%.1f total=%.1f",
            extract_ms,
            norm_ms,
            llm_ms,
            (time.perf_counter() - t_total) * 1000,
        )

    return result
