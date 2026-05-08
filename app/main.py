"""FastAPI application entrypoint for DONUT-PHARMALINK."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.services.normalization_service import NormalizationService
from app.services.ocr_service import OCRService


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifespan hook for optional warmup."""
    settings = get_settings()
    if settings.preload_models_on_startup:
        OCRService()._load_once()  # noqa: SLF001 - intentional warmup
        NormalizationService()._load_or_build_faiss()  # noqa: SLF001
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI app instance."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Prescription OCR and drug normalization backend.",
        lifespan=lifespan,
    )
    app.include_router(router)

    return app


app = create_app()
