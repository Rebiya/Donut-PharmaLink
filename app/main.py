"""FastAPI application entrypoint for DONUT-PHARMALINK."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import routes
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifespan hook for optional warmup."""
    settings = get_settings()
    if settings.preload_models_on_startup:
        routes.ocr_service._load_once()  # noqa: SLF001 - intentional warmup
        routes.normalization_service._load_or_build_faiss()  # noqa: SLF001
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
    app.include_router(routes.router)

    return app


app = create_app()
