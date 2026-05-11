"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "DONUT-PHARMALINK API"
    app_version: str = "1.0.0"
    donut_model_path: Path = Field(default=Path("model-cache"))
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-chat"
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    faiss_index_path: Path = Field(default=Path("artifacts/faiss_drugs.index"))
    drugs_cache_path: Path = Field(default=Path("artifacts/drug_names.json"))
    embeddings_cache_path: Path = Field(default=Path("artifacts/drug_embeddings.npy"))
    kaggle_dataset_ref: str = "protobioengineering/united-states-fda-drugs-feb-2024"
    max_generate_tokens: int = 512
    normalize_top_k: int = 5
    preload_models_on_startup: bool = True
    local_files_only: bool = True
    enable_latency_logging: bool = True
    device_preference: str = "auto"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
