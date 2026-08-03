from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM PROVIDER KEYS
    OPENAI_API_KEY: SecretStr | None = None

    # MODELS
    SPACY_MODEL: str = "en_core_web_sm"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo"

    # URLS & PORTS
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # PARAMETERS
    SCORE_THRESHOLD: float = 0.5
    SIMILARITY_THRESHOLD: int = 90
    OCR_DPI: int = 150
    COLLECTION_NAME: str = "pdf_documents"
    VECTOR_SIZE: int = 384
    TOP_K: int = 3
    CHUNK_SIZE: int = 500

    # GENERAL
    LOG_LEVEL: str = "info"
    LLM_TIMEOUT_SEC: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
