"""Application settings loaded from the environment."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the extraction and RAG pipelines.

    Attributes:
        OPENAI_API_KEY: Credential for the OpenAI chat completions API. Kept
            as a ``SecretStr`` so it is redacted in logs and reprs. ``None``
            when unset, which makes ``/api/ask`` fail at call time.
        SPACY_MODEL: Name of the spaCy pipeline used for named-entity
            recognition.
        EMBEDDING_MODEL: Name of the sentence-transformers model used to embed
            chunks and queries.
        LLM_MODEL: Chat model used to synthesise RAG answers.
        QDRANT_HOST: Hostname of the Qdrant vector database.
        QDRANT_PORT: Port of the Qdrant vector database.
        SCORE_THRESHOLD: Minimum cosine similarity a search hit must reach to
            be used as RAG context.
        SIMILARITY_THRESHOLD: Minimum fuzzy score (0-100) for an extracted
            name to be reported as a match for a query name.
        OCR_DPI: Resolution used when rasterising PDF pages for OCR. Bounding
            boxes are scaled back to PDF points from this value.
        COLLECTION_NAME: Qdrant collection that holds document chunks.
        VECTOR_SIZE: Dimensionality of the embedding vectors, which must match
            `EMBEDDING_MODEL`.
        TOP_K: Number of chunks retrieved per question.
        CHUNK_SIZE: Maximum chunk length in characters before splitting on
            whitespace.
        LOG_LEVEL: Root logging level name, case-insensitive.
        LLM_TIMEOUT_SEC: Timeout applied to a single LLM HTTP request.
    """

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
    """Return the process-wide settings instance.

    The result is cached so that the ``.env`` file is parsed once and every
    caller observes the same configuration object.

    Returns:
        The singleton :class:`Settings` instance for this process.
    """
    return Settings()
