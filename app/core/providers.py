"""Lazily constructed ML models and clients."""

import logging
import time
from functools import lru_cache

import spacy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from spacy.language import Language

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def load_spacy_model() -> Language:
    """Load and cache the spaCy pipeline named by `SPACY_MODEL`.

    Returns:
        The loaded spaCy pipeline. Repeated calls return the same object.

    Raises:
        OSError: If the configured model is not installed in the environment.
    """
    model_name = get_settings().SPACY_MODEL
    logger.info("Loading spaCy model '%s'", model_name)
    start = time.perf_counter()
    model = spacy.load(model_name)
    logger.info("spaCy model loaded in %.2fs", time.perf_counter() - start)
    return model


@lru_cache
def load_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model named by `EMBEDDING_MODEL`.

    Returns:
        The loaded embedding model. Repeated calls return the same object.
    """
    model_name = get_settings().EMBEDDING_MODEL
    logger.info("Loading embedding model '%s'", model_name)
    start = time.perf_counter()
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded in %.2fs", time.perf_counter() - start)
    return model


@lru_cache
def create_qdrant_client() -> QdrantClient:
    """Create and cache the Qdrant client for the configured host and port.

    Constructing the client does not open a connection, so this does not fail
    when Qdrant is unreachable; that surfaces on the first request instead.

    Returns:
        The shared Qdrant client. Repeated calls return the same object.
    """
    settings = get_settings()
    logger.info(
        "Creating Qdrant client for %s:%d", settings.QDRANT_HOST, settings.QDRANT_PORT
    )
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
