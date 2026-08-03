import logging
import time
from functools import lru_cache

import spacy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def load_spacy_model():
    model_name = get_settings().SPACY_MODEL
    logger.info("Loading spaCy model '%s'", model_name)
    start = time.perf_counter()
    model = spacy.load(model_name)
    logger.info("spaCy model loaded in %.2fs", time.perf_counter() - start)
    return model


@lru_cache
def load_embedding_model():
    model_name = get_settings().EMBEDDING_MODEL
    logger.info("Loading embedding model '%s'", model_name)
    start = time.perf_counter()
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded in %.2fs", time.perf_counter() - start)
    return model


@lru_cache
def create_qdrant_client():
    settings = get_settings()
    logger.info(
        "Creating Qdrant client for %s:%d", settings.QDRANT_HOST, settings.QDRANT_PORT
    )
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
