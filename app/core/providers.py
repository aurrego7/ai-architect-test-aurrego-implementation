from functools import lru_cache

import spacy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache
def load_spacy_model():
    return spacy.load(get_settings().SPACY_MODEL)


@lru_cache
def load_embedding_model():
    return SentenceTransformer(get_settings().EMBEDDING_MODEL)


@lru_cache
def create_qdrant_client():
    return QdrantClient(
        host=get_settings().QDRANT_HOST, port=get_settings().QDRANT_PORT
    )
