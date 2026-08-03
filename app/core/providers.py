from functools import lru_cache

import spacy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


@lru_cache
def load_spacy_model():
    return spacy.load("en_core_web_sm")


@lru_cache
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache
def create_qdrant_client():
    return QdrantClient(host="localhost", port=6333)
