from functools import lru_cache

import spacy
from sentence_transformers import SentenceTransformer


@lru_cache
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@lru_cache
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")