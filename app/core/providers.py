from functools import lru_cache

import spacy


@lru_cache
def load_spacy_model():
    return spacy.load("en_core_web_sm")
