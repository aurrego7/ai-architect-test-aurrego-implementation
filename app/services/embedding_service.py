from typing import Protocol

from app.core.providers import load_embedding_model


class EmbeddingService(Protocol): ...


class SentenceTransformerEmbedder:
    def __init__(self, model=None):
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert texts to vector embeddings."""
        model = self.model or load_embedding_model()
        embeddings = model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert a single query to a vector embedding.

        A query embedding is a text embedding of a single element list,
        so we reutilize `embed_text`.
        """
        return self.embed_texts([query])[0].tolist()


_default_embedder = SentenceTransformerEmbedder()
get_embeddings = _default_embedder.embed_texts
get_query_embedding = _default_embedder.embed_query
