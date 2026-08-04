"""Text-to-vector embedding."""

import logging
from typing import Protocol

from sentence_transformers import SentenceTransformer

from app.core.providers import load_embedding_model

logger = logging.getLogger(__name__)


class EmbeddingService(Protocol):
    """Interface for turning text into dense vectors."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per item in `texts`."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Return the embedding of a single query string."""
        ...


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model.

    Attributes:
        model: Model to encode with. When `None` the shared cached model is
            loaded on demand, which is what production code relies on; tests
            inject a stub here.
    """

    def __init__(self, model: SentenceTransformer | None = None) -> None:
        """Initilize the embedder.

        Args:
            model: Optional model to use instead of the shared cached one.
        """
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert texts to vector embeddings.

        Args:
            texts: Strings to embed. Order is preserved in the result.

        Returns:
            One vector per input string, each of length `VECTOR_SIZE`.
        """
        model = self.model or load_embedding_model()
        logger.debug("Embedding %d texts", len(texts))
        embeddings = model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert a single query to a vector embedding.

        A query embedding is a text embedding of a single element list,
        so we reutilize `embed_text`.

        Args:
            query: The question or phrase to embed.

        Returns:
            A single vector of length `VECTOR_SIZE`.
        """
        return self.embed_texts([query])[0]


_default_embedder = SentenceTransformerEmbedder()
get_embeddings = _default_embedder.embed_texts
get_query_embedding = _default_embedder.embed_query
