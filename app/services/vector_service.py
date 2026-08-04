"""Qdrant-backed storage and retrieval of document chunks."""

import logging
import uuid
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import get_settings
from app.core.errors import VectorStoreError
from app.core.providers import create_qdrant_client
from app.services.embedding_service import get_embeddings

logger = logging.getLogger(__name__)

client = create_qdrant_client()

POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "pdf-rag.chunks")


class VectorStore(Protocol):
    """Interface for saving and searching document embeddings."""

    def init_collection(self) -> None:
        """Ensure the backing collection exists."""
        ...

    def store_document_chunks(
        self, chunks: list[str], metadata: list[dict] | None = None
    ) -> None:
        """Embed and persist `chunks`, optionally with per-chunk metadata."""
        ...

    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """Return the chunks most similar to `query_embedding`."""
        ...


class QdrantVectorStore:
    """Vector store backed by a Qdrant collection using cosine distance.

    Attributes:
        client: Qdrant client to issue requests with. When `None` the shared
            module-level client is used, which is what production code relies
            on; tests inject a mock here.
        collection: Name of the collection all operations target.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        collection: str = get_settings().COLLECTION_NAME,
    ) -> None:
        """Initialize the store.

        Args:
            qdrant_client: Optional client to use instead of the shared one.
            collection: Collection name to read from and write to. Defaults to
                the configured `COLLECTION_NAME`.
        """
        self.client = qdrant_client
        self.collection = collection

    def init_collection(self) -> None:
        """Create the vector collection if it doesn't exist.

        Raises:
            VectorStoreError: If the vector store cannot be reached or refuses
                to create the collection.
        """
        vs_client = self.client or client
        try:
            collections = vs_client.get_collections().collections
        except ApiException as exc:
            logger.error("Vector store unreachable while listing collections: %s", exc)
            raise VectorStoreError("Vector store unavailable") from exc
        existing = [c.name for c in collections]

        if self.collection not in existing:
            logger.info("Creating collection '%s'", self.collection)
            try:
                vs_client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=get_settings().VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
            except ApiException as exc:
                logger.error(
                    "Failed to create collection '%s': %s", self.collection, exc
                )
                raise VectorStoreError("Vector store unavailable") from exc
        else:
            logger.debug("Collection '%s' already exists", self.collection)

    def store_document_chunks(
        self, chunks: list[str], metadata: list[dict] | None = None
    ) -> None:
        """Store text chunks with their embeddings in the vector database.

        Each point's ID is a UUID5 of the chunk text, so storing the same
        chunk twice updates one point instead of creating two.

        Args:
            chunks: Text chunks to embed and persist.
            metadata: Optional extra payload fields, positionally aligned with
                `chunks`. Entries beyond the length of `chunks` are
                ignored, and chunks beyond the length of `metadata` get only
                their text payload.

        Raises:
            VectorStoreError: If the upsert is rejected or the store is
                unreachable.
            ValueError: If the embedder returns a different number of vectors
                than there are chunks.
        """
        vs_client = self.client or client
        embeddings = get_embeddings(chunks)

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            payload = {"text": chunk}
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            points.append(
                PointStruct(
                    id=str(uuid.uuid5(POINT_NAMESPACE, chunk)),
                    vector=embedding,
                    payload=payload,
                )
            )

        try:
            vs_client.upsert(collection_name=self.collection, points=points)
        except ApiException as exc:
            logger.error("Failed to store %d chunks: %s", len(points), exc)
            raise VectorStoreError("vector store unavailable") from exc
        logger.info("Stored %d chunks in collection '%s'", len(points), self.collection)

    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """Search for similar text chunks.

        Hits scoring below the configured `SCORE_THRESHOLD` are discarded by
        the store itself, so a weakly-related question can legitimately return
        fewer than `top_k` results, or none at all.

        Args:
            query_embedding: Embedded query, of the collection's vector size.
            top_k: Maximum number of hits to return.

        Returns:
            One dictionary per hit, ordered by descending similarity, with
            keys `text` (str) and `score` (float).

        Raises:
            VectorStoreError: If the search fails or the store is unreachable.
        """
        vs_client = self.client or client
        score_threshold = get_settings().SCORE_THRESHOLD
        try:
            results = vs_client.search(
                collection_name=self.collection,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
            )
        except ApiException as exc:
            logger.error("Vector search failed: %s", exc)
            raise VectorStoreError("Vector store unavailable") from exc

        hits = [
            {
                "text": hit.payload["text"],
                "score": hit.score,
            }
            for hit in results
        ]
        logger.debug(
            "Retrieved %d hits (top_k=%d, threshold=%.2f), scores=%s",
            len(hits),
            top_k,
            score_threshold,
            [round(hit["score"], 3) for hit in hits],
        )
        return hits


_default_vector_store = QdrantVectorStore()
init_collection = _default_vector_store.init_collection
store_document_chunks = _default_vector_store.store_document_chunks
search_similar = _default_vector_store.search_similar
