import uuid
from typing import Protocol

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.providers import create_qdrant_client
from app.services.embedding_service import get_embeddings

client = create_qdrant_client()

COLLECTION_NAME = "pdf_documents"
VECTOR_SIZE = 384


class VectorStore(Protocol):
    def init_collection(self) -> None: ...
    def store_document_chunks(
        self, chunks: list[str], metadata: list[dict] | None = None
    ) -> None: ...
    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]: ...


class QdrantVectorStore:
    def __init__(self, qdrant_client=None, collection: str = COLLECTION_NAME):
        self.client = qdrant_client
        self.collection = collection

    def init_collection(self) -> None:
        """Create the vector collection if it doesn't exist."""
        vs_client = self.client or client
        collections = vs_client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection not in existing:
            vs_client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def store_document_chunks(
        self, chunks: list[str], metadata: list[dict] | None = None
    ) -> None:
        """Store text chunks with their embeddings in the vector database."""
        vs_client = self.client or client
        embeddings = get_embeddings(chunks)

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            payload = {"text": chunk}
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            )

        vs_client.upsert(collection_name=self.collection, points=points)

    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """Search for similar text chunks."""
        vs_client = self.client or client
        results = vs_client.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0.5,
        )

        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
            }
            for hit in results
        ]


_default_vector_store = QdrantVectorStore()
init_collection = _default_vector_store.init_collection
store_document_chunks = _default_vector_store.store_document_chunks
search_similar = _default_vector_store.search_similar
