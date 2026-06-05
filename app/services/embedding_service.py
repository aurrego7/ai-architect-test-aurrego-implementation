from sentence_transformers import SentenceTransformer


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert texts to vector embeddings."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return embeddings.tolist()


def get_query_embedding(query: str) -> list[float]:
    """Convert a single query to a vector embedding."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode([query])
    return embedding[0].tolist()
