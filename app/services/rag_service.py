import os
from string import whitespace

import httpx

from app.services.embedding_service import get_query_embedding
from app.services.vector_service import search_similar

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into chunks for embedding."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # If the end of the chunk is after the end of the text cap to end of text
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Find the last whitespace in the chunk
        last_whitespace = text.rfind(" ", start, end)
        # If found then cut the chunk to that whitespace index
        if last_whitespace != -1:
            end = last_whitespace + 1

        chunks.append(text[start:end])
        start = end

    return chunks


def generate_answer(question: str) -> dict:
    """Generate an answer using RAG strategy."""
    query_embedding = get_query_embedding(question)

    relevant_chunks = search_similar(query_embedding, top_k=3)

    if not relevant_chunks:
        return {"answer": "No relevant information found.", "sources": []}

    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

    prompt = f"""
    Based on the following context, answer the question.
    If the answer is not in the context, say "I don't have enough information."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    result = response.json()
    answer = result["choices"][0]["message"]["content"]

    return {
        "answer": answer,
        "sources": [chunk["text"] for chunk in relevant_chunks],
    }
