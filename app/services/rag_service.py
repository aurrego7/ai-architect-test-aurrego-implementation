import os
from string import whitespace
from typing import Protocol

import httpx

from app.core.prompts import rag_question_prompt
from app.services.embedding_service import get_query_embedding
from app.services.vector_service import search_similar

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class RAGService(Protocol):
    def chunk_text(self, text: str, chunk_size: int = 500) -> list[str]: ...
    def generate_answer(self, question: str) -> dict: ...


class OpenAIRAG:
    def __init__(self, embedding_function=None, search_function=None):
        self.embedding_function = embedding_function
        self.search_function = search_function

    def chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """Split text into chunks for embedding."""
        # Check chunk_size to prevent infinite while loop
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # If the end of the chunk is after the end of the text cap to end of text
            if end >= len(text):
                chunks.append(text[start:])
                break

            # Find the last whitespace in the chunk
            last_whitespace = max(text.rfind(c, start, end) for c in whitespace)
            # If found then cut the chunk to that whitespace index
            if last_whitespace != -1:
                end = last_whitespace + 1

            chunks.append(text[start:end])
            start = end

        return chunks

    def _call_llm(self, prompt: str) -> str:
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

        return answer

    def generate_answer(self, question: str) -> dict:
        """Generate an answer using RAG strategy."""
        embed_query = self.embedding_function or get_query_embedding
        get_relevant_chunks = self.search_function or search_similar

        query_embedding = embed_query(question)
        relevant_chunks = get_relevant_chunks(query_embedding, top_k=3)

        if not relevant_chunks:
            return {"answer": "No relevant information found.", "sources": []}

        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

        prompt = rag_question_prompt.format(context=context, question=question)

        answer = self._call_llm(prompt)

        return {
            "answer": answer,
            "sources": [chunk["text"] for chunk in relevant_chunks],
        }


_default_rag = OpenAIRAG()
chunk_text = _default_rag.chunk_text
generate_answer = _default_rag.generate_answer
