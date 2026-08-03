"""Retrieval-augmented generation over ingested documents."""

import logging
import time
from collections.abc import Callable
from string import whitespace
from typing import Any, Protocol

import httpx

from app.core.config import get_settings
from app.core.errors import LLMError
from app.core.prompts import rag_question_prompt
from app.services.embedding_service import get_query_embedding
from app.services.vector_service import search_similar

logger = logging.getLogger(__name__)

_key = get_settings().OPENAI_API_KEY
OPENAI_API_KEY = _key.get_secret_value() if _key else None
# Keeping this here to allow for test to patch without errors


class RAGService(Protocol):
    """Interface for the chunking and question-answering pipeline."""

    def chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """Split `text` into chunks of at most `chunk_size` characters."""
        ...

    def generate_answer(self, question: str) -> dict[str, Any]:
        """Answer `question` from the retrieved document chunks."""
        ...


class OpenAIRAG:
    """RAG pipeline that answers questions with the OpenAI chat API.

    Retrieval is injectable so the pipeline can be exercised without a running
    vector store, and so the retrieval strategy can be replaced independently
    of answer generation.

    Attributes:
        embedding_function: Callable turning a question into a vector. When
            `None` the default sentence-transformers embedder is used.
        search_function: Callable taking a query vector and a `top_k`
            keyword and returning scored chunks. When `None` the default
            Qdrant search is used.
    """

    def __init__(
        self,
        embedding_function: Callable[[str], list[float]] | None = None,
        search_function: Callable[..., list[dict]] | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            embedding_function: Optional replacement for the default query
                embedder. Must accept the question and return a vector.
            search_function: Optional replacement for the default vector
                search. Called as `search_function(embedding, top_k=...)`
                and must return dictionaries with a `text` key.
        """
        self.embedding_function = embedding_function
        self.search_function = search_function

    def chunk_text(
        self, text: str, chunk_size: int = get_settings().CHUNK_SIZE
    ) -> list[str]:
        """Split text into chunks for embedding.

        Chunks are cut at the last whitespace inside the window rather than at
        a fixed offset, so words are never split across two chunks. The cut
        keeps the last whitespace with the previous chunk, which means
        concatenating the result generates input exactly.

        Args:
            text: Text to split, typically a full OCR transcript.
            chunk_size: Maximum chunk length in characters. Defaults to the
                configured `CHUNK_SIZE`.

        Returns:
            The chunks in document order. Empty for empty input, and a single
            chunk when the text is shorter than `chunk_size`.

        Raises:
            ValueError: If `chunk_size` is not positive, which would
                cause inifite while loop.
        """
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

        logger.debug(
            "Chunked %d chars into %d chunks (chunk_size=%d)",
            len(text),
            len(chunks),
            chunk_size,
        )
        return chunks

    def _call_llm(self, prompt: str) -> str:
        """Send a prompt to the chat completions API and return its answer.

        Args:
            prompt: Fully rendered prompt, context included.

        Returns:
            The assistant message content from the first choice.

        Raises:
            LLMError: If the request times out, is rejected, fails at the
                transport level, or returns a body that is not JSON or does
                not have the expected shape. The original exception is
                always passed along.
        """
        settings = get_settings()
        model = settings.LLM_MODEL
        logger.debug("Calling LLM '%s' with prompt of %d chars", model, len(prompt))
        start = time.perf_counter()
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=settings.LLM_TIMEOUT_SEC,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            logger.error("LLM request timed out after %.0fs", settings.LLM_TIMEOUT_SEC)
            raise LLMError("LLM request timed out") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "LLM returned HTTP %d: %.500s",
                exc.response.status_code,
                exc.response.text,
            )
            raise LLMError("LLM request rejected") from exc
        except httpx.HTTPError as exc:
            logger.error("LLM request failed: %s", exc)
            raise LLMError("LLM unreachable") from exc

        try:
            result = response.json()
        except ValueError as exc:
            logger.error("LLM response was not JSON: %.300s", response.text)
            raise LLMError("Unexpected LLM response") from exc

        try:
            answer = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected LLM response shape: %.300s", str(result))
            raise LLMError("Unexpected LLM response") from exc

        logger.info("LLM call completed in %.2fs", time.perf_counter() - start)
        return answer

    def generate_answer(self, question: str) -> dict[str, Any]:
        """Generate an answer using RAG strategy.

        Retrieval runs first; when it returns nothing above the score
        threshold the LLM is never called and a fixed fallback is returned, so
        an unanswerable question costs no tokens.

        Args:
            question: Natural-language question to answer.

        Returns:
            A mapping with `answer` (str) and `sources` (list of str, the
            raw text of the chunks used as context, empty when nothing was
            retrieved).

        Raises:
            LLMError: If the LLM call fails.
            VectorStoreError: If retrieval fails or the store is unreachable.
        """
        embed_query = self.embedding_function or get_query_embedding
        get_relevant_chunks = self.search_function or search_similar

        query_embedding = embed_query(question)
        relevant_chunks = get_relevant_chunks(
            query_embedding, top_k=get_settings().TOP_K
        )

        if not relevant_chunks:
            logger.info("No relevant chunks found, returning fallback answer")
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
