"""HTTP endpoints for document ingestion and question answering."""

import logging
import os
import tempfile
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.errors import LLMError, OCRError, VectorStoreError
from app.models.schemas import RAGRequest, RAGResponse
from app.services.ocr_service import extract_text_from_pdf
from app.services.rag_service import chunk_text, generate_answer
from app.services.vector_service import init_collection, store_document_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest")
def ingest_pdf(
    pdf_file: UploadFile = File(...),  # noqa: B008 FastAPI format
) -> dict[str, Any]:
    """Ingest a PDF document into the vector database.

    OCRs the upload, splits it into chunks and stores their embeddings.
    Chunk IDs are content-derived, so re-ingesting the same document updates
    the existing points instead of duplicating them. The temporary file is
    always removed, including on failure.

    Args:
        pdf_file: Uploaded PDF to ingest.

    Returns:
        A mapping with ``status`` (str) and ``chunks_stored`` (int).

    Raises:
        HTTPException: 400 if the PDF cannot be read, 503 if the vector store
            is unavailable.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.file.read())
        tmp.close()

    try:
        logger.info(
            "Ingest started for '%s' (%d bytes)",
            pdf_file.filename,
            os.path.getsize(tmp.name),
        )
        init_collection()

        text = extract_text_from_pdf(tmp.name)
        chunks = chunk_text(text)

        store_document_chunks(chunks)

        logger.info(
            "Ingest finished for '%s': %d chunks stored",
            pdf_file.filename,
            len(chunks),
        )
        return {"status": "success", "chunks_stored": len(chunks)}
    except OCRError as exc:
        raise HTTPException(
            status_code=400, detail="Could not process the PDF file."
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503, detail="Document storage is temporarily unavailable."
        ) from exc
    finally:
        os.unlink(tmp.name)


@router.post("/ask", response_model=RAGResponse)
def ask_question(request: RAGRequest) -> dict[str, Any]:
    """Answer a question using RAG.

    Retrieves the most relevant ingested chunks and has the LLM answer from
    them. When nothing clears the similarity threshold a fallback answer with
    no sources is returned rather than an error.

    Args:
        request: Body carrying the question to answer.

    Returns:
        A mapping matching :class:`~app.models.schemas.RAGResponse`, with
        ``answer`` and the ``sources`` it was grounded in.

    Raises:
        HTTPException: 502 if the LLM fails to produce an answer, 503 if the
            vector store is unavailable.
    """
    logger.info("Question received: %.80s", request.question)
    try:
        result = generate_answer(request.question)
    except LLMError as exc:
        raise HTTPException(
            status_code=502, detail="Failed to generate an answer."
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503, detail="Document search is temporarily unavailable."
        ) from exc
    logger.info(
        "Answer generated (%d chars, %d sources)",
        len(result["answer"]),
        len(result["sources"]),
    )
    return result
