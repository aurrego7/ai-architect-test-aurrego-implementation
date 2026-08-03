import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.errors import LLMError, OCRError, VectorStoreError
from app.models.schemas import RAGRequest, RAGResponse
from app.services.ocr_service import extract_text_from_pdf
from app.services.rag_service import chunk_text, generate_answer
from app.services.vector_service import init_collection, store_document_chunks

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest")
def ingest_pdf(pdf_file: UploadFile = File(...)):  # noqa: B008 FastAPI format
    """Ingest a PDF document into the vector database."""
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
def ask_question(request: RAGRequest):
    """Answer a question using RAG."""
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
