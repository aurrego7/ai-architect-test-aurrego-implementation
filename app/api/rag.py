import logging
import os
import tempfile

from fastapi import APIRouter, File, UploadFile

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
    finally:
        os.unlink(tmp.name)


@router.post("/ask", response_model=RAGResponse)
def ask_question(request: RAGRequest):
    """Answer a question using RAG."""
    logger.info("Question received: %.80s", request.question)
    result = generate_answer(request.question)
    logger.info(
        "Answer generated (%d chars, %d sources)",
        len(result["answer"]),
        len(result["sources"]),
    )
    return result
