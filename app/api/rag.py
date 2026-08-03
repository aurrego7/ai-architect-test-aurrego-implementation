import os
import tempfile

from fastapi import APIRouter, File, UploadFile

from app.models.schemas import RAGRequest, RAGResponse
from app.services.ocr_service import extract_text_from_pdf
from app.services.rag_service import chunk_text, generate_answer
from app.services.vector_service import init_collection, store_document_chunks

router = APIRouter()


@router.post("/ingest")
def ingest_pdf(pdf_file: UploadFile = File(...)):  # noqa: B008 FastAPI format
    """Ingest a PDF document into the vector database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.file.read())
        tmp.close()

    try:
        init_collection()

        text = extract_text_from_pdf(tmp.name)
        chunks = chunk_text(text)

        store_document_chunks(chunks)

        return {"status": "success", "chunks_stored": len(chunks)}
    finally:
        os.unlink(tmp.name)


@router.post("/ask", response_model=RAGResponse)
def ask_question(request: RAGRequest):
    """Answer a question using RAG."""
    result = generate_answer(request.question)
    return result
