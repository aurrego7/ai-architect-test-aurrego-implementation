from fastapi import FastAPI

from app.api.extract import router as extract_router
from app.api.rag import router as rag_router

app = FastAPI(title="PDF Name Extractor & RAG API")

app.include_router(extract_router, prefix="/api", tags=["extraction"])
app.include_router(rag_router, prefix="/api", tags=["rag"])
