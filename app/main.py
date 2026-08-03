"""FastAPI application entrypoint."""

import logging

from fastapi import FastAPI

from app.api.extract import router as extract_router
from app.api.health import router as health_router
from app.api.rag import router as rag_router
from app.core.config import get_settings

logging.basicConfig(
    level=get_settings().LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

if get_settings().OPENAI_API_KEY is None:
    logger.warning("OPENAI_API_KEY is not set - /api/ask requests will fail")

app = FastAPI(title="PDF Name Extractor & RAG API")

app.include_router(extract_router, prefix="/api", tags=["extraction"])
app.include_router(rag_router, prefix="/api", tags=["rag"])
app.include_router(health_router, tags=["health"])
