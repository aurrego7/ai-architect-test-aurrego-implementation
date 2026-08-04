"""Health endpoint."""

import logging

from fastapi import APIRouter, HTTPException, status

from app.core.providers import create_qdrant_client

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def check_health() -> dict[str, str]:
    """Report whether the service is up.

    Returns:
        A mapping with ``status`` set to ``"ok"``.
    """
    try:
        # Check if dependencies are healthy
        create_qdrant_client().get_collections()
        return {"status": "ok"}
    except Exception as exc:
        logger.exception("Health check failed: Qdrant unreachable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant unreachable",
        ) from exc
