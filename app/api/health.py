"""Health endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def check_health() -> dict[str, str]:
    """Report whether the service is up.

    Returns:
        A mapping with ``status`` set to ``"ok"``.
    """
    try:
        # TODO - For refactor ping Qdrant client to check it is working here, hence try
        return {"status": "ok"}
    except Exception:
        return False
