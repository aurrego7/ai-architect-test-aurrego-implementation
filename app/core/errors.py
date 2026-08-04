"""Application-domain errors.

Services translate third-party/transport exceptions into these so that
API endpoints can map them to HTTP responses without knowing transport
details. The chain is: library exception -> domain error -> HTTP status.
"""


class AppError(Exception):
    """Base class for all application-domain errors."""


class OCRError(AppError):
    """The PDF could not be opened or processed for OCR."""


class VectorStoreError(AppError):
    """The vector database is unreachable or rejected the operation."""


class LLMError(AppError):
    """The LLM backend failed to produce an answer."""
