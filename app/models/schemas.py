"""Pydantic request and response models for the public API.

These schemas are the contract between the HTTP layer and its clients.
"""

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Rectangular region of a PDF page, expressed in PDF points.

    The origin is the top-left corner of the page and coordinates have already
    been scaled down from the OCR raster resolution.

    Attributes:
        x: Distance from the left edge of the page to the left edge of the box.
        y: Distance from the top edge of the page to the top edge of the box.
        width: Horizontal size of the box.
        height: Vertical size of the box.
        page_number: Zero-based index of the page the box was found on.
    """

    x: float
    y: float
    width: float
    height: float
    page_number: int


class ExtractedName(BaseModel):
    """A single occurrence of a person name located in a document.

    A name that appears several times yields one entry per occurrence, each
    with its own bounding box.

    Attributes:
        name: The name exactly as it was recognised in the document.
        bounding_box: Where that occurrence sits on the page.
    """

    name: str
    bounding_box: BoundingBox


class FuzzyMatch(BaseModel):
    """A query name paired with its closest extracted name.

    Only pairs scoring at or above the configured similarity threshold are
    returned.

    Attributes:
        extracted_name: Best-scoring name found in the document.
        matched_name: The caller-supplied name, as `first last`.
        score: Similarity in the range 0.0-1.0, where 1.0 is an exact match.
    """

    extracted_name: str
    matched_name: str
    score: float


class ExtractionResponse(BaseModel):
    """Response body of `POST /api/extract`.

    Attributes:
        extracted_names: Every name occurrence located in the PDF.
        fuzzy_matches: Matches between the caller's names and the extracted
            ones. Empty when nothing clears the similarity threshold.
    """

    extracted_names: list[ExtractedName]
    fuzzy_matches: list[FuzzyMatch]


class NamePair(BaseModel):
    """A name to search for, supplied by the caller of `/api/extract`.

    Attributes:
        first_name: Given name.
        last_name: Family name.
    """

    first_name: str
    last_name: str


class RAGRequest(BaseModel):
    """Request body of `POST /api/ask`.

    Attributes:
        question: Natural-language question to answer from ingested documents.
    """

    question: str


class RAGResponse(BaseModel):
    """Response body of `POST /api/ask`.

    Attributes:
        answer: The generated answer, or a fallback message when retrieval
            returned nothing above the score threshold.
        sources: Raw text of the chunks used as context, in retrieval order.
    """

    answer: str
    sources: list[str]
