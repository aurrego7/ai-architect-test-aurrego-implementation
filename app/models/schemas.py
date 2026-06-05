from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class ExtractedName(BaseModel):
    name: str
    bounding_box: BoundingBox


class FuzzyMatch(BaseModel):
    extracted_name: str
    matched_name: str
    score: float


class ExtractionResponse(BaseModel):
    extracted_names: list[ExtractedName]


class NamePair(BaseModel):
    first_name: str
    last_name: str


class RAGRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str
