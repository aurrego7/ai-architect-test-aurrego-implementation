import os
import tempfile

from fastapi import APIRouter, File, Form, UploadFile

from app.models.schemas import ExtractionResponse
from app.services.ocr_service import extract_text_from_pdf
from app.services.bbox_service import find_name_bounding_boxes
from app.services.fuzzy_service import fuzzy_match_names

router = APIRouter()


@router.post("/extract", response_model=ExtractionResponse)
def extract_names_from_pdf(
    pdf_file: UploadFile = File(...),
    names: str = Form(...),
):
    """Extract names from PDF and perform fuzzy matching."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_file.file.read())
    tmp.close()

    try:
        text = extract_text_from_pdf(tmp.name)

        name_boxes = find_name_bounding_boxes(tmp.name, text)

        import json

        query_names = json.loads(names)

        extracted_name_strings = [nb["name"] for nb in name_boxes]
        matches = fuzzy_match_names(extracted_name_strings, query_names)

        return {
            "extracted_names": [
                {
                    "name": nb["name"],
                    "bounding_box": {
                        "x": nb["x"],
                        "y": nb["y"],
                        "width": nb["width"],
                        "height": nb["height"],
                    },
                }
                for nb in name_boxes
            ],
            "fuzzy_matches": matches,
        }
    finally:
        os.unlink(tmp.name)
