import json
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import ExtractionResponse
from app.services.bbox_service import find_name_bounding_boxes
from app.services.fuzzy_service import fuzzy_match_names
from app.services.ocr_service import extract_text_from_pdf

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/extract", response_model=ExtractionResponse)
def extract_names_from_pdf(
    pdf_file: UploadFile = File(...),  # noqa: B008 FastAPI format
    names: str = Form(...),
):
    """Extract names from PDF and perform fuzzy matching."""
    # Easy check for proper file type before performing any operation
    if pdf_file.content_type != "application/pdf":
        logger.warning(
            "Rejected upload '%s': content type '%s' is not application/pdf",
            pdf_file.filename,
            pdf_file.content_type,
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. File must be a PDF.",
        )

    # More expensive check in the inital files byte to check file type
    header = pdf_file.file.read(5)
    pdf_file.file.seek(0)
    if header != b"%PDF-":
        logger.warning(
            "Rejected upload '%s': file header is not a PDF magic number",
            pdf_file.filename,
        )
        raise HTTPException(
            status_code=400, detail="Invalid file type. File must be a PDF."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.file.read())
        tmp.close()

    try:
        logger.info(
            "Extraction started for '%s' (%d bytes)",
            pdf_file.filename,
            os.path.getsize(tmp.name),
        )
        text = extract_text_from_pdf(tmp.name)

        name_boxes = find_name_bounding_boxes(tmp.name, text)

        query_names = json.loads(names)

        extracted_name_strings = [nb["name"] for nb in name_boxes]
        matches = fuzzy_match_names(extracted_name_strings, query_names)

        logger.info(
            "Extraction finished for '%s': %d name occurrences, %d fuzzy matches",
            pdf_file.filename,
            len(name_boxes),
            len(matches),
        )
        return {
            "extracted_names": [
                {
                    "name": nb["name"],
                    "bounding_box": {
                        "x": nb["x"],
                        "y": nb["y"],
                        "width": nb["width"],
                        "height": nb["height"],
                        "page_number": nb["page"],
                    },
                }
                for nb in name_boxes
            ],
            "fuzzy_matches": matches,
        }
    finally:
        os.unlink(tmp.name)
