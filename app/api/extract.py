import json
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.core.errors import OCRError
from app.models.schemas import ExtractionResponse, NamePair
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

    # Validate the names payload before doing any expensive OCR work
    # JSONDecodeError: not valid JSON
    # TypeError: an item is not a dict
    # ValidationError: an item is missing/has wrong fields
    try:
        raw_names = json.loads(names)
        query_names = [NamePair(**item).model_dump() for item in raw_names]
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        logger.warning("Rejected extraction request: invalid names payload")
        raise HTTPException(
            status_code=400,
            detail="Invalid names format. Expected a JSON array of "
            '{"first_name": ..., "last_name": ...} objects.',
        ) from exc

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
    except OCRError as exc:
        raise HTTPException(
            status_code=400, detail="Could not process the PDF file."
        ) from exc
    finally:
        os.unlink(tmp.name)
