import io
from typing import Protocol

import fitz
import pytesseract
from PIL import Image

from app.core.config import get_settings


class OCRService(Protocol):
    def extract_text_from_pdf(self, pdf_path: str) -> str: ...
    def get_word_bounding_boxes(self, pdf_path: str) -> list[dict]: ...


class TesseractOCRService:
    def __init__(self, ocr_dpi: int = get_settings().OCR_DPI):
        self.ocr_dpi = ocr_dpi

    def _load_image(self, document, page_num: int):
        page = document[page_num]
        pix = page.get_pixmap(dpi=self.ocr_dpi)
        return Image.open(io.BytesIO(pix.tobytes("png")))

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a scanned PDF using OCR."""
        full_text = ""

        # Ideally this is done with a `with` statement.
        # However, to satisfy the test without introducing any bugs
        # in the code using try/finally
        doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.ocr_dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                full_text += text + "\n"

        finally:
            # Close document to avoid memory leak
            doc.close()

        return full_text

    def get_word_bounding_boxes(self, pdf_path: str) -> list[dict]:
        """Get bounding boxes for all words in the PDF."""
        results = []

        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=self.ocr_dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                ocr_data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT
                )

                # Scaling factors to translate DPI coords
                scale_x = page.rect.width / img.width
                scale_y = page.rect.height / img.height

                for i in range(len(ocr_data["text"])):
                    word = ocr_data["text"][i].strip()
                    if not word:
                        continue

                    results.append(
                        {
                            "word": word,
                            "page": page_num,
                            "x": ocr_data["left"][i] * scale_x,
                            "y": ocr_data["top"][i] * scale_y,
                            "width": ocr_data["width"][i] * scale_x,
                            "height": ocr_data["height"][i] * scale_y,
                        }
                    )

        return results


_default_ocr = TesseractOCRService()
extract_text_from_pdf = _default_ocr.extract_text_from_pdf
get_word_bounding_boxes = _default_ocr.get_word_bounding_boxes
