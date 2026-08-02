import io

import fitz
import pytesseract
from PIL import Image


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a scanned PDF using OCR."""
    full_text = ""

    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            full_text += text + "\n"

    # Close document to avoid memory leak
    # This is redudant b/c with statement closes the executor.
    # Adding to satisfy the test
    doc.close()

    return full_text


def get_word_bounding_boxes(pdf_path: str) -> list[dict]:
    """Get bounding boxes for all words in the PDF."""
    results = []

    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
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
                        "width": ocr_data["width"][i],
                        "height": ocr_data["height"][i],
                    }
                )

    # Close document to avoid memory leak
    # This is redudant b/c with statement closes the executor.
    # Adding to satisfy the test
    doc.close()

    return results
