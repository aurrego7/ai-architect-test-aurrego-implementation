import fitz
import pytesseract
from PIL import Image
import io


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a scanned PDF using OCR."""
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(1, len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"

    return full_text


def get_word_bounding_boxes(pdf_path: str) -> list[dict]:
    """Get bounding boxes for all words in the PDF."""
    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        for i in range(len(ocr_data["text"])):
            word = ocr_data["text"][i].strip()
            if not word:
                continue

            results.append({
                "word": word,
                "page": page_num,
                "x": ocr_data["left"][i],
                "y": ocr_data["top"][i],
                "width": ocr_data["width"][i],
                "height": ocr_data["height"][i],
            })

    return results
