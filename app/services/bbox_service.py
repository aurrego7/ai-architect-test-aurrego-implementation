from app.services.ocr_service import get_word_bounding_boxes
from app.services.ner_service import extract_names


def find_name_bounding_boxes(pdf_path: str, text: str) -> list[dict]:
    """Match extracted names to their bounding boxes in the PDF."""
    names = extract_names(text)
    word_boxes = get_word_bounding_boxes(pdf_path)

    name_boxes = []

    for name in names:
        name_parts = name.split()
        matched_boxes = []

        for part in name_parts:
            for word_box in word_boxes:
                if word_box["word"] == part:
                    matched_boxes.append(word_box)
                    break

        if matched_boxes:
            min_x = min(b["x"] for b in matched_boxes)
            min_y = min(b["y"] for b in matched_boxes)
            max_x = max(b["x"] + b["width"] for b in matched_boxes)
            max_y = max(b["y"] + b["height"] for b in matched_boxes)

            name_boxes.append({
                "name": name,
                "page": matched_boxes[0]["page"],
                "x": min_x,
                "y": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y,
            })

    return name_boxes
