import string
from typing import Protocol

from app.services.ner_service import extract_names
from app.services.ocr_service import get_word_bounding_boxes


class BBoxService(Protocol):
    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]: ...


class BBoxLocator:
    def __init__(self, ner_function=None, word_boxes_function=None):
        self.ner_function = ner_function
        self.word_boxes_function = word_boxes_function

    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        """Match extracted names to their bounding boxes in the PDF.

        We iterate on the words and then check to see if there is any name that matches
        on that word and forward. This reduces the number of times we need to iterate on
        each word (only going forwards, rather than every time from start for each name)
        """
        # Find functions to be used
        # Allows for swapping with defaults
        get_names = self.ner_function or extract_names
        get_bounding_boxes = self.word_boxes_function or get_word_bounding_boxes

        # NER name extraction returns every name so it contains duplicates
        # Deplucating here and then the finding algorithm will find all ocurrences
        names = list(dict.fromkeys(get_names(text)))
        word_boxes = get_bounding_boxes(pdf_path)
        words = [
            b["word"].casefold().strip(string.punctuation) for b in word_boxes
        ]  # casefold for case-insensitive and punctuation-insensitive

        name_boxes = []

        i = 0
        while i < len(words):
            furthest = i
            for name in names:
                matched_boxes = []
                name_parts = [
                    part.casefold().strip(string.punctuation) for part in name.split()
                ]  # casefold for case-insensitive and punctuation-insensitive

                # If a blank name_part skip
                if not name_parts:
                    continue

                # If the first word of the name is not this word
                # skip since name doesn't start here
                if words[i] != name_parts[0]:
                    continue

                page = word_boxes[i]["page"]

                # max_gap helps for cases when name has an intermediate word
                # ex. name = "Jhon Smith" and you find "Jhon Michael Smith"
                # Potential improvement in better computation of limit to search for
                max_gap = len(name_parts) * 2

                word_box_index = i
                for part in name_parts:
                    # Iterate over all word_boxes after the previous word_box that
                    # matched up to the max_gap. Bound at the last word to
                    # prevent index error
                    for j in range(
                        word_box_index,
                        min(word_box_index + max_gap + 1, len(word_boxes)),
                    ):
                        if words[j] == part and word_boxes[j]["page"] == page:
                            matched_boxes.append(word_boxes[j])
                            word_box_index = j + 1
                            break  # Break at match since name_part was found on word

                # Only if all of the parts of the name are found save it
                if len(matched_boxes) == len(name_parts):
                    min_x = min(b["x"] for b in matched_boxes)
                    min_y = min(b["y"] for b in matched_boxes)
                    max_x = max(b["x"] + b["width"] for b in matched_boxes)
                    max_y = max(b["y"] + b["height"] for b in matched_boxes)

                    name_boxes.append(
                        {
                            "name": name,
                            "page": matched_boxes[0]["page"],
                            "x": min_x,
                            "y": min_y,
                            "width": max_x - min_x,
                            "height": max_y - min_y,
                        }
                    )
                    furthest = max(furthest, word_box_index - 1)

            i = furthest + 1

        return name_boxes

_default_bbox = BBoxLocator()
find_name_bounding_boxes = _default_bbox.find_name_bounding_boxes