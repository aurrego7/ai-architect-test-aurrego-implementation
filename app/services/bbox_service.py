"""Locating recognised names on the page."""

import logging
import string
from collections.abc import Callable
from typing import Protocol

from app.services.ner_service import extract_names
from app.services.ocr_service import get_word_bounding_boxes

logger = logging.getLogger(__name__)


class BBoxService(Protocol):
    """Interface for resolving names to page regions."""

    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        """Return a box for every occurrence of a recognised name."""
        ...


class BBoxLocator:
    """Locates person names in a PDF by aligning NER output with OCR words.

    Both collaborators are injectable so the matching algorithm can be tested
    without running spaCy or Tesseract, and so either stage can be replaced
    (for example NER swapped for an LLM extractor) as long as the injected
    callable keeps the same output shape.

    Attributes:
        ner_function: Callable mapping text to a list of names. When `None`
            the default spaCy-backed extractor is used.
        word_boxes_function: Callable mapping a PDF path to positioned words.
            When `None` the default Tesseract-backed extractor is used.
    """

    def __init__(
        self,
        ner_function: Callable[[str], list[str]] | None = None,
        word_boxes_function: Callable[[str], list[dict]] | None = None,
    ) -> None:
        """Initialize the locator.

        Args:
            ner_function: Optional replacement for the default name extractor.
                Must accept the document text and return a list of names.
            word_boxes_function: Optional replacement for the default word-box
                extractor. Must accept a PDF path and return dictionaries with
                `word`, `page`, `x`, `y`, `width` and `height`.
        """
        self.ner_function = ner_function
        self.word_boxes_function = word_boxes_function

    def find_name_bounding_boxes(self, pdf_path: str, text: str) -> list[dict]:
        """Match extracted names to their bounding boxes in the PDF.

        We iterate on the words and then check to see if there is any name that matches
        on that word and forward. This reduces the number of times we need to iterate on
        each word (only going forwards, rather than every time from start for each name)

        A name may be split by an unexpected token (a middle name, an OCR
        artefact), so each subsequent part is searched for within a window of
        `2 * len(name_parts)` words on the same page. A name only counts as
        found when EVERY one of its parts is matched, and the reported box is
        the union of the matched word boxes.

        Args:
            pdf_path: Filesystem path to the PDF the text came from.
            text: OCR transcript of that PDF, used for name recognition.

        Returns:
            One dictionary per name occurrence, with keys `name` (str),
            `page` (int, zero-based) and `x`, `y`, `width`, `height`
            (float, PDF points). A name appearing several times yields several
            entries; a recognised name whose parts cannot be aligned to the
            word stream yields none.
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

        logger.debug(
            "Located %d name occurrences for %d unique names across %d words",
            len(name_boxes),
            len(names),
            len(word_boxes),
        )
        return name_boxes


_default_bbox = BBoxLocator()
find_name_bounding_boxes = _default_bbox.find_name_bounding_boxes
