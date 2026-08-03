"""Named-entity recognition for person names."""

import logging
from typing import Protocol

from spacy.language import Language

from app.core.providers import load_spacy_model

logger = logging.getLogger(__name__)

nlp = load_spacy_model()


class NERService(Protocol):
    """Interface for pulling person names out of free text."""

    def extract_names(self, text: str) -> list[str]:
        """Return the person names found in `text`."""
        ...

    def extract_names_with_positions(self, text: str) -> list[dict]:
        """Return the person names in `text` with their character offsets."""
        ...


class SpacyNERService:
    """Person-name recogniser backed by a spaCy pipeline.

    Attributes:
        model: Pipeline to run. When `None` the module-level shared pipeline
            is used instead, which is what production code relies on; tests
            inject a stub here.
    """

    def __init__(self, model: Language | None = None) -> None:
        """Initialise the service.

        Args:
            model: Optional spaCy pipeline to use instead of the shared one.
        """
        self.model = model

    def extract_names(self, text: str) -> list[str]:
        """Extract person names from text using spaCy NER.

        Names are returned in the order they appear and duplicates are kept,
        so a name mentioned three times yields three entries.

        Args:
            text: Free text to analyse, typically a full OCR transcript.

        Returns:
            The surface form of every `PERSON` entity found. Empty when the
            text contains no person entities.
        """
        model = self.model or nlp
        doc = model(text)

        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        logger.debug(
            "NER found %d PERSON entities out of %d total entities",
            len(names),
            len(doc.ents),
        )
        return names

    def extract_names_with_positions(self, text: str) -> list[dict]:
        """Extract person names with their character positions in text.

        Args:
            text: Free text to analyse.

        Returns:
            One dictionary per `PERSON` entity, in order of appearance, with
            keys `name` (str), `start_char` (int), `end_char` (int) and
            `label` (str, always `"PERSON"`).
        """
        model = self.model or nlp
        doc = model(text)

        return [
            {
                "name": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_,
            }
            for ent in doc.ents
            if ent.label_ == "PERSON"
        ]


_default_ner = SpacyNERService()
extract_names = _default_ner.extract_names
extract_names_with_positions = _default_ner.extract_names_with_positions
