from typing import Protocol

from app.core.providers import load_spacy_model

nlp = load_spacy_model()


class NERService(Protocol):
    def extract_names(self, text: str) -> list[str]: ...
    def extract_names_with_positions(self, text: str) -> list[dict]: ...


class SpacyNERService:
    def __init__(self, model=None):
        self.model = model

    def extract_names(self, text: str) -> list[str]:
        """Extract person names from text using spaCy NER."""
        model = self.model or nlp
        doc = model(text)

        return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    def extract_names_with_positions(self, text: str) -> list[dict]:
        """Extract person names with their character positions in text."""
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
