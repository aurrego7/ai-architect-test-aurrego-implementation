import spacy


nlp = spacy.load("en_core_web_sm")


def extract_names(text: str) -> list[str]:
    """Extract person names from text using spaCy NER."""
    doc = nlp(text)

    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]


def extract_names_with_positions(text: str) -> list[dict]:
    """Extract person names with their character positions in text."""
    doc = nlp(text)

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
