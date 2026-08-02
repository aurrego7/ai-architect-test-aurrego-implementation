"""Tests for NER service.

These tests verify named entity extraction.
Some tests will FAIL due to bugs in the current implementation.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestExtractNames:
    """Tests for extract_names function."""

    def test_extracts_only_person_entities(self):
        """FAILS: Bug — function also extracts ORG entities.
        'Acme Corp' should NOT appear in results.
        """
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(text="John Smith", label_="PERSON"),
            MagicMock(text="Acme Corp", label_="ORG"),
            MagicMock(text="Maria Garcia", label_="PERSON"),
        ]
        mock_doc.ents = mock_ents

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.services.ner_service.nlp", mock_nlp):
            from app.services.ner_service import extract_names

            result = extract_names("John Smith works at Acme Corp with Maria Garcia")

        assert "John Smith" in result
        assert "Maria Garcia" in result
        assert "Acme Corp" not in result, (
            "ORG entities should not be returned as person names"
        )

    def test_returns_empty_list_for_no_names(self):
        """Should return empty list when no person entities found."""
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.services.ner_service.nlp", mock_nlp):
            from app.services.ner_service import extract_names

            result = extract_names("This text has no names in it.")

        assert result == []

    def test_handles_multiple_entity_types(self):
        """Should only return PERSON entities, ignoring GPE, DATE, etc."""
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(text="Sarah Williams", label_="PERSON"),
            MagicMock(text="New York", label_="GPE"),
            MagicMock(text="January 2024", label_="DATE"),
            MagicMock(text="MIT", label_="ORG"),
        ]
        mock_doc.ents = mock_ents
        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.services.ner_service.nlp", mock_nlp):
            from app.services.ner_service import extract_names

            result = extract_names("Sarah Williams visited New York in January 2024.")

        assert result == ["Sarah Williams"]


class TestExtractNamesWithPositions:
    """Tests for extract_names_with_positions function."""

    def test_returns_positions_for_person_only(self):
        """FAILS: Bug — includes ORG entities in results."""
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(
                text="John Smith",
                label_="PERSON",
                start_char=0,
                end_char=10,
            ),
            MagicMock(
                text="Google",
                label_="ORG",
                start_char=25,
                end_char=31,
            ),
        ]
        mock_doc.ents = mock_ents
        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.services.ner_service.nlp", mock_nlp):
            from app.services.ner_service import extract_names_with_positions

            result = extract_names_with_positions("John Smith works at Google.")

        assert len(result) == 1, "Should only return PERSON entities"
        assert result[0]["name"] == "John Smith"
        assert result[0]["label"] == "PERSON"
