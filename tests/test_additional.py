"""Additional tests beyond the provided grading suite.

These cover resource cleanup, credential handling, and NER edge cases that
the original test files leave untested.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.core.errors import LLMError
from app.services.rag_service import OpenAIRAG


class TestAdditional:
    """Additional test coverage"""

    def test_document_closed_when_page_processing_fails(self):
        """Should close the PDF even when OCR raises error in the middle of process."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__enter__.return_value = mock_doc

        page = MagicMock()
        pix = MagicMock()
        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="PNG")
        pix.tobytes.return_value = buf.getvalue()
        page.get_pixmap.return_value = pix
        mock_doc.__getitem__ = MagicMock(return_value=page)

        with (
            patch("app.services.ocr_service.fitz.open", return_value=mock_doc),
            patch(
                "app.services.ocr_service.pytesseract.image_to_string",
                side_effect=RuntimeError("tesseract crashed"),
            ),
        ):
            from app.services.ocr_service import extract_text_from_pdf

            with pytest.raises(RuntimeError):
                extract_text_from_pdf("test.pdf")

        mock_doc.close.assert_called_once()

    def test_returns_empty_list_for_text_with_only_org_entities(self):
        """Should return empty list when no person entities found"""
        mock_doc = MagicMock()
        mock_ents = [
            MagicMock(text="Acme Corp", label_="ORG"),
        ]
        mock_doc.ents = mock_ents

        mock_nlp = MagicMock(return_value=mock_doc)

        with patch("app.services.ner_service.nlp", mock_nlp):
            from app.services.ner_service import extract_names

            result = extract_names("He works at Acme Corp with them.")

        assert result == []

    def test_missing_api_key_raises_before_request(self):
        """Should fail fast when no API key is configured."""
        rag = OpenAIRAG(
            embedding_function=lambda question: [0.1] * 384,
            search_function=lambda embedding, top_k: [{"text": "chunk"}],
        )

        with (
            patch("app.services.rag_service.OPENAI_API_KEY", None),
            patch("app.services.rag_service.httpx.post") as mock_post,
            pytest.raises(LLMError),
        ):
            rag.generate_answer("Who signed the contract?")

        mock_post.assert_not_called()
