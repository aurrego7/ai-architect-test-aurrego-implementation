"""Tests for OCR service.

These tests verify text extraction from PDF documents.
Some tests will FAIL due to bugs in the current implementation.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


class TestExtractTextFromPDF:
    """Tests for extract_text_from_pdf function."""

    def test_extracts_text_from_all_pages(self):
        """FAILS: Bug — first page is skipped due to off-by-one error.
        The function starts iteration at page 1 instead of page 0.
        """
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)

        # mock_doc.__enter__ returns a brand new mock, not mock_doc
        # this causes with statements to break
        mock_doc.__enter__.return_value = mock_doc

        mock_pages = []
        for _ in range(3):
            page = MagicMock()
            pix = MagicMock()
            # Create a minimal valid PNG
            img = Image.new("RGB", (100, 100))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pix.tobytes.return_value = buf.getvalue()
            page.get_pixmap.return_value = pix
            mock_pages.append(page)

        mock_doc.__getitem__ = MagicMock(side_effect=lambda i: mock_pages[i])

        with (
            patch("app.services.ocr_service.fitz.open", return_value=mock_doc),
            patch(
                "app.services.ocr_service.pytesseract.image_to_string",
                side_effect=[f"Page {i} text" for i in range(3)],
            ),
        ):
            from app.services.ocr_service import extract_text_from_pdf

            result = extract_text_from_pdf("test.pdf")

        # Should contain text from ALL 3 pages including page 0
        assert "Page 0 text" in result, "First page text is missing — off-by-one bug"
        assert "Page 1 text" in result
        assert "Page 2 text" in result

    def test_returns_empty_string_for_empty_pdf(self):
        """Should handle empty PDFs gracefully."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)

        # mock_doc.__enter__ returns a brand new mock, not mock_doc
        # this causes with statements to break
        mock_doc.__enter__.return_value = mock_doc

        with patch("app.services.ocr_service.fitz.open", return_value=mock_doc):
            from app.services.ocr_service import extract_text_from_pdf

            result = extract_text_from_pdf("empty.pdf")

        assert result == ""

    def test_document_is_properly_closed(self):
        """FAILS: Bug — document is never closed after processing.
        Should use context manager or explicit close().
        """
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)

        # mock_doc.__enter__ returns a brand new mock, not mock_doc
        # this causes with statements to break
        mock_doc.__enter__.return_value = mock_doc

        with patch("app.services.ocr_service.fitz.open", return_value=mock_doc):
            from app.services.ocr_service import extract_text_from_pdf

            extract_text_from_pdf("test.pdf")

        mock_doc.close.assert_called_once()


class TestGetWordBoundingBoxes:
    """Tests for get_word_bounding_boxes function."""

    def test_coordinates_in_pdf_space(self):
        """FAILS: Bug — coordinates are in OCR image space (150 DPI),
        not converted back to PDF coordinate space (72 DPI).
        """
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)

        # mock_doc.__enter__ returns a brand new mock, not mock_doc
        # this causes with statements to break
        mock_doc.__enter__.return_value = mock_doc

        page = MagicMock()
        # PDF page is 612x792 points (8.5x11 inches at 72 DPI)
        page.rect = MagicMock()
        page.rect.width = 612
        page.rect.height = 792

        pix = MagicMock()
        img = Image.new("RGB", (1275, 1650))  # 8.5x11 at 150 DPI
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        pix.tobytes.return_value = buf.getvalue()
        page.get_pixmap.return_value = pix

        mock_doc.__getitem__ = MagicMock(return_value=page)

        ocr_data = {
            "text": ["Hello"],
            "left": [150],  # 150px at 150 DPI = 1 inch = 72 PDF points
            "top": [300],
            "width": [200],
            "height": [30],
            "conf": [95],
        }

        with (
            patch("app.services.ocr_service.fitz.open", return_value=mock_doc),
            patch(
                "app.services.ocr_service.pytesseract.image_to_data",
                return_value=ocr_data,
            ),
        ):
            from app.services.ocr_service import get_word_bounding_boxes

            results = get_word_bounding_boxes("test.pdf")

        assert len(results) == 1
        # At 150 DPI, 150px = 1 inch = 72 PDF points
        # Coordinates should be converted: x * (72/150)
        assert results[0]["x"] == pytest.approx(72.0, abs=1.0), (
            "X coordinate should be in PDF space (72 DPI), not OCR space (150 DPI)"
        )
        # top = 300 px at 150 DPI is 2 inches = 144 PDF points
        assert results[0]["y"] == pytest.approx(144.0, abs=1.0), (
            "Y coordinate should be in PDF space (72 DPI), not OCR space (150 DPI)"
        )
