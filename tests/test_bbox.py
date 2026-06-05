"""Tests for bounding box service.

These tests verify name-to-bounding-box matching.
Some tests will FAIL due to bugs in the current implementation.
"""

import pytest
from unittest.mock import patch


class TestFindNameBoundingBoxes:
    """Tests for find_name_bounding_boxes function."""

    def test_case_insensitive_matching(self):
        """FAILS: Bug — OCR returns 'JOHN' but NER returns 'John'.
        Case-sensitive comparison fails to match.
        """
        mock_names = ["John Smith"]
        mock_word_boxes = [
            {"word": "JOHN", "page": 0, "x": 100, "y": 50, "width": 40, "height": 12},
            {"word": "SMITH", "page": 0, "x": 145, "y": 50, "width": 50, "height": 12},
        ]

        with (
            patch(
                "app.services.bbox_service.extract_names", return_value=mock_names
            ),
            patch(
                "app.services.bbox_service.get_word_bounding_boxes",
                return_value=mock_word_boxes,
            ),
        ):
            from app.services.bbox_service import find_name_bounding_boxes

            result = find_name_bounding_boxes("test.pdf", "dummy text")

        assert len(result) == 1, (
            "Should match 'John' to 'JOHN' case-insensitively"
        )
        assert result[0]["name"] == "John Smith"

    def test_merges_multiword_name_boxes(self, sample_word_boxes):
        """Should merge bounding boxes for multi-word names correctly."""
        mock_names = ["John Smith"]

        with (
            patch(
                "app.services.bbox_service.extract_names", return_value=mock_names
            ),
            patch(
                "app.services.bbox_service.get_word_bounding_boxes",
                return_value=sample_word_boxes,
            ),
        ):
            from app.services.bbox_service import find_name_bounding_boxes

            result = find_name_bounding_boxes("test.pdf", "dummy text")

        assert len(result) == 1
        box = result[0]
        # Merged box should span from "John" to "Smith"
        assert box["x"] == 100  # min x
        assert box["y"] == 50  # min y
        assert box["width"] == 95  # 145 + 50 - 100

    def test_handles_duplicate_names(self):
        """FAILS: Bug — always returns first occurrence due to early break.
        If 'John Smith' appears twice, both should have bounding boxes.
        """
        mock_names = ["John Smith"]
        mock_word_boxes = [
            {"word": "John", "page": 0, "x": 100, "y": 50, "width": 40, "height": 12},
            {"word": "Smith", "page": 0, "x": 145, "y": 50, "width": 50, "height": 12},
            {"word": "John", "page": 1, "x": 200, "y": 100, "width": 40, "height": 12},
            {"word": "Smith", "page": 1, "x": 245, "y": 100, "width": 50, "height": 12},
        ]

        with (
            patch(
                "app.services.bbox_service.extract_names", return_value=mock_names
            ),
            patch(
                "app.services.bbox_service.get_word_bounding_boxes",
                return_value=mock_word_boxes,
            ),
        ):
            from app.services.bbox_service import find_name_bounding_boxes

            result = find_name_bounding_boxes("test.pdf", "dummy")

        # The current implementation only finds first occurrence.
        # This test documents the bug — candidate should decide if
        # returning all occurrences is the correct behavior.
        assert result[0]["page"] == 0
