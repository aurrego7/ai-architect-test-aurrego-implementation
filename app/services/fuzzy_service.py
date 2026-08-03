"""Fuzzy matching between caller-supplied names and OCR extracted names."""

import logging
from typing import Protocol

from thefuzz import fuzz

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = get_settings().SIMILARITY_THRESHOLD
# Keeping this here to allow for test to patch without errors


class NameMatcher(Protocol):
    """Interface for scoring query names against extracted names."""

    def match_names(
        self,
        extracted_names: list[str],
        query_names: list[dict],
    ) -> list[dict]:
        """Return the accepted matches between the two name collections."""
        ...


class FuzzyMatcher:
    """Name matcher based on Levenshtein ratio.

    `fuzz.ratio` is used rather than `fuzz.partial_ratio` so that a short
    name contained inside a longer one does not score as a perfect match.

    Attributes:
        similarity_threshold: Minimum score (0-100) a candidate must reach to
            be reported.
    """

    def __init__(self, similarity_threshold: int = SIMILARITY_THRESHOLD) -> None:
        """Initialise the matcher.

        Args:
            similarity_threshold: Minimum score out of 100 for a match to be
                accepted. Defaults to the configured `SIMILARITY_THRESHOLD`.
        """
        self.similarity_threshold = similarity_threshold

    def match_names(
        self,
        extracted_names: list[str],
        query_names: list[dict],
    ) -> list[dict]:
        """Perform fuzzy matching between extracted and query names.

        Each query name is compared against every extracted name and keeps at
        most its single best candidate, so the result has at most one entry
        per query name.

        Args:
            extracted_names: Names recognised in the document. May contain
                duplicates.
            query_names: Names to look for, each a mapping with `first_name`
                and `last_name` keys.

        Returns:
            One dictionary per accepted match, with keys `extracted_name`
            (str), `matched_name` (str, the query name as `"first last"`)
            and `score` (float in 0.0-1.0). Query names whose best candidate
            falls below the threshold are omitted.
        """
        matches = []

        for query in query_names:
            query_full = f"{query['first_name']} {query['last_name']}"

            best_match = None
            best_score = 0

            for extracted in extracted_names:
                score = fuzz.ratio(query_full, extracted)

                if score > best_score:
                    best_score = score
                    best_match = extracted

            if best_score >= self.similarity_threshold:
                matches.append(
                    {
                        "extracted_name": best_match,
                        "matched_name": query_full,
                        "score": best_score / 100.0,
                    }
                )

        logger.debug(
            "Fuzzy matched %d of %d query names (threshold=%d)",
            len(matches),
            len(query_names),
            self.similarity_threshold,
        )
        return matches


_default_matcher = FuzzyMatcher(SIMILARITY_THRESHOLD)
fuzzy_match_names = _default_matcher.match_names
