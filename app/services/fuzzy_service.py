import logging
from typing import Protocol

from thefuzz import fuzz

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = get_settings().SIMILARITY_THRESHOLD
# Keeping this here to allow for test to patch without errors


class NameMatcher(Protocol):
    def match_names(
        self,
        extracted_names: list[str],
        query_names: list[dict],
    ) -> list[dict]: ...


class FuzzyMatcher:
    def __init__(self, similarity_threshold: int = SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold

    def match_names(
        self,
        extracted_names: list[str],
        query_names: list[dict],
    ) -> list[dict]:
        """Perform fuzzy matching between extracted and query names."""
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
