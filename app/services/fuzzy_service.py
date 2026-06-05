from thefuzz import fuzz


SIMILARITY_THRESHOLD = 70


def fuzzy_match_names(
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
            score = fuzz.partial_ratio(query_full, extracted)

            if score > best_score:
                best_score = score
                best_match = extracted

        if best_score >= SIMILARITY_THRESHOLD:
            matches.append({
                "extracted_name": best_match,
                "matched_name": query_full,
                "score": best_score / 100.0,
            })

    return matches
