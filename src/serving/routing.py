"""Route already-computed recommendations according to available history/data."""

from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class RecommendationRoute:
    recommendations: list[int]
    strategy: str
    complete: bool


def fill_with_popularity(items, popular_items, k: int = 10) -> list[int]:
    """Preserve available personalized order and fill gaps without duplicates."""
    if k < 0:
        raise ValueError("k cannot be negative.")
    if k == 0:
        return []
    result = []
    used = set()
    for source in (items, popular_items):
        for item in source:
            product_id = int(item)
            if product_id not in used:
                used.add(product_id)
                result.append(product_id)
                if len(result) == k:
                    return result
    return result


def route_recommendations(
    *,
    history_order_count: int,
    popularity: Sequence[int],
    ranked: Sequence[int] | None = None,
    repeat: Sequence[int] | None = None,
    k: int = 10,
) -> RecommendationRoute:
    """Choose a source without needing Spark, pandas, neural models, or a DB.

    Callers supply history counts and candidate lists from the same snapshot.
    Zero-history requests always ignore personalized lists. No threshold greater
    than zero overrides the ranker. Fewer than K available unique items returns
    a short list with complete=False; products are never fabricated.
    """
    if history_order_count < 0 or k < 0:
        raise ValueError("History count and k must be nonnegative.")
    if k == 0:
        return RecommendationRoute([], "empty_request", True)
    if history_order_count == 0:
        items = fill_with_popularity([], popularity, k)
        return RecommendationRoute(items, "global_popularity", len(items) == k)

    if ranked is not None and len(ranked):
        unique_ranked = fill_with_popularity(ranked, [], k)
        items = fill_with_popularity(unique_ranked, popularity, k)
        strategy = "ranker" if len(unique_ranked) == k else "ranker_with_popularity_fill"
    elif repeat is not None and len(repeat):
        items = fill_with_popularity(repeat, popularity, k)
        strategy = "repeat_with_popularity_fill"
    else:
        items = fill_with_popularity([], popularity, k)
        strategy = "global_popularity"
    return RecommendationRoute(items, strategy, len(items) == k)
