import pytest

from src.serving.routing import route_recommendations


def test_zero_history_ignores_personalized_lists():
    result = route_recommendations(
        history_order_count=0, ranked=[99, 98], repeat=[97], popularity=[1, 2, 3], k=3,
    )
    assert result.recommendations == [1, 2, 3]
    assert result.strategy == "global_popularity"
    assert result.complete


@pytest.mark.parametrize("orders", [1, 2, 3, 5, 10, 30])
def test_all_known_history_segments_keep_available_ranker(orders):
    result = route_recommendations(
        history_order_count=orders, ranked=[9, 7, 8], repeat=[6, 5, 4], popularity=[1, 2, 3], k=3,
    )
    assert result.recommendations == [9, 7, 8]
    assert result.strategy == "ranker"


def test_missing_ranker_uses_repeat_then_unique_popularity():
    result = route_recommendations(
        history_order_count=2, ranked=[], repeat=[8, 7], popularity=[7, 1, 2], k=4,
    )
    assert result.recommendations == [8, 7, 1, 2]
    assert result.strategy == "repeat_with_popularity_fill"
    assert result.complete


def test_short_ranker_is_preserved_and_filled():
    result = route_recommendations(
        history_order_count=1, ranked=[8, 8], popularity=[8, 2, 3], k=3,
    )
    assert result.recommendations == [8, 2, 3]
    assert result.strategy == "ranker_with_popularity_fill"


def test_missing_personalization_and_insufficient_catalog_are_explicit():
    result = route_recommendations(history_order_count=3, popularity=[2, 2], k=3)
    assert result.recommendations == [2]
    assert result.strategy == "global_popularity"
    assert not result.complete


def test_invalid_counts_are_rejected():
    with pytest.raises(ValueError):
        route_recommendations(history_order_count=-1, popularity=[1])
