import numpy as np
import pandas as pd
import pytest

from src.evaluation.history_availability import (
    cold_start_data,
    evaluate_users,
    fill_with_popularity,
    history_segment,
    popular_products,
)


@pytest.mark.parametrize("count,expected", [
    (0, "00_zero_orders"), (1, "01_1_2_orders"), (2, "01_1_2_orders"),
    (3, "02_3_5_orders"), (5, "02_3_5_orders"), (6, "03_6_10_orders"),
    (10, "03_6_10_orders"), (11, "04_over_10_orders"),
])
def test_history_boundaries(count, expected):
    assert history_segment(count) == expected


def test_fallback_fills_without_reordering_or_duplicates():
    assert fill_with_popularity([3, 1, 3], [1, 2, 3, 4], k=4) == [3, 1, 2, 4]
    assert fill_with_popularity([], [2, 1], k=10) == [2, 1]
    assert fill_with_popularity([3], [1, 2], k=0) == []
    with pytest.raises(ValueError):
        fill_with_popularity([], [], k=-1)


def test_cold_users_are_excluded_and_only_first_recorded_basket_is_truth():
    train = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3],
        "order_number": [3, 1, 1, 1, 2, 2, 5],
        "product_id": [99, 7, 8, 4, 4, 9, 99],
    })
    warm, truth = cold_start_data(train, np.asarray([1, 3]))
    assert set(warm.user_id) == {2}
    assert truth == {1: {7, 8}, 3: {9}}
    assert popular_products(warm) == [4]
    # Changing hidden cold-user history cannot change the popularity fit.
    changed = train.copy()
    changed.loc[changed.user_id.isin([1, 3]), "product_id"] = 500
    changed_warm, _ = cold_start_data(changed, np.asarray([1, 3]))
    assert popular_products(changed_warm) == [4]


def test_popularity_ties_are_deterministic():
    frame = pd.DataFrame({"product_id": [5, 2, 5, 2, 9]})
    assert popular_products(frame, k=3) == [2, 5, 9]


def test_missing_users_are_not_silently_dropped():
    with pytest.raises(ValueError, match="identical users"):
        evaluate_users({1: [1]}, {1: {1}, 2: {2}}, {1: 1})


def test_short_lists_use_fixed_k_and_novel_metrics_use_eligible_truth():
    result = evaluate_users(
        {1: [1, 2], 2: [3]}, {1: {1, 2}, 2: {3}}, {1: 1, 2: 3},
        {1: {2}, 2: set()},
    )
    assert result.loc[1, "precision_at_10"] == 0.2
    assert result.loc[2, "precision_at_10"] == 0.1
    assert result.loc[1, "novel_recall_at_10"] == 1.0
    assert np.isnan(result.loc[2, "novel_recall_at_10"])
    assert result.loc[1, "segment"] == "01_1_2_orders"
