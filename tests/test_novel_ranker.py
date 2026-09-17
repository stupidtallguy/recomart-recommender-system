import numpy as np
import pandas as pd
import pytest

from src.evaluation.discovery_slot_diagnostic import rerank_with_minimum_novel_slots
from src.models.run_novel_ranker_v1 import (
    informative_novel_queries,
    internal_user_split,
    one_slot_policy,
    paired_intervals,
)


def ranked_pool():
    return pd.DataFrame({
        "product_id": [1, 2, 3, 4, 5],
        "seen_before": [1, 1, 1, 0, 0],
        "score": [5.0, 4.0, 3.0, 2.0, 1.0],
        "novel_score": [np.nan, np.nan, np.nan, 0.1, 0.9],
    })


def test_only_unseen_informative_queries_survive():
    frame = pd.DataFrame({
        "user_id": [3, 3, 1, 1, 2, 2, 4, 4],
        "product_id": [8, 7, 2, 1, 5, 6, 1, 2],
        "seen_before": [0, 0, 0, 1, 0, 0, 0, 0],
        "label": [0, 1, 0, 1, 1, 1, 0, 0],
    })
    actual = informative_novel_queries(frame)
    # User 1 has a seen positive, 2 only positives, 4 only negatives.
    assert actual.user_id.tolist() == [3, 3]
    assert actual.product_id.tolist() == [7, 8]
    assert actual.seen_before.eq(0).all()


def test_internal_split_is_reproducible_and_keeps_users_disjoint():
    users = np.arange(101, 201)
    fit, stop = internal_user_split(users)
    fit_again, stop_again = internal_user_split(users[::-1])
    np.testing.assert_array_equal(fit, fit_again)
    np.testing.assert_array_equal(stop, stop_again)
    assert len(stop) == 10
    assert not set(fit) & set(stop)
    assert set(fit) | set(stop) == set(users)


def test_novel_ranker_changes_item_but_not_slot_budget_or_base_positions():
    pool = ranked_pool()
    control = one_slot_policy(pool, "score", k=3)
    assert control == rerank_with_minimum_novel_slots(pool, 1, k=3)
    novel = one_slot_policy(pool, "novel_score", k=3)
    assert [item["product_id"] for item in control] == [1, 2, 4]
    assert [item["product_id"] for item in novel] == [1, 2, 5]
    assert sum(item["seen_before"] == 0 for item in novel) == 1
    # Arbitrary score scales must not move the inserted product above repeats.
    pool["novel_score"] *= 10000
    assert one_slot_policy(pool, "novel_score", k=3) == novel


def test_existing_novel_recommendations_are_preserved():
    pool = ranked_pool()
    pool.loc[1, "seen_before"] = 0
    original = pool.head(3)[["product_id", "seen_before", "score"]].to_dict("records")
    assert one_slot_policy(pool, "novel_score", k=3) == original


def test_no_available_novel_product_preserves_list():
    pool = ranked_pool()
    pool["seen_before"] = 1
    assert [item["product_id"] for item in one_slot_policy(pool, "novel_score", 3)] == [1, 2, 3]


def test_novel_tie_break_uses_product_id():
    pool = ranked_pool()
    pool.loc[3:, "novel_score"] = 0.9
    assert one_slot_policy(pool, "novel_score", 3)[-1]["product_id"] == 4


def test_bootstrap_uses_paired_differences_and_eligible_users():
    control = pd.DataFrame({"overall": [0.1, 0.2, 0.3], "novel": [np.nan, 0.0, 0.0]}, index=[1, 2, 3])
    candidate = control + 0.1
    result = paired_intervals(candidate, control, samples=100)
    for metric in result.values():
        assert metric["mean_delta"] == pytest.approx(0.1)
        assert metric["ci95_low"] == pytest.approx(0.1)
        assert metric["ci95_high"] == pytest.approx(0.1)
    assert result["novel"]["users"] == 2
    assert result["overall"]["users"] == 3
    with pytest.raises(ValueError, match="identical ordered users"):
        paired_intervals(candidate.iloc[::-1], control, samples=100)
