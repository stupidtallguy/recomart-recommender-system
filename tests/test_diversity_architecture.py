import pytest

from src.evaluation.diversity_architecture import (
    cap_aisles,
    category_metrics,
    eligible_diversity,
    noninferior,
)


def test_unconstrained_ranking_is_preserved():
    assert cap_aisles([4, 2, 8, 1], {}, None, k=3) == [4, 2, 8]


def test_cap_selects_other_aisles_and_restores_score_order():
    products = [1, 2, 3, 4, 5, 6]
    aisles = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}
    assert cap_aisles(products, aisles, cap=2, k=5) == [1, 2, 4, 5, 6]


def test_impossible_cap_backfills_without_shortening_or_duplicates():
    products = [1, 2, 3, 4, 5]
    aisles = {product: 1 for product in products}
    assert cap_aisles(products, aisles, cap=2, k=4) == [1, 2, 3, 4]
    assert cap_aisles(products[:2], aisles, cap=1, k=10) == [1, 2]


def test_feasible_cap_keeps_already_compliant_topk():
    aisles = {1: 1, 2: 2, 3: 1, 4: 3}
    assert cap_aisles([1, 2, 3, 4], aisles, cap=2, k=3) == [1, 2, 3]


def test_invalid_limits_and_duplicate_products_are_rejected():
    with pytest.raises(ValueError):
        cap_aisles([1], {1: 1}, cap=0)
    with pytest.raises(ValueError):
        cap_aisles([1, 1], {1: 1}, cap=1)
    assert cap_aisles([1], {1: 1}, cap=1, k=0) == []


def test_diversity_is_fraction_of_different_aisle_pairs():
    metrics = category_metrics([1, 2, 3], {1: 5, 2: 5, 3: 8}, {1: 1, 2: 1, 3: 2})
    assert metrics["aisle_pair_diversity"] == pytest.approx(2 / 3)
    assert metrics["unique_aisles"] == 2
    assert metrics["dominant_aisle_share"] == pytest.approx(2 / 3)


def test_complexity_gate_requires_both_accuracy_bounds():
    acceptable = {metric: {"ci95_low": -0.0009} for metric in ("recall_at_10", "ndcg_at_10")}
    assert noninferior(acceptable)
    acceptable["ndcg_at_10"]["ci95_low"] = -0.0011
    assert not noninferior(acceptable)


def test_diversity_gate_does_not_treat_uncertainty_as_a_win():
    accuracy = {metric: {"ci95_low": 0.0001} for metric in ("recall_at_10", "ndcg_at_10")}
    assert eligible_diversity(accuracy, 0.05)
    assert not eligible_diversity(accuracy, 0.0)
    accuracy["recall_at_10"]["ci95_low"] = -0.00001
    assert not eligible_diversity(accuracy, 0.05)
