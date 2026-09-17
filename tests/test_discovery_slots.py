import io
import json
import sys

import numpy as np
import pandas as pd
import pytest

from src.evaluation import discovery_slot_diagnostic
from src.evaluation.discovery_slot_diagnostic import (
    evaluate_policy,
    rerank_with_minimum_novel_slots,
)
from src.models.run_xgb_ranker_v3 import evaluate_predictions


def candidates(seen_flags):
    """A ranked pool with predictable scores and unique product IDs."""
    return pd.DataFrame(
        {
            "product_id": range(1, len(seen_flags) + 1),
            "seen_before": seen_flags,
            "score": [float(len(seen_flags) - i) for i in range(len(seen_flags))],
        }
    )


@pytest.mark.parametrize("quota", [0, 1, 2])
def test_satisfied_minimum_preserves_ranking(quota):
    pool = candidates([0, 1, 0, 1, 0])
    result = rerank_with_minimum_novel_slots(pool, quota, k=3)
    assert [item["product_id"] for item in result] == [1, 2, 3]


def test_quota_replaces_lowest_seen_and_keeps_existing_novel():
    pool = candidates([1, 0, 1, 1, 0, 0, 0])
    result = rerank_with_minimum_novel_slots(pool, 3, k=4)
    assert [item["product_id"] for item in result] == [1, 2, 5, 6]
    assert sum(item["seen_before"] == 0 for item in result) == 3


def test_insufficient_novel_candidates_uses_all_available():
    pool = candidates([1, 1, 1, 1, 0])
    result = rerank_with_minimum_novel_slots(pool, 2, k=3)
    assert [item["product_id"] for item in result] == [1, 2, 5]


def test_short_pool_is_preserved():
    pool = candidates([1, 0])
    result = rerank_with_minimum_novel_slots(pool, 3, k=10)
    assert [item["product_id"] for item in result] == [1, 2]


def test_evaluation_uses_ranker_users_and_fixed_precision_denominator():
    predictions = {
        1: candidates([1, 0]).to_dict("records"),
        2: candidates([1]).to_dict("records"),
    }
    # User 3 belongs to official validation but not ranker validation.
    result = evaluate_policy(
        predictions=predictions,
        latest_orders={1: 3, 2: 12},
        all_truth={1: {1, 2}, 2: {1, 3}, 3: {9}},
        repeat_truth={1: {1}, 2: {1}, 3: {9}},
        novel_truth={1: {2}, 2: {3}, 3: set()},
    )
    assert result["users"] == 2
    assert result["overall"]["recall_at_10"] == pytest.approx(0.75)
    assert result["overall"]["precision_at_10"] == pytest.approx(0.15)
    assert result["repeat"]["repeat_recall_at_10"] == pytest.approx(1.0)
    assert result["novel"]["novel_recall_at_10"] == pytest.approx(0.5)
    assert result["novel"]["novel_hit_rate"] == pytest.approx(0.5)
    assert result["history_buckets"]["01_<=5_orders"]["precision_at_10"] == pytest.approx(0.2)


def test_v3_control_precision_penalizes_short_and_missing_lists():
    result = evaluate_predictions(
        predictions={1: [1, 2]},
        truth={1: {1, 2}, 2: {3}},
        users=np.asarray([1, 2]),
        k=10,
    )
    assert result["precision_at_10"] == pytest.approx(0.1)


def test_complete_diagnostic_prints_on_windows_console(monkeypatch, tmp_path):
    pool = candidates([1] * 10 + [0] * 3)
    pool["user_id"] = 1
    pool["user_latest_order"] = 4
    output_path = tmp_path / "diagnostic.json"
    monkeypatch.setattr(discovery_slot_diagnostic, "load_truth", lambda: ({1: {1, 11}}, {1: {1}}, {1: {11}}))
    monkeypatch.setattr(discovery_slot_diagnostic, "load_scored_candidates", lambda: pool)
    monkeypatch.setattr(discovery_slot_diagnostic, "RESULT_PATH", output_path)

    # Strict legacy Windows encoding catches failures after metrics are saved.
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
    with monkeypatch.context() as console_patch:
        console_patch.setattr(sys, "stdout", stream)
        discovery_slot_diagnostic.main()
        stream.flush()
    assert "Saved:" in buffer.getvalue().decode("cp1252")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(result["policies"]) == 4
    assert result["policies"]["minimum_1_novel_slots"]["novel"]["novel_recall_at_10"] == 1.0
