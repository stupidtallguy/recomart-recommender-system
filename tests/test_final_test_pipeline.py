import json

import pandas as pd
import pytest

from src.evaluation import final_test_pipeline as final


@pytest.mark.parametrize("predictions,message", [
    ({1: [10]}, "cohort"),
    ({1: [10, 10], 2: []}, "duplicates"),
    ({1: [99], 2: []}, "Unknown"),
    ({1: list(range(11)), 2: []}, "length"),
])
def test_invalid_predictions_are_rejected(predictions, message):
    with pytest.raises(ValueError, match=message):
        final.validate_predictions(predictions, {1, 2}, set(range(20)))


def test_empty_baseline_predictions_are_preserved_but_primary_must_be_complete():
    final.validate_predictions({1: []}, {1}, {10})
    with pytest.raises(ValueError, match="Incomplete"):
        final.validate_predictions({1: []}, {1}, {10}, require_full=True)


def test_temporal_overlap_and_missing_users_are_rejected():
    train = pd.DataFrame({"user_id": [1, 2], "order_number": [2, 4]})
    val = pd.DataFrame({"user_id": [1, 2], "order_number": [3, 5]})
    final.validate_temporal_history(train, val, 2)
    val.loc[0, "order_number"] = 2
    with pytest.raises(ValueError, match="Temporal"):
        final.validate_temporal_history(train, val, 2)
    with pytest.raises(ValueError, match="cohort"):
        final.validate_temporal_history(train, val.iloc[:1], 2)


def test_multiple_validation_baskets_are_rejected():
    train = pd.DataFrame({"user_id": [1], "order_number": [1]})
    val = pd.DataFrame({"user_id": [1, 1], "order_number": [2, 3]})
    with pytest.raises(ValueError, match="one validation basket"):
        final.validate_temporal_history(train, val, 1)


def test_hash_guard_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "ROOT", tmp_path)
    artifact = tmp_path / "predictions.parquet"
    artifact.write_bytes(b"original predictions")
    hashes = final.inventory([artifact])
    final.verify_hashes(hashes)
    artifact.write_bytes(b"changed predictions")
    with pytest.raises(ValueError, match="Artifact changed"):
        final.verify_hashes(hashes)


def test_evaluation_does_not_read_truth_without_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "MODELS", tmp_path)
    def forbid_read(*args, **kwargs):
        pytest.fail("Evaluation attempted to read data before checking the seal")
    monkeypatch.setattr(pd, "read_parquet", forbid_read)
    with pytest.raises(FileNotFoundError):
        final.evaluate()


def test_seal_requires_every_baseline_before_reading_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "ROOT", tmp_path)
    monkeypatch.setattr(final, "MODELS", tmp_path)
    monkeypatch.setattr(final, "DATA", tmp_path)
    config = tmp_path / "contract.json"
    config.write_text(json.dumps({"final_test_protocol": {"expected_users": 1, "baselines": ["selected_lambdamart", "repeat_purchase"]}}))
    monkeypatch.setattr(final, "CONTRACT", config)
    final.save_json(tmp_path / "prediction_seal.json", {"contract_sha256": final.digest(config),
        "users": 1, "test_labels_read": False, "artifacts": {}})
    with pytest.raises(ValueError, match="Missing sealed artifact"):
        final.verify_prediction_seal()


def test_output_guard_rejects_existing_and_external_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "DATA", tmp_path / "data")
    monkeypatch.setattr(final, "MODELS", tmp_path / "models")
    output = tmp_path / "data" / "result.parquet"
    final.new_output(output)
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        final.new_output(output)
    with pytest.raises(ValueError, match="dedicated"):
        final.new_output(tmp_path / "outside.parquet")
    assert output.read_bytes() == b"keep"


def test_final_evaluator_keeps_empty_users_and_conditional_denominators(tmp_path, monkeypatch):
    """Exercise the metric stage on tiny, independently specified baskets."""
    root, data, models = tmp_path, tmp_path / "data", tmp_path / "models"
    data.mkdir()
    models.mkdir()
    for name, value in (("ROOT", root), ("DATA", data), ("MODELS", models),
                        ("REPORT", root / "report.json"), ("CONTRACT", root / "contract.json")):
        monkeypatch.setattr(final, name, value)
    policies = [*final.SOURCES, "global_popularity", "repeat_with_popularity_fill", "selected_lambdamart"]
    cfg = {"architecture_id": "test", "final_test_protocol": {"expected_users": 2,
        "baselines": policies, "test_truth": "test.parquet"}}
    final.save_json(root / "contract.json", cfg)
    final.save_json(models / "prediction_seal.json", {"fixture": True})
    monkeypatch.setattr(final, "verify_prediction_seal", lambda: {"sealed_at": "before-test"})
    monkeypatch.setattr(final, "verify_stage", lambda stage: {"statistics": {"fixture": stage}})
    pd.DataFrame({"user_id": [1, 2], "history_orders": [1, 3], "latest_order": [1, 3]}).to_parquet(data / "history_summary.parquet")
    pd.DataFrame({"user_id": [1, 2], "product_id": [1, 3]}).to_parquet(data / "history.parquet")
    pd.DataFrame({"product_id": list(range(1, 11)), "purchase_count": [1] * 10,
        "aisle_id": [1] * 5 + [2] * 5, "department_id": [1] * 10}).to_parquet(data / "item_features.parquet")
    pd.DataFrame({"user_id": [1, 1, 2], "product_id": [1, 2, 4], "order_id": [10, 10, 20],
        "order_number": [2, 2, 4]}).to_parquet(root / "test.parquet")
    (data / "predictions").mkdir()
    for policy in policies:
        recs = [[1], []] if policy == "repeat_purchase" else [list(range(1, 11))] * 2
        pd.DataFrame({"user_id": [1, 2], "recommendations": recs}).to_parquet(data / "predictions" / f"{policy}.parquet")
    (data / "candidates").mkdir()
    for file in final.SOURCES.values():
        pd.DataFrame({"user_id": [1, 2], "recommendations": [[1, 2], [3]]}).to_parquet(data / "candidates" / file)
    (data / "prediction_features").mkdir()
    pd.DataFrame({"user_id": [1, 1, 2], "product_id": [1, 2, 3]}).to_parquet(data / "prediction_features/part.parquet")
    final.evaluate()
    report = json.loads((root / "report.json").read_text())
    repeat = report["policies"]["repeat_purchase"]["overall"]
    assert repeat["users"] == 2
    assert repeat["recall_at_10"] == pytest.approx(.25)
    assert repeat["precision_at_10"] == pytest.approx(.05)
    assert repeat["repeat_recall_at_10"] == 1
    assert repeat["users_with_repeat_truth"] == 1
    assert repeat["novel_recall_at_10"] == 0
    assert repeat["users_with_novel_truth"] == 2
    assert repeat["fraction_short_lists"] == 1
    assert report["candidate_retrieval"]["union"]["candidate_recall"] == .5
    assert (models / "test_access.json").exists()
    from src.evaluation import audit_final_test as audit
    for name, value in (("ROOT", root), ("DATA", data), ("REPORT", root / "report.json")):
        monkeypatch.setattr(audit, name, value)
    monkeypatch.setattr(audit, "verify_prediction_seal", lambda: None)
    audit.main()
    report["policies"]["repeat_purchase"]["overall"]["precision_at_10"] = .5
    final.save_json(root / "report.json", report)
    with pytest.raises(ValueError, match="Independent metric mismatch"):
        audit.main()
