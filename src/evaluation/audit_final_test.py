"""Independently recompute core final metrics from the sealed saved predictions.

This read-only audit requires a completed benchmark. It never trains or scores a
model and uses vectorized joins instead of the experiment's per-user evaluator.
"""
import json

import numpy as np
import pandas as pd

from src.evaluation.final_test_pipeline import (
    DATA, REPORT, ROOT, contract, verify_hashes, verify_prediction_seal,
)


def main():
    if not REPORT.exists():
        raise RuntimeError("The final benchmark must finish before auditing test metrics")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    verify_prediction_seal()
    verify_hashes(report["per_user_metrics"])
    truth = pd.read_parquet(ROOT / contract()["final_test_protocol"]["test_truth"], columns=["user_id", "product_id"])
    denominators = truth.groupby("user_id").size().sort_index()
    discounts = 1 / np.log2(np.arange(10) + 2)
    ideal = np.r_[0, discounts.cumsum()][np.minimum(denominators.to_numpy(), 10)]
    if len(denominators) != report["test_users"] or len(truth) != report["test_interactions"]:
        raise ValueError("Reported test population is incorrect")
    checked = {}
    for name in contract()["final_test_protocol"]["baselines"]:
        saved = pd.read_parquet(DATA / "predictions" / f"{name}.parquet")
        if saved.user_id.duplicated().any() or set(saved.user_id) != set(denominators.index):
            raise ValueError(f"User coverage mismatch: {name}")
        flat = saved.explode("recommendations").dropna().rename(columns={"recommendations": "product_id"})
        flat["product_id"] = flat.product_id.astype("int32")
        flat["rank"] = flat.groupby("user_id").cumcount()
        if flat.duplicated(["user_id", "product_id"]).any() or flat["rank"].ge(10).any():
            raise ValueError(f"Invalid ranking: {name}")
        hits = flat.merge(truth, on=["user_id", "product_id"], validate="one_to_one")
        hit_counts = hits.groupby("user_id").size().reindex(denominators.index, fill_value=0)
        hits["gain"] = 1 / np.log2(hits["rank"] + 2)
        dcg = hits.groupby("user_id").gain.sum().reindex(denominators.index, fill_value=0)
        metrics = {
            "recall_at_10": float((hit_counts / denominators).mean()),
            "precision_at_10": float((hit_counts / 10).mean()),
            "ndcg_at_10": float((dcg / ideal).mean()),
            "hit_rate_at_10": float(hit_counts.gt(0).mean()),
            "catalog_coverage_at_10": flat.product_id.nunique() / report["observed_history_catalog_products"],
        }
        for metric, value in metrics.items():
            if not np.isclose(value, report["policies"][name]["overall"][metric], rtol=0, atol=1e-12):
                raise ValueError(f"Independent metric mismatch: {name}/{metric}")
        checked[name] = metrics
    print(json.dumps({"status": "passed", "users": len(denominators), "independently_recomputed": checked}, indent=2))


if __name__ == "__main__":
    main()
