"""Evaluate aisle caps and select among frozen three/four/five-source rankers.

All comparisons use ranker-validation users; this runner never opens final test.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm

from src.evaluation.history_availability import evaluate_users, summarize, paired_difference
from src.evaluation.ranking_segment_diagnostics import MODEL_CONFIGS
from src.models.run_novel_ranker_v1 import save_json, user_fingerprint


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
RESULT_PATH = ROOT / "results/benchmarks/diversity_architecture.json"
USER_PATH = PROCESSED / "diversity_architecture/user_metrics.parquet"
K = 10
ACCURACY_TOLERANCE = 0.001
CAPS = [None, 4, 3, 2]
ARCHITECTURES = {
    "three_source": MODEL_CONFIGS["lambda_v1_corrected"],
    "four_source": MODEL_CONFIGS["lambda_v2_tt_v1"],
    "five_source": MODEL_CONFIGS["lambda_v3_tt_v1_tt_v2"],
}


def cap_aisles(products: list[int], aisles: dict[int, int], cap: int | None, k: int = K) -> list[int]:
    """Best-effort cap on a score-ordered pool; restore base order after selection.

    When the pool cannot fill K under the cap, backfill highest-ranked skipped
    products. This avoids pretending that short output lists are diversity gains.
    """
    if k < 0 or (cap is not None and cap < 1):
        raise ValueError("k must be nonnegative and aisle cap must be positive.")
    if k == 0:
        return []
    if len(products) != len(set(products)):
        raise ValueError("Candidate products must be unique.")
    if cap is None:
        return list(products[:k])
    selected, skipped = [], []
    counts = Counter()
    for index, product in enumerate(products):
        aisle = aisles[product]
        if counts[aisle] < cap:
            selected.append(index)
            counts[aisle] += 1
            if len(selected) == k:
                break
        else:
            skipped.append(index)
    if len(selected) < k:
        selected.extend(skipped[:k - len(selected)])
    return [products[index] for index in sorted(selected)]


def category_metrics(products: list[int], aisles: dict, departments: dict) -> dict:
    counts = Counter(aisles[product] for product in products)
    n = len(products)
    return {
        "unique_aisles": len(counts),
        "unique_departments": len({departments[product] for product in products}),
        "max_products_in_one_aisle": max(counts.values(), default=0),
        "dominant_aisle_share": max(counts.values(), default=0) / n if n else 0.0,
        "aisle_pair_diversity": 1.0 - sum(count * (count - 1) for count in counts.values()) / (n * (n - 1)) if n > 1 else 0.0,
    }


def noninferior(comparisons: dict, tolerance: float = ACCURACY_TOLERANCE) -> bool:
    """Require both accuracy intervals to clear the predeclared loss tolerance."""
    return all(comparisons[metric]["ci95_low"] >= -tolerance for metric in ("recall_at_10", "ndcg_at_10"))


def eligible_diversity(comparisons: dict, diversity_gain: float) -> bool:
    """Without a business loss budget, require demonstrated accuracy gains too."""
    return diversity_gain > 0 and all(
        comparisons[metric]["ci95_low"] > 0
        for metric in ("recall_at_10", "ndcg_at_10")
    )


def load_rankings(config: dict) -> tuple[pd.DataFrame, dict]:
    start = perf_counter()
    features = config["features"]
    frame = pd.read_parquet(config["ranker_dir"] / "ranker_validation", columns=["user_id", "product_id", *features])
    if frame.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Duplicate candidate products within a user.")
    model = xgb.Booster()
    model.load_model(config["model_path"])
    benchmark = json.loads(config["benchmark_path"].read_text(encoding="utf-8"))
    best_iteration = benchmark["best_iteration"]
    matrix = xgb.DMatrix(frame[features].astype(np.float32), feature_names=features)
    scores = model.predict(matrix, iteration_range=(0, best_iteration + 1))
    selected_columns = frame[["user_id", "product_id"]].copy()
    selected_columns["score"] = scores
    ranked = selected_columns.sort_values(
        ["user_id", "score", "product_id"], ascending=[True, False, True], kind="mergesort",
    ).reset_index(drop=True)
    meta = {
        "model_path": str(config["model_path"].relative_to(ROOT)),
        "model_sha256": hashlib.sha256(config["model_path"].read_bytes()).hexdigest(),
        "best_iteration": best_iteration,
        "features": features,
        "candidate_rows": len(ranked),
        "mean_candidates_per_user": len(ranked) / ranked.user_id.nunique(),
        "score_and_sort_seconds": perf_counter() - start,
    }
    return ranked, meta


def metric_summary(frame: pd.DataFrame, predictions: dict, catalog_size: int, cap: int | None) -> dict:
    result = summarize(frame, predictions, catalog_size)
    result.update({name: float(frame[name].mean()) for name in (
        "unique_aisles", "unique_departments", "max_products_in_one_aisle", "dominant_aisle_share", "aisle_pair_diversity",
    )})
    result["fraction_users_over_cap"] = float(frame.max_products_in_one_aisle.gt(cap).mean()) if cap else None
    return result


def compare(candidate: pd.DataFrame, control: pd.DataFrame) -> dict:
    return {
        metric: paired_difference(candidate[metric], control[metric])
        for metric in ("recall_at_10", "ndcg_at_10")
    }


def main() -> None:
    start = perf_counter()
    # The cohort and history counts were independently validated in the preceding
    # experiment; using its small artifact avoids reloading 29.5M interactions.
    history_file = PROCESSED / "history_availability/user_metrics.parquet"
    histories_frame = pd.read_parquet(history_file, columns=["policy", "user_id", "history_orders"])
    histories_frame = histories_frame.loc[histories_frame.policy.eq("lambdamart_v3")]
    if histories_frame.user_id.duplicated().any():
        raise ValueError("Duplicate history users.")
    histories = dict(zip(histories_frame.user_id.astype(int), histories_frame.history_orders.astype(int)))
    users = np.sort(histories_frame.user_id.unique())
    validation = pd.read_parquet(PROCESSED / "validation_interactions.parquet", columns=["user_id", "product_id", "reordered"])
    validation = validation.loc[validation.user_id.isin(users)]
    truth = validation.groupby("user_id").product_id.agg(set).to_dict()
    novel = validation.loc[validation.reordered.eq(0)].groupby("user_id").product_id.agg(set).to_dict()
    novel_truth = {int(user): novel.get(user, set()) for user in users}
    metadata = pd.read_parquet(PROCESSED / "item_features.parquet", columns=["product_id", "aisle_id", "department_id", "purchase_count"])
    aisles = dict(zip(metadata.product_id.astype(int), metadata.aisle_id.astype(int)))
    departments = dict(zip(metadata.product_id.astype(int), metadata.department_id.astype(int)))
    catalog_size = int(metadata.purchase_count.gt(0).sum())
    truth_categories = pd.DataFrame([category_metrics(list(truth[int(user)]), aisles, departments) for user in users])
    results, all_metrics, models = {}, {}, {}
    diversity_differences = {}

    for architecture, config in ARCHITECTURES.items():
        print(f"Scoring {architecture}...", flush=True)
        ranked, models[architecture] = load_rankings(config)
        if not np.array_equal(np.sort(ranked.user_id.unique()), users):
            raise ValueError(f"Cohort mismatch: {architecture}")
        if not set(ranked.product_id).issubset(aisles):
            raise ValueError("Missing aisle metadata for candidates.")
        policies = {"no_cap" if cap is None else f"aisle_cap_{cap}": {} for cap in CAPS}
        for user, frame in tqdm(ranked.groupby("user_id", sort=False), total=len(users), desc=architecture):
            products = [int(product) for product in frame.product_id]
            for cap in CAPS:
                name = "no_cap" if cap is None else f"aisle_cap_{cap}"
                policies[name][int(user)] = cap_aisles(products, aisles, cap)
        del ranked
        gc.collect()
        results[architecture] = {}
        for name, predictions in policies.items():
            frame = evaluate_users(predictions, truth, histories, novel_truth)
            categories = pd.DataFrame.from_dict(
                {user: category_metrics(items, aisles, departments) for user, items in predictions.items()}, orient="index",
            ).reindex(frame.index)
            frame = frame.join(categories)
            cap = None if name == "no_cap" else int(name.rsplit("_", 1)[1])
            key = f"{architecture}/{name}"
            all_metrics[key] = frame
            results[architecture][name] = {
                "overall": metric_summary(frame, predictions, catalog_size, cap),
                "segments": {segment: metric_summary(rows, predictions, catalog_size, cap) for segment, rows in frame.groupby("segment")},
                "changed_users_vs_no_cap": sum(items != policies["no_cap"][user] for user, items in predictions.items()),
            }
        baseline = all_metrics[f"{architecture}/no_cap"]
        expected = json.loads(config["benchmark_path"].read_text(encoding="utf-8"))["ranker"]
        for metric, value in expected.items():
            if not np.isclose(float(baseline[metric].mean()), value, rtol=0, atol=1e-10):
                raise ValueError(f"Baseline drift: {architecture}/{metric}")
        diversity_differences[architecture] = {
            f"aisle_cap_{cap}": compare(all_metrics[f"{architecture}/aisle_cap_{cap}"], baseline)
            for cap in CAPS if cap is not None
        }
        del policies
        gc.collect()

    architecture_differences = {}
    for candidate, controls in [("three_source", ["four_source", "five_source"]), ("four_source", ["five_source"])]:
        architecture_differences[candidate] = {
            control: compare(all_metrics[f"{candidate}/no_cap"], all_metrics[f"{control}/no_cap"])
            for control in controls
        }
    chosen = "five_source"
    for candidate in ("three_source", "four_source"):
        if all(noninferior(comparison) for comparison in architecture_differences[candidate].values()):
            chosen = candidate
            break
    chosen_cap = None
    for cap in (4, 3, 2):
        gain = results[chosen][f"aisle_cap_{cap}"]["overall"]["aisle_pair_diversity"] - results[chosen]["no_cap"]["overall"]["aisle_pair_diversity"]
        if eligible_diversity(diversity_differences[chosen][f"aisle_cap_{cap}"], gain):
            chosen_cap = cap
            break

    # Recheck the selected primary model's history routing against its actual
    # fallback, rather than transferring the v3-only conclusion unexamined.
    previous = pd.read_parquet(PROCESSED / "history_availability/user_metrics.parquet")
    fallback = previous.loc[previous.policy.eq("repeat_with_popularity_fill")].set_index("user_id")
    chosen_key = f"{chosen}/no_cap" if chosen_cap is None else f"{chosen}/aisle_cap_{chosen_cap}"
    routing_checks = {
        segment: compare(rows, fallback.loc[rows.index])
        for segment, rows in all_metrics[chosen_key].groupby("segment")
    }
    USER_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_metrics, names=["policy"]).reset_index().to_parquet(USER_PATH, index=False)
    result = {
        "experiment": "diversity_and_architecture_selection_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "users_evaluated": len(users), "user_sha256": user_fingerprint(users),
        "catalog_size": catalog_size, "models": models,
        "protocol": {
            "aisle_caps": CAPS, "backfill": "If a cap cannot fill ten slots, use highest-ranked skipped products and record violations.",
            "architecture_absolute_accuracy_tolerance": ACCURACY_TOLERANCE,
            "architecture_rule": "Choose fewest sources whose paired 95% lower bounds against all more complex alternatives exceed -0.001 for Recall@10 and NDCG@10.",
            "diversity_rule": "Adopt the least restrictive tested cap only if pair diversity improves and both accuracy-difference 95% lower bounds exceed zero.",
            "uncertainty": "2,000 paired user-bootstrap samples, seed 42. Development intervals conditional on fitted models; not adjusted for multiple comparisons or historical model selection.",
        },
        "actual_basket_category_description": {name: float(truth_categories[name].mean()) for name in truth_categories.columns},
        "policies": results,
        "diversity_differences_vs_no_cap": diversity_differences,
        "architecture_differences": architecture_differences,
        "selection": {"architecture": chosen, "aisle_cap": chosen_cap, "minimum_novel_slots": 0},
        "selected_model_vs_repeat_fallback_by_history": routing_checks,
        "runtime_seconds": perf_counter() - start,
        "final_test_used": False,
    }
    save_json(RESULT_PATH, result)
    print(json.dumps(result["selection"], indent=2))
    print(json.dumps(results[chosen], indent=2))
    print(json.dumps(architecture_differences, indent=2))
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
