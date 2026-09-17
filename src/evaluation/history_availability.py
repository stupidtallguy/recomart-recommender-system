"""Evaluate history segments and an excluded-user zero-history scenario.

Uses existing retrieval artifacts and LambdaMART v3. No final-test data is read.
"""

from __future__ import annotations

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.evaluation import discovery_slot_diagnostic as diagnostic
from src.evaluation.metrics import recall_at_k, precision_at_k, ndcg_at_k
from src.models.run_novel_ranker_v1 import save_json, user_fingerprint
from src.models.run_xgb_ranker_v3 import MODEL_PATH as RANKER_MODEL_PATH
from src.serving.routing import fill_with_popularity, route_recommendations


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
CANDIDATES = PROCESSED / "candidates"
RESULT_PATH = ROOT / "results/benchmarks/history_availability.json"
USER_PATH = PROCESSED / "history_availability/user_metrics.parquet"
FALLBACK_PATH = ROOT / "models/popularity_fallback_v1.json"
SEED = 42
BOOTSTRAP_SAMPLES = 2000
K = 10
SOURCE_FILES = {
    "repeat_purchase": "repeat_purchase_top50.parquet",
    "als": "als_v2_top50.parquet",
    "content": "content_v1_top50.parquet",
    "two_tower_v1": "two_tower_v1_top50.parquet",
    "two_tower_v2": "two_tower_v2_top50.parquet",
}


def history_segment(order_count: int) -> str:
    if order_count < 0:
        raise ValueError("History order count cannot be negative.")
    if order_count == 0:
        return "00_zero_orders"
    if order_count <= 2:
        return "01_1_2_orders"
    if order_count <= 5:
        return "02_3_5_orders"
    if order_count <= 10:
        return "03_6_10_orders"
    return "04_over_10_orders"


def popular_products(interactions: pd.DataFrame, k: int = 50) -> list[int]:
    counts = interactions.groupby("product_id").size().rename("count").reset_index()
    return counts.sort_values(["count", "product_id"], ascending=[False, True]).head(k).product_id.astype(int).tolist()


def cold_start_data(train: pd.DataFrame, held_users: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Exclude held users entirely from fitting; their first recorded order is truth."""
    mask = train.user_id.isin(held_users)
    cold = train.loc[mask]
    if set(cold.user_id.unique()) != set(held_users):
        raise ValueError("Missing held-out users in historical data.")
    first_number = cold.groupby("user_id").order_number.transform("min")
    first = cold.loc[cold.order_number.eq(first_number), ["user_id", "product_id"]]
    truth = first.groupby("user_id").product_id.agg(set).to_dict()
    warm = train.loc[~mask]
    if set(warm.user_id.unique()) & set(held_users):
        raise ValueError("Cold-start fitting contains held-out users.")
    return warm, truth


def load_source(path: Path, users: np.ndarray) -> dict[int, list[int]]:
    frame = pd.read_parquet(path, columns=["user_id", "recommendations"])
    frame = frame.loc[frame.user_id.isin(users)]
    if frame.user_id.duplicated().any() or set(frame.user_id) != set(users):
        raise ValueError(f"Candidate cohort mismatch: {path}")
    return {int(user): [int(item) for item in items[:K]] for user, items in frame.itertuples(index=False, name=None)}


def evaluate_users(predictions: dict, truth: dict, histories: dict, novel_truth: dict | None = None) -> pd.DataFrame:
    if set(predictions) != set(truth) or set(predictions) != set(histories):
        raise ValueError("Predictions, truth and histories must use identical users.")
    rows = []
    for user in sorted(truth):
        items = predictions[user][:K]
        if len(items) != len(set(items)):
            raise ValueError(f"Duplicate recommendations for user {user}.")
        target = truth[user]
        novel = novel_truth[user] if novel_truth is not None else None
        rows.append({
            "user_id": user,
            "history_orders": histories[user],
            "segment": history_segment(histories[user]),
            "recall_at_10": recall_at_k(items, target, K),
            "precision_at_10": precision_at_k(items, target, K),
            "ndcg_at_10": ndcg_at_k(items, target, K),
            "hit_rate_at_10": float(bool(set(items) & target)),
            "returned_items": len(items),
            "novel_recall_at_10": recall_at_k(items, novel, K) if novel else np.nan,
            "novel_hit_rate_at_10": float(bool(set(items) & novel)) if novel else np.nan,
        })
    return pd.DataFrame(rows).set_index("user_id")


def summarize(frame: pd.DataFrame, predictions: dict, catalog_size: int) -> dict:
    unique = {item for user in frame.index for item in predictions[int(user)][:K]}
    result = {
        "users": len(frame),
        **{metric: float(frame[metric].mean()) for metric in ["recall_at_10", "precision_at_10", "ndcg_at_10", "hit_rate_at_10"]},
        "mean_returned_items": float(frame.returned_items.mean()),
        "fraction_short_lists": float(frame.returned_items.lt(K).mean()),
        "unique_recommended_products": len(unique),
        "catalog_coverage_at_10": len(unique) / catalog_size,
    }
    if frame.novel_recall_at_10.notna().any():
        result["users_with_novel_truth"] = int(frame.novel_recall_at_10.notna().sum())
        result["novel_recall_at_10"] = float(frame.novel_recall_at_10.mean())
        result["novel_hit_rate_at_10"] = float(frame.novel_hit_rate_at_10.mean())
    return result


def paired_difference(alternative: pd.Series, control: pd.Series) -> dict:
    if not alternative.index.equals(control.index):
        raise ValueError("Paired comparison needs the same ordered users.")
    delta = (alternative - control).to_numpy()
    rng = np.random.default_rng(SEED)
    means = np.empty(BOOTSTRAP_SAMPLES)
    for start in range(0, BOOTSTRAP_SAMPLES, 50):
        batch = min(50, BOOTSTRAP_SAMPLES - start)
        draws = rng.integers(0, len(delta), size=(batch, len(delta)))
        means[start:start + batch] = delta[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean_delta": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def main() -> None:
    start = perf_counter()
    print("Scoring the unchanged LambdaMART v3 validation cohort...", flush=True)
    scored = diagnostic.load_scored_candidates()
    users = np.sort(scored.user_id.unique())
    top = scored.groupby("user_id", sort=False).head(K)
    predictions = {
        "lambdamart_v3": top.groupby("user_id", sort=False).product_id.agg(list).to_dict(),
    }
    candidate_latest = scored.groupby("user_id").user_latest_order.first()
    del scored, top
    gc.collect()

    print("Counting observed training orders and isolating cold-start users...", flush=True)
    train = pd.read_parquet(PROCESSED / "train_interactions.parquet", columns=["user_id", "order_number", "product_id"])
    counts = train.groupby("user_id").order_number.nunique()
    histories = {int(user): int(counts.loc[user]) for user in users}
    actual_latest = train.groupby("user_id").order_number.max().reindex(users)
    if not np.array_equal(actual_latest.to_numpy(), candidate_latest.reindex(users).to_numpy()):
        raise ValueError("Candidate histories do not match the training snapshot.")
    catalog_size = int(train.product_id.nunique())
    popular = popular_products(train)
    for name, filename in SOURCE_FILES.items():
        predictions[name] = load_source(CANDIDATES / filename, users)
    predictions["global_popularity"] = {int(user): popular[:K] for user in users}
    predictions["repeat_with_popularity_fill"] = {
        user: route_recommendations(
            history_order_count=histories[user], repeat=items, popularity=popular, k=K,
        ).recommendations
        for user, items in predictions["repeat_purchase"].items()
    }
    for user, ranked in predictions["lambdamart_v3"].items():
        routed = route_recommendations(
            history_order_count=histories[user], ranked=ranked,
            repeat=predictions["repeat_purchase"][user], popularity=popular, k=K,
        )
        if routed.strategy != "ranker" or routed.recommendations != ranked:
            raise ValueError(f"Unexpected primary routing change for user {user}.")

    warm, first_truth = cold_start_data(train, users)
    cold_popular = popular_products(warm)
    warm_catalog = set(warm.product_id.unique())
    cold_stats = {
        "fitting_users": int(warm.user_id.nunique()),
        "fitting_interactions": len(warm),
        "fitting_catalog_products": len(warm_catalog),
        "held_users": len(users),
        "held_user_sha256": user_fingerprint(users),
        "target_interactions": sum(len(items) for items in first_truth.values()),
        "target_products_missing_from_fitting_catalog": sum(len(items - warm_catalog) for items in first_truth.values()),
        "target": "First recorded basket for each held user; all of that user's interactions excluded from popularity fitting.",
        "limitations": "Simulation on historical dataset users, not live new customers. No shared calendar timestamps are available to impose a global-time cutoff. No onboarding category preferences are assumed.",
    }
    serving_popularity = {
        "model": "global_purchase_count_popularity",
        "fit_source": "train_interactions.parquet",
        "fitting_users": int(train.user_id.nunique()),
        "fitting_interactions": len(train),
        "recommendations": popular,
        "tie_break": "purchase_count descending, product_id ascending",
        "note": "Full-training fallback for future unknown users. Cold-start evaluation instead uses the separate held-user-excluded fit recorded in the benchmark.",
    }
    del warm, train
    gc.collect()

    validation = pd.read_parquet(PROCESSED / "validation_interactions.parquet", columns=["user_id", "product_id", "reordered"])
    validation = validation.loc[validation.user_id.isin(users)]
    truth = validation.groupby("user_id").product_id.agg(set).to_dict()
    novel_groups = validation.loc[validation.reordered.eq(0)].groupby("user_id").product_id.agg(set).to_dict()
    novel_truth = {int(user): novel_groups.get(user, set()) for user in users}
    del validation
    known_results = {}
    metrics = {}
    for name, recs in predictions.items():
        print(f"Evaluating known-user policy: {name}", flush=True)
        frame = evaluate_users(recs, truth, histories, novel_truth)
        metrics[name] = frame
        known_results[name] = {
            "overall": summarize(frame, recs, catalog_size),
            "segments": {segment: summarize(rows, recs, catalog_size) for segment, rows in frame.groupby("segment")},
        }

    old = json.loads(diagnostic.RESULT_PATH.read_text(encoding="utf-8"))["policies"]["minimum_0_novel_slots"]["overall"]
    for metric, expected in old.items():
        if not np.isclose(known_results["lambdamart_v3"]["overall"][metric], expected, rtol=0, atol=1e-10):
            raise ValueError(f"Ranker control drift: {metric}")

    comparisons = {}
    for segment, control in metrics["lambdamart_v3"].groupby("segment"):
        comparisons[segment] = {}
        for name, frame in metrics.items():
            if name == "lambdamart_v3":
                continue
            alternative = frame.loc[control.index]
            comparisons[segment][name] = {
                metric: paired_difference(alternative[metric], control[metric])
                for metric in ["recall_at_10", "ndcg_at_10"]
            }

    print("Evaluating zero-history first-basket predictions...", flush=True)
    cold_predictions = {
        int(user): route_recommendations(
            history_order_count=0, popularity=cold_popular, k=K,
        ).recommendations
        for user in users
    }
    cold_metrics = evaluate_users(cold_predictions, first_truth, {int(user): 0 for user in users})
    cold_stats["popularity_top10"] = cold_popular[:K]
    cold_stats["metrics"] = summarize(cold_metrics, cold_predictions, len(warm_catalog))
    cold_stats["all_users_receive_same_list"] = True
    cold_stats["comparison_warning"] = "The target is a first recorded basket. Do not interpret its metrics as a gain or loss against known-user next-basket metrics."

    USER_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat({**metrics, "zero_history_popularity": cold_metrics}, names=["policy"]).reset_index().to_parquet(USER_PATH, index=False)
    result = {
        "experiment": "history_availability_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ranker_model_sha256": hashlib.sha256(RANKER_MODEL_PATH.read_bytes()).hexdigest(),
        "validation_users": len(users), "validation_user_sha256": user_fingerprint(users),
        "catalog_products": catalog_size,
        "history_count_definition": "Distinct order_number values in training interactions per user, not maximum order_number.",
        "naturally_zero_history_users_in_known_cohort": sum(count == 0 for count in histories.values()),
        "known_user_results": known_results,
        "paired_segment_differences_vs_ranker": comparisons,
        "uncertainty": {
            "method": "Paired user bootstrap, 95% percentile intervals, within each history segment.",
            "samples": BOOTSTRAP_SAMPLES, "seed": SEED,
            "limitation": "Exploratory development comparisons, unadjusted for multiple comparisons and prior model selection. Do not tune routing to isolated small segment wins.",
        },
        "zero_history_scenario": cold_stats,
        "routing_checks": {
            "available_ranker_preserved_for_all_users": True,
            "ranker_unavailable_fallback_evaluated": "repeat_with_popularity_fill",
            "cold_predictions_generated_by_zero_history_route": True,
        },
        "runtime_seconds": perf_counter() - start,
        "final_test_used": False,
    }
    save_json(RESULT_PATH, result)
    save_json(FALLBACK_PATH, serving_popularity)
    print(json.dumps({name: value["overall"] for name, value in known_results.items()}, indent=2))
    print(json.dumps(cold_stats, indent=2))
    print(f"Saved: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
