"""Controlled unseen-item ranking experiment on the frozen v3 candidate store.

Run with ``python -m src.models.run_novel_ranker_v1``. Use ``--evaluate-only``
to reuse the saved novel model. No final-test data is read by this module.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm

from src.evaluation import discovery_slot_diagnostic as diagnostic
from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from src.models import run_xgb_ranker_v3 as base


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/xgb_novel_ranker_v1.json"
RESULT_PATH = ROOT / "results/benchmarks/novel_ranker_v1.json"
TRAINING_PATH = ROOT / "results/benchmarks/novel_ranker_v1_training.json"
IMPORTANCE_PATH = ROOT / "results/benchmarks/novel_ranker_v1_feature_importance.csv"
USER_METRICS_PATH = ROOT / "data/processed/novel_ranker_v1/evaluation_user_metrics.parquet"
FEATURES = base.FEATURE_COLUMNS
SEED = 42
BOOTSTRAP_SAMPLES = 2000
TOP_K = 10
POLICY_ZERO = "base_zero_slots"
POLICY_ONE = "base_minimum_one_slot"
POLICY_NOVEL = "novel_ranker_minimum_one_slot"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=4, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def user_fingerprint(users: np.ndarray) -> str:
    return hashlib.sha256(np.sort(users).astype("<i8").tobytes()).hexdigest()


def informative_novel_queries(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep unseen rows only, with both labels present within each query."""
    novel = frame.loc[frame.seen_before.eq(0)].copy()
    groups = novel.groupby("user_id").label.agg(["sum", "count"])
    users = groups.index[(groups["sum"] > 0) & (groups["sum"] < groups["count"])]
    return novel.loc[novel.user_id.isin(users)].sort_values(
        ["user_id", "product_id"], kind="mergesort"
    ).reset_index(drop=True)


def internal_user_split(users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reserve 10% of training users for early stopping, never outer validation."""
    users = np.sort(np.unique(users))
    if len(users) < 2:
        raise ValueError("At least two informative training users are required.")
    shuffled = np.random.default_rng(SEED).permutation(users)
    size = max(1, int(np.ceil(len(users) * 0.10)))
    return np.sort(shuffled[size:]), np.sort(shuffled[:size])


def rank_matrix(frame: pd.DataFrame, ref=None) -> xgb.QuantileDMatrix:
    return xgb.QuantileDMatrix(
        frame[FEATURES].astype(np.float32),
        label=frame.label.to_numpy(dtype=np.float32),
        qid=frame.user_id.to_numpy(dtype=np.int32),
        max_bin=256,
        ref=ref,
    )


def train_model() -> dict:
    start = perf_counter()
    print("Loading the existing, corrected v3 training samples...", flush=True)
    raw = pd.read_parquet(base.TRAIN_PATH, columns=base.LOAD_COLUMNS)
    outer_users = np.sort(pd.read_parquet(base.VALIDATION_PATH, columns=["user_id"]).user_id.unique())
    raw_users = np.sort(raw.user_id.unique())
    if np.intersect1d(raw_users, outer_users).size:
        raise ValueError("Ranker train and outer validation users overlap.")
    if raw.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Duplicate training user-product pairs.")
    if not raw.label.isin([0, 1]).all():
        raise ValueError("Expected binary labels.")

    unseen = raw.loc[raw.seen_before.eq(0)]
    stats = {
        "source_rows": len(raw),
        "source_users": len(raw_users),
        "outer_validation_users": len(outer_users),
        "outer_validation_user_sha256": user_fingerprint(outer_users),
        "unseen_rows_before_query_filter": len(unseen),
        "unseen_positives_before_query_filter": int(unseen.label.sum()),
        "unseen_negatives_before_query_filter": int(unseen.label.eq(0).sum()),
        "negative_sampling": "Unseen subset of existing corrected v3 samples; no new sampling or positive injection.",
    }
    train = informative_novel_queries(raw)
    del raw, unseen
    gc.collect()
    train_users = np.sort(train.user_id.unique())
    fit_users, stop_users = internal_user_split(train_users)
    fit = train.loc[train.user_id.isin(fit_users)]
    stop = train.loc[train.user_id.isin(stop_users)]
    stats.update({
        "informative_unseen_rows": len(train),
        "informative_unseen_users": len(train_users),
        "informative_unseen_positives": int(train.label.sum()),
        "internal_fit_users": len(fit_users),
        "internal_early_stop_users": len(stop_users),
        "internal_fit_user_sha256": user_fingerprint(fit_users),
        "internal_early_stop_user_sha256": user_fingerprint(stop_users),
        "constant_features": [name for name in FEATURES if train[name].nunique() <= 1],
    })
    print(json.dumps(stats, indent=2), flush=True)
    # A single discovery slot makes NDCG@1 the internal checkpoint criterion.
    # Tree/pair settings remain identical to v3; no hyperparameter sweep.
    params = {**base.XGB_PARAMS, "eval_metric": "ndcg@1"}
    dfit = rank_matrix(fit)
    dstop = rank_matrix(stop, ref=dfit)
    del fit, stop
    print("Choosing training duration on internal TRAIN users only...", flush=True)
    history = {}
    selected = xgb.train(
        params, dfit, num_boost_round=base.NUM_BOOST_ROUND,
        evals=[(dstop, "internal_train_holdout")],
        early_stopping_rounds=base.EARLY_STOPPING_ROUNDS,
        evals_result=history, verbose_eval=25,
    )
    best_iteration = int(selected.best_iteration)
    internal_score = float(selected.best_score)
    del selected, dfit, dstop
    gc.collect()
    print(f"Refitting all informative TRAIN users for {best_iteration + 1} rounds...", flush=True)
    dall = rank_matrix(train)
    final_model = xgb.train(params, dall, num_boost_round=best_iteration + 1)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(MODEL_PATH)
    importance = final_model.get_score(importance_type="gain")
    pd.DataFrame([
        {"feature": name, "gain": float(importance.get(name, 0.0))}
        for name in FEATURES
    ]).sort_values("gain", ascending=False).to_csv(IMPORTANCE_PATH, index=False)
    result = {
        "model": "xgb_novel_ranker_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES, "params": params, "data": stats,
        "selected_iteration": best_iteration,
        "internal_ndcg_at_1": internal_score,
        "checkpoint_selection": "10% internal training-user holdout; refit all informative training users for selected rounds.",
        "internal_history": history,
        "training_seconds": perf_counter() - start,
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
    }
    save_json(TRAINING_PATH, result)
    return result


def one_slot_policy(frame: pd.DataFrame, selection_score: str, k: int = TOP_K) -> list[dict]:
    """Select a missing discovery item, preserving the original score ordering.

    Input is sorted by base score descending then product_id ascending. Scores
    from the two models are never added or directly compared with each other.
    """
    columns = ["product_id", "seen_before", "score"]
    selected = frame.head(k)[columns].to_dict("records")
    if len(frame) <= k or any(item["seen_before"] == 0 for item in selected):
        return selected
    novel = frame.loc[frame.seen_before.eq(0)]
    if novel.empty:
        return selected
    values = novel[selection_score].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite unseen ranking scores.")
    order = np.lexsort((novel.product_id.to_numpy(), -values))
    selected[-1] = novel.iloc[order[0]][columns].to_dict()
    return sorted(selected, key=lambda item: (-float(item["score"]), int(item["product_id"])))


def per_user_metrics(predictions: dict, truth: dict, repeat_truth: dict, novel_truth: dict) -> pd.DataFrame:
    rows = []
    for user_id in sorted(predictions):
        products = [int(item["product_id"]) for item in predictions[user_id]]
        novel = novel_truth[user_id]
        repeat = repeat_truth[user_id]
        rows.append({
            "user_id": user_id,
            "recall_at_10": recall_at_k(products, truth[user_id], TOP_K),
            "precision_at_10": precision_at_k(products, truth[user_id], TOP_K),
            "ndcg_at_10": ndcg_at_k(products, truth[user_id], TOP_K),
            "novel_recall_at_10": recall_at_k(products, novel, TOP_K) if novel else np.nan,
            "novel_hit_rate": float(bool(set(products) & novel)) if novel else np.nan,
            "repeat_recall_at_10": recall_at_k(products, repeat, TOP_K) if repeat else np.nan,
        })
    return pd.DataFrame(rows).set_index("user_id")


def paired_intervals(candidate: pd.DataFrame, control: pd.DataFrame, samples: int = BOOTSTRAP_SAMPLES) -> dict:
    """Paired bootstrap over users, conditional on truth eligibility per metric."""
    if not candidate.index.equals(control.index):
        raise ValueError("Paired comparison requires identical ordered users.")
    rng = np.random.default_rng(SEED)
    result = {}
    for metric in candidate.columns:
        delta = (candidate[metric] - control[metric]).dropna().to_numpy()
        if not len(delta):
            raise ValueError(f"No eligible users for {metric}.")
        means = np.empty(samples)
        for start in range(0, samples, 50):
            count = min(50, samples - start)
            indices = rng.integers(0, len(delta), size=(count, len(delta)))
            means[start:start + count] = delta[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        result[metric] = {
            "users": len(delta), "mean_delta": float(delta.mean()),
            "ci95_low": float(low), "ci95_high": float(high),
        }
    return result


def evaluate_model(training: dict) -> dict:
    start = perf_counter()
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != training["model_sha256"]:
        raise ValueError("Novel model does not match its training report.")
    scored = diagnostic.load_scored_candidates()
    users = np.sort(scored.user_id.unique())
    if user_fingerprint(users) != training["data"]["outer_validation_user_sha256"]:
        raise ValueError("Outer validation cohort changed since training.")
    if scored.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Duplicate validation user-product pairs.")
    scored["novel_score"] = np.nan
    unseen_indices = np.flatnonzero(scored.seen_before.to_numpy() == 0)
    print(f"Scoring {len(unseen_indices):,} unseen validation candidates...", flush=True)
    for start_index in range(0, len(unseen_indices), 100_000):
        indices = unseen_indices[start_index:start_index + 100_000]
        scored.loc[indices, "novel_score"] = model.inplace_predict(
            scored.iloc[indices][FEATURES].astype(np.float32)
        )
    scored = scored[["user_id", "product_id", "seen_before", "score", "novel_score", "user_latest_order"]].copy()
    del model
    gc.collect()
    truth, repeat_truth, novel_truth = diagnostic.load_truth()
    if not set(users).issubset(truth):
        raise ValueError("Missing official truth for a validation user.")

    policies = {name: {} for name in [POLICY_ZERO, POLICY_ONE, POLICY_NOVEL]}
    latest_orders = {}
    changed_users = 0
    deficit_users = 0
    print("Building matched zero-slot and one-slot policies...", flush=True)
    for user_id, frame in tqdm(scored.groupby("user_id", sort=False), total=len(users), desc="Novel-slot policies"):
        user_id = int(user_id)
        latest_orders[user_id] = int(frame.user_latest_order.iloc[0])
        original = frame.head(TOP_K)[["product_id", "seen_before", "score"]].to_dict("records")
        control = one_slot_policy(frame, "score")
        novel = one_slot_policy(frame, "novel_score")
        policies[POLICY_ZERO][user_id] = original
        policies[POLICY_ONE][user_id] = control
        policies[POLICY_NOVEL][user_id] = novel
        assert sum(item["seen_before"] == 0 for item in control) == sum(item["seen_before"] == 0 for item in novel)
        deficit_users += int(not any(item["seen_before"] == 0 for item in original))
        changed_users += int([item["product_id"] for item in novel] != [item["product_id"] for item in control])
    candidate_rows = len(scored)
    del scored
    gc.collect()
    results = {}
    user_metrics = {}
    for name, predictions in policies.items():
        print(f"Evaluating {name}...", flush=True)
        results[name] = diagnostic.evaluate_policy(predictions, latest_orders, truth, repeat_truth, novel_truth)
        user_metrics[name] = per_user_metrics(predictions, truth, repeat_truth, novel_truth)

    # Catch changes in sorting, cohort, or metric semantics before interpretation.
    previous = json.loads(diagnostic.RESULT_PATH.read_text(encoding="utf-8"))["policies"]
    for name, old_name in [(POLICY_ZERO, "minimum_0_novel_slots"), (POLICY_ONE, "minimum_1_novel_slots")]:
        for section in ["overall", "repeat", "novel", "composition"]:
            for metric, expected in previous[old_name][section].items():
                actual = results[name][section][metric]
                if isinstance(expected, dict):
                    if actual != expected:
                        raise ValueError(f"Control drift: {name}/{section}/{metric}")
                elif not np.isclose(actual, expected, rtol=0, atol=1e-10):
                    raise ValueError(f"Control drift: {name}/{section}/{metric}: {actual} != {expected}")
    print("Both controls reproduce the discovery-slot benchmark. Bootstrapping paired differences...", flush=True)
    comparisons = {
        control: paired_intervals(user_metrics[POLICY_NOVEL], user_metrics[control])
        for control in [POLICY_ONE, POLICY_ZERO]
    }
    USER_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(user_metrics, names=["policy"]).reset_index().to_parquet(USER_METRICS_PATH, index=False)
    result = {
        "experiment": "dedicated_unseen_ranker_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_report": str(TRAINING_PATH.relative_to(ROOT)),
        "base_model_sha256": hashlib.sha256(base.MODEL_PATH.read_bytes()).hexdigest(),
        "novel_model_sha256": training["model_sha256"],
        "users_evaluated": len(users), "candidate_rows": candidate_rows,
        "base_users_without_novel_top10": deficit_users,
        "users_with_different_selection_vs_one_slot": changed_users,
        "policy_definition": "Minimum one novel slot. Change only the inserted unseen candidate for users lacking novelty; keep already-satisfied lists and use base scores for final positions.",
        "policies": results,
        "paired_differences": comparisons,
        "uncertainty": {
            "method": "Paired user bootstrap; percentile 95% marginal intervals; conditional on eligible truth for repeat/novel metrics.",
            "samples": BOOTSTRAP_SAMPLES, "seed": SEED,
            "limitation": "Development-cohort intervals conditional on fitted models; no adjustment for earlier validation-driven architecture selection or multiple metrics.",
        },
        "evaluation_seconds": perf_counter() - start,
        "final_test_used": False,
    }
    save_json(RESULT_PATH, result)
    print(json.dumps({name: {**value["overall"], **value["novel"]} for name, value in results.items()}, indent=2))
    print(json.dumps(comparisons, indent=2))
    print(f"Saved: {RESULT_PATH}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8")) if args.evaluate_only else train_model()
    gc.collect()
    evaluate_model(training)


if __name__ == "__main__":
    main()
