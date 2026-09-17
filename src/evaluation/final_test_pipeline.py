"""Execute the frozen final experiment with a prediction-before-label barrier.

Each stage runs in its own process to release Spark/ML memory. Completed stages
are content-hashed and reusable, but never silently overwritten. The immutable
architecture manifest remains the record of what was decided before testing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
DATA = PROCESSED / "final_v1"
MODELS = ROOT / "models/final_v1"
CONTRACT = ROOT / "configs/final_architecture.json"
REPORT = ROOT / "results/benchmarks/final_test_v1.json"
SOURCES = {
    "repeat_purchase": "repeat_purchase_top50.parquet",
    "als_v2": "als_v2_top50.parquet",
    "content_v1": "content_v1_top50.parquet",
}
STAGES = ["preflight", "refit-data", "refit", "history", "retrieve-repeat", "retrieve-content", "retrieve-als", "features", "predict"]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inventory(paths):
    result = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if file.is_file():
                result[str(file.relative_to(ROOT)).replace("\\", "/")] = digest(file)
    return result


def verify_hashes(hashes):
    for relative, expected in hashes.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Artifact changed: {relative}")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def receipt_path(stage):
    return MODELS / f"stage_{stage}.json"


def receipt(stage, paths, stats):
    save_json(receipt_path(stage), {
        "stage": stage, "completed_at": now(), "contract_sha256": digest(CONTRACT),
        "outputs": inventory(paths), "statistics": stats,
    })


def verify_stage(stage):
    saved = json.loads(receipt_path(stage).read_text(encoding="utf-8"))
    if saved["contract_sha256"] != digest(CONTRACT):
        raise ValueError("Frozen contract changed")
    verify_hashes(saved["outputs"])
    return saved


def new_output(path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    if not any(path.resolve().is_relative_to(root.resolve()) for root in (DATA, MODELS)):
        raise ValueError("Output must be inside the dedicated final experiment directories")
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def module_paths(module, **overrides):
    old = {key: getattr(module, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(module, key, value)


def write_spark(df, path):
    from src.utils.spark_io import write_spark_df_local_parquet
    new_output(path)
    # The legacy writer removes existing output directories. Give it a fresh,
    # validated child path so it cannot remove any completed artifact.
    temporary = path.parent / (path.name + ".building-" + uuid.uuid4().hex)
    new_output(temporary)
    count = write_spark_df_local_parquet(df, temporary, num_partitions=32)
    temporary.rename(path)
    return count


def build_features(spark, history, items, candidates):
    """Apply the unchanged frozen feature functions with an explicit cutoff."""
    from pyspark.sql import functions as F
    from src.features import build_ranker_dataset_v1_corrected as legacy
    with module_paths(legacy, TRAIN_PATH=history, CANDIDATE_DIR=candidates):
        union = legacy.build_candidate_union(spark)
        train, user_product, user_summary = legacy.build_history_features(spark)
    item = spark.read.parquet(str(items)).select(
        F.col("product_id").cast("int"), F.col("aisle_id").cast("int"),
        F.col("department_id").cast("int"),
        F.col("purchase_count").cast("float").alias("product_purchase_count"),
        F.col("reorder_rate").cast("float").alias("product_reorder_rate"),
        F.col("log_purchase_count").cast("float").alias("product_log_purchase_count"),
    )
    aisle, department = legacy.build_category_affinities(train, item)
    features = (union.join(user_product, ["user_id", "product_id"], "left")
                .join(user_summary, "user_id", "left").join(item, "product_id", "left")
                .join(aisle, ["user_id", "aisle_id"], "left")
                .join(department, ["user_id", "department_id"], "left")
                .fillna({"user_product_purchase_count": 0., "user_product_reorder_rate": 0.,
                         "aisle_affinity": 0., "department_affinity": 0.})
                .withColumn("seen_before", (F.col("user_product_purchase_count") > 0).cast("int"))
                .withColumn("orders_since_last_purchase", F.when(F.col("last_purchase_order").isNull(),
                    F.col("user_latest_order") + 1).otherwise(
                    F.col("user_latest_order") - F.col("last_purchase_order")).cast("float"))
                .fillna(legacy.RANK_COLUMNS))
    return features.select("user_id", "product_id", *contract()["ranker"]["feature_columns"])


def sample_negatives(labeled, limit=15):
    from pyspark.sql import Window, functions as F
    positives = labeled.filter(F.col("label") == 1)
    window = Window.partitionBy("user_id").orderBy(F.xxhash64("user_id", "product_id"), F.col("product_id"))
    negatives = (labeled.filter(F.col("label") == 0)
                 .withColumn("negative_rank", F.row_number().over(window))
                 .filter(F.col("negative_rank") <= limit).drop("negative_rank"))
    return positives.unionByName(negatives)


def preflight():
    import platform
    import importlib.metadata
    cfg = contract()
    if cfg["status"] != "frozen" or cfg["final_test_used_at_freeze"]:
        raise ValueError("Expected an untouched, frozen contract")
    verify_hashes(cfg["reference_implementation_sha256"])
    verify_hashes({cfg["selection_evidence"]: cfg["selection_evidence_sha256"],
                   cfg["ranker"]["development_checkpoint"]["path"]: cfg["ranker"]["development_checkpoint"]["sha256"]})
    paths = [CONTRACT, ROOT / cfg["selection_evidence"],
             ROOT / cfg["ranker"]["development_checkpoint"]["path"],
             *[ROOT / p for p in cfg["reference_implementation_sha256"]],
             PROCESSED / "train_interactions.parquet", PROCESSED / "validation_interactions.parquet",
             PROCESSED / "products.parquet", PROCESSED / "item_features.parquet",
             *[PROCESSED / "candidates" / file for file in SOURCES.values()]]
    receipt("preflight", paths, {"python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("pandas", "numpy", "xgboost", "pyspark", "pyarrow")},
        "test_labels_read": False})


def refit_data():
    from src.features import build_ranker_dataset_v1_corrected as legacy
    spark = legacy.create_spark_session()
    try:
        features = build_features(spark, PROCESSED / "train_interactions.parquet",
                                  PROCESSED / "item_features.parquet", PROCESSED / "candidates")
        labeled = legacy.add_labels(spark, features)
        sampled = sample_negatives(labeled)
        path = DATA / "refit_features"
        rows = write_spark(sampled, path)
        receipt("refit-data", [path], {"sampled_rows": rows, "history": "train", "labels": "validation"})
    finally:
        spark.stop()


def refit():
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from src.models.run_xgb_ranker_v1_corrected import prepare_dataframe, keep_users_with_positive, build_quantile_matrix
    frame = prepare_dataframe(DATA / "refit_features")
    if frame.duplicated(["user_id", "product_id"]).any() or not frame.label.isin([0, 1]).all():
        raise ValueError("Invalid refit queries/labels")
    users = int(frame.user_id.nunique())
    if users != contract()["final_test_protocol"]["expected_users"]:
        raise ValueError("Refit cohort mismatch")
    validation = pd.read_parquet(PROCESSED / "validation_interactions.parquet", columns=["user_id", "product_id"])
    target_keys = pd.MultiIndex.from_frame(validation).unique()
    actual_labels = pd.MultiIndex.from_frame(frame[["user_id", "product_id"]]).isin(target_keys)
    if not np.array_equal(frame.label.to_numpy(), actual_labels.astype(np.int8)):
        raise ValueError("Refit labels disagree with validation baskets")
    # Compare rebuilt features with the established development pipeline on a
    # deterministic cross-section of both former user splits (before fitting).
    sample_users = sorted(frame.user_id.unique())[::997]
    reference_parts = [pd.read_parquet(PROCESSED / "ranker_v1_corrected" / name,
        filters=[("user_id", "in", sample_users)]) for name in ("ranker_train", "ranker_validation")]
    reference = pd.concat(reference_parts, ignore_index=True)
    rebuilt = frame.loc[frame.user_id.isin(sample_users)].set_index(["user_id", "product_id"]).sort_index()
    reference = reference.set_index(["user_id", "product_id"]).reindex(rebuilt.index)
    for column in [*contract()["ranker"]["feature_columns"], "label"]:
        if not np.allclose(rebuilt[column], reference[column].fillna(0), rtol=1e-6, atol=1e-6):
            raise ValueError(f"Rebuilt feature differs from development reference: {column}")
    negatives = frame.loc[frame.label.eq(0)].groupby("user_id").size()
    if (negatives > 15).any():
        raise ValueError("Negative sampling exceeded frozen limit")
    frame = keep_users_with_positive(frame)
    stats = {"users_before_positive_filter": users, "fitting_users": int(frame.user_id.nunique()),
             "fitting_rows": len(frame), "positives": int(frame.label.sum()), "rounds": 486,
             "reference_feature_check_users": len(sample_users), "reference_feature_check_rows": len(rebuilt),
             "all_sampled_labels_verified_against_validation": True}
    if not np.isfinite(frame[contract()["ranker"]["feature_columns"]].to_numpy()).all():
        raise ValueError("Nonfinite refit features")
    matrix = build_quantile_matrix(frame)
    del frame
    gc.collect()
    params = dict(contract()["ranker"]["parameters"], nthread=max(1, os.cpu_count() - 1))
    print(f"Refitting fixed 486 rounds on {stats['fitting_users']:,} positive queries", flush=True)
    model = xgb.train(params, matrix, num_boost_round=486)
    if model.num_boosted_rounds() != 486 or model.feature_names != contract()["ranker"]["feature_columns"]:
        raise ValueError("Refit model does not match contract")
    path = MODELS / "ranker.json"
    new_output(path)
    model.save_model(path)
    receipt("refit", [path], stats)


def validate_temporal_history(train, validation, expected_users):
    if train.user_id.nunique() != expected_users or set(train.user_id.unique()) != set(validation.user_id.unique()):
        raise ValueError("History cohort mismatch")
    if validation.groupby("user_id").order_number.nunique().ne(1).any():
        raise ValueError("Expected one validation basket per user")
    latest = train.groupby("user_id").order_number.max()
    next_order = validation.groupby("user_id").order_number.min().reindex(latest.index)
    if not (next_order > latest).all():
        raise ValueError("Temporal overlap between history and validation")


def history():
    import pandas as pd
    import pyarrow.parquet as pq
    from src.features import item_features
    columns = ["user_id", "product_id", "order_id", "order_number"]
    train = pd.read_parquet(PROCESSED / "train_interactions.parquet", columns=columns)
    val = pd.read_parquet(PROCESSED / "validation_interactions.parquet", columns=columns)
    validate_temporal_history(train, val, contract()["final_test_protocol"]["expected_users"])
    if train.duplicated(["order_id", "product_id"]).any() or val.duplicated(["order_id", "product_id"]).any():
        raise ValueError("Duplicate basket products")
    if set(train.order_id.unique()) & set(val.order_id.unique()):
        raise ValueError("Order IDs overlap")
    combined = pd.concat([train, val], ignore_index=True)
    histories = combined.groupby("user_id").agg(history_orders=("order_number", "nunique"), latest_order=("order_number", "max"))
    path = DATA / "history_summary.parquet"
    new_output(path)
    histories.reset_index().to_parquet(path, index=False)
    stats = {"train_rows": len(train), "validation_rows": len(val), "history_rows": len(combined),
             "users": len(histories), "observed_catalog_products": int(combined.product_id.nunique())}
    del combined, train, val
    gc.collect()
    output = DATA / "history.parquet"
    new_output(output)
    temporary = output.with_suffix(".parquet.tmp")
    schema = pq.ParquetFile(PROCESSED / "train_interactions.parquet").schema_arrow
    with pq.ParquetWriter(temporary, schema, compression="snappy") as writer:
        for name in ("train_interactions.parquet", "validation_interactions.parquet"):
            for batch in pq.ParquetFile(PROCESSED / name).iter_batches(batch_size=250000):
                writer.write_batch(batch)
    temporary.replace(output)
    item_path = DATA / "item_features.parquet"
    new_output(item_path)
    with module_paths(item_features, TRAIN_PATH=output, OUTPUT_PATH=item_path):
        items = item_features.build_item_features()
    popular = items.loc[items.purchase_count.gt(0)].sort_values(["purchase_count", "product_id"], ascending=[False, True]).product_id.head(50).astype(int).tolist()
    pop_path = MODELS / "popularity.json"
    new_output(pop_path)
    save_json(pop_path, {"recommendations": popular, "history": "train + validation", **stats})
    receipt("history", [path, output, item_path, pop_path], stats)


def retrieve(source):
    from src.candidates.build_candidate_store import write_candidate_dict
    path = DATA / "candidates" / SOURCES[source]
    new_output(path)
    if source == "als_v2":
        from src.features.build_ranker_dataset_v1_corrected import create_spark_session
        from src.models.als import ALSRecommender
        spark = create_spark_session()
        try:
            cfg = contract()["retrieval"]["als_v2"]
            model = ALSRecommender(**{key: cfg[key] for key in ("rank", "alpha", "reg_param", "max_iter", "seed")}, precompute_k=50)
            model.fit(spark, DATA / "history.parquet")
            recs = model.recommendations
            write_candidate_dict(recs, path, 50)
        finally:
            spark.stop()
    else:
        import pandas as pd
        if source == "repeat_purchase":
            from src.models.repeat_purchase import RepeatPurchaseRecommender
            frame = pd.read_parquet(DATA / "history.parquet", columns=["user_id", "product_id", "order_number", "reordered"])
            recs = RepeatPurchaseRecommender().fit(frame).user_products
        else:
            from src.models.content_based import ContentBasedRecommender
            frame = pd.read_parquet(DATA / "history.parquet", columns=["user_id", "product_id"])
            items = pd.read_parquet(DATA / "item_features.parquet")
            recs = ContentBasedRecommender(**contract()["retrieval"]["content_v1"], precompute_k=50).fit(frame, items).recommendations
        write_candidate_dict(recs, path, 50)
    if len(recs) != contract()["final_test_protocol"]["expected_users"]:
        raise ValueError("Retrieval cohort mismatch")
    stage = {"repeat_purchase": "retrieve-repeat", "content_v1": "retrieve-content", "als_v2": "retrieve-als"}[source]
    receipt(stage, [path], {"users": len(recs), "top_n": 50})


def features():
    from src.features.build_ranker_dataset_v1_corrected import create_spark_session
    spark = create_spark_session()
    try:
        frame = build_features(spark, DATA / "history.parquet", DATA / "item_features.parquet", DATA / "candidates")
        path = DATA / "prediction_features"
        rows = write_spark(frame, path)
        receipt("features", [path], {"candidate_rows": rows, "history": "train + validation", "labels": "none"})
    finally:
        spark.stop()


def validate_predictions(predictions, users, catalog, k=10, require_full=False):
    if set(predictions) != set(users):
        raise ValueError("Prediction cohort mismatch")
    for user, items in predictions.items():
        if len(items) > k or len(set(items)) != len(items):
            raise ValueError(f"Invalid recommendation length/duplicates: {user}")
        if require_full and len(items) != k:
            raise ValueError(f"Incomplete primary prediction: {user}")
        if not set(items).issubset(catalog):
            raise ValueError(f"Unknown recommended product: {user}")


def load_lists(path):
    import pandas as pd
    frame = pd.read_parquet(path)
    if frame.user_id.duplicated().any():
        raise ValueError("Duplicate prediction users")
    return {int(user): list(map(int, items)) for user, items in frame.itertuples(index=False, name=None)}


def predict():
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from src.candidates.build_candidate_store import write_candidate_dict
    from src.serving.routing import route_recommendations, fill_with_popularity
    cfg = contract()
    columns = cfg["ranker"]["feature_columns"]
    summary = pd.read_parquet(DATA / "history_summary.parquet").set_index("user_id")
    users = set(map(int, summary.index))
    items = pd.read_parquet(DATA / "item_features.parquet")
    catalog = set(map(int, items.product_id))
    popular = json.loads((MODELS / "popularity.json").read_text())["recommendations"]
    model = xgb.Booster()
    model.load_model(MODELS / "ranker.json")
    if model.feature_names != columns or model.num_boosted_rounds() != 486:
        raise ValueError("Unexpected model feature order or number of rounds")
    ranked = {}
    all_candidate_users = set()
    for path in sorted((DATA / "prediction_features").glob("*.parquet")):
        frame = pd.read_parquet(path)
        if "label" in frame or frame.duplicated(["user_id", "product_id"]).any():
            raise ValueError("Prediction features contain labels or duplicate candidates")
        part_users = set(map(int, frame.user_id.unique()))
        if part_users & all_candidate_users:
            raise ValueError("A user spans partitions; ranking would be incomplete")
        all_candidate_users.update(part_users)
        latest = frame.groupby("user_id").user_latest_order.first()
        if not latest.eq(summary.latest_order.reindex(latest.index)).all():
            raise ValueError("Prediction features have the wrong temporal cutoff")
        values = frame[columns].fillna(0).astype(np.float32)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Nonfinite prediction features")
        frame["score"] = model.predict(xgb.DMatrix(values, feature_names=columns))
        if not np.isfinite(frame.score).all():
            raise ValueError("Nonfinite model scores")
        top = frame.sort_values(["user_id", "score", "product_id"], ascending=[True, False, True]).groupby("user_id", sort=False).head(10)
        ranked.update(top.groupby("user_id").product_id.agg(list).to_dict())
        print(f"Scored {len(all_candidate_users):,} users", flush=True)
    if all_candidate_users != users:
        raise ValueError("Candidate cohort mismatch")
    outputs = []
    repeat = None
    for source, filename in SOURCES.items():
        recs = {u: p[:10] for u, p in load_lists(DATA / "candidates" / filename).items()}
        validate_predictions(recs, users, catalog)
        path = DATA / "predictions" / f"{source}.parquet"
        new_output(path)
        write_candidate_dict(recs, path, 10)
        outputs.append(path)
        if source == "repeat_purchase":
            repeat = recs
    policies = {
        "global_popularity": {u: popular[:10] for u in users},
        "repeat_with_popularity_fill": {u: fill_with_popularity(repeat[u], popular, 10) for u in users},
        "selected_lambdamart": {u: route_recommendations(history_order_count=int(summary.loc[u, "history_orders"]),
            ranked=ranked[u], repeat=repeat[u], popularity=popular).recommendations for u in users},
    }
    for name, recs in policies.items():
        validate_predictions(recs, users, catalog, require_full=True)
        path = DATA / "predictions" / f"{name}.parquet"
        new_output(path)
        write_candidate_dict(recs, path, 10)
        outputs.append(path)
    receipt("predict", outputs, {"users": len(users), "policies": cfg["final_test_protocol"]["baselines"], "test_labels_read": False})
    # Verify every input, intermediate artifact and prediction before sealing.
    hashes = {}
    for stage in STAGES:
        saved = verify_stage(stage)
        hashes.update(saved["outputs"])
    hashes.update(inventory([Path(__file__)]))
    save_json(MODELS / "prediction_seal.json", {"sealed_at": now(), "contract_sha256": digest(CONTRACT),
        "users": len(users), "test_labels_read": False, "artifacts": hashes})


def verify_prediction_seal():
    seal = json.loads((MODELS / "prediction_seal.json").read_text(encoding="utf-8"))
    if seal["contract_sha256"] != digest(CONTRACT) or seal["test_labels_read"]:
        raise ValueError("Invalid prediction seal")
    if seal["users"] != contract()["final_test_protocol"]["expected_users"]:
        raise ValueError("Sealed cohort mismatch")
    required = [DATA / "predictions" / f"{name}.parquet" for name in contract()["final_test_protocol"]["baselines"]]
    required += [MODELS / "ranker.json", DATA / "history.parquet", DATA / "item_features.parquet", Path(__file__)]
    for path in required:
        key = str(path.relative_to(ROOT)).replace("\\", "/")
        if key not in seal["artifacts"]:
            raise ValueError(f"Missing sealed artifact: {key}")
    verify_hashes(seal["artifacts"])
    return seal


def extended_summary(frame, predictions, catalog_size):
    from src.evaluation.history_availability import summarize
    result = summarize(frame, predictions, catalog_size)
    result["users_with_repeat_truth"] = int(frame.repeat_recall_at_10.notna().sum())
    for name in ("repeat_recall_at_10", "unique_aisles", "unique_departments", "max_products_in_one_aisle", "dominant_aisle_share", "aisle_pair_diversity"):
        result[name] = float(frame[name].mean()) if frame[name].notna().any() else None
    return result


def evaluate():
    # No fitting, scoring, or recommendation mutation is permitted below this
    # barrier. All six complete prediction artifacts must already be sealed.
    seal = verify_prediction_seal()
    if REPORT.exists():
        print("Final test report already exists; refusing a second evaluation", flush=True)
        return
    import numpy as np
    import pandas as pd
    from src.evaluation.history_availability import evaluate_users, history_segment
    from src.evaluation.diversity_architecture import category_metrics
    users_frame = pd.read_parquet(DATA / "history_summary.parquet").set_index("user_id")
    histories = users_frame.history_orders.astype(int).to_dict()
    items = pd.read_parquet(DATA / "item_features.parquet")
    catalog = set(map(int, items.product_id))
    catalog_size = int(items.purchase_count.gt(0).sum())
    for name in contract()["final_test_protocol"]["baselines"]:
        validate_predictions(load_lists(DATA / "predictions" / f"{name}.parquet"), histories, catalog,
            require_full=name in ("global_popularity", "repeat_with_popularity_fill", "selected_lambdamart"))
    access = MODELS / "test_access.json"
    if not access.exists():
        with access.open("x", encoding="utf-8") as stream:
            json.dump({"first_access_at": now(), "prediction_seal_sha256": digest(MODELS / "prediction_seal.json"),
                       "purpose": "Metric calculation only; fitting and predictions locked"}, stream, indent=2)
    elif json.loads(access.read_text())["prediction_seal_sha256"] != digest(MODELS / "prediction_seal.json"):
        raise ValueError("Cannot change predictions after test access")
    print("Prediction integrity verified. Opening final-test labels for metrics only.", flush=True)
    test_path = ROOT / contract()["final_test_protocol"]["test_truth"]
    target = pd.read_parquet(test_path, columns=["user_id", "product_id", "order_id", "order_number"])
    if set(target.user_id.unique()) != set(histories) or target.groupby("user_id").order_id.nunique().ne(1).any():
        raise ValueError("Final-test population/basket mismatch")
    if target.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Duplicate final-test basket products")
    if not (target.groupby("user_id").order_number.min() > users_frame.latest_order).all():
        raise ValueError("Final test does not follow prediction history")
    history_pairs = pd.read_parquet(DATA / "history.parquet", columns=["user_id", "product_id"]).drop_duplicates()
    target["seen_before"] = pd.MultiIndex.from_frame(target[["user_id", "product_id"]]).isin(pd.MultiIndex.from_frame(history_pairs))
    del history_pairs
    gc.collect()
    truth = target.groupby("user_id").product_id.agg(set).to_dict()
    novel_groups = target.loc[~target.seen_before].groupby("user_id").product_id.agg(set).to_dict()
    repeat_groups = target.loc[target.seen_before].groupby("user_id").product_id.agg(set).to_dict()
    novel_truth = {u: novel_groups.get(u, set()) for u in histories}
    repeats = {u: repeat_groups.get(u, set()) for u in histories}
    aisles = items.set_index("product_id").aisle_id.to_dict()
    departments = items.set_index("product_id").department_id.to_dict()
    results = {}
    metrics_paths = []
    for name in contract()["final_test_protocol"]["baselines"]:
        print(f"Evaluating {name}", flush=True)
        recs = load_lists(DATA / "predictions" / f"{name}.parquet")
        frame = evaluate_users(recs, truth, histories, novel_truth)
        frame["repeat_recall_at_10"] = [len(set(recs[u]) & repeats[u]) / len(repeats[u]) if repeats[u] else np.nan for u in frame.index]
        diversity = pd.DataFrame([category_metrics(recs[u], aisles, departments) for u in frame.index], index=frame.index)
        frame = frame.join(diversity)
        results[name] = {"overall": extended_summary(frame, recs, catalog_size),
            "segments": {segment: extended_summary(rows, recs, catalog_size) for segment, rows in frame.groupby("segment")}}
        path = DATA / "metrics" / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index().to_parquet(path, index=False)
        metrics_paths.append(path)
    # Compute union candidate recall from the saved, deduplicated feature rows.
    # No candidate retrieval is rerun after accessing labels.
    hit_counts = pd.Series(0, index=users_frame.index, dtype="int64")
    candidate_counts = hit_counts.copy()
    keys = target[["user_id", "product_id"]]
    for path in sorted((DATA / "prediction_features").glob("*.parquet")):
        pairs = pd.read_parquet(path, columns=["user_id", "product_id"])
        hits = pairs.merge(keys, on=["user_id", "product_id"], how="inner", validate="one_to_one").groupby("user_id").size()
        hit_counts = hit_counts.add(hits, fill_value=0)
        candidate_counts = candidate_counts.add(pairs.groupby("user_id").size(), fill_value=0)
    candidate = pd.DataFrame({"candidate_recall": hit_counts / target.groupby("user_id").size(),
        "candidate_count": candidate_counts, "segment": users_frame.history_orders.map(history_segment)})
    candidate_path = DATA / "metrics/candidate_recall.parquet"
    candidate.reset_index().to_parquet(candidate_path, index=False)
    metrics_paths.append(candidate_path)
    retrieval = {"union": {"users": len(candidate), "candidate_recall": float(candidate.candidate_recall.mean()),
        "mean_candidates": float(candidate.candidate_count.mean()),
        "segments": {s: {"users": len(g), "candidate_recall": float(g.candidate_recall.mean()),
            "mean_candidates": float(g.candidate_count.mean())} for s, g in candidate.groupby("segment")}}}
    for name, filename in SOURCES.items():
        recs = load_lists(DATA / "candidates" / filename)
        recall = np.array([len(set(recs[u]) & truth[u]) / len(truth[u]) for u in sorted(truth)])
        retrieval[name] = {"users": len(recall), "candidate_recall_at_50": float(recall.mean())}
    verify_prediction_seal()
    report = {"architecture_id": contract()["architecture_id"], "completed_at": now(),
        "contract_sha256": digest(CONTRACT), "prediction_sealed_at": seal["sealed_at"],
        "prediction_seal_sha256": digest(MODELS / "prediction_seal.json"),
        "test_first_access": json.loads(access.read_text()), "test_truth_sha256": digest(test_path),
        "test_users": len(truth), "test_interactions": len(target), "observed_history_catalog_products": catalog_size,
        "refit": verify_stage("refit")["statistics"], "prediction_history": verify_stage("history")["statistics"],
        "policies": results, "candidate_retrieval": retrieval, "per_user_metrics": inventory(metrics_paths),
        "protocol": {"test_used_for_tuning": False, "precision_denominator": 10,
            "repeat_novel_truth": "Membership in train + validation history, conditional on eligible users",
            "aggregation": f"Macro over all {len(truth)} users, including empty recommendation lists",
            "cold_start": "Separate historical simulation; excluded from this known-user next-basket test"}}
    save_json(REPORT, report)
    rows = [{"policy": name, **value["overall"]} for name, value in results.items()]
    pd.DataFrame(rows).to_csv(REPORT.with_suffix(".csv"), index=False)
    print(json.dumps({name: value["overall"] for name, value in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "evaluate", "run"])
    stage = parser.parse_args().stage
    if stage == "run":
        for name in [*STAGES, "evaluate"]:
            subprocess.run([sys.executable, "-u", "-m", "src.evaluation.final_test_pipeline", name], check=True, cwd=ROOT)
        return
    if stage == "evaluate":
        evaluate()
        return
    if (MODELS / "test_access.json").exists():
        raise RuntimeError("Test labels have been opened; fitting/prediction stages are locked")
    if receipt_path(stage).exists():
        verify_stage(stage)
        print(f"Verified completed stage: {stage}", flush=True)
        return
    for prerequisite in STAGES[:STAGES.index(stage)]:
        verify_stage(prerequisite)
    print(f"Starting {stage} at {now()}", flush=True)
    actions = {"preflight": preflight, "refit-data": refit_data, "refit": refit, "history": history,
        "retrieve-repeat": lambda: retrieve("repeat_purchase"), "retrieve-content": lambda: retrieve("content_v1"),
        "retrieve-als": lambda: retrieve("als_v2"), "features": features, "predict": predict}
    actions[stage]()
    print(f"Completed {stage} at {now()}", flush=True)


if __name__ == "__main__":
    main()
