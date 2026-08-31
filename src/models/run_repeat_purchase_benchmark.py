from pathlib import Path
import json

import pandas as pd


from src.models.repeat_purchase import (
    RepeatPurchaseRecommender
)

from src.evaluation.evaluate import (
    evaluate_model
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


train_path = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "train_interactions.parquet"
)


validation_path = (
    PROJECT_ROOT /
    "data" /
    "processed" /
    "validation_interactions.parquet"
)


results_path = (
    PROJECT_ROOT /
    "results" /
    "benchmarks" /
    "repeat_purchase.json"
)


results_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


print(
    "Loading training data..."
)


train = pd.read_parquet(
    train_path
)


model = RepeatPurchaseRecommender()


print(
    "Training repeat purchase model..."
)


model.fit(
    train
)


print(
    "Loading validation data..."
)


validation = pd.read_parquet(
    validation_path
)


print(
    "Evaluating..."
)


results = evaluate_model(
    model=model,
    evaluation_data=validation,
    k=10,
)


print(results)


results_path.write_text(
    json.dumps(
        results,
        indent=4
    ),
    encoding="utf-8",
)


print(
    f"Saved results to {results_path}"
)