"""Read-only serving snapshot built exclusively from sealed pre-test artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from src.serving import routing

ROOT = Path(__file__).resolve().parents[2]
DATA = "data/processed/final_v1"
MODELS = "models/final_v1"
CONFIG = "configs/final_architecture.json"
PRODUCTS = f"{DATA}/item_features.parquet"
HISTORIES = f"{DATA}/history_summary.parquet"
RANKED = f"{DATA}/predictions/selected_lambdamart.parquet"
REPEAT = f"{DATA}/predictions/repeat_purchase.parquet"
POPULARITY = f"{MODELS}/popularity.json"


class ArtifactError(ValueError):
    """An artifact is missing, inconsistent, or does not match its seal."""


@dataclass(frozen=True, slots=True)
class Product:
    product_id: int
    product_name: str
    aisle_id: int
    aisle: str
    department_id: int
    department: str


@dataclass(frozen=True)
class RecommendationStore:
    model_version: str
    snapshot_id: str
    snapshot_created_at: str
    products: Mapping[int, Product]
    histories: Mapping[int, int]
    ranked: Mapping[int, tuple[int, ...]]
    repeat: Mapping[int, tuple[int, ...]]
    popularity: tuple[int, ...]
    degraded: bool

    def recommend(self, user_id: int, k: int = 10):
        if user_id < 1 or not 1 <= k <= 10:
            raise ValueError("user_id must be positive and k must be between 1 and 10")
        return routing.route_recommendations(
            history_order_count=self.histories.get(user_id, 0),
            ranked=self.ranked.get(user_id), repeat=self.repeat.get(user_id),
            popularity=self.popularity, k=k,
        )


def _positive_integer(value, field):
    if type(value) is not int or value < 1:
        raise ArtifactError(f"Invalid {field}")
    return value


def _items(values, catalog, maximum):
    if not isinstance(values, list) or len(values) > maximum:
        raise ArtifactError("Invalid recommendation list")
    result = tuple(_positive_integer(value, "product_id") for value in values)
    if len(result) != len(set(result)) or any(product not in catalog for product in result):
        raise ArtifactError("Duplicate or unknown recommendation products")
    return result


def load_snapshot(root: Path = ROOT) -> RecommendationStore:
    """Verify and load only artifacts needed for serving, never labels or models.

    Missing ranked/repeat files permit degraded fallback service. A present but
    corrupt file aborts startup. Hashes verify the exact bytes that are parsed.
    """
    root = Path(root).resolve()
    seal_bytes = (root / MODELS / "prediction_seal.json").read_bytes()
    seal = json.loads(seal_bytes)
    hashes = seal["artifacts"]

    def verified(relative):
        blob = (root / relative).read_bytes()
        if relative not in hashes or hashlib.sha256(blob).hexdigest() != hashes[relative]:
            raise ArtifactError(f"Artifact checksum mismatch: {relative}")
        return blob

    def rows(relative, columns):
        parquet = pq.ParquetFile(pa.BufferReader(verified(relative)))
        if not set(columns).issubset(parquet.schema_arrow.names):
            raise ArtifactError(f"Missing required columns: {relative}")
        for batch in parquet.iter_batches(batch_size=20000, columns=columns):
            yield from batch.to_pylist()

    config_bytes = verified(CONFIG)
    cfg = json.loads(config_bytes)
    if (hashlib.sha256(config_bytes).hexdigest() != seal["contract_sha256"]
            or cfg["status"] != "frozen" or seal["test_labels_read"]
            or cfg["recommendation_policy"]["top_k"] != 10):
        raise ArtifactError("Invalid frozen serving contract")
    expected_routing = cfg["reference_implementation_sha256"]["src/serving/routing.py"]
    if hashlib.sha256(Path(routing.__file__).read_bytes()).hexdigest() != expected_routing:
        raise ArtifactError("Serving routing differs from the evaluated policy")

    products = {}
    for row in rows(PRODUCTS, list(Product.__dataclass_fields__)):
        product_id = _positive_integer(row["product_id"], "product_id")
        if product_id in products:
            raise ArtifactError("Duplicate product metadata")
        for field in ("product_name", "aisle", "department"):
            if not isinstance(row[field], str) or not row[field]:
                raise ArtifactError(f"Missing product {field}")
        for field in ("aisle_id", "department_id"):
            _positive_integer(row[field], field)
        products[product_id] = Product(**row)
    if not products:
        raise ArtifactError("Empty product catalog")

    histories = {}
    for row in rows(HISTORIES, ["user_id", "history_orders"]):
        user = _positive_integer(row["user_id"], "user_id")
        if user in histories:
            raise ArtifactError("Duplicate history user")
        histories[user] = _positive_integer(row["history_orders"], "history_orders")
    if len(histories) != seal["users"] or len(histories) != cfg["final_test_protocol"]["expected_users"]:
        raise ArtifactError("Serving history population mismatch")

    def load_recommendations(relative):
        if not (root / relative).exists():
            return {}
        result = {}
        for row in rows(relative, ["user_id", "recommendations"]):
            user = _positive_integer(row["user_id"], "user_id")
            if user not in histories or user in result:
                raise ArtifactError("Unexpected or duplicate recommendation user")
            result[user] = _items(row["recommendations"], products, 10)
        return result

    popularity = _items(json.loads(verified(POPULARITY))["recommendations"], products, 50)
    if len(popularity) < min(10, len(products)):
        raise ArtifactError("Popularity fallback cannot fill the available catalog")
    ranked = load_recommendations(RANKED)
    repeat = load_recommendations(REPEAT)
    return RecommendationStore(
        model_version=cfg["architecture_id"], snapshot_id=hashlib.sha256(seal_bytes).hexdigest(),
        snapshot_created_at=seal["sealed_at"], products=MappingProxyType(products),
        histories=MappingProxyType(histories), ranked=MappingProxyType(ranked),
        repeat=MappingProxyType(repeat), popularity=popularity,
        degraded=(set(ranked) != set(histories) or set(repeat) != set(histories)
                  or any(len(items) < 10 for items in ranked.values())),
    )
