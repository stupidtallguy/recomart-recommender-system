import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.api.main import create_app
from src.serving import routing
from src.serving.store import (
    ArtifactError, CONFIG, HISTORIES, MODELS, POPULARITY, PRODUCTS, RANKED, REPEAT, _items, load_snapshot,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def seal(root):
    hashes = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
              for relative in (CONFIG, PRODUCTS, HISTORIES, RANKED, REPEAT, POPULARITY)
              if (root / relative).exists()}
    write_json(root / MODELS / "prediction_seal.json", {
        "artifacts": hashes, "contract_sha256": hashes[CONFIG], "users": 3,
        "test_labels_read": False, "sealed_at": "2026-09-16T07:04:16+00:00",
    })


@pytest.fixture
def artifacts(tmp_path):
    write_json(tmp_path / CONFIG, {
        "architecture_id": "recomart_final_v1", "status": "frozen",
        "recommendation_policy": {"top_k": 10}, "final_test_protocol": {"expected_users": 3},
        "reference_implementation_sha256": {
            "src/serving/routing.py": hashlib.sha256(Path(routing.__file__).read_bytes()).hexdigest()},
    })
    write_rows(tmp_path / PRODUCTS, [{"product_id": p, "product_name": f"Product {p}",
        "aisle_id": 1, "aisle": "Produce", "department_id": 1, "department": "Food"} for p in range(1, 13)])
    write_rows(tmp_path / HISTORIES, [{"user_id": u, "history_orders": u} for u in (1, 2, 3)])
    write_rows(tmp_path / RANKED, [{"user_id": 1, "recommendations": list(range(12, 2, -1))},
                                 {"user_id": 2, "recommendations": [2, 1]}])
    write_rows(tmp_path / REPEAT, [{"user_id": 1, "recommendations": [1]},
        {"user_id": 2, "recommendations": [3]}, {"user_id": 3, "recommendations": [4, 2]}])
    write_json(tmp_path / POPULARITY, {"recommendations": list(range(1, 13))})
    seal(tmp_path)
    return tmp_path


@pytest.fixture
def client(artifacts):
    with TestClient(create_app(artifact_root=artifacts)) as test_client:
        yield test_client


def ids(response):
    return [product["product_id"] for product in response.json()["recommendations"]]


def test_known_user_preserves_saved_order_and_supports_prefix(client):
    response = client.get("/recommendations/1")
    assert response.status_code == 200
    assert ids(response) == list(range(12, 2, -1))
    assert response.json()["strategy"] == "ranker"
    assert response.json()["complete"] is True
    assert ids(client.get("/recommendations/1?k=3")) == [12, 11, 10]
    assert response.json()["recommendations"][0]["product_name"] == "Product 12"


def test_unknown_user_uses_popularity(client):
    response = client.get("/recommendations/999999")
    assert response.status_code == 200
    assert response.json()["strategy"] == "global_popularity"
    assert ids(response) == list(range(1, 11))


def test_partial_ranker_and_missing_user_ranker_use_correct_fallbacks(client):
    partial = client.get("/recommendations/2")
    assert partial.json()["strategy"] == "ranker_with_popularity_fill"
    assert ids(partial) == [2, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    absent = client.get("/recommendations/3")
    assert absent.json()["strategy"] == "repeat_with_popularity_fill"
    assert ids(absent) == [4, 2, 1, 3, 5, 6, 7, 8, 9, 10]


@pytest.mark.parametrize("remove,expected", [([RANKED], "repeat_with_popularity_fill"),
                                           ([RANKED, REPEAT], "global_popularity")])
def test_missing_optional_files_start_degraded_with_fallback(artifacts, remove, expected):
    for relative in remove:
        (artifacts / relative).unlink()
    with TestClient(create_app(artifact_root=artifacts)) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.get("/health").json()["ready"] is True
        assert client.get("/recommendations/1").json()["strategy"] == expected


@pytest.mark.parametrize("relative", [PRODUCTS, RANKED, POPULARITY, HISTORIES, CONFIG])
def test_changed_artifact_aborts_startup(artifacts, relative):
    with (artifacts / relative).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ArtifactError, match="checksum"):
        with TestClient(create_app(artifact_root=artifacts)):
            pytest.fail("Invalid snapshot started")


def test_missing_required_history_aborts_startup(artifacts):
    (artifacts / HISTORIES).unlink()
    with pytest.raises(FileNotFoundError):
        load_snapshot(artifacts)


@pytest.mark.parametrize("rows,message", [
    ([{"user_id": 1, "recommendations": [1, 1]}], "Duplicate or unknown"),
    ([{"user_id": 1, "recommendations": [99]}], "Duplicate or unknown"),
    ([{"user_id": 1, "recommendations": [1.5]}], "Invalid product_id"),
    ([{"user_id": 1, "recommendations": list(range(1, 12))}], "Invalid recommendation list"),
    ([{"user_id": 1, "recommendations": [1]}, {"user_id": 1, "recommendations": [2]}], "duplicate recommendation user"),
    ([{"user_id": 999, "recommendations": [1]}], "Unexpected"),
])
def test_malformed_but_checksummed_predictions_are_rejected(artifacts, rows, message):
    write_rows(artifacts / RANKED, rows)
    seal(artifacts)
    with pytest.raises(ArtifactError, match=message):
        load_snapshot(artifacts)


@pytest.mark.parametrize("url", ["/recommendations/0", "/recommendations/-1", "/recommendations/abc",
    "/recommendations/9223372036854775808", "/recommendations/1?k=0", "/recommendations/1?k=11",
    "/recommendations/1?k=bad", "/products/-1"])
def test_invalid_parameters_return_422(client, url):
    assert client.get(url).status_code == 422


def test_products_health_and_openapi(client):
    health = client.get("/health").json()
    assert health["known_users"] == 3
    assert health["catalog_products"] == 12
    assert health["users_with_ranked_lists"] == 2
    assert len(health["snapshot_id"]) == 64
    assert client.get("/products/2").json()["product_id"] == 2
    assert client.get("/products/99").status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/products/{product_id}", "/recommendations/{user_id}",
                          "/catalog", "/catalog/products", "/catalog/products/{product_id}/neighbors",
                          "/users/{user_id}/profile"}
    assert client.get("/docs").status_code == 200


def test_requests_before_startup_return_503():
    client = TestClient(create_app())
    assert client.get("/health").status_code == 503
    assert client.get("/recommendations/1").status_code == 503


def test_snapshot_is_immutable_and_does_not_read_training_or_test_data(artifacts, monkeypatch):
    allowed = {CONFIG, PRODUCTS, HISTORIES, RANKED, REPEAT, POPULARITY, f"{MODELS}/prediction_seal.json"}
    original = Path.read_bytes
    def checked(path):
        if path.is_relative_to(artifacts):
            assert path.relative_to(artifacts).as_posix() in allowed
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", checked)
    snapshot = load_snapshot(artifacts)
    with pytest.raises(TypeError):
        snapshot.ranked[1] = ()
    first = snapshot.recommend(1)
    first.recommendations.clear()
    assert snapshot.recommend(1).recommendations == list(range(12, 2, -1))


def test_short_catalog_is_explicitly_incomplete(artifacts):
    products = pq.read_table(artifacts / PRODUCTS).to_pylist()[:3]
    write_rows(artifacts / PRODUCTS, products)
    (artifacts / RANKED).unlink()
    (artifacts / REPEAT).unlink()
    write_json(artifacts / POPULARITY, {"recommendations": [1, 2, 3]})
    seal(artifacts)
    with TestClient(create_app(artifact_root=artifacts)) as client:
        response = client.get("/recommendations/999")
        assert ids(response) == [1, 2, 3]
        assert response.json()["complete"] is False


def test_product_validation_does_not_enumerate_the_entire_catalog():
    from collections.abc import Mapping
    class LargeCatalog(Mapping):
        def __len__(self):
            return 50000
        def __getitem__(self, key):
            if 1 <= key <= 50000:
                return key
            raise KeyError(key)
        def __iter__(self):
            raise AssertionError("Validation should look up recommended IDs, not copy the catalog")
    assert _items([1, 2, 3], LargeCatalog(), 10) == (1, 2, 3)
