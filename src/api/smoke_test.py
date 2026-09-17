"""Compare a running local API with its sealed snapshot; no test labels needed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import urlopen

from src.serving.store import ROOT, load_snapshot


def run(base_url: str, root: Path):
    def get(path, expected=200):
        try:
            with urlopen(base_url.rstrip("/") + path, timeout=10) as response:
                status, body = response.status, json.load(response)
        except HTTPError as error:
            status, body = error.code, json.load(error)
        if status != expected:
            raise AssertionError(f"{path}: expected {expected}, received {status}")
        return body

    start = perf_counter()
    store = load_snapshot(root)
    load_seconds = perf_counter() - start
    # Full-population parity check against the saved prediction order.
    for user, saved in store.ranked.items():
        if store.recommend(user).recommendations != list(saved):
            raise AssertionError(f"Routing changed the full ranking for user {user}")
    health = get("/health")
    if health["snapshot_id"] != store.snapshot_id or not health["ready"]:
        raise AssertionError("The server has a different or unready snapshot")
    users = sorted(store.histories)
    samples = users[::max(1, len(users) // 100)][:100]
    latencies = []
    for index, user in enumerate(samples):
        k = index % 10 + 1
        start = perf_counter()
        response = get(f"/recommendations/{user}?k={k}")
        latencies.append((perf_counter() - start) * 1000)
        actual = [item["product_id"] for item in response["recommendations"]]
        if actual != store.recommend(user, k).recommendations or response["snapshot_id"] != store.snapshot_id:
            raise AssertionError(f"HTTP recommendation mismatch for user {user}")
    unknown = max(users) + 1
    response = get(f"/recommendations/{unknown}")
    if response["strategy"] != "global_popularity" or [p["product_id"] for p in response["recommendations"]] != list(store.popularity[:10]):
        raise AssertionError("Unknown-user fallback mismatch")
    product = store.popularity[0]
    if get(f"/products/{product}")["product_name"] != store.products[product].product_name:
        raise AssertionError("Product metadata mismatch")
    get(f"/products/{max(store.products) + 1}", expected=404)
    get("/recommendations/0", expected=422)
    get("/recommendations/1?k=11", expected=422)
    get("/openapi.json")
    return {
        "status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url, "snapshot_id": store.snapshot_id,
        "known_users": len(store.histories), "catalog_products": len(store.products),
        "all_user_ranking_parity_checks": len(store.ranked), "sampled_http_recommendation_checks": len(samples),
        "unknown_user_fallback": "passed", "product_lookup": "passed",
        "invalid_input_and_missing_product": "passed", "openapi": "passed",
        "snapshot_load_seconds": load_seconds,
        "sequential_loopback_latency_ms": {"median": statistics.median(latencies),
            "p95": sorted(latencies)[int(.95 * (len(latencies) - 1))], "requests": len(latencies)},
        "latency_scope": "Sequential local smoke check; not a concurrency or production capacity benchmark",
        "training_modules_imported": [name for name in ("xgboost", "pyspark", "torch") if name in sys.modules],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/api_smoke.json")
    args = parser.parse_args()
    result = run(args.base_url, args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
