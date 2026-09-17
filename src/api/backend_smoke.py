"""Verify PostgreSQL parity and real Redis miss/hit behavior through HTTP."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
import redis

from src.serving.cache import RecommendationCache
from src.serving.store import ROOT, load_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmarks/backend_smoke.json")
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    dsn, redis_url = os.environ.get("RECOMART_DATABASE_URL"), os.environ.get("RECOMART_REDIS_URL")
    if not dsn or not redis_url:
        parser.error("Configure RECOMART_DATABASE_URL and RECOMART_REDIS_URL")
    store = load_snapshot(Path(os.environ.get("RECOMART_ARTIFACT_ROOT", str(ROOT))))
    users = products = 0
    with psycopg.connect(dsn, row_factory=dict_row, options="-c default_transaction_read_only=on") as connection:
        with connection.cursor(name="verify_users") as cursor:
            cursor.execute("SELECT user_id,history_orders,ranked,repeat FROM recomart.user_recommendations WHERE snapshot_id=%s", (store.snapshot_id,))
            for row in cursor:
                user = row["user_id"]
                assert row["history_orders"] == store.histories[user]
                assert row["ranked"] == list(store.ranked[user])
                assert row["repeat"] == list(store.repeat[user])
                users += 1
        with connection.cursor(name="verify_products") as cursor:
            cursor.execute("SELECT product_id,product_name,aisle_id,aisle,department_id,department FROM recomart.products WHERE snapshot_id=%s", (store.snapshot_id,))
            for row in cursor:
                assert row == asdict(store.products[row["product_id"]])
                products += 1
    assert users == len(store.histories) and products == len(store.products)
    client = redis.Redis.from_url(redis_url, socket_timeout=1, socket_connect_timeout=1, decode_responses=True)
    cache = RecommendationCache(store, client=client)
    user = min(store.histories)
    # Invalidate only this snapshot/user/K entry, never flush the database.
    client.delete(cache.key(user, 10))

    def get(path):
        with urlopen(args.base_url.rstrip("/") + path, timeout=10) as response:
            return json.load(response), dict(response.headers)

    first, h1 = get(f"/recommendations/{user}")
    second, h2 = get(f"/recommendations/{user}")
    assert first == second
    # HTTP header names are case-insensitive; urllib preserves wire casing.
    h1, h2 = {k.lower(): v for k, v in h1.items()}, {k.lower(): v for k, v in h2.items()}
    assert (h1["x-recomart-cache"], h1["x-recomart-source"]) == ("miss", "postgresql")
    assert (h2["x-recomart-cache"], h2["x-recomart-source"]) == ("hit", "redis")
    assert [p["product_id"] for p in second["recommendations"]] == list(store.ranked[user])
    ttl = client.ttl(cache.key(user, 10))
    assert 0 < ttl <= 86400
    client.expire(cache.key(user, 10), 0)
    _, h3 = get(f"/recommendations/{user}")
    assert {k.lower(): v for k, v in h3.items()}["x-recomart-cache"] == "miss"
    unknown = max(store.histories) + 1
    body, _ = get(f"/recommendations/{unknown}")
    assert body["strategy"] == "global_popularity" and body["user_id"] == unknown
    health, _ = get("/health")
    assert health["database"] == health["cache"] == "ok"
    report = {"status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": store.snapshot_id, "postgres_users_verified": users, "postgres_products_verified": products,
        "http_cache_miss_source": "postgresql", "http_cache_hit_source": "redis",
        "cache_expiry": "passed", "cache_ttl_seconds": ttl, "unknown_user": "passed",
        "redis_version": client.info("server")["redis_version"], "health": health,
        "test_labels_read": False}
    client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
