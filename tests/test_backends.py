from dataclasses import replace
import json
from types import MappingProxyType

import fakeredis
from fastapi.testclient import TestClient
import psycopg
import pytest

from src.api.main import create_app
from src.serving.cache import RecommendationCache
from src.serving.service import RecommendationService
from src.serving.store import Product, RecommendationStore


@pytest.fixture
def snapshot():
    return RecommendationStore(
        model_version="test", snapshot_id="a" * 64, snapshot_created_at="2026-09-16T00:00:00Z",
        products=MappingProxyType({p: Product(p, f"Product {p}", 1, "Aisle", 1, "Department") for p in range(1, 13)}),
        histories=MappingProxyType({1: 5}), ranked=MappingProxyType({1: tuple(range(12, 2, -1))}),
        repeat=MappingProxyType({1: (2, 1)}), popularity=tuple(range(1, 13)), degraded=False,
    )


class Repository:
    def __init__(self, store):
        self.store, self.calls, self.failed, self.closed = store, 0, False, False
    def recommend(self, user, k):
        self.calls += 1
        if self.failed:
            raise psycopg.OperationalError("offline")
        return self.store.recommend(user, k)
    def product(self, product_id):
        if self.failed:
            raise psycopg.OperationalError("offline")
        return self.store.products.get(product_id)
    def ping(self):
        if self.failed:
            raise psycopg.OperationalError("offline")
    def close(self):
        self.closed = True


@pytest.fixture
def backends(snapshot):
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=True)
    cache = RecommendationCache(snapshot, client=client, cooldown_seconds=0)
    repository = Repository(snapshot)
    service = RecommendationService(snapshot, repository=repository, cache=cache)
    yield service, repository, cache, server
    service.close()


def test_miss_reads_database_then_hit_avoids_it(backends):
    service, repository, cache, _ = backends
    first = service.recommend(1, 10)
    assert first[1:] == ("miss", "postgresql")
    assert repository.calls == 1
    second = service.recommend(1, 10)
    assert second[0] == first[0]
    assert second[1:] == ("hit", "redis")
    assert repository.calls == 1
    assert 3599 <= cache.client.ttl(cache.key(1, 10)) <= 3600


def test_expired_cache_reads_database_again(backends):
    service, repository, cache, _ = backends
    service.recommend(1, 10)
    cache.client.expire(cache.key(1, 10), 0)
    assert service.recommend(1, 10)[2] == "postgresql"
    assert repository.calls == 2


def test_snapshot_and_k_are_separate_cache_namespaces(snapshot):
    client = fakeredis.FakeRedis(decode_responses=True)
    cache = RecommendationCache(snapshot, client=client)
    cache.write(1, 10, snapshot.recommend(1, 10))
    assert cache.read(1, 3) is None
    next_cache = RecommendationCache(replace(snapshot, snapshot_id="b" * 64), client=client)
    assert next_cache.read(1, 10) is None


def test_unknown_users_share_popularity_without_sharing_identity(backends):
    service, repository, cache, _ = backends
    first = service.recommend(999, 10)
    second = service.recommend(1000, 10)
    assert first[0] == second[0]
    assert second[1] == "hit"
    assert cache.key(999, 10) == cache.key(1000, 10)
    assert repository.calls == 1


@pytest.mark.parametrize("value", [b"\xff", "not-json", "[]", "{}", "x" * 5000,
    json.dumps({"key": "wrong", "route": {"recommendations": [1], "strategy": "ranker", "complete": True}})])
def test_invalid_cache_entries_are_discarded(backends, value):
    service, repository, cache, _ = backends
    cache.client.set(cache.key(1, 10), value)
    assert service.recommend(1, 10)[2] == "postgresql"
    assert cache.stats()["invalid_entries"] == 1
    assert service.recommend(1, 10)[2] == "redis"


def test_changed_rank_order_cannot_be_served_from_cache(backends):
    service, _, cache, _ = backends
    service.recommend(1, 10)
    value = json.loads(cache.client.get(cache.key(1, 10)))
    value["route"]["recommendations"].reverse()
    cache.client.set(cache.key(1, 10), json.dumps(value))
    assert service.recommend(1, 10)[2] == "postgresql"


def test_redis_outage_uses_database_and_recovers(backends):
    service, repository, cache, server = backends
    server.connected = False
    result = service.recommend(1, 10)
    assert result[1:] == ("bypass", "postgresql")
    assert service.health()["cache"] == "unavailable"
    server.connected = True
    assert service.health()["cache"] == "ok"
    service.recommend(1, 10)
    assert service.recommend(1, 10)[1] == "hit"


def test_database_outage_uses_identical_sealed_fallback(backends):
    service, repository, cache, server = backends
    repository.failed = True
    server.connected = False
    result = service.recommend(1, 10)
    assert result[0] == service.store.recommend(1, 10)
    assert result[2] == "snapshot"
    assert service.product(1) == service.store.products[1]
    assert service.health()["database"] == "unavailable"
    repository.failed = False
    service.database_retry_at = 0
    assert service.recommend(1, 10)[2] == "postgresql"


def test_dependency_health_and_headers_over_http(snapshot, backends):
    service, repository, cache, server = backends
    with TestClient(create_app(store=snapshot, service_factory=lambda *a, **kw: service)) as client:
        assert client.get("/recommendations/1").headers["X-RecoMart-Cache"] == "miss"
        assert client.get("/recommendations/1").headers["X-RecoMart-Cache"] == "hit"
        assert client.get("/recommendations/999").json()["user_id"] == 999
        assert client.get("/recommendations/1000").json()["user_id"] == 1000
        server.connected = False
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ready"] is True
        assert health.json()["status"] == "degraded"
        assert health.json()["cache"] == "unavailable"
        assert client.get("/recommendations/1").status_code == 200
    assert repository.closed


def test_cache_circuit_breaker_avoids_repeated_connection_attempts(snapshot):
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server)
    cache = RecommendationCache(snapshot, client=client, cooldown_seconds=60)
    server.connected = False
    for _ in range(10):
        assert cache.read(1, 10) is None
    assert cache.stats()["errors"] == 1


@pytest.mark.parametrize("ttl", [0, -1, 86401, "10", True])
def test_invalid_cache_ttl_is_rejected(snapshot, ttl):
    with pytest.raises(ValueError):
        RecommendationCache(snapshot, ttl_seconds=ttl)
