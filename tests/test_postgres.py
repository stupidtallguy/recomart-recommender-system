"""Live integration tests; create and drop only a dedicated UUID-named test DB."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from src.serving.database import PostgresRepository, publish_snapshot
from src.serving.store import Product, RecommendationStore


@pytest.fixture(scope="module")
def database_url():
    dsn = os.environ.get("RECOMART_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set RECOMART_TEST_DATABASE_URL for isolated live PostgreSQL tests")
    parameters = conninfo_to_dict(dsn)
    name = "recomart_test_" + uuid4().hex
    admin = dict(parameters, dbname="postgres")
    with psycopg.connect(make_conninfo(**admin), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield make_conninfo(**dict(parameters, dbname=name))
    finally:
        # Identifier comes only from the UUID generated in this fixture, never
        # from the supplied DSN or an existing user database.
        assert name.startswith("recomart_test_") and len(name) == 46
        with psycopg.connect(make_conninfo(**admin), autocommit=True) as connection:
            connection.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


@pytest.fixture
def snapshot():
    return RecommendationStore("test", uuid4().hex * 2, "2026-09-16T00:00:00Z",
        {p: Product(p, f"Product {p}", 1, "Aisle", 1, "Department") for p in range(1, 13)},
        {1: 5}, {1: tuple(range(12, 2, -1))}, {1: (2, 1)}, tuple(range(1, 13)), False)


def test_publication_retry_and_reads(database_url, snapshot):
    assert publish_snapshot(database_url, snapshot)["status"] == "imported"
    assert publish_snapshot(database_url, snapshot)["status"] == "already_present"
    repository = PostgresRepository(database_url, snapshot)
    try:
        for user, k in ((1, 10), (1, 3), (999, 10)):
            assert repository.recommend(user, k) == snapshot.recommend(user, k)
        assert repository.product(1) == snapshot.products[1]
        assert repository.product(99) is None
        repository.ping()
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            with repository.pool.connection() as connection:
                connection.execute("DELETE FROM recomart.snapshots WHERE snapshot_id=%s", (snapshot.snapshot_id,))
    finally:
        repository.close()


def test_failed_import_is_rolled_back(database_url, snapshot):
    duplicated = dict(snapshot.products)
    duplicated[2] = snapshot.products[1]
    broken = replace(snapshot, products=duplicated)
    with pytest.raises(psycopg.errors.UniqueViolation):
        publish_snapshot(database_url, broken)
    with psycopg.connect(database_url) as connection:
        assert connection.execute("SELECT 1 FROM recomart.snapshots WHERE snapshot_id=%s", (snapshot.snapshot_id,)).fetchone() is None
    assert publish_snapshot(database_url, snapshot)["status"] == "imported"


def test_two_publishers_do_not_create_partial_or_duplicate_snapshots(database_url, snapshot):
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: publish_snapshot(database_url, snapshot), range(2)))
    assert sorted(result["status"] for result in results) == ["already_present", "imported"]


def test_database_mutation_is_rejected_by_read_repository(database_url, snapshot):
    publish_snapshot(database_url, snapshot)
    repository = PostgresRepository(database_url, snapshot)
    try:
        with psycopg.connect(database_url) as connection:
            connection.execute("UPDATE recomart.user_recommendations SET ranked=%s WHERE snapshot_id=%s", ([1, 2], snapshot.snapshot_id))
            connection.execute("UPDATE recomart.products SET product_name='Changed' WHERE snapshot_id=%s AND product_id=1", (snapshot.snapshot_id,))
        with pytest.raises(ValueError, match="differs"):
            repository.recommend(1, 10)
        with pytest.raises(ValueError, match="differs"):
            repository.product(1)
    finally:
        repository.close()


def test_missing_version_refuses_database_startup(database_url, snapshot):
    # Ensure the schema exists while leaving this version unpublished.
    publish_snapshot(database_url, replace(snapshot, snapshot_id=uuid4().hex * 2))
    with pytest.raises(ValueError, match="Publish the matching"):
        PostgresRepository(database_url, snapshot)


def test_degraded_snapshot_cannot_be_published(database_url, snapshot):
    with pytest.raises(ValueError, match="complete"):
        publish_snapshot(database_url, replace(snapshot, degraded=True))
