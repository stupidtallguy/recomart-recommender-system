"""Versioned PostgreSQL publication and bounded read-only request queries."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.serving.routing import route_recommendations
from src.serving.store import Product, RecommendationStore

MIGRATION = Path(__file__).parent / "migrations/001_snapshots.sql"


def migrate(connection):
    script = MIGRATION.read_text(encoding="utf-8").encode("utf-8")
    checksum = hashlib.sha256(script).hexdigest()
    connection.execute("SELECT pg_advisory_xact_lock(72143001)")
    exists = connection.execute("SELECT to_regclass('recomart.schema_migrations')").fetchone()[0]
    if exists:
        rows = connection.execute("SELECT version, checksum FROM recomart.schema_migrations ORDER BY version").fetchall()
        if rows != [(1, checksum)]:
            raise ValueError("Database migration version/checksum mismatch")
        return
    connection.execute(script.decode("utf-8"))
    connection.execute("INSERT INTO recomart.schema_migrations(version, checksum) VALUES (1, %s)", (checksum,))


def publish_snapshot(dsn: str, store: RecommendationStore):
    """Atomically publish one full verified snapshot; retries never replace rows."""
    if store.degraded or len(store.ranked) != len(store.histories):
        raise ValueError("Only a complete sealed snapshot can be published")
    with psycopg.connect(dsn, connect_timeout=3) as connection:
        migrate(connection)
        # Serializes publications as well as first-time schema creation.
        existing = connection.execute(
            "SELECT known_users, catalog_products FROM recomart.snapshots WHERE snapshot_id=%s",
            (store.snapshot_id,),
        ).fetchone()
        if existing:
            counts = connection.execute(
                "SELECT (SELECT count(*) FROM recomart.user_recommendations WHERE snapshot_id=%s), "
                "(SELECT count(*) FROM recomart.products WHERE snapshot_id=%s)",
                (store.snapshot_id, store.snapshot_id),
            ).fetchone()
            if existing != (len(store.histories), len(store.products)) or counts != existing:
                raise ValueError("Existing snapshot is incomplete; refusing to replace it")
            return {"status": "already_present", "snapshot_id": store.snapshot_id, "users": counts[0], "products": counts[1]}
        connection.execute(
            "INSERT INTO recomart.snapshots(snapshot_id, model_version, snapshot_created_at, known_users, catalog_products, popularity) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (store.snapshot_id, store.model_version, store.snapshot_created_at,
             len(store.histories), len(store.products), list(store.popularity)),
        )
        with connection.cursor().copy(
            "COPY recomart.products(snapshot_id,product_id,product_name,aisle_id,aisle,department_id,department) FROM STDIN"
        ) as copy:
            for item in store.products.values():
                copy.write_row((store.snapshot_id, *asdict(item).values()))
        with connection.cursor().copy(
            "COPY recomart.user_recommendations(snapshot_id,user_id,history_orders,ranked,repeat) FROM STDIN"
        ) as copy:
            for user, history in store.histories.items():
                copy.write_row((store.snapshot_id, user, history, list(store.ranked[user]),
                                list(store.repeat[user]) if user in store.repeat else None))
    return {"status": "imported", "snapshot_id": store.snapshot_id,
            "users": len(store.histories), "products": len(store.products)}


class PostgresRepository:
    def __init__(self, dsn: str, store: RecommendationStore):
        self.store = store
        self.pool = ConnectionPool(dsn, min_size=1, max_size=5, max_waiting=32, timeout=.3,
            kwargs={"connect_timeout": 3, "autocommit": True, "row_factory": dict_row,
                    "options": "-c statement_timeout=1000 -c default_transaction_read_only=on"}, open=False)
        self.pool.open()
        try:
            self.pool.wait(timeout=5)
            with self.pool.connection() as connection:
                row = connection.execute("SELECT * FROM recomart.snapshots WHERE snapshot_id=%s", (store.snapshot_id,)).fetchone()
                if (not row or row["known_users"] != len(store.histories)
                        or row["catalog_products"] != len(store.products)
                        or row["model_version"] != store.model_version
                        or row["popularity"] != list(store.popularity)):
                    raise ValueError("Publish the matching sealed snapshot before starting database serving")
        except BaseException:
            self.pool.close()
            raise

    def recommend(self, user_id, k):
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT history_orders, ranked, repeat FROM recomart.user_recommendations "
                "WHERE snapshot_id=%s AND user_id=%s", (self.store.snapshot_id, user_id),
            ).fetchone()
        if row is None:
            if user_id in self.store.histories:
                raise ValueError("Known user missing from database snapshot")
            return route_recommendations(history_order_count=0, popularity=self.store.popularity, k=k)
        route = route_recommendations(history_order_count=row["history_orders"], ranked=row["ranked"],
            repeat=row["repeat"], popularity=self.store.popularity, k=k)
        # The sealed local snapshot is also the outage fallback and integrity
        # reference. Never cache a mutated database ranking under its old version.
        if route != self.store.recommend(user_id, k):
            raise ValueError("Database recommendation differs from the sealed snapshot")
        return route

    def product(self, product_id):
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT product_id,product_name,aisle_id,aisle,department_id,department "
                "FROM recomart.products WHERE snapshot_id=%s AND product_id=%s",
                (self.store.snapshot_id, product_id),
            ).fetchone()
        item = Product(**row) if row else None
        if item != self.store.products.get(product_id):
            raise ValueError("Database product differs from the sealed snapshot")
        return item

    def ping(self):
        with self.pool.connection() as connection:
            connection.execute("SELECT 1")

    def close(self):
        self.pool.close()
