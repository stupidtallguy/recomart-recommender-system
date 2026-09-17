"""Cache-aside serving, with the sealed in-memory snapshot as outage fallback."""
import logging
from time import monotonic

import psycopg
from psycopg_pool import PoolTimeout, TooManyRequests

from src.serving.cache import RecommendationCache
from src.serving.database import PostgresRepository

logger = logging.getLogger(__name__)
DATABASE_ERRORS = (psycopg.Error, PoolTimeout, TooManyRequests, ValueError)


class RecommendationService:
    def __init__(self, store, database_url=None, redis_url=None, ttl_seconds=3600, *, repository=None, cache=None):
        self.store = store
        self.repository = repository
        if self.repository is None and database_url:
            self.repository = PostgresRepository(database_url, store)
        try:
            self.cache = cache if cache is not None else RecommendationCache(store, redis_url, ttl_seconds=ttl_seconds)
        except BaseException:
            if self.repository is not None:
                self.repository.close()
            raise
        self.database_available = self.repository is not None
        self.database_retry_at = 0.

    def database_failed(self):
        # Do not log connection URLs or credentials from driver exceptions.
        logger.warning("PostgreSQL unavailable or inconsistent; serving the sealed local snapshot")
        self.database_available = False
        self.database_retry_at = monotonic() + 5

    def recommend(self, user, k):
        cached = self.cache.read(user, k)
        if cached is not None:
            return cached, "hit", "redis"
        source, route = "snapshot", None
        if self.repository is not None and monotonic() >= self.database_retry_at:
            try:
                route = self.repository.recommend(user, k)
                self.database_available = True
                source = "postgresql"
            except DATABASE_ERRORS:
                self.database_failed()
        if route is None:
            route = self.store.recommend(user, k)
        self.cache.write(user, k, route)
        cache_status = "miss" if self.cache.client is not None and self.cache.available else "bypass"
        return route, cache_status, source

    def product(self, product_id):
        if self.repository is not None and monotonic() >= self.database_retry_at:
            try:
                result = self.repository.product(product_id)
                self.database_available = True
                return result
            except DATABASE_ERRORS:
                self.database_failed()
        return self.store.products.get(product_id)

    def health(self):
        database = "disabled"
        if self.repository is not None:
            if monotonic() >= self.database_retry_at:
                try:
                    self.repository.ping()
                    self.database_available = True
                except DATABASE_ERRORS:
                    self.database_failed()
            database = "ok" if self.database_available else "unavailable"
        return {"database": database, "cache": self.cache.health(), "cache_counters": self.cache.stats()}

    def close(self):
        self.cache.close()
        if self.repository is not None:
            self.repository.close()
