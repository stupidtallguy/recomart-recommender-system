"""Bounded Redis cache scoped to the frozen snapshot, user, and requested K."""
from dataclasses import asdict
import json
from threading import Lock
from time import monotonic

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from src.serving.routing import RecommendationRoute


class RecommendationCache:
    def __init__(self, store, url=None, *, client=None, ttl_seconds=3600, cooldown_seconds=5):
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 86400:
            raise ValueError("Cache TTL must be between 1 and 86400 seconds")
        self.store, self.ttl = store, ttl_seconds
        self.client = client if client is not None else (
            redis.Redis.from_url(url, decode_responses=True, socket_timeout=.2, socket_connect_timeout=.2,
                max_connections=16, retry=Retry(NoBackoff(), 0)) if url else None)
        self.cooldown = cooldown_seconds
        self.retry_at = 0.
        self.available = self.client is not None
        self.lock = Lock()
        self.counts = {name: 0 for name in ("hits", "misses", "writes", "errors", "invalid_entries")}

    def increment(self, name):
        with self.lock:
            self.counts[name] += 1

    def key(self, user, k):
        # All unknown users share the identical popularity route. Their IDs are
        # never embedded in a shared cached response or an unbounded key space.
        bucket = user if user in self.store.histories else 0
        return f"recomart:v1:{self.store.snapshot_id}:user:{bucket}:k:{k}"

    def _failure(self):
        with self.lock:
            self.available = False
            self.retry_at = monotonic() + self.cooldown
            self.counts["errors"] += 1

    def usable(self):
        return self.client is not None and monotonic() >= self.retry_at

    def read(self, user, k):
        if not self.usable():
            return None
        try:
            try:
                # Decoding can fail inside GET before JSON parsing if another
                # writer placed non-UTF-8 bytes in this cache entry.
                raw = self.client.get(self.key(user, k))
                self.available = True
                if raw is None:
                    self.increment("misses")
                    return None
                if len(raw) > 4096:
                    raise ValueError("Oversized cache entry")
                payload = json.loads(raw)
                route = RecommendationRoute(**payload["route"])
                expected = self.store.recommend(user, k)
                if (payload["key"] != self.key(user, k) or route != expected
                        or type(route.complete) is not bool
                        or any(type(p) is not int for p in route.recommendations)):
                    raise ValueError("Invalid cached route")
            except (ValueError, TypeError, KeyError, AttributeError):
                self.increment("invalid_entries")
                self.client.delete(self.key(user, k))
                return None
            self.increment("hits")
            return route
        except redis.RedisError:
            self._failure()
            return None

    def write(self, user, k, route):
        if not self.usable():
            return
        try:
            self.client.set(self.key(user, k), json.dumps({"key": self.key(user, k), "route": asdict(route)}), ex=self.ttl)
            self.available = True
            self.increment("writes")
        except redis.RedisError:
            self._failure()

    def health(self):
        if self.client is None:
            return "disabled"
        if self.usable():
            try:
                self.client.ping()
                self.available = True
            except redis.RedisError:
                self._failure()
        return "ok" if self.available else "unavailable"

    def stats(self):
        with self.lock:
            return dict(self.counts)

    def close(self):
        if self.client is not None:
            self.client.close()
