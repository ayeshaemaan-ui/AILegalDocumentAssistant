"""Rate limiting (Tier 1 M2: Rate Limiting).

Two layers, because they guard different things.

`TokenBucket` limits *request rate* per client. A bucket refills continuously at
`rate` tokens/second up to `burst`, so a user can fire a short burst of
questions naturally but cannot sustain a flood. Chosen over a fixed window
because fixed windows allow a double-burst across the boundary - 30 requests at
11:59:59 and 30 more at 12:00:00.

`CostCircuitBreaker` limits *spend*. Rate limiting alone does not bound cost:
30 requests/minute against a 1M-token context is a large bill. The breaker
reads the day's actual recorded spend and trips before the ceiling is crossed.

Both are in-process. For a single container that is correct and has no network
cost; behind multiple replicas each would need to move to Redis, which is why
the storage is isolated behind these two small classes.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float, limit: int, window: int) -> None:
        self.retry_after = retry_after
        self.limit = limit
        self.window = window
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window}s. "
            f"Retry in {retry_after:.1f}s."
        )


@dataclass(slots=True)
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucket:
    def __init__(self, rate_per_second: float, burst: int) -> None:
        self.rate = rate_per_second
        self.burst = burst
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def consume(self, key: str, tokens: float = 1.0) -> tuple[bool, float]:
        """Take `tokens` from `key`'s bucket. Returns (allowed, retry_after)."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.burst), last_refill=now)
                self._buckets[key] = bucket

            elapsed = now - bucket.last_refill
            bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate)
            bucket.last_refill = now

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True, 0.0
            deficit = tokens - bucket.tokens
            return False, deficit / self.rate if self.rate > 0 else 60.0

    def remaining(self, key: str) -> int:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return self.burst
            elapsed = time.monotonic() - bucket.last_refill
            return int(min(self.burst, bucket.tokens + elapsed * self.rate))

    def sweep(self, max_idle_seconds: float = 3600) -> int:
        """Drop buckets that have been full and idle - stops unbounded growth."""
        now = time.monotonic()
        with self._lock:
            stale = [
                key for key, b in self._buckets.items()
                if now - b.last_refill > max_idle_seconds
            ]
            for key in stale:
                del self._buckets[key]
            return len(stale)


class CostCircuitBreaker:
    """Hard daily spend ceiling, checked before each model call."""

    def __init__(self, ceiling_usd: float) -> None:
        self.ceiling_usd = ceiling_usd
        self._lock = threading.Lock()

    def check(self, spent_today_usd: float) -> tuple[bool, str | None]:
        if self.ceiling_usd <= 0:
            return True, None
        with self._lock:
            if spent_today_usd >= self.ceiling_usd:
                return False, (
                    f"Daily API budget of ${self.ceiling_usd:.2f} has been reached "
                    f"(${spent_today_usd:.2f} spent). The limit resets at midnight UTC, "
                    "or an administrator can raise DAILY_COST_CEILING_USD."
                )
            if spent_today_usd >= self.ceiling_usd * 0.9:
                logger.warning(
                    "Cost ceiling 90%% consumed: $%.2f of $%.2f",
                    spent_today_usd, self.ceiling_usd,
                )
            return True, None


_limiter: TokenBucket | None = None
_breaker: CostCircuitBreaker | None = None


def get_rate_limiter(settings: Settings | None = None) -> TokenBucket:
    global _limiter
    if _limiter is None:
        settings = settings or get_settings()
        rate = settings.rate_limit_requests / max(settings.rate_limit_window_seconds, 1)
        _limiter = TokenBucket(rate_per_second=rate, burst=settings.rate_limit_burst)
    return _limiter


def get_cost_breaker(settings: Settings | None = None) -> CostCircuitBreaker:
    global _breaker
    if _breaker is None:
        settings = settings or get_settings()
        _breaker = CostCircuitBreaker(settings.daily_cost_ceiling_usd)
    return _breaker
