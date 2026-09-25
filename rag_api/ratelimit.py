"""In-memory fixed-window rate limiter.

Enough for a single Render instance. State resets when the instance
restarts, which is acceptable for protecting an LLM quota.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class FixedWindowLimiter:
    def __init__(
        self,
        limit: int,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self._limit = limit
        self._window = window_s
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._hits)

    def allow(self, key: str) -> bool:
        with self._lock:
            now = self._clock()
            start, count = self._hits.get(key, (now, 0))
            if now - start >= self._window:
                start, count = now, 0
            if count >= self._limit:
                return False
            self._hits[key] = (start, count + 1)
            if len(self._hits) > self._max_keys:
                self._prune(now)
            return True

    def _prune(self, now: float) -> None:
        live = {k: v for k, v in self._hits.items() if now - v[0] < self._window}
        if len(live) > self._max_keys:
            sorted_items = sorted(live.items(), key=lambda kv: kv[1][0])
            newest = sorted_items[-self._max_keys // 2 :]
            live = dict(newest)
        self._hits = live
