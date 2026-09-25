from __future__ import annotations

from rag_api.ratelimit import FixedWindowLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_limit_then_blocks():
    limiter = FixedWindowLimiter(3, clock=FakeClock())
    assert [limiter.allow("ip") for _ in range(4)] == [True, True, True, False]


def test_window_resets_after_window_seconds():
    clock = FakeClock()
    limiter = FixedWindowLimiter(1, window_s=60, clock=clock)
    assert limiter.allow("ip")
    assert not limiter.allow("ip")
    clock.now += 60
    assert limiter.allow("ip")


def test_keys_are_independent():
    limiter = FixedWindowLimiter(1, clock=FakeClock())
    assert limiter.allow("a")
    assert limiter.allow("b")
    assert not limiter.allow("a")


def test_memory_stays_bounded_under_many_keys():
    clock = FakeClock()
    limiter = FixedWindowLimiter(1, clock=clock, max_keys=100)
    for i in range(5000):
        clock.now += 0.001
        limiter.allow(f"ip{i}")
    assert len(limiter) <= 100


def test_blocked_key_stays_blocked_within_window():
    limiter = FixedWindowLimiter(2, clock=FakeClock())
    limiter.allow("ip")
    limiter.allow("ip")
    assert not limiter.allow("ip")
    assert not limiter.allow("ip")


def test_prune_trims_below_cap_so_it_does_not_run_every_request():
    clock = FakeClock()
    limiter = FixedWindowLimiter(1, clock=clock, max_keys=100)
    for i in range(101):
        clock.now += 0.001
        limiter.allow(f"ip{i}")
    assert len(limiter) <= 50
