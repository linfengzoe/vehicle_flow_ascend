from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class FpsMeter:
    window_size: int = 30
    time_fn: Callable[[], float] = perf_counter
    _timestamps: deque[float] = field(default_factory=deque)

    def tick(self) -> float:
        now = self.time_fn()
        self._timestamps.append(now)
        while len(self._timestamps) > self.window_size:
            self._timestamps.popleft()
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
