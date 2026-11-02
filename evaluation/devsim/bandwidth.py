"""Time-varying bandwidth models.

A model produces a *factor* per time step (1.0 = the device's nominal bandwidth). The factor
multiplies both directions. Models are seeded, so the same scenario gives the same bandwidth
over time for every scheduler being compared.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

MIN_FACTOR = 0.001  # an outage is a near-zero rate, never exactly zero (a transfer must be able to finish)


class BandwidthModel(Protocol):
    def generate(self, rng: random.Random, horizon_s: float, step_s: float) -> List[float]: ...


def _steps(horizon_s: float, step_s: float) -> int:
    return int(math.ceil(horizon_s / step_s)) + 1


@dataclass(frozen=True)
class Stable:
    """Constant nominal bandwidth."""

    def generate(self, rng, horizon_s, step_s):
        return [1.0] * _steps(horizon_s, step_s)


@dataclass(frozen=True)
class RandomWalk:
    """Mean-reverting random walk in log space: the bandwidth drifts up and down around nominal."""

    sigma: float = 0.2  # per-step volatility of log bandwidth
    reversion: float = 0.1  # pull back towards nominal per step
    lo: float = 0.1
    hi: float = 1.5

    def generate(self, rng, horizon_s, step_s):
        x, out = 0.0, []
        for _ in range(_steps(horizon_s, step_s)):
            x += -self.reversion * x + self.sigma * rng.gauss(0.0, 1.0)
            out.append(min(self.hi, max(self.lo, math.exp(x))))
        return out


@dataclass(frozen=True)
class Markov:
    """Two states, good and congested, like a Wi-Fi link that is sometimes shared with a busy neighbour."""

    good: float = 1.0
    bad: float = 0.15
    p_bad: float = 0.03  # chance per step of falling into the congested state
    p_good: float = 0.12  # chance per step of recovering
    jitter: float = 0.05

    def generate(self, rng, horizon_s, step_s):
        in_bad = rng.random() < self.p_bad / (self.p_bad + self.p_good)  # start from the stationary mix
        out = []
        for _ in range(_steps(horizon_s, step_s)):
            in_bad = (rng.random() >= self.p_good) if in_bad else (rng.random() < self.p_bad)
            level = self.bad if in_bad else self.good
            out.append(max(MIN_FACTOR, level * (1.0 + rng.uniform(-self.jitter, self.jitter))))
        return out


@dataclass(frozen=True)
class Step:
    """Drops (or raises) the bandwidth to ``factor`` from ``at_s``, back to nominal at ``until_s``."""

    at_s: float
    factor: float
    until_s: Optional[float] = None

    def generate(self, rng, horizon_s, step_s):
        out = []
        for i in range(_steps(horizon_s, step_s)):
            t = i * step_s
            out.append(max(MIN_FACTOR, self.factor) if t >= self.at_s and (self.until_s is None or t < self.until_s) else 1.0)
        return out


@dataclass(frozen=True)
class Outage:
    """The link is (almost) dead during each window."""

    windows: Tuple[Tuple[float, float], ...] = ()

    def generate(self, rng, horizon_s, step_s):
        out = []
        for i in range(_steps(horizon_s, step_s)):
            t = i * step_s
            out.append(MIN_FACTOR if any(a <= t < b for a, b in self.windows) else 1.0)
        return out


@dataclass(frozen=True)
class RandomOutages:
    """``count`` short outages at random times (a phone going through a tunnel)."""

    count: int = 2
    min_s: float = 2.0
    max_s: float = 6.0
    after_s: float = 10.0

    def generate(self, rng, horizon_s, step_s):
        windows = []
        for _ in range(self.count):
            start = rng.uniform(self.after_s, max(self.after_s + 1.0, horizon_s - self.max_s))
            windows.append((start, start + rng.uniform(self.min_s, self.max_s)))
        return Outage(tuple(windows)).generate(rng, horizon_s, step_s)


@dataclass(frozen=True)
class Compose:
    """The product of several models (for example a drift plus random outages)."""

    models: Tuple[BandwidthModel, ...]

    def generate(self, rng, horizon_s, step_s):
        series = [m.generate(rng, horizon_s, step_s) for m in self.models]
        return [max(MIN_FACTOR, math.prod(values)) for values in zip(*series)]


@dataclass(frozen=True)
class BandwidthTrace:
    """Bandwidth factor over virtual time; holds the last value after the end."""

    values: Sequence[float]
    step_s: float = 1.0

    def factor(self, t: float) -> float:
        index = int(max(0.0, t) / self.step_s)
        return self.values[min(index, len(self.values) - 1)]

    def mean(self, start_s: float = 0.0, end_s: Optional[float] = None) -> float:
        end = len(self.values) * self.step_s if end_s is None else end_s
        a, b = int(start_s / self.step_s), max(int(start_s / self.step_s) + 1, int(end / self.step_s))
        window = self.values[a:b] or [self.values[-1]]
        return sum(window) / len(window)


def build_trace(model: BandwidthModel, rng: random.Random, horizon_s: float = 3600.0, step_s: float = 1.0) -> BandwidthTrace:
    values = [max(MIN_FACTOR, v) for v in model.generate(rng, horizon_s, step_s)]
    return BandwidthTrace(tuple(values), step_s)
