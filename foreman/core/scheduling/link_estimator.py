"""Per-worker link and compute estimates for the bandwidth-aware scheduler.

Design notes (see docs/bandwidth_aware_scheduling.md for the literature):

* Transfer time is linear in bytes, so the estimator averages **seconds per
  byte**, not bytes per second. An EWMA of the reciprocal is a harmonic-style
  mean: one lucky fast sample cannot make a slow link look fast (harmonic means
  are the usual robust throughput predictor in adaptive-streaming work).
* The smoothing factor defaults to 0.3, the value Cotter et al. use for their
  EWMA bandwidth estimate in mobile-edge offloading.
* Estimates carry their age and sample count. A worker with no samples is
  imputed with the fleet median (so it looks average and gets tried, which is
  how it gets measured). A stale estimate drifts back towards the fleet median
  rather than being trusted forever.
* Compute time is learned as ``turnaround - predicted transfer`` per
  (worker, task key), so transfer is not counted twice.

Nothing here does I/O; the clock is injected.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple
from dataclasses import dataclass

MIN_TRANSFER_BYTES = 10_000  # smaller transfers are dominated by RTT, not bandwidth
MIN_RATE_BPS = 1_000.0
MAX_RATE_BPS = 10_000_000_000.0
MIN_SECONDS = 1e-4
MIN_COMPUTE_S = 0.01

DEFAULT_PRIOR_BPS = 1_250_000.0  # 10 Mbit/s
DEFAULT_PRIOR_RTT_S = 0.05


class _Ewma:
    __slots__ = ("alpha", "value", "n", "updated_at")

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value = 0.0
        self.n = 0
        self.updated_at = -math.inf

    def update(self, x: float, now: float) -> None:
        self.value = x if self.n == 0 else self.alpha * x + (1.0 - self.alpha) * self.value
        self.n += 1
        self.updated_at = now


@dataclass(frozen=True)
class LinkEstimate:
    """Rates are bytes/s. ``down`` is Foreman -> worker, ``up`` is worker -> Foreman."""

    down_bps: float
    up_bps: float
    rtt_s: float
    n_down: int = 0
    n_up: int = 0
    age_s: float = math.inf
    imputed: bool = True


class _Link:
    __slots__ = ("down", "up", "rtt")

    def __init__(self, alpha: float) -> None:
        self.down = _Ewma(alpha)  # seconds per byte
        self.up = _Ewma(alpha)
        self.rtt = _Ewma(alpha)  # seconds


class LinkEstimator:
    def __init__(
        self,
        *,
        alpha: float = 0.3,
        max_age_s: float = 300.0,
        prior_bps: float = DEFAULT_PRIOR_BPS,
        prior_rtt_s: float = DEFAULT_PRIOR_RTT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.max_age_s = max_age_s
        self.prior_spb = 1.0 / prior_bps
        self.prior_rtt_s = prior_rtt_s
        self.clock = clock
        self._links: Dict[str, _Link] = {}
        self._compute: Dict[Tuple[str, str], _Ewma] = {}
        self._compute_any: Dict[str, _Ewma] = {}

    # ------------------------------------------------------------ observations
    def _now(self, now: Optional[float]) -> float:
        return self.clock() if now is None else now

    def _link(self, worker_id: str) -> _Link:
        link = self._links.get(worker_id)
        if link is None:
            link = self._links[worker_id] = _Link(self.alpha)
        return link

    def observe_transfer(self, worker_id: str, direction: str, n_bytes: int, seconds: float,
                         now: Optional[float] = None) -> bool:
        """A timed transfer of ``n_bytes``. Returns False if it was too small or implausible."""
        if n_bytes < MIN_TRANSFER_BYTES or seconds < MIN_SECONDS:
            return False
        return self.observe_rate(worker_id, direction, n_bytes / seconds, now)

    def observe_rate(self, worker_id: str, direction: str, bytes_per_s: float,
                     now: Optional[float] = None) -> bool:
        if direction not in ("down", "up") or not (MIN_RATE_BPS <= bytes_per_s <= MAX_RATE_BPS):
            return False
        link = self._link(worker_id)
        (link.down if direction == "down" else link.up).update(1.0 / bytes_per_s, self._now(now))
        return True

    def observe_rtt(self, worker_id: str, rtt_s: float, now: Optional[float] = None) -> bool:
        if not (0.0 < rtt_s < 60.0):
            return False
        self._link(worker_id).rtt.update(rtt_s, self._now(now))
        return True

    def observe_report(self, worker_id: str, report: Mapping[str, float], now: Optional[float] = None) -> int:
        """Apply a worker's own link measurement ``{down_bps, up_bps, rtt_s}``; returns how many were used."""
        used = 0
        if report.get("down_bps"):
            used += self.observe_rate(worker_id, "down", float(report["down_bps"]), now)
        if report.get("up_bps"):
            used += self.observe_rate(worker_id, "up", float(report["up_bps"]), now)
        if report.get("rtt_s"):
            used += self.observe_rtt(worker_id, float(report["rtt_s"]), now)
        return used

    def observe_compute(self, worker_id: str, key: str, seconds: float, now: Optional[float] = None) -> None:
        seconds = max(MIN_COMPUTE_S, float(seconds))
        now = self._now(now)
        for table, k in ((self._compute, (worker_id, key)), (self._compute_any, worker_id)):
            ewma = table.get(k)
            if ewma is None:
                ewma = table[k] = _Ewma(self.alpha)
            ewma.update(seconds, now)

    def forget(self, worker_id: str) -> None:
        self._links.pop(worker_id, None)
        self._compute_any.pop(worker_id, None)
        for key in [k for k in self._compute if k[0] == worker_id]:
            del self._compute[key]

    # ------------------------------------------------------------ queries
    def known_workers(self) -> Iterable[str]:
        return tuple(self._links)

    def _fleet(self, attr: str, prior: float) -> float:
        values = [getattr(link, attr).value for link in self._links.values() if getattr(link, attr).n]
        return statistics.median(values) if values else prior

    def estimates(self, worker_ids: Iterable[str], now: Optional[float] = None) -> Dict[str, LinkEstimate]:
        """Estimates for many workers; the fleet medians are computed once."""
        now = self._now(now)
        fleet = {
            "down": self._fleet("down", self.prior_spb),
            "up": self._fleet("up", self.prior_spb),
            "rtt": self._fleet("rtt", self.prior_rtt_s),
        }
        return {w: self._estimate(w, now, fleet) for w in worker_ids}

    def estimate(self, worker_id: str, now: Optional[float] = None) -> LinkEstimate:
        return self.estimates([worker_id], now)[worker_id]

    def _blend(self, ewma: Optional[_Ewma], fallback: float, now: float) -> Tuple[float, float]:
        """(value, age). A stale value drifts towards ``fallback`` over one more ``max_age_s``."""
        if ewma is None or ewma.n == 0:
            return fallback, math.inf
        age = now - ewma.updated_at
        stale = min(1.0, max(0.0, (age - self.max_age_s) / self.max_age_s))
        return (1.0 - stale) * ewma.value + stale * fallback, age

    def _estimate(self, worker_id: str, now: float, fleet: Mapping[str, float]) -> LinkEstimate:
        link = self._links.get(worker_id)
        if link is None:
            return LinkEstimate(1.0 / fleet["down"], 1.0 / fleet["up"], fleet["rtt"])
        # A direction nobody measured copies the other direction of the same worker (assume symmetric),
        # and only then falls back to the fleet median.
        fb_down = link.up.value if link.up.n and not link.down.n else fleet["down"]
        fb_up = link.down.value if link.down.n and not link.up.n else fleet["up"]
        down, age_down = self._blend(link.down, fb_down, now)
        up, age_up = self._blend(link.up, fb_up, now)
        rtt, age_rtt = self._blend(link.rtt, fleet["rtt"], now)
        ages = [a for a in (age_down, age_up) if math.isfinite(a)]
        return LinkEstimate(
            down_bps=1.0 / down, up_bps=1.0 / up, rtt_s=rtt, n_down=link.down.n, n_up=link.up.n,
            age_s=min(ages) if ages else math.inf, imputed=not (link.down.n or link.up.n),
        )

    def compute_seconds(self, worker_id: str, key: str, default: float) -> float:
        """Expected compute time: this worker for this key, else the fleet for this key, else the
        worker's average over all keys, else the fleet average, else ``default``."""
        own = self._compute.get((worker_id, key))
        if own is not None and own.n:
            return own.value
        same_key = [e.value for (w, k), e in self._compute.items() if k == key and e.n]
        if same_key:
            return statistics.median(same_key)
        any_own = self._compute_any.get(worker_id)
        if any_own is not None and any_own.n:
            return any_own.value
        everyone = [e.value for e in self._compute_any.values() if e.n]
        return statistics.median(everyone) if everyone else default

    def snapshot(self, now: Optional[float] = None) -> Dict[str, Dict[str, object]]:
        now = self._now(now)
        out: Dict[str, Dict[str, object]] = {}
        for worker_id, est in self.estimates(self._links, now).items():
            out[worker_id] = {
                "down_mbps": round(est.down_bps * 8 / 1e6, 3),
                "up_mbps": round(est.up_bps * 8 / 1e6, 3),
                "rtt_ms": round(est.rtt_s * 1000, 1),
                "samples_down": est.n_down,
                "samples_up": est.n_up,
                "age_s": None if not math.isfinite(est.age_s) else round(est.age_s, 1),
                "imputed": est.imputed,
                "compute_s": round(self._compute_any[worker_id].value, 3) if worker_id in self._compute_any else None,
            }
        return out
