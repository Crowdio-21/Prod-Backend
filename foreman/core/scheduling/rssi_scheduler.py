"""RSSI-aware scheduler: the bandwidth-aware scheduler, corrected by the signal strength phones report.

Why: phones report a measured bandwidth only after a timed, network-limited transfer, so between
transfers the scheduler's bandwidth estimate can be minutes old. RSSI (received signal strength,
dBm) arrives with every heartbeat. It is not throughput (load, interference and the access point
matter too), so this scheduler never *replaces* a measurement with it. It uses RSSI three ways:

1. **Prior.** A phone with no bandwidth measurement gets ``curve(rssi)`` instead of the fleet
   median, capped by the Wi-Fi link speed it reports.
2. **Early warning.** When a phone's RSSI has fallen by at least ``drop_db`` since its bandwidth
   was last measured, the estimate is scaled by ``curve(rssi_now) / curve(rssi_then)``. It is
   only ever scaled *down*: a better signal must show up in a measurement before it is believed.
3. **Weak signal.** Below ``weak_dbm`` a phone is near the edge of coverage, where transfers stall
   and connections drop; its cost is multiplied by ``weak_penalty``.

A radio report older than ``radio_max_age_s`` is ignored. With no RSSI at all the scheduler is
exactly the bandwidth-aware one.

The curve is log-linear (throughput falls by 10x every ``db_per_decade`` dB). Its default numbers
are **placeholders, not a calibration**; :func:`fit_rssi_curve` and :func:`calibrate_from_log` fit
it from the heartbeat log (``CROWDIO_EXPERIMENT_LOG``) of a real run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .bandwidth_scheduler import BandwidthAwareScheduler, TaskProfile
from .link_estimator import LinkEstimate

MBIT = 1e6 / 8  # bytes/s in one Mbit/s


@dataclass(frozen=True)
class RssiCurve:
    """Expected throughput (Mbit/s) at a signal strength: ``ref_mbps * 10 ** ((rssi - ref_dbm) / db_per_decade)``."""

    ref_dbm: float = -50.0
    ref_mbps: float = 60.0
    db_per_decade: float = 20.0
    floor_mbps: float = 0.5
    ceiling_mbps: float = 400.0
    samples: int = 0  # how many measurements it was fitted on (0 = default placeholder)

    def mbps(self, rssi_dbm: float) -> float:
        value = self.ref_mbps * 10.0 ** ((rssi_dbm - self.ref_dbm) / self.db_per_decade)
        return min(self.ceiling_mbps, max(self.floor_mbps, value))

    def bytes_per_s(self, rssi_dbm: float) -> float:
        return self.mbps(rssi_dbm) * MBIT

    def ratio(self, now_dbm: float, then_dbm: float) -> float:
        return self.mbps(now_dbm) / self.mbps(then_dbm)


def fit_rssi_curve(pairs: Iterable[Tuple[float, float]], *, min_samples: int = 5,
                   default: RssiCurve = RssiCurve()) -> RssiCurve:
    """Least-squares fit of log10(throughput Mbit/s) against RSSI (dBm).

    Returns ``default`` when there are too few samples, too little spread in RSSI to fit a slope,
    or the fitted slope says throughput rises as the signal weakens (noise, not physics).
    """
    points = [(float(r), math.log10(float(m))) for r, m in pairs if m and m > 0 and r is not None]
    if len(points) < min_samples:
        return default
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1.0:  # RSSI barely varied: no slope to learn, keep the default slope at the observed level
        return replace(default, ref_dbm=mx, ref_mbps=10.0 ** my, samples=len(points))
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx  # decades per dB
    if slope <= 0:
        return replace(default, ref_dbm=mx, ref_mbps=10.0 ** my, samples=len(points))
    return replace(default, ref_dbm=mx, ref_mbps=10.0 ** my, db_per_decade=1.0 / slope, samples=len(points))


def calibrate_from_log(path: Path, **kwargs) -> RssiCurve:
    """Fit the curve from a heartbeat log written with ``CROWDIO_EXPERIMENT_LOG`` (rows with both
    ``rssi_dbm`` and ``bw_down_bps``, the phone's own measured downlink in bits/s)."""
    pairs: List[Tuple[float, float]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rssi, bw = row.get("rssi_dbm"), row.get("bw_down_bps")
            if rssi is not None and bw:
                pairs.append((float(rssi), float(bw) / 1e6))
    return fit_rssi_curve(pairs, **kwargs)


@dataclass
class _Radio:
    rssi_dbm: float
    at: float
    link_mbps: Optional[float] = None
    rssi_at_measurement: Optional[float] = None  # RSSI when the bandwidth was last measured


class RssiAwareScheduler(BandwidthAwareScheduler):
    def __init__(
        self,
        *args,
        curve: Optional[RssiCurve] = None,
        drop_db: float = 4.0,
        weak_dbm: float = -82.0,
        weak_penalty: float = 1.5,
        radio_max_age_s: float = 60.0,
        rssi_alpha: float = 0.5,
        link_efficiency: float = 0.5,
        up_ratio: float = 0.4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < rssi_alpha <= 1.0:
            raise ValueError("rssi_alpha must be in (0, 1]")
        if weak_penalty < 1.0:
            raise ValueError("weak_penalty must be at least 1")
        self.curve = curve or RssiCurve()
        self.drop_db = drop_db
        self.weak_dbm = weak_dbm
        self.weak_penalty = weak_penalty
        self.radio_max_age_s = radio_max_age_s
        self.rssi_alpha = rssi_alpha
        self.link_efficiency = link_efficiency  # usable share of the negotiated Wi-Fi rate
        self.up_ratio = up_ratio  # uplink prior as a share of the downlink prior
        self._radio: Dict[str, _Radio] = {}

    # ------------------------------------------------------------ observations
    def observe_radio(self, worker_id: str, report: Mapping[str, float]) -> None:
        """A heartbeat's ``{rssi_dbm, link_mbps}``. RSSI is smoothed a little (radios are noisy)."""
        rssi = report.get("rssi_dbm")
        if rssi is None or not (-120.0 <= float(rssi) <= 0.0):
            return
        now = self.clock()
        radio = self._radio.get(worker_id)
        if radio is None or now - radio.at > self.radio_max_age_s:
            previous = radio.rssi_at_measurement if radio else None
            radio = self._radio[worker_id] = _Radio(float(rssi), now, rssi_at_measurement=previous)
        else:
            radio.rssi_dbm = self.rssi_alpha * float(rssi) + (1.0 - self.rssi_alpha) * radio.rssi_dbm
            radio.at = now
        link = report.get("link_mbps")
        if link and float(link) > 0:
            radio.link_mbps = float(link)

    def observe_link_report(self, worker_id: str, report: dict) -> None:
        super().observe_link_report(worker_id, report)
        radio = self._fresh(worker_id)
        if radio is not None and (report.get("down_bps") or report.get("up_bps")):
            radio.rssi_at_measurement = radio.rssi_dbm  # the baseline later drops are measured from

    def _fresh(self, worker_id: str) -> Optional[_Radio]:
        radio = self._radio.get(worker_id)
        if radio is None or self.clock() - radio.at > self.radio_max_age_s:
            return None
        return radio

    # ------------------------------------------------------------ cost model
    def adjust(self, worker_id: str, est: LinkEstimate) -> Tuple[LinkEstimate, float, str]:
        """(estimate corrected by RSSI, cost multiplier, reason)."""
        radio = self._fresh(worker_id)
        if radio is None:
            return est, 1.0, "no radio"
        reason = "measured"
        if est.imputed:
            down = self.curve.bytes_per_s(radio.rssi_dbm)
            if radio.link_mbps:
                down = min(down, radio.link_mbps * self.link_efficiency * MBIT)
            est = replace(est, down_bps=down, up_bps=down * self.up_ratio)
            reason = "rssi prior"
        elif radio.rssi_at_measurement is not None and radio.rssi_at_measurement - radio.rssi_dbm >= self.drop_db:
            ratio = min(1.0, self.curve.ratio(radio.rssi_dbm, radio.rssi_at_measurement))
            est = replace(est, down_bps=est.down_bps * ratio, up_bps=est.up_bps * ratio)
            reason = f"signal fell {radio.rssi_at_measurement - radio.rssi_dbm:.0f} dB"
        multiplier = self.weak_penalty if radio.rssi_dbm < self.weak_dbm else 1.0
        if multiplier > 1.0:
            reason += ", weak signal"
        return est, multiplier, reason

    def cost(self, profile: TaskProfile, worker_id: str, est: LinkEstimate,
             level_down: float = math.inf, level_up: float = math.inf) -> float:
        adjusted, multiplier, _ = self.adjust(worker_id, est)
        return super().cost(profile, worker_id, adjusted, level_down, level_up) * multiplier

    # ------------------------------------------------------------ feedback
    def _commit(self, profile: TaskProfile, worker_id: str, seconds: float, est: LinkEstimate) -> None:
        super()._commit(profile, worker_id, seconds, est)
        radio = self._fresh(worker_id)
        _, multiplier, reason = self.adjust(worker_id, est)
        self.decisions[-1].update({
            "rssi_dbm": None if radio is None else round(radio.rssi_dbm, 1),
            "rssi_reason": reason, "rssi_multiplier": multiplier,
        })

    def link_snapshot(self) -> dict:
        snapshot = super().link_snapshot()
        now = self.clock()
        snapshot["radio"] = {
            w: {"rssi_dbm": round(r.rssi_dbm, 1), "age_s": round(now - r.at, 1), "link_mbps": r.link_mbps,
                "rssi_at_measurement": None if r.rssi_at_measurement is None else round(r.rssi_at_measurement, 1)}
            for w, r in sorted(self._radio.items())
        }
        snapshot["rssi_curve"] = {"ref_dbm": self.curve.ref_dbm, "ref_mbps": round(self.curve.ref_mbps, 2),
                                  "db_per_decade": round(self.curve.db_per_decade, 2), "samples": self.curve.samples}
        return snapshot
