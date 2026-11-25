"""A physical network model for scheduler studies: RSSI and throughput come from *separate* mechanisms.

Per device and per time step (``dt`` seconds):

* **Position** follows random-waypoint mobility at ``speed_mps`` (0 = static; walking about 1.4 m/s,
  vehicular 15-30 m/s) inside a disc of radius ``area_m`` around the access point.
* **RSSI** = log-distance path loss + shadow fading (AR(1) in *distance travelled*, so fast movers
  decorrelate quickly) + short-term fading noise. A handoff outage drives RSSI to the floor.
* **Link rate** = capped Shannon-style curve of the SNR (noise floor, implementation gap, MAC
  efficiency), so a stronger signal gives a higher PHY rate up to ``max_link_mbps``.
* **Achievable throughput** = min(link rate x (1 - others' airtime), per-device backhaul cap).
  The cap (server, ISP or phone limit; your real phones reach 15-17 Mbit/s on a 670 Mbit/s link)
  and the airtime taken by other devices are *invisible to RSSI*. So is an interference burst
  (it lowers SNR without lowering RSSI) and a throttled "sudden drop". That is what makes RSSI an
  informative but imperfect predictor, as on real hardware.

Everything is generated from one seed, so every scheduler can be run on exactly the same world
(paired runs). :class:`ReplayWorld` loads a recorded trace instead.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

MBIT = 1e6 / 8  # bytes/s per Mbit/s
OUTAGE_RATE_MBPS = 0.01
DROP_RATE_MBPS = 1.2  # about 150 kB/s, the throttle used in the phone experiments


@dataclass(frozen=True)
class NetParams:
    devices: int = 12
    speed_mps: float = 1.4
    area_m: float = 40.0  # radius of the disc devices move in
    ring: bool = False  # all devices start on a ring of radius ``area_m`` (identical-ish radio)
    path_loss_exp: float = 3.2
    ref_dbm: float = -40.0  # RSSI at 1 m
    shadow_sigma_db: float = 4.0
    shadow_corr_m: float = 8.0
    fast_sigma_db: float = 2.0
    noise_floor_dbm: float = -95.0
    snr_gap_db: float = 10.0
    bandwidth_mhz: float = 40.0
    mac_efficiency: float = 0.6
    max_link_mbps: float = 300.0
    up_ratio: float = 0.7  # uplink PHY rate as a share of downlink
    backhaul_mbps: float = 20.0  # median per-device cap (math.inf: radio-limited only)
    spread: float = 5.0  # device-to-device spread of the cap (1 = identical, 50 = 50x)
    backhaul_var: float = 0.15  # slow lognormal variation of the cap over time
    contention: float = 0.0  # mean share of airtime used by other devices
    outage_per_min: float = 0.0  # per device; handoff outages of 2-6 s
    burst_per_min: float = 0.0  # per device; interference bursts of 5-20 s (-12 dB SNR, RSSI unchanged)
    drop_frac: float = 0.0  # share of devices throttled to 150 kB/s once, for 30-70 s
    rtt_ms: float = 30.0  # median round trip
    cpu_spread: float = 0.3  # lognormal sigma of device compute speed
    horizon_s: float = 3000.0
    dt: float = 1.0
    rssi_noise_db: float = 1.5  # what the phone *reports* (Android reports whole dB)


@dataclass
class DeviceTrace:
    device_id: str
    dt: float
    rssi: np.ndarray  # dBm, true
    down: np.ndarray  # achievable bytes/s, Foreman -> device
    up: np.ndarray
    rtt_s: float
    cpu_factor: float  # compute seconds = work x cpu_factor
    link_mbps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cum_down: np.ndarray = field(default=None, repr=False)
    cum_up: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.cum_down = np.concatenate(([0.0], np.cumsum(self.down) * self.dt))
        self.cum_up = np.concatenate(([0.0], np.cumsum(self.up) * self.dt))

    def _index(self, t: float) -> int:
        return min(len(self.down) - 1, max(0, int(t / self.dt)))

    def rate(self, direction: str, t: float) -> float:
        return float((self.down if direction == "down" else self.up)[self._index(t)])

    def rssi_at(self, t: float) -> float:
        return float(self.rssi[self._index(t)])

    def mean_rate(self, direction: str) -> float:
        return float((self.down if direction == "down" else self.up).mean())

    def transfer_end(self, direction: str, t0: float, n_bytes: float) -> float:
        """Time at which ``n_bytes`` started at ``t0`` have been delivered (rate varies per step)."""
        if n_bytes <= 0:
            return t0
        cum, series = (self.cum_down, self.down) if direction == "down" else (self.cum_up, self.up)
        horizon = len(series) * self.dt
        if t0 >= horizon:
            return t0 + n_bytes / max(series[-1], 1.0)
        k = int(t0 / self.dt)
        start = cum[k] + series[k] * (t0 - k * self.dt)
        target = start + n_bytes
        if target > cum[-1]:
            return horizon + (target - cum[-1]) / max(series[-1], 1.0)
        j = int(np.searchsorted(cum, target, side="left"))  # cum[j-1] < target <= cum[j]
        j = max(1, j)
        step_rate = series[j - 1]
        return (j - 1) * self.dt + (target - cum[j - 1]) / max(step_rate, 1e-9)


@dataclass
class World:
    params: NetParams
    devices: List[DeviceTrace]
    seed: int = 0

    def device(self, device_id: str) -> DeviceTrace:
        return next(d for d in self.devices if d.device_id == device_id)

    @property
    def by_id(self) -> Dict[str, DeviceTrace]:
        return {d.device_id: d for d in self.devices}

    def reference_rate(self) -> float:
        """Median over devices of the time-mean downlink (bytes/s): the yardstick for transfer/compute ratios."""
        return float(np.median([d.mean_rate("down") for d in self.devices]))


# ------------------------------------------------------------------ generation
def _ar1(rng: np.random.Generator, n: int, rho: np.ndarray | float, sigma: float) -> np.ndarray:
    """Stationary AR(1) series with per-step correlation ``rho`` and marginal std ``sigma``."""
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (n,))
    noise = rng.standard_normal(n)
    out = np.empty(n)
    out[0] = sigma * noise[0]
    for i in range(1, n):
        out[i] = rho[i] * out[i - 1] + sigma * math.sqrt(max(0.0, 1 - rho[i] ** 2)) * noise[i]
    return out


def _walk(rng: np.random.Generator, p: NetParams, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the AP (m) and distance travelled per step (m) under random-waypoint mobility."""
    radius = max(p.area_m, 2.0)

    def point(first: bool = False) -> np.ndarray:
        r = radius if (p.ring and first) else radius * math.sqrt(rng.uniform(0.02, 1.0))
        a = rng.uniform(0, 2 * math.pi)
        return np.array([r * math.cos(a), r * math.sin(a)])

    pos = point(first=True)
    if p.speed_mps <= 0:
        return np.full(n, max(1.0, float(np.hypot(*pos)))), np.zeros(n)
    dist, travelled = np.empty(n), np.empty(n)
    target = point()
    step = p.speed_mps * p.dt
    for i in range(n):
        gap = target - pos
        norm = float(np.hypot(*gap))
        moved = 0.0
        remaining = step
        while remaining > 0:
            if norm <= remaining:  # reach the waypoint, pick another
                pos, moved, remaining = target.copy(), moved + norm, remaining - norm
                target = point()
                gap = target - pos
                norm = float(np.hypot(*gap))
            else:
                pos = pos + gap / norm * remaining
                moved += remaining
                remaining = 0.0
        dist[i], travelled[i] = max(1.0, float(np.hypot(*pos))), moved
    return dist, travelled


def link_rate_mbps(p: NetParams, snr_db: np.ndarray) -> np.ndarray:
    """Capped Shannon-style PHY goodput: B x efficiency x log2(1 + SNR / gap)."""
    snr = 10.0 ** ((snr_db - p.snr_gap_db) / 10.0)
    rate = p.bandwidth_mhz * p.mac_efficiency * np.log2(1.0 + snr)
    return np.clip(rate, 0.2, p.max_link_mbps)


def _events(rng: np.random.Generator, n: int, dt: float, per_min: float, lo_s: float, hi_s: float) -> np.ndarray:
    """Boolean mask of steps covered by Poisson-arriving events of random duration."""
    mask = np.zeros(n, dtype=bool)
    if per_min <= 0:
        return mask
    t = rng.exponential(60.0 / per_min)
    while t < n * dt:
        mask[int(t / dt): int((t + rng.uniform(lo_s, hi_s)) / dt) + 1] = True
        t += rng.exponential(60.0 / per_min)
    return mask


def generate_world(params: NetParams, seed: int) -> World:
    p = params
    n = int(p.horizon_s / p.dt)
    rng = np.random.default_rng([seed, 7919])
    # Airtime used by others: one shared AP-wide process plus a per-device part.
    shared = np.zeros(n)
    if p.contention > 0:
        shared = np.clip(p.contention + _ar1(rng, n, 0.97, 0.5 * p.contention), 0.0, 0.95)
    devices: List[DeviceTrace] = []
    drop_ids = set(rng.choice(p.devices, size=int(round(p.drop_frac * p.devices)), replace=False)) if p.drop_frac > 0 else set()
    for i in range(p.devices):
        drng = np.random.default_rng([seed, 104729, i])
        dist, moved = _walk(drng, p, n)
        eff_move = np.maximum(moved, 0.1 * p.dt)  # even a static phone sees a slowly changing environment
        rho = np.exp(-eff_move / p.shadow_corr_m)
        shadow = _ar1(drng, n, rho, p.shadow_sigma_db)
        fast = drng.normal(0.0, p.fast_sigma_db, n)
        rssi = p.ref_dbm - 10.0 * p.path_loss_exp * np.log10(dist) + shadow + fast
        outage = _events(drng, n, p.dt, p.outage_per_min, 2.0, 6.0)
        rssi = np.where(outage, -100.0, rssi)
        burst = _events(drng, n, p.dt, p.burst_per_min, 5.0, 20.0)
        snr = rssi - p.noise_floor_dbm - np.where(burst, 12.0, 0.0)
        link = link_rate_mbps(p, snr)
        airtime = np.clip(shared + (_ar1(drng, n, 0.9, 0.5 * p.contention) if p.contention > 0 else 0.0), 0.0, 0.95)
        scale = math.exp(drng.uniform(-0.5, 0.5) * math.log(max(p.spread, 1.0)))
        cap = (p.backhaul_mbps * scale * np.exp(_ar1(drng, n, 0.99, p.backhaul_var))) if math.isfinite(p.backhaul_mbps) \
            else np.full(n, np.inf)
        down = np.minimum(link * (1.0 - airtime), cap)
        up = np.minimum(link * p.up_ratio * (1.0 - airtime), cap * 0.5)
        if i in drop_ids:
            start = drng.uniform(20.0, 200.0)
            end = start + drng.uniform(30.0, 70.0)
            steps = np.arange(n) * p.dt
            throttled = (steps >= start) & (steps < end)
            down, up = np.where(throttled, np.minimum(down, DROP_RATE_MBPS), down), np.where(
                throttled, np.minimum(up, DROP_RATE_MBPS), up)
        down = np.where(outage, OUTAGE_RATE_MBPS, down)
        up = np.where(outage, OUTAGE_RATE_MBPS, up)
        devices.append(DeviceTrace(
            device_id=f"d{i:03d}", dt=p.dt, rssi=rssi, down=np.maximum(down, OUTAGE_RATE_MBPS) * MBIT,
            up=np.maximum(up, OUTAGE_RATE_MBPS) * MBIT, link_mbps=link,
            rtt_s=p.rtt_ms / 1000.0 * math.exp(drng.normal(0.0, 0.4)),
            cpu_factor=math.exp(drng.normal(0.0, p.cpu_spread))))
    return World(params=p, devices=devices, seed=seed)


# ------------------------------------------------------------------ replay
class ReplayWorld:
    """Build a :class:`World` from a recorded trace (CSV: device, t, rssi_dbm, down_mbps, up_mbps).

    Columns ``up_mbps`` and ``rssi_dbm`` are optional (uplink defaults to half the downlink; RSSI to
    -60 dBm). Samples are resampled to ``dt`` by holding the last value. Public 4G/5G or Wi-Fi
    throughput datasets can be converted to this shape.
    """

    @staticmethod
    def load(path: Path, *, dt: float = 1.0, horizon_s: Optional[float] = None, rtt_ms: float = 30.0,
             cpu_spread: float = 0.3, seed: int = 0) -> World:
        rows: Dict[str, List[tuple]] = {}
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                down = float(row["down_mbps"])
                up = float(row["up_mbps"]) if row.get("up_mbps") else down / 2.0
                rssi = float(row["rssi_dbm"]) if row.get("rssi_dbm") else -60.0
                rows.setdefault(row["device"], []).append((float(row["t"]), rssi, down, up))
        if not rows:
            raise ValueError("empty trace")
        horizon = horizon_s or max(r[-1][0] for r in map(sorted, rows.values())) + dt
        n = int(horizon / dt)
        rng = np.random.default_rng([seed, 31337])
        devices: List[DeviceTrace] = []
        for name, samples in sorted(rows.items()):
            samples.sort()
            times = np.array([s[0] for s in samples])
            idx = np.clip(np.searchsorted(times, np.arange(n) * dt, side="right") - 1, 0, len(samples) - 1)
            col = lambda k: np.array([s[k] for s in samples])[idx]  # noqa: E731
            devices.append(DeviceTrace(
                device_id=name, dt=dt, rssi=col(1), down=np.maximum(col(2), OUTAGE_RATE_MBPS) * MBIT,
                up=np.maximum(col(3), OUTAGE_RATE_MBPS) * MBIT, rtt_s=rtt_ms / 1000.0,
                cpu_factor=math.exp(rng.normal(0.0, cpu_spread))))
        return World(params=NetParams(devices=len(devices), horizon_s=horizon, dt=dt), devices=devices, seed=seed)


def correlation_rssi_throughput(world: World, stride: int = 5) -> float:
    """Pearson correlation of log10 throughput against RSSI over all devices (how informative RSSI is)."""
    xs = np.concatenate([d.rssi[::stride] for d in world.devices])
    ys = np.log10(np.concatenate([d.down[::stride] for d in world.devices]))
    return float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 and ys.std() > 0 else 0.0
