"""A fleet of simulated devices with different CPUs and links."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

MBIT = 125_000.0  # bytes per second in one megabit per second


@dataclass(frozen=True)
class DeviceSpec:
    device_id: str
    kind: str  # "mobile" or "pc"
    tier: str
    cpu_factor: float  # 1.0 = the reference device; 2.0 takes twice as long to compute
    down_mbps: float  # nominal Foreman -> device
    up_mbps: float  # nominal device -> Foreman
    rtt_ms: float
    ram_mb: int
    cores: int
    battery: float  # percent; PCs report 100 and charging

    @property
    def down_bps(self) -> float:
        return self.down_mbps * MBIT

    @property
    def up_bps(self) -> float:
        return self.up_mbps * MBIT


# tier, share of the mobile fleet, cpu_factor, down Mbit/s, up Mbit/s, rtt ms, RAM MB, cores
MOBILE_TIERS = (
    ("flagship", 0.2, 1.0, 40.0, 15.0, 15.0, 8192, 8),
    ("mid", 0.4, 1.6, 15.0, 6.0, 25.0, 4096, 8),
    ("low", 0.4, 3.0, 5.0, 2.0, 45.0, 2048, 4),
)
PC_TIER = ("pc", 1.0, 0.6, 200.0, 100.0, 3.0, 16384, 8)


def _allocate(total: int, shares: List[float]) -> List[int]:
    """Whole numbers that sum to ``total``, proportional to ``shares`` (largest remainder)."""
    raw = [total * s / sum(shares) for s in shares]
    counts = [int(math.floor(r)) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: total - sum(counts)]:
        counts[i] += 1
    return counts


def make_fleet(n: int = 15, seed: int = 0, pc: int = 0, spread: float = 0.15) -> List[DeviceSpec]:
    """``n`` devices: ``pc`` of them PCs, the rest phones split across flagship, mid and low tiers.

    Each device's speeds are scaled by a seeded log-normal factor (``spread``), so no two are
    identical. Ids are ``sim-phone-01`` ... and ``sim-pc-01`` ...
    """
    if n < 1 or not 0 <= pc <= n:
        raise ValueError("need at least one device and 0 <= pc <= n")
    rng = random.Random(f"fleet:{seed}")
    fleet: List[DeviceSpec] = []

    def add(kind: str, tier, index: int) -> None:
        name, _, cpu, down, up, rtt, ram, cores = tier
        jitter = lambda: math.exp(rng.gauss(0.0, spread))  # noqa: E731
        fleet.append(DeviceSpec(
            device_id=f"sim-{kind}-{index:02d}", kind=kind, tier=name, cpu_factor=round(cpu * jitter(), 3),
            down_mbps=round(down * jitter(), 2), up_mbps=round(up * jitter(), 2), rtt_ms=round(rtt * jitter(), 1),
            ram_mb=ram, cores=cores, battery=100.0 if kind == "pc" else round(rng.uniform(40, 100), 1)))

    for i in range(pc):
        add("pc", PC_TIER, i + 1)
    index = 0
    for tier, count in zip(MOBILE_TIERS, _allocate(n - pc, [t[1] for t in MOBILE_TIERS])):
        for _ in range(count):
            index += 1
            add("phone", tier, index)
    return fleet
