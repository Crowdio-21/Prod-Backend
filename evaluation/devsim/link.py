"""Link engine: how long a transfer takes on a link whose bandwidth changes.

A transfer is a *flow* with bytes remaining. A background loop advances all flows in small
virtual-time steps. Each step, a flow gets:

* its device's current rate (nominal x the scenario's bandwidth factor), split fairly among
  that device's own concurrent flows in the same direction; and
* if the scenario has a shared medium, no more than its max-min fair share of that capacity.

Because the rate is re-read every step, a bandwidth drop in the middle of a transfer slows it
down from that moment, and an outage stalls it until the link recovers. Steps use the measured
elapsed time, so scheduling jitter does not change how many bytes move.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import List, Mapping, Optional

from foreman.core.scheduling.bandwidth_scheduler import fair_level

from .bandwidth import BandwidthTrace
from .clock import VirtualClock
from .fleet import MBIT, DeviceSpec

SMALL_MESSAGE_BYTES = 1500  # one packet: not worth modelling as a flow


@dataclass
class _Flow:
    device_id: str
    direction: str
    remaining: float
    started: float
    future: asyncio.Future


class LinkEngine:
    def __init__(
        self,
        clock: VirtualClock,
        specs: Mapping[str, DeviceSpec],
        traces: Mapping[str, BandwidthTrace],
        *,
        shared_capacity_mbps: Optional[float] = None,
        tick_s: float = 0.05,
        min_real_tick_s: float = 0.004,
    ) -> None:
        self.clock = clock
        self._specs = specs
        self._traces = traces
        self._capacity = None if shared_capacity_mbps is None else shared_capacity_mbps * MBIT
        self._tick_real = max(tick_s / clock.speed, min_real_tick_s)
        self._flows: List[_Flow] = []
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def rate_bps(self, device_id: str, direction: str, t: Optional[float] = None) -> float:
        """The device's rate in bytes per virtual second at virtual time ``t`` (now by default)."""
        spec = self._specs[device_id]
        base = spec.down_bps if direction == "down" else spec.up_bps
        return base * self._traces[device_id].factor(self.clock.now() if t is None else t)

    async def transfer(self, device_id: str, direction: str, n_bytes: int) -> float:
        """Move ``n_bytes``; returns the virtual seconds it took."""
        if direction not in ("down", "up"):
            raise ValueError("direction must be 'down' or 'up'")
        if n_bytes <= SMALL_MESSAGE_BYTES:
            return 0.0
        flow = _Flow(device_id, direction, float(n_bytes), self.clock.now(),
                     asyncio.get_running_loop().create_future())
        self._flows.append(flow)
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())
        self._wake.set()
        try:
            return await flow.future
        finally:
            if flow in self._flows:
                self._flows.remove(flow)

    def abort(self, device_id: str) -> int:
        """Fail every transfer of a device (it dropped off the network); returns how many."""
        aborted = [f for f in self._flows if f.device_id == device_id]
        for flow in aborted:
            self._flows.remove(flow)
            if not flow.future.done():
                flow.future.set_exception(ConnectionError(f"{device_id} lost its connection"))
        return len(aborted)

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        for flow in list(self._flows):
            if not flow.future.done():
                flow.future.cancel()
        self._flows.clear()

    async def _run(self) -> None:
        last = self.clock.now()
        while True:
            if not self._flows:
                self._wake.clear()
                await self._wake.wait()
                last = self.clock.now()
                continue
            await asyncio.sleep(self._tick_real)
            now = self.clock.now()
            self._advance(last, now)
            last = now

    def _advance(self, start: float, now: float) -> None:
        dt = max(0.0, now - start)
        middle = start + dt / 2
        for direction in ("down", "up"):
            flows = [f for f in self._flows if f.direction == direction]
            if not flows:
                continue
            per_device = Counter(f.device_id for f in flows)
            caps = [self.rate_bps(f.device_id, direction, middle) / per_device[f.device_id] for f in flows]
            level = fair_level(caps, self._capacity) if self._capacity else float("inf")
            for flow, cap in zip(flows, caps):
                flow.remaining -= min(cap, level) * dt
        for flow in [f for f in self._flows if f.remaining <= 0]:
            self._flows.remove(flow)
            if not flow.future.done():
                flow.future.set_result(now - flow.started)
