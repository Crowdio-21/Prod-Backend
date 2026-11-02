"""Virtual time for the device simulator.

Everything a simulated device does (transfers, compute, bandwidth changes) is measured in
*virtual* seconds. ``speed`` virtual seconds pass per real second, so a 120 s bandwidth
scenario runs in 24 real seconds at speed 5.

The Foreman is not simulated: it runs in real time, so it sees the compressed clock.
Devices therefore report link rates in bytes per *real* second (virtual rate x speed) so
that what the Foreman measures is consistent with the world it is scheduling. The Foreman's
own processing time (a few milliseconds) is not compressed, which is why very high speeds
distort results; 1 to 10 is reasonable.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable


class VirtualClock:
    def __init__(self, speed: float = 1.0, real: Callable[[], float] = time.monotonic) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = float(speed)
        self._real = real
        self._t0 = real()

    def reset(self) -> None:
        self._t0 = self._real()

    def now(self) -> float:
        """Virtual seconds since the clock was created or last reset."""
        return (self._real() - self._t0) * self.speed

    def to_real(self, virtual_s: float) -> float:
        return virtual_s / self.speed

    async def sleep(self, virtual_s: float) -> None:
        if virtual_s > 0:
            await asyncio.sleep(virtual_s / self.speed)

    async def sleep_until(self, virtual_t: float) -> None:
        await self.sleep(virtual_t - self.now())
