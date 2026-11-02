"""Measure this worker's link to the Foreman.

The worker times a small request (round trip), a download and an upload against the
Foreman's ``/api/probe`` endpoints and reports ``{down_bps, up_bps, rtt_s}`` (bytes/s and
seconds) in its heartbeat and PING replies. The Foreman's bandwidth-aware scheduler uses
these as its bandwidth estimate.

Probing costs bandwidth, so it is rate limited (``min_interval_s``) and the caller only
triggers it while the worker is idle (probing during a task would slow that task down,
the overhead trade-off Cotter et al. observe for active probing).
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

FOREMAN_HTTP_PORT = 8000
MIN_ELAPSED_S = 1e-4
RTT_SAMPLES = 3  # the first request pays one-off costs (connection set-up, proxy lookup); keep the fastest
MIN_NET_FRACTION = 0.2  # subtracting the RTT may remove at most 80% of a transfer's time


def foreman_http_base(foreman_url: str) -> str:
    """HTTP base URL of the Foreman's REST API, derived from its WebSocket URL.

    ``CROWDIO_FOREMAN_HTTP`` overrides it (for example ``http://192.168.1.7:8000``).
    """
    override = os.getenv("CROWDIO_FOREMAN_HTTP")
    if override:
        return override.rstrip("/")
    return f"http://{urlparse(foreman_url).hostname or 'localhost'}:{FOREMAN_HTTP_PORT}"


def _http_get(url: str, timeout: float) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return len(response.read())


def _http_post(url: str, body: bytes, timeout: float) -> None:
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


class LinkProbe:
    def __init__(
        self,
        base_url: str,
        *,
        down_bytes: int = 262_144,
        up_bytes: int = 131_072,
        min_interval_s: float = 60.0,
        timeout_s: float = 5.0,
        get: Callable[[str, float], int] = _http_get,
        post: Callable[[str, bytes, float], None] = _http_post,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.down_bytes = down_bytes
        self.up_bytes = up_bytes
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._get = get
        self._post = post
        self._clock = clock
        self._last: Optional[Dict[str, float]] = None
        self._last_attempt = -float("inf")

    def due(self) -> bool:
        return self._clock() - self._last_attempt >= self.min_interval_s

    def measure(self) -> Optional[Dict[str, float]]:
        """Run one measurement (blocking; call it from an executor). Keeps the old value on failure."""
        self._last_attempt = self._clock()
        try:
            rtts = []
            for _ in range(RTT_SAMPLES):
                t0 = self._clock()
                self._get(f"{self.base_url}/api/probe/1", self.timeout_s)
                rtts.append(max(self._clock() - t0, MIN_ELAPSED_S))
            rtt = min(rtts)

            t0 = self._clock()
            received = self._get(f"{self.base_url}/api/probe/{self.down_bytes}", self.timeout_s)
            down = received / self._net_seconds(self._clock() - t0, rtt)

            body = bytes(self.up_bytes)
            t0 = self._clock()
            self._post(f"{self.base_url}/api/probe", body, self.timeout_s)
            up = len(body) / self._net_seconds(self._clock() - t0, rtt)
        except Exception:
            return self._last
        self._last = {"down_bps": down, "up_bps": up, "rtt_s": rtt, "measured_at": self._last_attempt}
        return self._last

    @staticmethod
    def _net_seconds(elapsed: float, rtt: float) -> float:
        """Transfer time with the round trip taken out, never less than 20% of what was measured.

        When a transfer is as fast as the round trip itself (a very fast link, or timing noise),
        the difference is meaningless; dividing by it would report an absurd rate.
        """
        return max(elapsed - rtt, MIN_NET_FRACTION * elapsed, MIN_ELAPSED_S)

    def last(self) -> Optional[Dict[str, float]]:
        """The latest measurement (with its age), or None if none has succeeded yet."""
        if self._last is None:
            return None
        return {**self._last, "age_s": max(0.0, self._clock() - self._last["measured_at"])}
