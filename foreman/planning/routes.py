"""Opportunistic route selection (below the planner, in the cost layer).

The cost model already prices a boundary as min(direct, relay). This module
adds what the live system needs on top: hysteresis (margin + minimum dwell) so
routes never flap on noisy estimates, immediate fallback to the relay when a
peer becomes unreachable, and capability-check bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .config import RouteConfig
from .contract import RouteMode
from .cost_model import CostOptions, _direct_link, t_comm
from .types import Q, Snapshot


@dataclass
class _RouteState:
    mode: RouteMode
    since: float


@dataclass(frozen=True)
class RouteChoice:
    mode: RouteMode
    flipped: bool
    reason: str
    direct_s: Optional[float]
    relay_s: float


class RouteSelector:
    def __init__(self, cfg: RouteConfig = RouteConfig()) -> None:
        self.cfg = cfg
        self._state: Dict[Tuple[str, str], _RouteState] = {}
        self.flips = 0

    def current(self, i: str, j: str) -> RouteMode:
        state = self._state.get((i, j))
        return state.mode if state else RouteMode.VIA_FOREMAN

    def choose(self, snap: Snapshot, i: str, j: str, n_bytes: float, opts: CostOptions, now: float) -> RouteChoice:
        relay_opts = CostOptions(**{**opts.__dict__, "direct_path": False})
        relay = t_comm(snap, i, j, n_bytes, relay_opts).time
        state = self._state.setdefault((i, j), _RouteState(RouteMode.VIA_FOREMAN, -math.inf))
        link = _direct_link(snap, i, j, opts)
        if link is None:
            reason = "direct disabled" if not opts.direct_path else "peer unreachable or estimate stale"
            return self._set(state, RouteMode.VIA_FOREMAN, now, reason, None, relay.mid, force=True)
        direct = Q.divide(n_bytes * snap.compression_ratio, Q.from_est(link.bw, opts.k, opts.age_growth), opts.min_bw) \
            + Q.from_est(link.rtt, opts.k, opts.age_growth)
        dwell_ok = now - state.since >= self.cfg.min_dwell_s
        margin = 1.0 + self.cfg.margin_frac
        if state.mode is RouteMode.VIA_FOREMAN and dwell_ok and direct.hi * margin < relay.mid:
            return self._set(state, RouteMode.DIRECT, now, "direct cheaper by margin", direct.mid, relay.mid)
        if state.mode is RouteMode.DIRECT and dwell_ok and relay.hi * margin < direct.mid:
            return self._set(state, RouteMode.VIA_FOREMAN, now, "relay cheaper by margin", direct.mid, relay.mid)
        return RouteChoice(state.mode, False, "hold", direct.mid, relay.mid)

    def fallback(self, i: str, j: str, now: float, reason: str) -> RouteChoice:
        state = self._state.setdefault((i, j), _RouteState(RouteMode.VIA_FOREMAN, -math.inf))
        return self._set(state, RouteMode.VIA_FOREMAN, now, reason, None, math.nan, force=True)

    def _set(self, state: _RouteState, mode: RouteMode, now: float, reason: str,
             direct: Optional[float], relay: float, force: bool = False) -> RouteChoice:
        flipped = state.mode is not mode
        if flipped:
            state.mode = mode
            state.since = now
            self.flips += 1
        return RouteChoice(mode, flipped, reason, direct, relay)


class CapabilityTracker:
    """When to (re)test peer reachability: on join, on network change, and rarely otherwise."""

    def __init__(self, cfg: RouteConfig = RouteConfig()) -> None:
        self.cfg = cfg
        self._tested: Dict[Tuple[str, str], Tuple[float, bool]] = {}
        self._network_epoch: Dict[str, int] = {}

    def needs_test(self, i: str, j: str, now: float) -> bool:
        entry = self._tested.get((i, j))
        return entry is None or now - entry[0] > self.cfg.capability_retest_s

    def record(self, i: str, j: str, reachable: bool, now: float) -> None:
        self._tested[(i, j)] = (now, reachable)

    def reachable(self, i: str, j: str) -> bool:
        entry = self._tested.get((i, j))
        return bool(entry and entry[1])

    def network_changed(self, worker_id: str) -> None:
        for pair in [p for p in self._tested if worker_id in p]:
            del self._tested[pair]

    def pairs_to_test(self, workers: Iterable[str], now: float) -> List[Tuple[str, str]]:
        workers = sorted(workers)
        return [(i, j) for i in workers for j in workers if i != j and self.needs_test(i, j, now)]

    def as_mapping(self) -> Dict[Tuple[str, str], bool]:
        return {pair: ok for pair, (_, ok) in self._tested.items()}
