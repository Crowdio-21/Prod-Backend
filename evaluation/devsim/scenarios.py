"""Named scenarios: how each device's bandwidth changes over time, and who joins or leaves.

Scenarios are seeded. The same ``(scenario, fleet, seed)`` gives the same bandwidth for every
device whichever scheduler is being tested, which is what makes a comparison fair.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .bandwidth import (
    BandwidthModel,
    BandwidthTrace,
    Compose,
    Markov,
    RandomOutages,
    RandomWalk,
    Stable,
    Step,
    build_trace,
)
from .fleet import DeviceSpec


@dataclass(frozen=True)
class ScenarioEvent:
    at_s: float
    action: str  # "leave" or "join"
    device_id: str


ModelFn = Callable[[Sequence[DeviceSpec], random.Random], Dict[str, BandwidthModel]]
EventFn = Callable[[Sequence[DeviceSpec], random.Random], List[ScenarioEvent]]


def _no_events(fleet, rng):
    return []


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    models: ModelFn
    events: EventFn = _no_events
    shared_capacity_mbps: Optional[float] = None  # one medium all devices share (a single access point)


@dataclass
class PreparedScenario:
    scenario: Scenario
    traces: Dict[str, BandwidthTrace]
    events: List[ScenarioEvent] = field(default_factory=list)

    @property
    def shared_capacity_mbps(self) -> Optional[float]:
        return self.scenario.shared_capacity_mbps


def prepare(scenario: Scenario, fleet: Sequence[DeviceSpec], seed: int = 0, horizon_s: float = 3600.0,
            step_s: float = 1.0) -> PreparedScenario:
    rng = random.Random(f"{seed}:{scenario.name}")
    models = scenario.models(fleet, rng)
    traces = {
        d.device_id: build_trace(models.get(d.device_id, Stable()),
                                 random.Random(f"{seed}:{scenario.name}:{d.device_id}"), horizon_s, step_s)
        for d in fleet
    }
    return PreparedScenario(scenario, traces, sorted(scenario.events(fleet, rng), key=lambda e: e.at_s))


# ---------------------------------------------------------------- presets
def _stable(fleet, rng):
    return {}


def _congestion(fleet, rng):
    """Every device's link flips between good and congested on its own schedule."""
    return {d.device_id: Markov(bad=0.15, p_bad=0.03, p_good=0.12) for d in fleet}


def _drift(fleet, rng):
    return {d.device_id: RandomWalk(sigma=0.18, reversion=0.08) for d in fleet}


def _drop(fleet, rng):
    """The fastest third of the fleet collapses to 8% of its bandwidth at 30 s and recovers at 100 s."""
    fastest = sorted(fleet, key=lambda d: d.down_mbps, reverse=True)[: max(1, len(fleet) // 3)]
    return {d.device_id: Step(at_s=30.0, factor=0.08, until_s=100.0) for d in fastest}


def _commute(fleet, rng):
    """Drifting links; a third of the devices also pass through short outages (tunnels)."""
    unlucky = {d.device_id for d in rng.sample(list(fleet), max(1, len(fleet) // 3))}
    return {d.device_id: (Compose((RandomWalk(sigma=0.12), RandomOutages(count=3, min_s=2.0, max_s=6.0)))
                          if d.device_id in unlucky else RandomWalk(sigma=0.12))
            for d in fleet}


def _churn_models(fleet, rng):
    return {}


def _churn_events(fleet, rng):
    """Three devices drop off around 25 s; two of them come back at about 70 s."""
    leaving = rng.sample(list(fleet), min(3, len(fleet)))
    events = [ScenarioEvent(round(rng.uniform(20, 30), 1), "leave", d.device_id) for d in leaving]
    events += [ScenarioEvent(round(rng.uniform(60, 80), 1), "join", d.device_id) for d in leaving[:2]]
    return events


SCENARIOS: Dict[str, Scenario] = {s.name: s for s in (
    Scenario("stable", "Every device keeps its nominal bandwidth.", _stable),
    Scenario("congestion", "Links flip between good and congested (about 8 s in 40 congested at 15% bandwidth).", _congestion),
    Scenario("drift", "Bandwidth drifts up and down around nominal (random walk).", _drift),
    Scenario("drop", "The fastest third of the fleet falls to 8% bandwidth from 30 s to 100 s.", _drop),
    Scenario("commute", "Drifting links, with a third of the devices going through short outages.", _commute),
    Scenario("churn", "Stable links; three devices leave around 25 s and two return around 70 s.", _churn_models, _churn_events),
    Scenario("shared-wifi", "Congested links that also share one 100 Mbit/s access point.", _congestion,
             shared_capacity_mbps=100.0),
)}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(f"Unknown scenario {name!r}. Available: {', '.join(SCENARIOS)}") from None
