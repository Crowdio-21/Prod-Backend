"""Per-device-class phase profiles: priors plus online linear fits.

Priors come from the unverified reference numbers (cold ~48 s, warm ~6 s for a
~180 MB DistilBERT shard on a mid-range Android phone). They carry n=0, so the
controller treats them as stale and never makes a non-forced switch on priors
alone. Real ``load_result`` timings replace them as they arrive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Tuple

from .types import DeviceProfile, Est, LinearFit

PHASES = ("load", "init", "write", "unload", "peak_mem")
REF_BYTES = 180e6


def _prior(seconds_at_ref: float, intercept: float = 0.0) -> LinearFit:
    slope = max(0.0, seconds_at_ref - intercept) / REF_BYTES
    return LinearFit(intercept, slope, resid_std=0.5 * seconds_at_ref, n=0)


DEFAULT_PROFILES: Dict[str, DeviceProfile] = {
    "android": DeviceProfile(
        device_class="android",
        load=_prior(4.0, 0.3),
        init=_prior(2.0, 0.2),
        write=_prior(4.0, 0.1),
        unload=LinearFit(0.2, 0.0, 0.1, 0),
        peak_mem=LinearFit(32e6, 2.0, 0.0, 0),
        flops_per_s=Est.prior(5e9),
    ),
    "pc": DeviceProfile(
        device_class="pc",
        load=_prior(0.8, 0.05),
        init=_prior(0.4, 0.05),
        write=_prior(0.6, 0.0),
        unload=LinearFit(0.05, 0.0, 0.02, 0),
        peak_mem=LinearFit(64e6, 2.0, 0.0, 0),
        flops_per_s=Est.prior(5e10),
    ),
}
DEFAULT_PROFILES["default"] = replace(DEFAULT_PROFILES["android"], device_class="default")


def profile_for(profiles: Mapping[str, DeviceProfile], device_class: str) -> DeviceProfile:
    return profiles.get(device_class) or profiles.get("default") or DEFAULT_PROFILES["default"]


@dataclass
class PhaseFitter:
    """Least-squares fit of y = a + b*x, anchored to a prior until data disagree."""

    prior: LinearFit
    samples: List[Tuple[float, float]] = field(default_factory=list)
    max_samples: int = 200

    def add(self, x: float, y: float) -> None:
        self.samples.append((float(x), float(y)))
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def fit(self) -> LinearFit:
        n = len(self.samples)
        if n == 0:
            return self.prior
        xs = [x for x, _ in self.samples]
        ys = [y for _, y in self.samples]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        sxx = sum((x - mean_x) ** 2 for x in xs)
        if n >= 2 and sxx > 0:
            b = sum((x - mean_x) * (y - mean_y) for x, y in self.samples) / sxx
            b = max(0.0, b)
            a = mean_y - b * mean_x
        else:
            # One size seen so far: keep the prior slope, move the intercept.
            b = self.prior.b
            a = mean_y - b * mean_x
        residuals = [y - (a + b * x) for x, y in self.samples]
        if n >= 3:
            resid = math.sqrt(sum(r * r for r in residuals) / (n - 2))
        else:
            resid = max(0.25 * abs(mean_y), self.prior.resid_std * 0.5)
        return LinearFit(a, b, resid, n)


class ProfileLearner:
    """Learns DeviceProfiles from load_result timings (stateful, not used by planners)."""

    def __init__(self, priors: Mapping[str, DeviceProfile] = DEFAULT_PROFILES) -> None:
        self.priors = dict(priors)
        self._fitters: Dict[Tuple[str, str], PhaseFitter] = {}

    def _fitter(self, device_class: str, phase: str) -> PhaseFitter:
        key = (device_class, phase)
        if key not in self._fitters:
            base = profile_for(self.priors, device_class)
            self._fitters[key] = PhaseFitter(getattr(base, phase))
        return self._fitters[key]

    def observe(self, device_class: str, shard_bytes: int, **phases: float) -> None:
        for phase, value in phases.items():
            if phase not in PHASES or value is None:
                continue
            if phase != "peak_mem" and value <= 0:
                continue
            self._fitter(device_class, phase).add(shard_bytes, value)

    def profiles(self) -> Dict[str, DeviceProfile]:
        result = dict(self.priors)
        classes = {device_class for device_class, _ in self._fitters}
        for device_class in classes:
            base = profile_for(self.priors, device_class)
            updates = {
                phase: self._fitters[(device_class, phase)].fit()
                for phase in PHASES
                if (device_class, phase) in self._fitters
            }
            result[device_class] = replace(base, device_class=device_class, **updates)
        return result
