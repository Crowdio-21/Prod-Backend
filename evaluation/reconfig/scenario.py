"""Scenario hooks: scheduled events (inject load, change the network, drop a worker).

Network shaping is done **outside** the app, e.g. ``tc`` on a Linux access point
(laptop or OpenWrt router). In live mode the harness only runs the configured
external command for an event, so experiments are repeatable; in simulation
the same events modify the simulated ground truth.

Event fields (YAML):
  at: seconds from run start
  action: network | load | drop_worker | join_worker | command
  worker: worker id (network/load/drop/join)
  bw: bytes/s (network)      rtt: seconds (network)
  cpu: 0..1 (load)           temp: Celsius (load)
  command: shell command for live mode (any action may carry one); "{worker}",
           "{bw}", "{rtt}" are substituted
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ACTIONS = ("network", "load", "drop_worker", "join_worker", "command")


@dataclass(frozen=True)
class ScenarioEvent:
    at: float
    action: str
    worker: Optional[str] = None
    bw: Optional[float] = None
    rtt: Optional[float] = None
    cpu: Optional[float] = None
    temp: Optional[float] = None
    command: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ScenarioEvent":
        event = cls(**{k: data.get(k) for k in cls.__dataclass_fields__})
        if event.action not in ACTIONS:
            raise ValueError(f"unknown scenario action {event.action!r}")
        return event

    def render_command(self) -> Optional[str]:
        if not self.command:
            return None
        return self.command.format(worker=self.worker or "", bw=self.bw or "", rtt=self.rtt or "")


def load_scenario(items: Sequence[Mapping[str, Any]]) -> List[ScenarioEvent]:
    return sorted((ScenarioEvent.from_mapping(i) for i in items), key=lambda e: e.at)


class LiveScenarioRunner:
    """Runs external commands at scheduled times (wall clock, relative to start)."""

    def __init__(self, events: Sequence[ScenarioEvent], log: Optional[Callable[..., Any]] = None,
                 runner: Callable[[str], int] = None) -> None:
        self.events = list(events)
        self.log = log
        self.runner = runner or self._run
        self.started = time.monotonic()
        self._next = 0

    @staticmethod
    def _run(command: str) -> int:
        return subprocess.call(shlex.split(command))

    def poll(self, now: Optional[float] = None) -> List[ScenarioEvent]:
        elapsed = (time.monotonic() if now is None else now) - self.started
        fired = []
        while self._next < len(self.events) and self.events[self._next].at <= elapsed:
            event = self.events[self._next]
            self._next += 1
            command = event.render_command()
            code = self.runner(command) if command else None
            if self.log is not None:
                self.log("scenario", action=event.action, worker=event.worker, at=event.at, command=command,
                         exit_code=code)
            fired.append(event)
        return fired
