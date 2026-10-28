"""Replay logged planner calls through the pure planner and check the decisions match."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from foreman.planning.config import ControllerConfig
from foreman.planning.controller import guard_initial
from foreman.planning.planner import Objective
from foreman.planning.policies import build_plan, make_policy

from foreman.planning.serialize import opts_from_dict, snapshot_from_dict


def replay(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    skipped = 0
    for event in events:
        if event.get("event") != "planner_call":
            continue
        if not event.get("snapshot"):
            skipped += 1
            continue
        snap = snapshot_from_dict(event["snapshot"])
        opts = opts_from_dict(event["opts"])
        policy = make_policy(event["policy"], opts)
        kind, deadline = event.get("objective") or ("min_total", None)
        current = None
        if event.get("current"):
            current = build_plan(snap, tuple(map(tuple, event["current"])), epoch=0, reason="replay", opts=opts)
        objective = Objective(kind, deadline)
        decision = policy.plan(snap, current, int(event["n"]), objective)
        triggers = set(event.get("triggers") or [])
        if triggers == {"initial"} and policy.switch_rule == "break_even" and event.get("controller"):
            decision = guard_initial(snap, int(event["n"]), policy, objective,
                                     ControllerConfig(**event["controller"]), decision)
        chosen = [list(s) for s in (decision.stages or ())]
        checked += 1
        if chosen != event.get("chosen", []):
            mismatches.append({"ts": event["ts"], "logged": event.get("chosen"), "replayed": chosen})
    return {"checked": checked, "skipped_without_snapshot": skipped, "mismatches": mismatches,
            "identical": checked > 0 and not mismatches}
