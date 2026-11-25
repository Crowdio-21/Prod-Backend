"""Pure reconfiguration controller.

Rule (break-even policies): switch only if J(new) + margin < J(current), where
J = T_total_remaining computed with ready_i = C_switch_i (so the switching cost
is inside J(new) once, not added twice) and

    margin = (J_new.hi - J_new.mid) + (J_cur.mid - J_cur.lo)   # estimate uncertainty
             + margin_abs_s + margin_frac * J_cur.mid             # tunables

Equivalent break-even form, logged for every decision:
    N_remaining * (T_steady_cur - T_steady_new)  vs  C_switch + margin.

Rule (regret policies, rent or buy): while the best alternative is cheaper per
request, accumulate the regret of staying,

    regret += served_since_last_decision * (T_steady_cur - T_steady_best)

and switch once regret >= C_switch(best) + margin_regret, where

    margin_regret = (C_switch.hi - C_switch.mid) + margin_abs_s + margin_frac * C_switch.mid.

Regret resets to zero after a switch and whenever no alternative is cheaper.
Unlike break-even it uses no prediction of the remaining requests. In the
two-state abstraction (stay versus one fixed alternative, exact costs) this is
the deterministic ski-rental rule, which is 2-competitive; with many plans and
drifting estimates no such bound is claimed.

Hysteresis: a trigger must persist for ``persist_k`` samples, a cooldown
follows every switch, and switches per window are capped. Forced replans
(worker lost, load failed, OOM, next hop down) bypass margin and hysteresis but
still take the cheapest feasible plan. Non-forced switches are refused while
any estimate they rely on is stale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from .config import ControllerConfig
from .contract import Plan
from .cost_model import PipelineCost
from .planner import Objective, Stages
from .policies import B0Policy, Policy, PolicyDecision
from .types import Snapshot

FORCED = frozenset({"worker_lost", "load_failed", "oom", "next_hop_down", "initial", "ram_pressure"})
# Forced only once it persists (free RAM flaps on a phone), unlike a lost worker.
PERSISTENT_FORCED = frozenset({"ram_pressure"})


@dataclass(frozen=True)
class Baseline:
    workers: Mapping[str, Tuple[float, float, float, float, Optional[float]]]  # up, down, rtt, cpu, temp

    @classmethod
    def capture(cls, snap: Snapshot) -> "Baseline":
        return cls({
            w: (v.up.mean, v.down.mean, v.rtt.mean, v.cpu, v.temp)
            for w, v in snap.workers.items() if v.alive
        })


@dataclass(frozen=True)
class ControllerState:
    last_switch_at: float = -math.inf
    switch_times: Tuple[float, ...] = ()
    trigger_counts: Mapping[str, int] = field(default_factory=dict)
    baseline: Optional[Baseline] = None
    regret_s: float = 0.0  # regret policies: accumulated cost of staying
    last_n: Optional[int] = None  # n_remaining at the previous decision


@dataclass(frozen=True)
class ControllerDecision:
    action: str  # initial | switch | forced_switch | stay | stall
    reason: str
    triggers: Tuple[str, ...] = ()
    decision: Optional[PolicyDecision] = None
    j_current: Optional[float] = None
    j_new: Optional[float] = None
    margin: Optional[float] = None
    break_even: Optional[Dict[str, float]] = None
    planner_called: bool = False

    @property
    def switches(self) -> bool:
        return self.action in ("initial", "switch", "forced_switch")

    @property
    def target(self) -> Optional[Stages]:
        return self.decision.stages if self.decision else None


def _rel(a: float, b: float) -> float:
    base = max(abs(b), 1e-9)
    return abs(a - b) / base


def detect_triggers(
    snap: Snapshot,
    plan: Optional[Plan],
    baseline: Optional[Baseline],
    cfg: ControllerConfig,
    *,
    problem_count: int = 0,
    urgent_problems: FrozenSet[str] = frozenset(),
    deadline_remaining_s: Optional[float] = None,
    j_current_hi: Optional[float] = None,
    ram_critical_bytes: int = 0,
) -> Tuple[str, ...]:
    """Pure trigger detection. Returns sorted trigger names (forced ones included)."""
    found = set(urgent_problems)
    if plan is None:
        return ("initial",)
    plan_workers = set(plan.workers)
    alive = {w for w, v in snap.workers.items() if v.alive}
    if plan_workers - alive:
        found.add("worker_lost")
    for stage in plan.stages:
        if (stage.worker_id, stage.shard_id) in snap.excluded:
            found.add("load_failed")
    if baseline is not None:
        known = set(baseline.workers)
        if alive - known:
            found.add("worker_join")
        if (known - alive) - plan_workers:
            found.add("worker_leave")
        for worker_id, (up, down, rtt, cpu, temp) in baseline.workers.items():
            view = snap.workers.get(worker_id)
            if view is None or not view.alive:
                continue
            if _rel(view.up.mean, up) > cfg.bw_drift or _rel(view.down.mean, down) > cfg.bw_drift:
                found.add("bw_drift")
            if _rel(view.rtt.mean, rtt) > cfg.rtt_drift:
                found.add("rtt_drift")
            if abs(view.cpu - cpu) > cfg.cpu_drift:
                found.add("cpu_drift")
            if temp is not None and view.temp is not None and abs(view.temp - temp) > cfg.temp_drift:
                found.add("temp_drift")
    for stage in plan.stages:
        view = snap.workers.get(stage.worker_id)
        if (view is not None and ram_critical_bytes > 0 and view.ram_free is not None
                and view.ram_free < ram_critical_bytes):
            found.add("ram_pressure")
        if view is not None and view.queue_len >= cfg.queue_growth:
            found.add("queue_growth")
    if problem_count >= cfg.problem_threshold:
        found.add("problems")
    if deadline_remaining_s is not None and j_current_hi is not None and j_current_hi > deadline_remaining_s:
        found.add("deadline_risk")
    return tuple(sorted(found))


def decide(
    snap: Snapshot,
    plan: Optional[Plan],
    n_remaining: int,
    policy: Policy,
    objective: Objective,
    state: ControllerState,
    cfg: ControllerConfig,
    triggers: Tuple[str, ...],
    now: float,
) -> Tuple[ControllerDecision, ControllerState]:
    counts = {t: state.trigger_counts.get(t, 0) + 1 for t in triggers}
    forced = tuple(t for t in triggers if t in FORCED
                   and (t not in PERSISTENT_FORCED or counts[t] >= cfg.persist_k))
    served = max(0, state.last_n - n_remaining) if state.last_n is not None else 0
    state = replace(state, trigger_counts=counts, last_n=n_remaining)

    if forced:
        decision = policy.plan(snap, plan, n_remaining, objective)
        if not decision.feasible or decision.stages is None:
            if set(forced) <= PERSISTENT_FORCED and plan is not None:
                # Memory is tight but the current plan still runs: keep serving rather than stop.
                return ControllerDecision("stay", "no plan fits the free memory", triggers, decision,
                                          planner_called=True), state
            return ControllerDecision("stall", "no feasible plan", triggers, decision, planner_called=True), state
        action = "initial" if "initial" in forced else "forced_switch"
        if set(forced) == {"initial"} and policy.switch_rule == "break_even":
            decision = guard_initial(snap, n_remaining, policy, objective, cfg, decision)
        if plan is not None and decision.stages == plan.assignment() and not set(forced) - {"initial"}:
            return ControllerDecision("stay", "forced trigger, same plan", triggers, decision, planner_called=True), state
        return ControllerDecision(action, ",".join(forced), triggers, decision,
                                  j_new=decision.cost.T_total.mid if decision.cost else None,
                                  planner_called=True), _after_switch(state, now, cfg, snap)

    active = tuple(t for t in triggers if counts.get(t, 0) >= cfg.persist_k)
    accumulating = policy.switch_rule == "regret" and state.regret_s > 0
    if not active and not accumulating:
        return ControllerDecision("stay", "no persistent trigger", triggers), state
    if policy.switch_rule == "static":
        return ControllerDecision("stay", "static policy", triggers), state
    if now - state.last_switch_at < cfg.cooldown_s:
        return ControllerDecision("stay", "cooldown", triggers), state
    recent = tuple(t for t in state.switch_times if now - t <= cfg.switch_window_s)
    if len(recent) >= cfg.max_switches:
        return ControllerDecision("stay", "switch cap reached", triggers), replace(state, switch_times=recent)

    decision = policy.plan(snap, plan, n_remaining, objective)
    if not decision.feasible or decision.stages is None:
        return ControllerDecision("stay", "no feasible alternative", triggers, decision, planner_called=True), \
            replace(state, regret_s=0.0)
    if plan is not None and decision.stages == plan.assignment() and not decision.replicas:
        return ControllerDecision("stay", "planner keeps current plan", triggers, decision, planner_called=True), \
            replace(state, regret_s=0.0)
    current: Optional[PipelineCost] = policy.evaluate(snap, plan, n_remaining)
    new = decision.cost
    if current is None:
        return ControllerDecision("forced_switch", "current plan not runnable", triggers, decision,
                                  j_new=new.T_total.mid, planner_called=True), _after_switch(state, now, cfg, snap)
    if policy.switch_rule == "regret":
        return _decide_regret(new, current, decision, triggers, served, state, cfg, now, snap)
    margin = _margin(new, current, cfg)
    break_even = {
        "gain": n_remaining * (current.T_steady.mid - new.T_steady.mid),
        "switch_cost": new.C_switch_total.mid,
        "margin": margin,
    }
    common = dict(triggers=triggers, decision=decision, j_current=current.T_total.mid, j_new=new.T_total.mid,
                  margin=margin, break_even=break_even, planner_called=True)
    if new.stale or current.stale:
        return ControllerDecision("stay", "stale estimates", **common), state
    if policy.switch_rule == "always":
        if new.T_total.mid < current.T_total.mid:
            return ControllerDecision("switch", "planner prefers new plan", **common), _after_switch(state, now, cfg, snap)
        return ControllerDecision("stay", "no improvement", **common), state
    if new.T_total.mid + margin < current.T_total.mid:
        return ControllerDecision("switch", "J(new) + margin < J(current)", **common), _after_switch(state, now, cfg, snap)
    return ControllerDecision("stay", "gain below switching cost + margin", **common), state


def _decide_regret(new: PipelineCost, current: PipelineCost, decision: PolicyDecision, triggers, served: int,
                   state: ControllerState, cfg: ControllerConfig, now: float, snap: Snapshot):
    gap = current.T_steady.mid - new.T_steady.mid
    switch_cost = new.C_switch_total
    margin = (switch_cost.hi - switch_cost.mid) + cfg.margin_abs_s + cfg.margin_frac * switch_cost.mid
    regret = state.regret_s + served * gap if gap > 0 else 0.0
    break_even = {"regret": regret, "gain_per_request": gap, "served": float(served),
                  "switch_cost": switch_cost.mid, "margin": margin}
    common = dict(triggers=triggers, decision=decision, j_current=current.T_total.mid, j_new=new.T_total.mid,
                  margin=margin, break_even=break_even, planner_called=True)
    if new.stale or current.stale:
        return ControllerDecision("stay", "stale estimates", **common), state
    if gap <= 0:
        return ControllerDecision("stay", "no cheaper alternative", **common), replace(state, regret_s=0.0)
    if regret >= switch_cost.mid + margin:
        return ControllerDecision("switch", "regret >= C_switch + margin", **common), _after_switch(state, now, cfg, snap)
    return ControllerDecision("stay", "accumulating regret", **common), replace(state, regret_s=regret)


def _margin(new: PipelineCost, current: PipelineCost, cfg: ControllerConfig) -> float:
    return ((new.T_total.hi - new.T_total.mid) + (current.T_total.mid - current.T_total.lo)
            + cfg.margin_abs_s + cfg.margin_frac * current.T_total.mid)


def guard_initial(snap, n, policy, objective, cfg, decision: PolicyDecision) -> PolicyDecision:
    """An initial plan has nothing to recover, so a costly first placement must still
    beat the residency-preserving affinity plan by the margin (noisy first estimates)."""
    base = B0Policy(policy.cost_options()).plan(snap, None, n, objective)
    if not base.feasible or base.stages == decision.stages or decision.cost is None:
        return decision
    if decision.cost.T_total.mid + _margin(decision.cost, base.cost, cfg) < base.cost.T_total.mid:
        return decision
    return replace(base, policy=decision.policy, search=decision.search,
                   reason="initial: residency-preserving plan (gain below margin)")


def _after_switch(state: ControllerState, now: float, cfg: ControllerConfig, snap: Snapshot) -> ControllerState:
    recent = tuple(t for t in state.switch_times if now - t <= cfg.switch_window_s) + (now,)
    return replace(state, last_switch_at=now, switch_times=recent, trigger_counts={}, baseline=Baseline.capture(snap),
                   regret_s=0.0)
