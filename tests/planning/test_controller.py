from dataclasses import replace

import pytest

from foreman.planning.config import ControllerConfig
from foreman.planning.controller import Baseline, ControllerState, decide, detect_triggers
from foreman.planning.planner import Objective
from foreman.planning.policies import B0Policy, B2Policy, SAwarePolicy, build_plan
from foreman.planning.types import Est
from tests.planning.fixtures import MB, opts, profile, snapshot, worker

CFG = ControllerConfig(persist_k=1, cooldown_s=0, margin_abs_s=0.5, margin_frac=0.0)


def _setup(slow_bw=0.5 * MB, absent=True):
    """Current plan sits on resident workers; s2 is slow for b. 'spare' is fast but cold."""
    p = profile(compute={"a": 0.1, "b": 2.0, "c": 0.1})
    fast = profile("fast", compute={"a": 0.1, "b": 0.2, "c": 0.1})
    workers = [
        worker("s1", resident="a"),
        worker("s2", resident="b"),
        worker("s3", resident="c"),
        worker("spare", device_class="fast", disk=set() if absent else {"b"}),
    ]
    snap = snapshot(workers, profiles={"phone": p, "fast": fast})
    o = opts()
    plan = build_plan(snap, (("a", "s1"), ("b", "s2"), ("c", "s3")), epoch=1, reason="init", opts=o)
    return snap, plan, o


def _decide(snap, plan, n, policy, state=None, triggers=("bw_drift",), now=100.0, cfg=CFG):
    return decide(snap, plan, n, policy, Objective(), state or ControllerState(), cfg, triggers, now)


def test_small_n_switch_is_rejected():
    snap, plan, o = _setup()
    decision, _ = _decide(snap, plan, 2, SAwarePolicy(o, recut=False))
    assert decision.action == "stay"
    assert decision.reason in ("gain below switching cost + margin", "planner keeps current plan")


def test_large_n_switch_is_taken_when_gain_beats_cost():
    snap, plan, o = _setup()
    decision, state = _decide(snap, plan, 500, SAwarePolicy(o, recut=False))
    assert decision.action == "switch"
    assert ("b", "spare") in decision.target
    assert decision.break_even["gain"] > decision.break_even["switch_cost"] + decision.break_even["margin"]
    assert state.last_switch_at == 100.0


def test_warm_replacement_breaks_even_sooner_than_cold():
    snap_cold, plan, o = _setup(absent=True)
    snap_warm, _, _ = _setup(absent=False)
    policy = SAwarePolicy(o, recut=False)
    # Gain 1.8 s/request. Warm switch ~1.5 s breaks even at n ~1; cold (~11.8 s) needs n ~7.
    n = 4
    cold, _ = _decide(snap_cold, plan, n, policy)
    warm, _ = _decide(snap_warm, plan, n, policy)
    assert cold.action == "stay"
    assert warm.action == "switch"


def test_trigger_must_persist_k_samples():
    snap, plan, o = _setup()
    cfg = replace(CFG, persist_k=2)
    first, state = _decide(snap, plan, 500, SAwarePolicy(o, recut=False), cfg=cfg)
    assert first.action == "stay" and first.reason == "no persistent trigger"
    assert not first.planner_called
    second, _ = _decide(snap, plan, 500, SAwarePolicy(o, recut=False), state=state, cfg=cfg)
    assert second.action == "switch"


def test_cooldown_and_switch_cap():
    snap, plan, o = _setup()
    cfg = replace(CFG, cooldown_s=60)
    state = ControllerState(last_switch_at=90.0)
    decision, _ = _decide(snap, plan, 500, SAwarePolicy(o, recut=False), state=state, cfg=cfg)
    assert decision.reason == "cooldown"
    cfg = replace(CFG, max_switches=2, switch_window_s=100)
    state = ControllerState(switch_times=(10.0, 50.0))
    decision, _ = _decide(snap, plan, 500, SAwarePolicy(o, recut=False), state=state, cfg=cfg)
    assert decision.reason == "switch cap reached"


def test_stale_estimates_block_non_forced_switch():
    snap, plan, o = _setup()
    stale = replace(snap.workers["spare"], down=Est(10 * MB, 0, 0, 1e9))
    snap = snap.with_workers(spare=stale)
    decision, _ = _decide(snap, plan, 500, SAwarePolicy(replace(o, min_samples=1), recut=False))
    assert decision.action == "stay" and decision.reason == "stale estimates"


def test_forced_replan_bypasses_margin_and_takes_cheapest():
    snap, plan, o = _setup()
    snap = snap.with_workers(s2=replace(snap.workers["s2"], alive=False))
    triggers = detect_triggers(snap, plan, Baseline.capture(snap), CFG)
    assert "worker_lost" in triggers
    decision, _ = _decide(snap, plan, 2, SAwarePolicy(o, recut=False), triggers=triggers,
                          state=ControllerState(last_switch_at=99.0), cfg=replace(CFG, cooldown_s=60))
    assert decision.action == "forced_switch"
    assert ("b", "spare") in decision.target


def test_static_policy_only_moves_when_forced():
    snap, plan, o = _setup()
    decision, _ = _decide(snap, plan, 500, B0Policy(o))
    assert decision.action == "stay" and decision.reason == "static policy"


def test_b2_switches_on_any_improvement():
    snap, plan, o = _setup()
    decision, _ = _decide(snap, plan, 2, B2Policy(o))
    assert decision.action == "switch"


def test_drift_detection():
    snap, plan, o = _setup()
    base = Baseline.capture(snap)
    assert detect_triggers(snap, plan, base, CFG) == ()
    slower = replace(snap.workers["s2"], down=Est(1 * MB, 0, 10, 0))
    assert "bw_drift" in detect_triggers(snap.with_workers(s2=slower), plan, base, CFG)
    hot = replace(snap.workers["s2"], queue_len=50)
    assert "queue_growth" in detect_triggers(snap.with_workers(s2=hot), plan, base, CFG)
    assert "problems" in detect_triggers(snap, plan, base, CFG, problem_count=10)
    assert "deadline_risk" in detect_triggers(snap, plan, base, CFG, deadline_remaining_s=1, j_current_hi=5)
    assert detect_triggers(snap, None, None, CFG) == ("initial",)
