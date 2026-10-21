import asyncio

from foreman.core.foreman_policy import (
    Assignment,
    AssignmentRecord,
    BreakEvenPolicy,
    DecisionKind,
    LatencyPredictor,
    MoveCostModel,
    StageProfile,
    SystemState,
    WorkerRegistry,
    WorkerTelemetry,
    enumerate_assignments,
)


def _state(**kwargs):
    workers = {
        "slow": WorkerTelemetry("slow", device_class="slow", bandwidth_mean=100),
        "fast": WorkerTelemetry("fast", device_class="fast", bandwidth_mean=100),
    }
    profiles = {
        "a": StageProfile("a", size_bytes=10, stage_seconds=10),
        "b": StageProfile("b", size_bytes=10, stage_seconds=1),
    }
    current = AssignmentRecord(Assignment({"a": "slow", "b": "fast"}))
    defaults = dict(
        workers=workers,
        profiles=profiles,
        current=current,
        n_remaining=100,
        now=100,
        cooldown_seconds=0,
        persistence_ticks=1,
        persisted_ticks=1,
        move_cost=MoveCostModel(min_bandwidth_bytes_per_second=100),
    )
    defaults.update(kwargs)
    return SystemState(**defaults)


def test_enumeration_is_injective_and_includes_stay_candidate():
    candidates = list(enumerate_assignments(("a", "b"), ("slow", "fast")))
    assert len(candidates) == 2
    assert all(len(set(c.worker_by_partition.values())) == 2 for c in candidates)


def test_break_even_policy_moves_when_workload_pays_for_move():
    state = _state(current=AssignmentRecord(Assignment({"a": "fast", "b": "slow"})))
    predictor = LatencyPredictor(state.profiles)
    predictor.observe("slow", "a", 1)
    predictor.observe("fast", "a", 10)
    predictor.observe("slow", "b", 10)
    predictor.observe("fast", "b", 1)
    policy = BreakEvenPolicy(predictor)
    decision = policy.decide(state)
    assert decision.kind == DecisionKind.REASSIGN
    assert decision.plan.worker_by_partition == {"a": "slow", "b": "fast"}


def test_break_even_policy_stays_when_workload_is_too_small():
    policy = BreakEvenPolicy(LatencyPredictor(_state().profiles))
    decision = policy.decide(_state(n_remaining=0, arrival_rate=0))
    assert decision.kind == DecisionKind.STAY


def test_cooldown_and_persistence_are_first_class_gates():
    policy = BreakEvenPolicy(LatencyPredictor(_state().profiles))
    assert policy.decide(_state(persisted_ticks=0)).kind == DecisionKind.STAY
    assert policy.decide(_state(cooldown_seconds=100)).kind == DecisionKind.STAY


def test_worker_registry_requires_three_missed_heartbeats():
    registry = WorkerRegistry(missed_heartbeats=3)
    worker = WorkerTelemetry("w")
    registry.heartbeat(worker, now=0)
    assert registry.departed(29, interval=10) == set()
    assert registry.departed(31, interval=10) == {"w"}


def test_predictor_updates_with_ewma():
    profile = StageProfile("a", 1, 10)
    predictor = LatencyPredictor({"a": profile}, alpha=0.5)
    worker = WorkerTelemetry("w", device_class="pc")
    predictor.observe("pc", "a", 4)
    assert predictor.predict(worker, "a") == 7
