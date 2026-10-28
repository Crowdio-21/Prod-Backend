"""Telemetry ingestion and estimation (stateful; produces pure snapshots).

* Passive first: bandwidth comes from timed shard and tensor transfers.
* The active probe is only requested when an estimate is older than
  ``probe_after_s`` or the worker is idle, and at most once per
  ``probe_min_interval_s`` per worker.
* Every estimate carries its age and sample count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .config import EstimatorConfig
from .contract import WorkerState
from .types import DeviceProfile, Est, LinkView, ShardCatalog, Snapshot, WorkerView


@dataclass
class Ewma:
    """Exponentially weighted mean and variance with sample count and timestamp."""

    alpha: float
    mean: float = 0.0
    var: float = 0.0
    n: int = 0
    updated_at: float = -math.inf

    def update(self, value: float, now: float) -> None:
        if self.n == 0:
            self.mean, self.var = float(value), 0.0
        else:
            diff = value - self.mean
            incr = self.alpha * diff
            self.mean += incr
            self.var = (1 - self.alpha) * (self.var + diff * incr)
        self.n += 1
        self.updated_at = now

    def estimate(self, now: float, default: Optional[Est] = None, rel_floor: float = 0.0) -> Est:
        if self.n == 0:
            return default if default is not None else Est(0.0, 0.0, 0, math.inf)
        # A standard-error floor keeps one or two samples from looking exact.
        std = max(math.sqrt(max(0.0, self.var)), abs(self.mean) * rel_floor / math.sqrt(self.n))
        return Est(self.mean, std, self.n, max(0.0, now - self.updated_at))


@dataclass
class _WorkerRecord:
    worker_id: str
    device_class: str = "unknown"
    last_seen: float = -math.inf
    state: Optional[WorkerState] = None
    up: Ewma = None  # type: ignore[assignment]
    down: Ewma = None  # type: ignore[assignment]
    rtt: Ewma = None  # type: ignore[assignment]
    last_probe_at: float = -math.inf
    compute: Dict[str, Ewma] = field(default_factory=dict)  # shard -> seconds/request
    alive: bool = True
    kind: Optional[str] = None  # "mobile" | "pc", from the worker type at connect


DEFAULT_BW = Est.prior(1.25e6)  # 10 Mbit/s prior, stale until measured
DEFAULT_RTT = Est.prior(0.05)


class TelemetryStore:
    def __init__(
        self,
        config: EstimatorConfig = EstimatorConfig(),
        clock: Callable[[], float] = None,  # type: ignore[assignment]
        on_update: Optional[Callable[[str, str, Est], None]] = None,
    ) -> None:
        import time

        self.config = config
        self.clock = clock or time.monotonic
        self.on_update = on_update
        self._workers: Dict[str, _WorkerRecord] = {}
        self._peer: Dict[Tuple[str, str], Tuple[float, LinkView]] = {}
        self._capability: Dict[Tuple[str, str], Tuple[float, bool]] = {}
        self._compression = Ewma(config.ewma_alpha)

    # ----------------------------------------------------------- records
    def _record(self, worker_id: str) -> _WorkerRecord:
        record = self._workers.get(worker_id)
        if record is None:
            alpha = self.config.ewma_alpha
            record = _WorkerRecord(worker_id, up=Ewma(alpha), down=Ewma(alpha), rtt=Ewma(alpha))
            self._workers[worker_id] = record
        return record

    def _emit(self, worker_id: str, name: str, ewma: Ewma, now: float) -> None:
        if self.on_update is not None:
            self.on_update(worker_id, name, ewma.estimate(now))

    def workers(self) -> Iterable[str]:
        return tuple(self._workers)

    def set_kind(self, worker_id: str, kind: Optional[str]) -> None:
        if kind in ("mobile", "pc"):
            self._record(worker_id).kind = kind

    def kind(self, worker_id: str) -> str:
        from .scope import infer_kind

        record = self._workers.get(worker_id)
        if record is None:
            return "unknown"
        return record.kind or infer_kind(record.device_class)

    def note_resident(self, worker_id: str, shard_id: str, n_bytes: int) -> None:
        """Record a confirmed load (the next report from the worker supersedes it)."""
        from .contract import DiskShard, SessionState

        record = self._record(worker_id)
        state = record.state or WorkerState(worker_id=worker_id, device_class=record.device_class)
        disk = list(state.disk_shards)
        if shard_id not in {d.id for d in disk}:
            disk.append(DiskShard(id=shard_id, bytes=n_bytes))
        record.state = state.model_copy(update={
            "resident_shard": shard_id, "disk_shards": disk, "session_state": SessionState.READY})

    def mark_departed(self, worker_id: str) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].alive = False

    # ----------------------------------------------------------- ingestion
    def ingest_state(self, state: WorkerState, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        record = self._record(state.worker_id)
        record.device_class = state.device_class
        record.state = state
        record.last_seen = now
        record.alive = True
        if state.rtt_to_foreman_ms is not None:
            record.rtt.update(state.rtt_to_foreman_ms / 1000.0, now)
            self._emit(state.worker_id, "rtt", record.rtt, now)
        # Worker-side observed throughput is a measurement of transfers it timed.
        if state.observed_up_bps:
            record.up.update(state.observed_up_bps, now)
            self._emit(state.worker_id, "up", record.up, now)
        if state.observed_down_bps:
            record.down.update(state.observed_down_bps, now)
            self._emit(state.worker_id, "down", record.down, now)
        for link in state.peer_links:
            if link.bw_est is None and link.rtt_ms is None:
                continue
            observed_at = now - link.age_s
            view = LinkView(
                reachable=link.reachable,
                bw=Est(link.bw_est or 0.0, 0.0, self.config.min_samples, link.age_s),
                rtt=Est((link.rtt_ms or 0.0) / 1000.0, 0.0, self.config.min_samples, link.age_s),
            )
            self._peer[(state.worker_id, link.peer)] = (observed_at, view)
            self._capability[(state.worker_id, link.peer)] = (observed_at, link.reachable)
        stats = state.stage_stats
        shard = (stats.shard_id if stats else None) or state.resident_shard
        if stats and shard and stats.compute_ms_p50 is not None:
            ewma = record.compute.setdefault(shard, Ewma(self.config.ewma_alpha))
            # p50 updates the mean; the p95-p50 gap feeds the variance via the update.
            ewma.update(stats.compute_ms_p50 / 1000.0, now)
            if stats.compute_ms_p95 is not None and stats.compute_ms_p95 > stats.compute_ms_p50:
                sigma = (stats.compute_ms_p95 - stats.compute_ms_p50) / 1000.0 / 1.645
                ewma.var = max(ewma.var, sigma * sigma)
            ewma.n = max(ewma.n, stats.samples or ewma.n)

    def ingest_legacy_heartbeat(self, worker_id: str, data: Mapping[str, Any], now: Optional[float] = None) -> WorkerState:
        """Accept today's PC (flat or nested ``performance_metrics``) and Android heartbeats."""
        state = legacy_to_worker_state(worker_id, data, self._record(worker_id).device_class)
        self.ingest_state(state, now)
        return state

    def observe_transfer(self, worker_id: str, direction: str, n_bytes: int, seconds: float,
                         now: Optional[float] = None) -> None:
        """Passive bandwidth sample. ``direction`` is "up" (worker->Foreman) or "down"."""
        if n_bytes <= 0 or seconds <= 0:
            return
        now = self.clock() if now is None else now
        record = self._record(worker_id)
        ewma = record.up if direction == "up" else record.down
        ewma.update(n_bytes / seconds, now)
        self._emit(worker_id, direction, ewma, now)

    def observe_rtt(self, worker_id: str, seconds: float, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        record = self._record(worker_id)
        record.rtt.update(seconds, now)
        self._emit(worker_id, "rtt", record.rtt, now)

    def observe_compute(self, worker_id: str, shard_id: str, seconds: float, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        ewma = self._record(worker_id).compute.setdefault(shard_id, Ewma(self.config.ewma_alpha))
        ewma.update(seconds, now)

    def observe_compression(self, raw_bytes: int, wire_bytes: int, now: Optional[float] = None) -> None:
        if raw_bytes > 0 and wire_bytes > 0:
            self._compression.update(wire_bytes / raw_bytes, self.clock() if now is None else now)

    def set_capability(self, a: str, b: str, reachable: bool, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        self._capability[(a, b)] = (now, reachable)

    def capability_age(self, a: str, b: str, now: Optional[float] = None) -> float:
        now = self.clock() if now is None else now
        entry = self._capability.get((a, b))
        return math.inf if entry is None else now - entry[0]

    # ----------------------------------------------------------- probing
    def needs_probe(self, worker_id: str, now: Optional[float] = None) -> bool:
        now = self.clock() if now is None else now
        record = self._workers.get(worker_id)
        if record is None or not record.alive:
            return False
        if now - record.last_probe_at < self.config.probe_min_interval_s:
            return False
        idle = record.state is not None and record.state.session_state.value in ("IDLE", "READY")
        stale = min(record.up.estimate(now).age_s, record.down.estimate(now).age_s) > self.config.probe_after_s
        return stale or (idle and record.down.n < self.config.min_samples)

    def mark_probed(self, worker_id: str, now: Optional[float] = None) -> None:
        self._record(worker_id).last_probe_at = self.clock() if now is None else now

    # ----------------------------------------------------------- snapshot
    def snapshot(
        self,
        catalog: ShardCatalog,
        profiles: Mapping[str, DeviceProfile],
        *,
        excluded=frozenset(),
        uploads_in_flight: Optional[Mapping[str, int]] = None,
        default_compression: float = 0.9,
        dnn_scope: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Snapshot:
        """``dnn_scope`` marks workers outside the scope ineligible (None = all eligible)."""
        from .scope import kind_allowed

        now = self.clock() if now is None else now
        workers: Dict[str, WorkerView] = {}
        for worker_id, record in self._workers.items():
            state = record.state
            workers[worker_id] = WorkerView(
                worker_id=worker_id,
                device_class=record.device_class,
                alive=record.alive,
                cpu=(state.cpu or 0.0) if state else 0.0,
                ram_free=state.ram_free if state else None,
                battery=state.battery if state else None,
                charging=state.charging if state else None,
                temp=state.temp if state else None,
                thermal=state.thermal_status if state else None,
                resident_shard=state.resident_shard if state else None,
                session_state=state.session_state.value if state else "IDLE",
                disk_shards=frozenset(d.id for d in state.disk_shards) if state else frozenset(),
                disk_used=state.disk_used if state else 0,
                disk_cap=state.disk_cap if state else None,
                up=record.up.estimate(now, DEFAULT_BW, self.config.small_sample_rel_std),
                down=record.down.estimate(now, DEFAULT_BW, self.config.small_sample_rel_std),
                rtt=record.rtt.estimate(now, DEFAULT_RTT, self.config.small_sample_rel_std),
                queue_len=(state.stage_stats.queue_len if state and state.stage_stats else 0),
                eligible=dnn_scope is None or kind_allowed(self.kind(worker_id), dnn_scope),
            )
        links = {
            pair: replace(view, bw=replace(view.bw, age_s=now - observed), rtt=replace(view.rtt, age_s=now - observed))
            for pair, (observed, view) in self._peer.items()
        }
        capability = {pair: reachable for pair, (observed, reachable) in self._capability.items()}
        profiles = self._apply_compute(profiles, now)
        compression = self._compression.mean if self._compression.n else default_compression
        return Snapshot(
            now=now,
            workers=workers,
            catalog=catalog,
            profiles=profiles,
            links=links,
            capability=capability,
            excluded=frozenset(excluded),
            uploads_in_flight=dict(uploads_in_flight or {}),
            compression_ratio=compression,
        )

    def _apply_compute(self, profiles: Mapping[str, DeviceProfile], now: float) -> Mapping[str, DeviceProfile]:
        """Fold per-worker compute observations into per-device-class profiles."""
        merged: Dict[str, Dict[str, Est]] = {}
        for record in self._workers.values():
            for shard, ewma in record.compute.items():
                est = ewma.estimate(now, None, self.config.small_sample_rel_std)
                current = merged.setdefault(record.device_class, {}).get(shard)
                if current is None or est.n > current.n:
                    merged[record.device_class][shard] = est
        result = dict(profiles)
        for device_class, by_shard in merged.items():
            base = result.get(device_class) or result.get("default")
            if base is None:
                continue
            result[device_class] = replace(
                base, device_class=device_class, compute_by_shard={**dict(base.compute_by_shard), **by_shard}
            )
        return result


def legacy_to_worker_state(worker_id: str, data: Mapping[str, Any], device_class: str = "unknown") -> WorkerState:
    """Map today's heartbeat shapes onto WorkerState without inventing values.

    ``network_speed_mbps`` (PC) is NIC *utilisation*, not capacity, so it is
    deliberately not used as a bandwidth estimate.
    """
    if "state" in data and isinstance(data["state"], Mapping):
        return WorkerState.model_validate({"worker_id": worker_id, **data["state"]})
    metrics: Mapping[str, Any] = data.get("performance_metrics") or data
    fields: Dict[str, Any] = {"worker_id": worker_id, "device_class": data.get("device_class", device_class)}
    if metrics.get("cpu_usage_percent") is not None:
        fields["cpu"] = min(1.0, float(metrics["cpu_usage_percent"]) / 100.0)
    if metrics.get("ram_available_mb") is not None:
        fields["ram_free"] = int(float(metrics["ram_available_mb"]) * 1024 * 1024)
    if metrics.get("battery_level") is not None:
        fields["battery"] = float(metrics["battery_level"])
    if metrics.get("is_charging") is not None:
        fields["charging"] = bool(metrics["is_charging"])
    if metrics.get("rtt_seconds") is not None:
        fields["rtt_to_foreman_ms"] = float(metrics["rtt_seconds"]) * 1000.0
    net = data.get("net") or {}
    res = data.get("res") or {}
    if net:
        # Android reports bits/s from timed, network-limited transfers.
        if net.get("bw_up"):
            fields["observed_up_bps"] = float(net["bw_up"]) / 8.0
        if net.get("bw_down"):
            fields["observed_down_bps"] = float(net["bw_down"]) / 8.0
    if res:
        if res.get("free_ram_mb") is not None:
            fields["ram_free"] = int(float(res["free_ram_mb"]) * 1024 * 1024)
        if res.get("battery_pct") is not None:
            fields["battery"] = float(res["battery_pct"])
        if res.get("thermal") is not None:
            fields["thermal_status"] = int(res["thermal"])
        if res.get("cpu_load") is not None:
            fields["cpu"] = min(1.0, max(0.0, float(res["cpu_load"])))
        if fields.get("device_class") in (None, "unknown"):
            fields["device_class"] = "android"
    on_disk = data.get("on_disk") or []
    if on_disk:
        fields["disk_shards"] = [
            {"id": item["id"], "bytes": int(item.get("bytes", 0))} if isinstance(item, Mapping) else {"id": str(item), "bytes": 0}
            for item in on_disk
        ]
    # residency.shard is a task id in today's Android build, so it is not trusted.
    return WorkerState.model_validate(fields)
