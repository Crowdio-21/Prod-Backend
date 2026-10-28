"""In-memory fake workers that speak the v2 protocol, for Foreman integration tests.

Each shard adds (number of atoms it covers) to ``value``, so a correct pipeline
returns input + n_atoms. Workers enforce single residency, retain sent
activations until acked, de-duplicate by (request_id, epoch, stage), and
record any protocol violation the Foreman causes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from foreman.core.plan_coordinator import PlanCoordinator
from foreman.planning.config import RunConfig
from foreman.planning.contract import (
    AtomSpec,
    Envelope,
    Manifest,
    ShardSpec,
    SubmitInferenceMsg,
    TensorContract,
    TensorSpec,
    make_envelope,
)
from foreman.planning.registry import ShardRegistry

MB = 1_000_000
SEQ = 64


def make_manifest(b_bytes: int = 50 * MB, act_bpt: float = 3000.0) -> Manifest:
    tensor = TensorContract(inputs=[TensorSpec(name="x", dtype="float32", shape=["batch", "seq", 768])],
                            outputs=[TensorSpec(name="y", dtype="float32", shape=["batch", "seq", 768])])

    def spec(shard_id, start, end, size, pinned=False, role="encoder", out=act_bpt):
        return ShardSpec(shard_id=shard_id, model_id="m", role=role, start_atom=start, end_atom=end, pinned=pinned,
                         bytes=size, flops=1e6, in_act_bytes_per_token=act_bpt, out_act_bytes_per_token=out,
                         tensor_contract=tensor, sha256="0" * 64)

    return Manifest(
        model_id="m", cut_grid=1,
        atoms=[AtomSpec(atom_index=i, name=n) for i, n in enumerate(["embed", "L0", "L1", "head"])],
        shards=[spec("a", 0, 0, 50 * MB), spec("b", 1, 2, b_bytes), spec("c", 3, 3, 5 * MB, True, "head", 8.0 / SEQ)],
        default_composition=["a", "b", "c"],
    )


@dataclass
class FakeClock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


class FakeNetwork:
    def __init__(self) -> None:
        self.coordinator: Optional[PlanCoordinator] = None
        self.workers: Dict[str, "FakeWorker"] = {}
        self.blocked: Set[Tuple[str, str]] = set()  # direct peer path a -> b dropped
        self.drop_acks: Set[Tuple[str, str]] = set()  # peer acks a -> b dropped
        self.blackhole: Set[Tuple[str, str]] = set()  # a -> b silently lost (send "succeeds")
        self.tasks: Set[asyncio.Task] = set()
        self.foreman_bytes = 0

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    # Transport interface used by the coordinator
    async def send_worker(self, worker_id: str, message: dict) -> bool:
        worker = self.workers.get(worker_id)
        if worker is None or not worker.connected:
            return False
        self._spawn(worker.receive("foreman", message))
        return True

    async def send_client(self, client: Any, message: dict) -> bool:
        client.append(message)
        return True

    def to_foreman(self, worker_id: str, message: dict) -> None:
        self.foreman_bytes += len(str(message))
        self._spawn(self.coordinator.handle_envelope(None, message, worker_id=worker_id))

    def to_peer(self, src: str, dst: str, message: dict) -> bool:
        if message["type"] == "ack" and (src, dst) in self.drop_acks:
            return True  # silently lost
        if (src, dst) in self.blackhole:
            return True
        if (src, dst) in self.blocked or dst not in self.workers or not self.workers[dst].connected:
            return False
        self._spawn(self.workers[dst].receive(src, message))
        return True

    async def settle(self, rounds: int = 200) -> None:
        for _ in range(rounds):
            await asyncio.sleep(0)
            if not self.tasks:
                await asyncio.sleep(0)
                if not self.tasks:
                    return
        await asyncio.sleep(0.01)


@dataclass
class FakeWorker:
    worker_id: str
    net: FakeNetwork
    device_class: str = "phone"
    bw: float = 10 * MB
    resident: Optional[str] = None
    disk: Set[str] = field(default_factory=set)
    timings: Dict[str, float] = field(default_factory=lambda: {"load": 1.0, "init": 0.5, "write": 0.2, "unload": 0.3})
    compute_ms: float = 20.0
    ack_timeout_s: float = 0.05
    stall: bool = False
    fail_load: bool = False
    connected: bool = True
    peer_links: Dict[str, Tuple[bool, float, float]] = field(default_factory=dict)  # peer -> (ok, rtt_ms, bw)
    # bookkeeping
    routes: Dict[int, dict] = field(default_factory=dict)
    retained: Dict[Tuple[str, int, int], dict] = field(default_factory=dict)
    seen_inputs: Set[Tuple[str, int, int]] = field(default_factory=set)
    in_flight: Dict[Tuple[str, int], int] = field(default_factory=dict)  # (request, epoch) -> stage
    stalled: List[dict] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    duplicates: int = 0
    loads: List[Tuple[str, int]] = field(default_factory=list)
    prefetches: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    computed: int = 0

    def __post_init__(self) -> None:
        self.net.workers[self.worker_id] = self
        if self.resident:
            self.disk.add(self.resident)

    # ------------------------------------------------------------ helpers
    def _send_foreman(self, msg_type: str, data: dict, epoch: Optional[int] = None) -> None:
        self.net.to_foreman(self.worker_id, make_envelope(msg_type, data, epoch=epoch))

    def report(self, *, bw: Optional[float] = None, queue_len: int = 0) -> None:
        if bw is not None:
            self.bw = bw
        catalog = self.net.coordinator.registry.catalog
        state = {
            "worker_id": self.worker_id, "device_class": self.device_class,
            "resident_shard": self.resident,
            "session_state": "SERVING" if self.in_flight else ("READY" if self.resident else "IDLE"),
            "disk_shards": [{"id": s, "bytes": catalog.shards[s].bytes} for s in sorted(self.disk)],
            "observed_up_bps": self.bw, "observed_down_bps": self.bw, "rtt_to_foreman_ms": 10.0,
            "peer_links": [{"peer": p, "reachable": ok and (self.worker_id, p) not in self.net.blocked
                            and (self.worker_id, p) not in self.net.blackhole,
                            "rtt_ms": rtt, "bw_est": bw, "age_s": 0}
                           for p, (ok, rtt, bw) in self.peer_links.items()],
            "stage_stats": {"compute_ms_p50": self.compute_ms, "compute_ms_p95": self.compute_ms,
                            "queue_len": queue_len or len(self.in_flight), "samples": max(1, self.computed),
                            "shard_id": self.resident},
            "peer_endpoint": f"{self.worker_id}:7000",
        }
        self._send_foreman("report", {"state": state})

    def disconnect(self) -> None:
        self.connected = False

    # ------------------------------------------------------------ inbound
    async def receive(self, sender: str, raw: dict) -> None:
        if not self.connected:
            return
        envelope = Envelope.model_validate(raw)
        body = envelope.body()
        kind = envelope.type
        if sender == "foreman" and envelope.msg_id and kind not in ("load", "run", "ack"):
            self._send_foreman("msg_ack", {"ack_of": envelope.msg_id})
        if kind == "load":
            await self._load(body, envelope.msg_id)
        elif kind == "run":
            await self._run(body.request_id, body.epoch, body.stage, body.activation)
        elif kind == "activation":  # from a peer (direct route)
            await self.net.workers[sender].receive_ack_later(self.worker_id, body)
            await self._run(body.request_id, body.epoch, body.stage + 1, body.tensor)
        elif kind == "ack":
            self.retained.pop((body.request_id, body.epoch, body.stage), None)
        elif kind == "route":
            self.routes[body.stage] = body.next_hop.model_dump(mode="json")
            await self._resend(body.stage)
        elif kind == "prefetch":
            self.prefetches.append(body.shard_id)
            self.disk.add(body.shard_id)
        elif kind == "cancel_prefetch":
            pass
        elif kind == "probe":
            if body.peer:
                ok = (self.worker_id, body.peer) not in self.net.blocked and body.peer in self.peer_links
                self._send_foreman("probe_result", {"peer": body.peer, "reachable": ok, "size": body.size})
            else:
                self._send_foreman("probe_result", {"size": body.size, "down_s": body.size / self.bw,
                                                    "up_s": body.size / self.bw})

    async def receive_ack_later(self, receiver: str, body) -> None:
        """The receiver acks a direct activation back to the producer (may be dropped)."""
        ack = make_envelope("ack", {"request_id": body.request_id, "epoch": body.epoch, "stage": body.stage})
        self.net.to_peer(receiver, self.worker_id, ack)

    async def _load(self, body, msg_id) -> None:
        if self.in_flight:
            self.violations.append(f"load {body.shard_id} while serving {sorted(self.in_flight)}")
        self.loads.append((body.shard_id, body.epoch))
        if self.fail_load:
            self._send_foreman("load_result", {"shard_id": body.shard_id, "epoch": body.epoch, "ok": False,
                                               "error": "LOAD_FAILED"})
            return
        timings = {"unload_s": self.timings["unload"] if self.resident else 0.0}
        self.resident = None  # single residency: unload first
        size = self.net.coordinator.registry.catalog.shards[body.shard_id].bytes
        if body.shard_id not in self.disk:
            timings["fetch_s"] = size / self.bw
            timings["write_s"] = self.timings["write"]
            timings["bytes_fetched"] = size
            self.disk.add(body.shard_id)
        timings["load_s"] = self.timings["load"]
        timings["init_s"] = self.timings["init"]
        self.resident = body.shard_id
        self._send_foreman("load_result", {"shard_id": body.shard_id, "epoch": body.epoch, "ok": True,
                                           "timings": timings})

    async def _run(self, request_id: str, epoch: int, stage: int, activation: dict) -> None:
        key = (request_id, epoch, stage)
        if key in self.seen_inputs:
            self.duplicates += 1
            return
        self.seen_inputs.add(key)
        if self.resident is None:
            self.problems.append("MISSING_SHARD")
            self._send_foreman("problem", {"type": "MISSING_SHARD", "stage": stage, "request_id": request_id,
                                           "epoch": epoch})
            return
        if self.stall:
            self.in_flight[(request_id, epoch)] = stage
            self.stalled.append({"request_id": request_id, "epoch": epoch, "stage": stage})
            return
        self.in_flight[(request_id, epoch)] = stage
        await asyncio.sleep(0)
        shard = self.net.coordinator.registry.catalog.shards[self.resident]
        out = {"value": activation["value"] + (shard.end_atom - shard.start_atom + 1)}
        self.computed += 1
        self.in_flight.pop((request_id, epoch), None)
        message = make_envelope("activation", {"stage": stage, "epoch": epoch, "request_id": request_id,
                                               "tensor": out, "compute_ms": self.compute_ms}, epoch=epoch)
        self.retained[(request_id, epoch, stage)] = message
        self._deliver(stage, message)
        asyncio.get_running_loop().create_task(self._ack_watch((request_id, epoch, stage), stage))

    def _deliver(self, stage: int, message: dict) -> None:
        hop = self.routes.get(stage, {"worker_id": None, "allowed_routes": ["via_foreman"]})
        if hop.get("worker_id") and "direct" in hop.get("allowed_routes", []) and \
                hop["allowed_routes"][0] == "direct":
            if self.net.to_peer(self.worker_id, hop["worker_id"], message):
                return
        self.net.to_foreman(self.worker_id, message)

    async def _ack_watch(self, key, stage) -> None:
        await asyncio.sleep(self.ack_timeout_s)
        if key in self.retained and self.connected:
            self.problems.append("NEXT_HOP_DOWN")
            self._send_foreman("problem", {"type": "NEXT_HOP_DOWN", "stage": stage, "request_id": key[0],
                                           "epoch": key[1]})

    async def _resend(self, stage: int) -> None:
        for key, message in list(self.retained.items()):
            if key[2] == stage:
                self._deliver(stage, message)


def make_coordinator(net: FakeNetwork, clock: FakeClock, *, policy: str = "S-aware-placement",
                     flags: Optional[dict] = None, manifest: Optional[Manifest] = None, **config) -> PlanCoordinator:
    defaults = dict(policy=policy, flags=flags or {"adaptive_planning": True},
                    estimator={"min_samples": 1, "max_age_s": 1e6, "age_std_growth": 0.0},
                    controller={"persist_k": 1, "cooldown_s": 0.0, "margin_abs_s": 0.5, "margin_frac": 0.0},
                    cost={"seq_len": SEQ, "compression_ratio": 1.0, "runtime_overhead_bytes": 0})
    defaults.update(config)
    run_config = RunConfig.from_mapping(defaults)
    registry = ShardRegistry(manifest or make_manifest(), ".")
    coordinator = PlanCoordinator(run_config, registry, net, clock=clock, msg_timeout_s=0.2,
                                  request_timeout_s=1e9)
    net.coordinator = coordinator
    # Seed phase fits so estimates are not priors (two shard sizes, exact timings).
    for size in (5 * MB, 50 * MB):
        coordinator.learner.observe("phone", size, load=1.0, init=0.5, write=0.2, unload=0.3, peak_mem=size)
    return coordinator


async def submit(net: FakeNetwork, n: int, client: Optional[list] = None, start: int = 0) -> list:
    client = client if client is not None else []
    body = SubmitInferenceMsg(model_id="m", inputs=[{"value": start + i} for i in range(n)])
    await net.coordinator.handle_envelope(client, make_envelope("submit_inference", body))
    return client


def results(client: list) -> List[dict]:
    out = []
    for message in client:
        if message["type"] == "inference_result":
            out.extend(message["data"]["results"])
    return out
