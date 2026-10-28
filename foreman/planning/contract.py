"""Shared SDK / Foreman / worker contract for reconfiguration-aware planning.

Typed, versioned and validated wire schemas. Everything here mirrors the
shared contract; additions that the contract does not list are marked
``# contract extension`` so the other agents can review them.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 2
PROTOCOL_NAME = "crowdio.reconfiguration"
PROTOCOL_VERSION = 2

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _Model(BaseModel):
    # Unknown fields are ignored so older/newer peers stay compatible.
    model_config = ConfigDict(extra="ignore", frozen=True)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ShardRole(str, Enum):
    EMBED = "embed"
    ENCODER = "encoder"
    HEAD = "head"
    FULL = "full"


class RouteMode(str, Enum):
    VIA_FOREMAN = "via_foreman"
    DIRECT = "direct"
    OPPORTUNISTIC = "opportunistic"


class ProblemType(str, Enum):
    NEXT_HOP_DOWN = "NEXT_HOP_DOWN"
    MISSING_SHARD = "MISSING_SHARD"
    WRONG_SHARD = "WRONG_SHARD"
    WRONG_EPOCH = "WRONG_EPOCH"
    OOM = "OOM"
    SLOW_STAGE = "SLOW_STAGE"
    LINK_DEGRADED = "LINK_DEGRADED"
    LOAD_FAILED = "LOAD_FAILED"


URGENT_PROBLEMS = frozenset({ProblemType.NEXT_HOP_DOWN, ProblemType.OOM, ProblemType.LOAD_FAILED})


class SessionState(str, Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    READY = "READY"
    SERVING = "SERVING"
    DRAINING = "DRAINING"
    UNLOADING = "UNLOADING"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TensorSpec(_Model):
    name: str
    dtype: str
    # Symbolic dims are strings ("batch", "seq"); fixed dims are ints.
    shape: List[Union[int, str]]


class TensorContract(_Model):
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]


class ShardSpec(_Model):
    shard_id: str
    model_id: str
    role: ShardRole
    start_atom: int = Field(ge=0)
    end_atom: int = Field(ge=0)  # inclusive
    start_layer: Optional[int] = None
    end_layer: Optional[int] = None
    # pinned: the shard's boundaries are fixed cuts, so it is always its own
    # stage and is never merged with neighbours (e.g. a classification head).
    pinned: bool = False
    bytes: int = Field(gt=0)
    # flops per token. Method recorded in flops_method (see manifest docs).
    flops: float = Field(ge=0)
    flops_method: str = "analytic_matmul"  # contract extension
    in_act_bytes_per_token: float = Field(ge=0)
    out_act_bytes_per_token: float = Field(ge=0)
    tensor_contract: TensorContract
    sha256: str
    precision: str = "fp32"
    runtime_format: str = "onnx"
    external_data_files: List[str] = Field(default_factory=list)
    exporter_version: str = "unknown"
    file_name: Optional[str] = None  # contract extension: path relative to manifest

    @field_validator("sha256")
    @classmethod
    def _check_sha(cls, value: str) -> str:
        value = value.lower()
        if not _SHA256.match(value):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def _check_range(self) -> "ShardSpec":
        if self.end_atom < self.start_atom:
            raise ValueError("end_atom must be >= start_atom")
        return self


class AtomSpec(_Model):
    atom_index: int = Field(ge=0)
    name: str
    layers: List[int] = Field(default_factory=list)


class Manifest(_Model):
    schema_version: int = SCHEMA_VERSION
    model_id: str
    cut_grid: int = Field(ge=1, description="layers per atom")
    atoms: List[AtomSpec]
    shards: List[ShardSpec]
    # The composition a fixed-DAG SDK uses today (policy B0).
    default_composition: List[str]  # contract extension

    @model_validator(mode="after")
    def _validate(self) -> "Manifest":
        n = len(self.atoms)
        if [a.atom_index for a in self.atoms] != list(range(n)):
            raise ValueError("atoms must be indexed 0..n-1 in order")
        ids = [s.shard_id for s in self.shards]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate shard_id in manifest")
        for shard in self.shards:
            if shard.model_id != self.model_id:
                raise ValueError(f"shard {shard.shard_id} belongs to another model")
            if shard.end_atom >= n:
                raise ValueError(f"shard {shard.shard_id} exceeds atom range")
        by_id = {s.shard_id: s for s in self.shards}
        expected = 0
        for shard_id in self.default_composition:
            if shard_id not in by_id:
                raise ValueError(f"default_composition references unknown shard {shard_id}")
            if by_id[shard_id].start_atom != expected:
                raise ValueError("default_composition must tile atoms contiguously")
            expected = by_id[shard_id].end_atom + 1
        if expected != n:
            raise ValueError("default_composition must cover every atom")
        return self

    def shard(self, shard_id: str) -> ShardSpec:
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise KeyError(shard_id)


# ---------------------------------------------------------------------------
# Worker state (heartbeat)
# ---------------------------------------------------------------------------


class DiskShard(_Model):
    id: str
    bytes: int = Field(ge=0)


class PeerLink(_Model):
    peer: str
    reachable: bool
    rtt_ms: Optional[float] = Field(default=None, ge=0)
    bw_est: Optional[float] = Field(default=None, ge=0, description="bytes/s")
    age_s: float = Field(default=0.0, ge=0)


class StageStats(_Model):
    compute_ms_p50: Optional[float] = Field(default=None, ge=0)
    compute_ms_p95: Optional[float] = Field(default=None, ge=0)
    queue_len: int = Field(default=0, ge=0)
    bytes_in: int = Field(default=0, ge=0)
    bytes_out: int = Field(default=0, ge=0)
    route_counts: Dict[str, int] = Field(default_factory=dict)
    samples: int = Field(default=0, ge=0)  # contract extension: n behind p50/p95
    shard_id: Optional[str] = None  # contract extension


class WorkerState(_Model):
    worker_id: str
    device_class: str = "unknown"
    proto_version: int = PROTOCOL_VERSION
    cpu: Optional[float] = Field(default=None, ge=0, le=1, description="utilisation 0..1")
    ram_free: Optional[int] = Field(default=None, ge=0, description="bytes")
    battery: Optional[float] = Field(default=None, ge=0, le=100)
    charging: Optional[bool] = None
    temp: Optional[float] = None  # Celsius
    thermal_status: Optional[int] = Field(default=None, ge=0)
    rtt_to_foreman_ms: Optional[float] = Field(default=None, ge=0)
    observed_up_bps: Optional[float] = Field(default=None, ge=0, description="bytes/s")
    observed_down_bps: Optional[float] = Field(default=None, ge=0, description="bytes/s")
    resident_shard: Optional[str] = None
    session_state: SessionState = SessionState.IDLE
    disk_shards: List[DiskShard] = Field(default_factory=list)
    disk_used: int = Field(default=0, ge=0)
    disk_cap: Optional[int] = Field(default=None, ge=0)
    peer_links: List[PeerLink] = Field(default_factory=list)
    stage_stats: Optional[StageStats] = None
    last_epoch: Optional[int] = None
    peer_endpoint: Optional[str] = None  # contract extension: "host:port" for direct/peer traffic


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class NextHop(_Model):
    worker_id: Optional[str]  # None means the Foreman (final output)
    address: Optional[str] = None  # contract extension: peer endpoint for the direct path
    allowed_routes: List[RouteMode] = Field(default_factory=lambda: [RouteMode.VIA_FOREMAN])


class PlanStage(_Model):
    stage_index: int = Field(ge=0)
    shard_id: str
    worker_id: str
    next_hop: NextHop
    backups: List[str] = Field(default_factory=list)


class Replica(_Model):  # contract extension (policy B4 data-parallel replicas)
    shard_id: str
    worker_id: str


class Plan(_Model):
    epoch: int = Field(ge=0)
    stages: List[PlanStage]
    created_at: float
    reason: str
    replicas: List[Replica] = Field(default_factory=list)  # contract extension

    @model_validator(mode="after")
    def _validate(self) -> "Plan":
        if [s.stage_index for s in self.stages] != list(range(len(self.stages))):
            raise ValueError("stage_index must be 0..K-1 in order")
        return self

    @property
    def workers(self) -> List[str]:
        return [s.worker_id for s in self.stages] + [r.worker_id for r in self.replicas]

    def assignment(self) -> tuple:
        return tuple((s.shard_id, s.worker_id) for s in self.stages)


# ---------------------------------------------------------------------------
# Messages (all travel inside the crowdio.reconfiguration envelope, v2)
# ---------------------------------------------------------------------------


class TensorPayload(_Model):
    """Serialized tensor; same fields as common.serializer.serialize_tensor."""

    model_config = ConfigDict(extra="allow", frozen=True)
    dtype: Optional[str] = None
    shape: Optional[List[int]] = None
    compression: Optional[str] = None


# Foreman -> worker
class LoadMsg(_Model):
    shard_id: str
    epoch: int
    stage: Optional[int] = None  # contract extension
    next_hop: Optional[NextHop] = None  # contract extension
    source: Optional[str] = None  # contract extension: donor id, None = Foreman


class PrefetchMsg(_Model):
    shard_id: str
    source: Optional[str] = None  # contract extension


class CancelPrefetchMsg(_Model):
    shard_id: Optional[str] = None


class RunMsg(_Model):
    stage: int
    activation: Dict[str, Any]
    epoch: int
    request_id: str


class ProbeMsg(_Model):
    size: int = Field(gt=0)
    peer: Optional[str] = None  # contract extension: capability/peer-link test target
    url: Optional[str] = None  # contract extension: Foreman probe endpoint


class PeerFetchMsg(_Model):
    shard_id: str
    donor: str
    ranges: List[List[int]] = Field(default_factory=list)


class ServeAllowedMsg(_Model):
    shard_id: str
    to: str
    rate_limit: Optional[float] = Field(default=None, gt=0, description="bytes/s")


class RouteMsg(_Model):  # contract extension: next-hop update on reroute
    epoch: int
    stage: int
    next_hop: NextHop


# Worker -> Foreman
class ReportMsg(_Model):
    state: WorkerState


class LoadTimings(_Model):
    unload_s: float = 0.0
    fetch_s: float = 0.0
    write_s: float = 0.0
    load_s: float = 0.0
    init_s: float = 0.0
    peak_mem: Optional[int] = None
    bytes_fetched: int = 0


class LoadResultMsg(_Model):
    shard_id: str
    epoch: int
    ok: bool
    timings: LoadTimings = Field(default_factory=LoadTimings)
    error: Optional[str] = None


class ProblemMsg(_Model):
    type: ProblemType
    stage: Optional[int] = None
    request_id: Optional[str] = None
    epoch: Optional[int] = None
    detail: Dict[str, Any] = Field(default_factory=dict)


class NeedMsg(_Model):
    shard_id: str
    why: str


# Worker -> worker (brokered); also worker -> Foreman on the relay route
class ActivationMsg(_Model):
    stage: int  # the stage that produced this activation
    epoch: int
    request_id: str
    tensor: Dict[str, Any]
    compute_ms: Optional[float] = None  # contract extension: stage summary


class AckMsg(_Model):
    request_id: str
    epoch: int
    stage: int


# Client -> Foreman (contract extension: planned inference submit)
class SubmitInferenceMsg(_Model):
    model_id: str
    inputs: List[Dict[str, Any]]
    objective: Optional[str] = None
    deadline_s: Optional[float] = None
    seq_len: Optional[int] = None
    batch: int = 1


MESSAGE_SCHEMAS: Dict[str, type] = {
    "load": LoadMsg,
    "prefetch": PrefetchMsg,
    "cancel_prefetch": CancelPrefetchMsg,
    "run": RunMsg,
    "probe": ProbeMsg,
    "peer_fetch": PeerFetchMsg,
    "serve_allowed": ServeAllowedMsg,
    "route": RouteMsg,
    "report": ReportMsg,
    "load_result": LoadResultMsg,
    "problem": ProblemMsg,
    "need": NeedMsg,
    "activation": ActivationMsg,
    "ack": AckMsg,
    "submit_inference": SubmitInferenceMsg,
    "probe_result": None,  # contract extension: {down_s, up_s, size} or {peer, reachable, rtt_ms, bw_est}
    "msg_ack": None,  # contract extension: transport-level ack {ack_of}
    "inference_accepted": None,  # contract extension: {job_id, n}
    "inference_result": None,  # contract extension: {job_id, results: [{request_id, tensor}]}
    "inference_error": None,  # contract extension: {job_id, error}
}


class Envelope(_Model):
    protocol: Literal["crowdio.reconfiguration"] = PROTOCOL_NAME
    version: int = PROTOCOL_VERSION
    type: str
    msg_id: Optional[str] = None
    epoch: Optional[int] = None
    data: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _known(cls, value: str) -> str:
        if value not in MESSAGE_SCHEMAS:
            raise ValueError(f"unknown message type {value!r}")
        return value

    def body(self) -> Any:
        schema = MESSAGE_SCHEMAS[self.type]
        return dict(self.data) if schema is None else schema.model_validate(self.data)


def is_v2_envelope(raw: Any) -> bool:
    return (
        isinstance(raw, dict)
        and raw.get("protocol") == PROTOCOL_NAME
        and raw.get("version") == PROTOCOL_VERSION
    )


def make_envelope(msg_type: str, body: Any, *, msg_id: Optional[str] = None, epoch: Optional[int] = None) -> dict:
    data = body.model_dump(mode="json") if isinstance(body, BaseModel) else dict(body)
    return Envelope(type=msg_type, msg_id=msg_id, epoch=epoch, data=data).model_dump(mode="json")
