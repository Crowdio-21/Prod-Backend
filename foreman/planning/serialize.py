"""Lossless (for planning) Snapshot <-> dict, so logged planner inputs can be replayed."""

from __future__ import annotations

import math
from typing import Any, Dict

from foreman.planning.cost_model import CostOptions
from foreman.planning.types import (
    DeviceProfile,
    Est,
    LinearFit,
    LinkView,
    ShardCatalog,
    ShardInfo,
    Snapshot,
    WorkerView,
)


def _num(value: float) -> Any:
    return value if math.isfinite(value) else ("inf" if value > 0 else "-inf")


def _float(value: Any) -> float:
    return float(value) if not isinstance(value, str) else float(value)


def est_to(e: Est) -> list:
    return [e.mean, e.std, e.n, _num(e.age_s)]


def est_from(v: list) -> Est:
    return Est(_float(v[0]), _float(v[1]), int(v[2]), _float(v[3]))


def fit_to(f: LinearFit) -> list:
    return [f.a, f.b, f.resid_std, f.n]


def fit_from(v: list) -> LinearFit:
    return LinearFit(*v[:3], int(v[3]))


def snapshot_to_dict(snap: Snapshot) -> Dict[str, Any]:
    return {
        "now": snap.now,
        "compression_ratio": snap.compression_ratio,
        "workers": {
            w: {**{k: getattr(v, k) for k in ("worker_id", "device_class", "alive", "cpu", "ram_free", "battery",
                                              "charging", "temp", "thermal", "resident_shard", "session_state",
                                              "disk_used", "disk_cap", "queue_len", "eligible")},
                "disk_shards": sorted(v.disk_shards), "up": est_to(v.up), "down": est_to(v.down), "rtt": est_to(v.rtt)}
            for w, v in snap.workers.items()
        },
        "catalog": {
            "model_id": snap.catalog.model_id, "n_atoms": snap.catalog.n_atoms,
            "default_composition": list(snap.catalog.default_composition),
            "full_shard_id": snap.catalog.full_shard_id,
            "shards": {s: dict(v.__dict__) for s, v in snap.catalog.shards.items()},
        },
        "profiles": {
            c: {"load": fit_to(p.load), "init": fit_to(p.init), "write": fit_to(p.write), "unload": fit_to(p.unload),
                "peak_mem": fit_to(p.peak_mem), "flops_per_s": est_to(p.flops_per_s),
                "compute_by_shard": {s: est_to(e) for s, e in p.compute_by_shard.items()}, "ref_tokens": p.ref_tokens}
            for c, p in snap.profiles.items()
        },
        "links": [[a, b, l.reachable, est_to(l.bw), est_to(l.rtt)] for (a, b), l in snap.links.items()],
        "capability": [[a, b, ok] for (a, b), ok in snap.capability.items()],
        "excluded": sorted([list(p) for p in snap.excluded]),
        "uploads_in_flight": dict(snap.uploads_in_flight),
    }


def snapshot_from_dict(d: Dict[str, Any]) -> Snapshot:
    workers = {}
    for w, v in d["workers"].items():
        fields = {k: v[k] for k in ("worker_id", "device_class", "alive", "cpu", "ram_free", "battery", "charging",
                                    "temp", "thermal", "resident_shard", "session_state", "disk_used", "disk_cap",
                                    "queue_len")}
        fields["eligible"] = v.get("eligible", True)
        workers[w] = WorkerView(**fields, disk_shards=frozenset(v["disk_shards"]), up=est_from(v["up"]),
                                down=est_from(v["down"]), rtt=est_from(v["rtt"]))
    c = d["catalog"]
    catalog = ShardCatalog(c["model_id"], c["n_atoms"], {s: ShardInfo(**v) for s, v in c["shards"].items()},
                           tuple(c["default_composition"]), c["full_shard_id"])
    profiles = {
        k: DeviceProfile(k, fit_from(p["load"]), fit_from(p["init"]), fit_from(p["write"]), fit_from(p["unload"]),
                         fit_from(p["peak_mem"]), est_from(p["flops_per_s"]),
                         {s: est_from(e) for s, e in p["compute_by_shard"].items()}, p["ref_tokens"])
        for k, p in d["profiles"].items()
    }
    return Snapshot(
        now=d["now"], workers=workers, catalog=catalog, profiles=profiles,
        links={(a, b): LinkView(ok, est_from(bw), est_from(rtt)) for a, b, ok, bw, rtt in d["links"]},
        capability={(a, b): ok for a, b, ok in d["capability"]},
        excluded=frozenset(tuple(p) for p in d["excluded"]),
        uploads_in_flight=dict(d["uploads_in_flight"]), compression_ratio=d["compression_ratio"],
    )


def opts_to_dict(opts: CostOptions) -> Dict[str, Any]:
    out = dict(opts.__dict__)
    out["critical_workers"] = sorted(opts.critical_workers)
    return out


def opts_from_dict(d: Dict[str, Any]) -> CostOptions:
    d = dict(d)
    d["critical_workers"] = frozenset(d.get("critical_workers", ()))
    return CostOptions(**d)
