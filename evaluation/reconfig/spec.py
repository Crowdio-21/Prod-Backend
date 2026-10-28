"""Simulation spec loader (YAML): manifest, true device profiles, workers, peer links.

profiles:
  android: {load: [a, b], init: [a, b], write: [a, b], unload: [a, b], peak_mult: 2.0, flops_per_s: 5e9,
            compute: {cell_a: 0.30, cell_b: 0.25}}     # optional measured seconds per request
workers:
  - {id: p1, class: android, bw: 2.5e6, rtt: 0.03, resident: cell_a, disk: [cell_a], ram: 1.5e9}
peer_links: [[p1, p2, 2.0e7, 0.005]]   # directed, bytes/s and seconds
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from foreman.planning.registry import catalog_from_manifest, load_manifest
from foreman.planning.types import DeviceProfile, Est, LinearFit

from .sim import SimSpec, SimWorker


def _fit(value) -> LinearFit:
    a, b = (value, 0.0) if isinstance(value, (int, float)) else value
    return LinearFit(float(a), float(b), 0.0, 100)


def profile_from(name: str, data: Mapping[str, Any], seq_tokens: int) -> DeviceProfile:
    return DeviceProfile(
        device_class=name,
        load=_fit(data["load"]), init=_fit(data["init"]), write=_fit(data["write"]), unload=_fit(data["unload"]),
        peak_mem=LinearFit(float(data.get("peak_base", 0.0)), float(data.get("peak_mult", 2.0)), 0.0, 100),
        flops_per_s=Est.known(float(data.get("flops_per_s", 5e9))),
        compute_by_shard={k: Est.known(float(v)) for k, v in (data.get("compute") or {}).items()},
        ref_tokens=int(data.get("ref_tokens", seq_tokens)),
    )


def load_spec(path: str | Path, *, seq_tokens: int = 64) -> SimSpec:
    import yaml

    path = Path(path)
    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest_ref = Path(data["manifest"])
    if not manifest_ref.is_absolute():
        root = Path(__file__).resolve().parents[2]
        manifest_ref = (root / manifest_ref) if (root / manifest_ref).exists() else (path.parent / manifest_ref)
    manifest, _ = load_manifest(manifest_ref, hash_files=False)
    catalog = catalog_from_manifest(manifest)
    profiles = {name: profile_from(name, p, seq_tokens) for name, p in data["profiles"].items()}
    workers = [
        SimWorker(worker_id=w["id"], device_class=w["class"], bw=float(w["bw"]), rtt=float(w.get("rtt", 0.02)),
                  cpu=float(w.get("cpu", 0.0)), resident=w.get("resident"), disk=set(w.get("disk", [])),
                  ram=int(float(w["ram"])) if w.get("ram") is not None else None,
                  battery=w.get("battery", 90.0))
        for w in data["workers"]
    ]
    peer_links = {(a, b): (float(bw), float(rtt)) for a, b, bw, rtt in data.get("peer_links", [])}
    return SimSpec(catalog=catalog, workers=workers, profiles=profiles, peer_links=peer_links)
