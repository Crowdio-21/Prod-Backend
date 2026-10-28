"""Shard registry: manifest loading/validation, file verification, holder map, LRU.

FLOP estimation method ("analytic_matmul"): 2 x multiply-accumulates of the
dense layers per token, plus 4 * seq * d for attention scores and the weighted
sum. Embedding lookups and layer norms are ignored. Per-request terms (the
classification head works on the CLS token only) are divided by seq_len.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contract import AtomSpec, Manifest, ShardRole, ShardSpec, TensorContract, TensorSpec
from .types import ShardCatalog, ShardInfo

EXPORTER_VERSION = "crowdio-onnx-export/1"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def distilbert_layer_flops_per_token(d: int = 768, ff: int = 3072, seq: int = 64) -> float:
    dense = 4 * d * d + 2 * d * ff
    return 2.0 * dense + 4.0 * seq * d


def manifest_from_legacy(directory: Path, *, hash_files: bool = True) -> Manifest:
    """Build a Manifest from today's ``partition_metadata.json`` (cell_a/b/c)."""
    directory = Path(directory)
    meta = json.loads((directory / "partition_metadata.json").read_text(encoding="utf-8"))
    n_layers = int(meta["encoder_layers_total"])
    split = int(meta["split_idx"])
    contract = meta["pipeline_contract"]
    seq = int(contract["cell_a_input"]["shape"][-1])
    hidden = int(contract["cell_b_output"]["shape"][-1])
    model_id = meta["model"]
    # Atoms: embed, L0..L{n-1}, head  (cut grid = 1 layer per atom).
    atoms = [AtomSpec(atom_index=0, name="embed")]
    atoms += [AtomSpec(atom_index=i + 1, name=f"L{i}", layers=[i]) for i in range(n_layers)]
    atoms.append(AtomSpec(atom_index=n_layers + 1, name="head"))
    layer_flops = distilbert_layer_flops_per_token(hidden, 4 * hidden, seq)
    head_flops = 2.0 * (hidden * hidden + hidden * int(meta.get("num_labels", 2))) / seq

    def spec(name: str, cell_file: str, role: ShardRole, start: int, end: int,
             first_layer: Optional[int], last_layer: Optional[int], flops: float,
             inputs: TensorSpec, outputs: TensorSpec, in_bpt: float, out_bpt: float, pinned=False) -> ShardSpec:
        path = directory / cell_file
        size = path.stat().st_size
        digest = sha256_file(path) if hash_files else "0" * 64
        return ShardSpec(
            shard_id=name, model_id=model_id, role=role, start_atom=start, end_atom=end,
            start_layer=first_layer, end_layer=last_layer, pinned=pinned, bytes=size,
            flops=flops, in_act_bytes_per_token=in_bpt, out_act_bytes_per_token=out_bpt,
            tensor_contract=TensorContract(inputs=[inputs], outputs=[outputs]), sha256=digest,
            precision="fp32", runtime_format="onnx", exporter_version=EXPORTER_VERSION,
            file_name=cell_file,
        )

    def tensor(entry: Mapping, symbolic: Sequence) -> TensorSpec:
        return TensorSpec(name=entry["name"], dtype=entry["dtype"], shape=list(symbolic))

    files = meta["files"]
    shards = [
        spec("cell_a", files["cell_a"], ShardRole.ENCODER, 0, split, 0, split - 1, layer_flops * split,
             tensor(contract["cell_a_input"], ["batch", "seq"]),
             tensor(contract["cell_a_output"], ["batch", "seq", hidden + 1]),
             8.0, 4.0 * (hidden + 1)),
        spec("cell_b", files["cell_b"], ShardRole.ENCODER, split + 1, n_layers, split, n_layers - 1,
             layer_flops * (n_layers - split),
             tensor(contract["cell_a_output"], ["batch", "seq", hidden + 1]),
             tensor(contract["cell_b_output"], ["batch", "seq", hidden]),
             4.0 * (hidden + 1), 4.0 * hidden),
        spec("cell_c", files["cell_c"], ShardRole.HEAD, n_layers + 1, n_layers + 1, None, None, head_flops,
             tensor(contract["cell_b_output"], ["batch", "seq", hidden]),
             tensor(contract["cell_c_output"], ["batch", int(meta.get("num_labels", 2))]),
             4.0 * hidden, 4.0 * int(meta.get("num_labels", 2)) / seq, pinned=True),
    ]
    return Manifest(
        model_id=model_id, cut_grid=1, atoms=atoms, shards=shards,
        default_composition=["cell_a", "cell_b", "cell_c"],
    )


def load_manifest(path: Path, *, hash_files: bool = True) -> Tuple[Manifest, Path]:
    """Load ``manifest.json`` (new format) or adapt ``partition_metadata.json``."""
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    candidate = directory / "manifest.json" if path.is_dir() else path
    if candidate.name == "manifest.json" and candidate.exists():
        return Manifest.model_validate_json(candidate.read_text(encoding="utf-8")), directory
    return manifest_from_legacy(directory, hash_files=hash_files), directory


def catalog_from_manifest(manifest: Manifest) -> ShardCatalog:
    full = next((s.shard_id for s in manifest.shards if s.role is ShardRole.FULL), None)
    return ShardCatalog(
        model_id=manifest.model_id,
        n_atoms=len(manifest.atoms),
        shards={
            s.shard_id: ShardInfo(
                shard_id=s.shard_id, start_atom=s.start_atom, end_atom=s.end_atom, bytes=s.bytes,
                flops_per_token=s.flops, in_act_bytes_per_token=s.in_act_bytes_per_token,
                out_act_bytes_per_token=s.out_act_bytes_per_token, pinned=s.pinned, role=s.role.value,
            )
            for s in manifest.shards
        },
        default_composition=tuple(manifest.default_composition),
        full_shard_id=full,
    )


@dataclass(frozen=True)
class DiskEntry:
    shard_id: str
    bytes: int
    last_used: float


def plan_evictions(
    entries: Iterable[DiskEntry],
    disk_used: int,
    disk_cap: Optional[int],
    incoming_bytes: int,
    protected: Iterable[str] = (),
) -> Optional[List[str]]:
    """Pure LRU: shards to delete so ``incoming_bytes`` fits under ``disk_cap``.

    Never evicts a protected shard (the resident shard's file, an in-flight
    fetch). Returns None when the incoming shard cannot fit at all.
    """
    if disk_cap is None:
        return []
    needed = disk_used + incoming_bytes - disk_cap
    if needed <= 0:
        return []
    keep = set(protected)
    evict: List[str] = []
    for entry in sorted(entries, key=lambda e: (e.last_used, e.shard_id)):
        if entry.shard_id in keep:
            continue
        evict.append(entry.shard_id)
        needed -= entry.bytes
        if needed <= 0:
            return evict
    return None


class ShardRegistry:
    """Holds the manifest, serves file paths and tracks per-worker disk usage."""

    def __init__(self, manifest: Manifest, directory: Path) -> None:
        self.manifest = manifest
        self.directory = Path(directory)
        self.catalog = catalog_from_manifest(manifest)
        self._last_used: Dict[Tuple[str, str], float] = {}
        self._verified: Dict[str, bool] = {}

    @classmethod
    def load(cls, path: Path, *, hash_files: bool = True) -> "ShardRegistry":
        manifest, directory = load_manifest(path, hash_files=hash_files)
        return cls(manifest, directory)

    def spec(self, shard_id: str) -> ShardSpec:
        return self.manifest.shard(shard_id)

    def path(self, shard_id: str) -> Path:
        spec = self.spec(shard_id)
        path = (self.directory / (spec.file_name or f"{shard_id}.onnx")).resolve()
        if self.directory.resolve() not in path.parents:
            raise ValueError("shard path escapes the manifest directory")
        return path

    def verify(self, shard_id: str) -> bool:
        if shard_id not in self._verified:
            path = self.path(shard_id)
            spec = self.spec(shard_id)
            self._verified[shard_id] = (
                path.is_file() and path.stat().st_size == spec.bytes and sha256_file(path) == spec.sha256
            )
        return self._verified[shard_id]

    def touch(self, worker_id: str, shard_id: str, now: float) -> None:
        self._last_used[(worker_id, shard_id)] = now

    def disk_entries(self, worker_id: str, disk_shards: Iterable[Tuple[str, int]]) -> List[DiskEntry]:
        return [
            DiskEntry(shard_id, size, self._last_used.get((worker_id, shard_id), 0.0))
            for shard_id, size in disk_shards
        ]
