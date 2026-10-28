#!/usr/bin/env python3
"""Export DistilBERT SST-2 as re-cuttable ONNX shards plus a contract ``manifest.json``.

Atoms: ``embed``, one atom per ``--cut-grid`` encoder layers, ``head``. Exported:
* every contiguous encoder range, with the embedding folded into ranges that start
  at the first encoder atom (``embed`` alone is also exported),
* the classification head (pinned: always its own stage),
* the full model (``full``), always, so B4 can replicate where it fits.

Encoder shards exchange ``[batch, seq, hidden + 1]`` (hidden state + attention
mask packed as the last channel, as cell_a does today). The manifest's
``default_composition`` reproduces today's fixed cut (``--split-idx``).

Usage:
  python tests/dnn_partitioning/sentiment_onnx_pipeline/export_manifest.py \
      --output-dir tools/onnx_shards_sentiment --cut-grid 1 --split-idx 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_partitions import load_model_and_tokenizer, run_distilbert_block, tokenize_text  # noqa: E402
from foreman.planning.contract import (  # noqa: E402
    AtomSpec, Manifest, ShardRole, ShardSpec, TensorContract, TensorSpec,
)
from foreman.planning.registry import EXPORTER_VERSION, distilbert_layer_flops_per_token, sha256_file  # noqa: E402


class EncoderRange(nn.Module):
    """Optional embedding + encoder layers [lo, hi). Packs the mask as an extra channel."""

    def __init__(self, distilbert, lo: int, hi: int, with_embed: bool, pad_token_id: int):
        super().__init__()
        self.with_embed = with_embed
        self.embeddings = distilbert.embeddings if with_embed else None
        self.layers = nn.ModuleList(list(distilbert.transformer.layer)[lo:hi])
        self.hidden = int(distilbert.config.dim)
        self.pad = int(pad_token_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_embed:
            mask = x != self.pad
            hidden = self.embeddings(input_ids=x)
        else:
            hidden = x[..., : self.hidden]
            mask = x[..., self.hidden] > 0.5
        for layer in self.layers:
            hidden = run_distilbert_block(layer, hidden, mask)
        return torch.cat([hidden, mask.unsqueeze(-1).to(hidden.dtype)], dim=-1)


class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pre_classifier = model.pre_classifier
        self.classifier = model.classifier
        self.dropout = model.dropout
        self.hidden = int(model.config.dim)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        pooled = torch.relu(self.pre_classifier(packed[:, 0, : self.hidden]))
        return self.classifier(self.dropout(pooled))


class Full(nn.Module):
    def __init__(self, model, pad_token_id: int):
        super().__init__()
        n = len(model.distilbert.transformer.layer)
        self.encoder = EncoderRange(model.distilbert, 0, n, True, pad_token_id)
        self.head = Head(model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(ids))


def export(module, sample, path: Path, in_name: str, out_name: str, opset: int, out_seq: bool = True) -> None:
    axes = {in_name: {0: "batch", 1: "seq"}, out_name: {0: "batch", 1: "seq"} if out_seq else {0: "batch"}}
    torch.onnx.export(module, sample, path.as_posix(), export_params=True, opset_version=opset,
                      do_constant_folding=True, input_names=[in_name], output_names=[out_name],
                      dynamic_axes=axes, dynamo=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tools" / "onnx_shards_sentiment")
    parser.add_argument("--cut-grid", type=int, default=1, help="encoder layers per atom")
    parser.add_argument("--split-idx", type=int, default=3, help="today's fixed cut (default composition)")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    n_layers = len(model.distilbert.transformer.layer)
    if n_layers % args.cut_grid or args.split_idx % args.cut_grid:
        raise SystemExit("cut grid must divide the layer count and the split index")
    hidden = int(model.config.dim)
    seq = args.max_length
    labels = int(model.config.num_labels)
    groups = n_layers // args.cut_grid
    atoms = [AtomSpec(atom_index=0, name="embed")]
    atoms += [AtomSpec(atom_index=g + 1, name=f"L{g * args.cut_grid}-{(g + 1) * args.cut_grid - 1}",
                       layers=list(range(g * args.cut_grid, (g + 1) * args.cut_grid))) for g in range(groups)]
    head_atom = groups + 1
    atoms.append(AtomSpec(atom_index=head_atom, name="head"))
    layer_flops = distilbert_layer_flops_per_token(hidden, 4 * hidden, seq)
    ids = tokenize_text(tokenizer, "I loved the app, the experience was smooth and fast.", seq)["input_ids"]
    packed = torch.cat([torch.zeros(1, seq, hidden), torch.ones(1, seq, 1)], dim=-1)
    ids_spec = TensorSpec(name="input_ids", dtype="int64", shape=["batch", "seq"])
    packed_in = TensorSpec(name="features_in", dtype="float32", shape=["batch", "seq", hidden + 1])
    packed_out = TensorSpec(name="features_out", dtype="float32", shape=["batch", "seq", hidden + 1])
    logits = TensorSpec(name="logits", dtype="float32", shape=["batch", labels])
    shards = []

    def add(shard_id, role, start, end, first_layer, last_layer, flops, module, sample, ins, outs, in_bpt, out_bpt,
            pinned=False):
        path = out / f"{shard_id}.onnx"
        export(module.eval(), sample, path, ins.name, outs.name, args.opset, out_seq=outs.name != "logits")
        shards.append(ShardSpec(
            shard_id=shard_id, model_id=args.model_name, role=role, start_atom=start, end_atom=end,
            start_layer=first_layer, end_layer=last_layer, pinned=pinned, bytes=path.stat().st_size, flops=flops,
            in_act_bytes_per_token=in_bpt, out_act_bytes_per_token=out_bpt,
            tensor_contract=TensorContract(inputs=[ins], outputs=[outs]), sha256=sha256_file(path),
            precision="fp32", runtime_format="onnx", exporter_version=EXPORTER_VERSION, file_name=path.name))
        print(f"exported {shard_id}: atoms {start}-{end} ({path.stat().st_size / 1e6:.1f} MB)")

    packed_bpt = 4.0 * (hidden + 1)
    for end_group in range(0, groups + 1):  # embed + groups [0, end_group)
        hi = end_group * args.cut_grid
        add(f"s{0}_{end_group}", ShardRole.EMBED if hi == 0 else ShardRole.ENCODER, 0, end_group,
            0 if hi else None, hi - 1 if hi else None, layer_flops * hi,
            EncoderRange(model.distilbert, 0, hi, True, pad), ids, ids_spec, packed_out, 8.0, packed_bpt)
    for g0 in range(groups):
        for g1 in range(g0 + 1, groups + 1):
            lo, hi = g0 * args.cut_grid, g1 * args.cut_grid
            add(f"s{g0 + 1}_{g1}", ShardRole.ENCODER, g0 + 1, g1, lo, hi - 1, layer_flops * (hi - lo),
                EncoderRange(model.distilbert, lo, hi, False, pad), packed, packed_in, packed_out, packed_bpt,
                packed_bpt)
    head_flops = 2.0 * (hidden * hidden + hidden * labels) / seq
    add("head", ShardRole.HEAD, head_atom, head_atom, None, None, head_flops, Head(model), packed, packed_in, logits,
        packed_bpt, 4.0 * labels / seq, pinned=True)
    add("full", ShardRole.FULL, 0, head_atom, 0, n_layers - 1, layer_flops * n_layers + head_flops,
        Full(model, pad), ids, ids_spec, logits, 8.0, 4.0 * labels / seq)
    split_group = args.split_idx // args.cut_grid
    manifest = Manifest(model_id=args.model_name, cut_grid=args.cut_grid, atoms=atoms, shards=shards,
                        default_composition=[f"s0_{split_group}", f"s{split_group + 1}_{groups}", "head"])
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tokenizer.save_pretrained((out / "tokenizer").as_posix())
    print(f"manifest: {out / 'manifest.json'} ({len(shards)} shards)")

    if not args.skip_verify:
        import onnxruntime as ort

        with torch.no_grad():
            reference = model(input_ids=ids, attention_mask=(ids != pad).long()).logits.numpy()

        def run(name, x):
            session = ort.InferenceSession((out / f"{name}.onnx").as_posix(), providers=["CPUExecutionProvider"])
            return session.run(None, {session.get_inputs()[0].name: x})[0]

        chain = ids.numpy().astype(np.int64)
        for name in manifest.default_composition:
            chain = run(name, chain)
        full = run("full", ids.numpy().astype(np.int64))
        report = {"default_chain_max_abs_diff": float(np.max(np.abs(chain - reference))),
                  "full_max_abs_diff": float(np.max(np.abs(full - reference)))}
        (out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
