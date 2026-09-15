#!/usr/bin/env python3
"""Export DistilBERT SST-2 into three ONNX pipeline partitions."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "transformers is required. Install with: pip install transformers"
    ) from exc


class CellA(nn.Module):
    def __init__(self, distilbert_model: nn.Module, split_idx: int, pad_token_id: int) -> None:
        super().__init__()
        self.embeddings = distilbert_model.embeddings
        self.layers = nn.ModuleList(list(distilbert_model.transformer.layer)[:split_idx])
        self.hidden_size = int(distilbert_model.config.dim)
        self.pad_token_id = int(pad_token_id)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = input_ids != self.pad_token_id
        hidden = self.embeddings(input_ids=input_ids)
        for layer in self.layers:
            hidden = run_distilbert_block(layer, hidden, attention_mask)
        return torch.cat([hidden, attention_mask.unsqueeze(-1).to(hidden.dtype)], dim=-1)


class CellB(nn.Module):
    def __init__(self, distilbert_model: nn.Module, split_idx: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(distilbert_model.transformer.layer)[split_idx:])
        self.hidden_size = int(distilbert_model.config.dim)

    def forward(self, features_a: torch.Tensor) -> torch.Tensor:
        hidden = features_a[..., : self.hidden_size]
        attention_mask = features_a[..., self.hidden_size] > 0.5
        for layer in self.layers:
            hidden = run_distilbert_block(layer, hidden, attention_mask)
        return hidden


class CellC(nn.Module):
    def __init__(self, sequence_model: nn.Module) -> None:
        super().__init__()
        self.pre_classifier = sequence_model.pre_classifier
        self.classifier = sequence_model.classifier
        self.dropout = sequence_model.dropout

    def forward(self, features_b: torch.Tensor) -> torch.Tensor:
        pooled = torch.relu(self.pre_classifier(features_b[:, 0]))
        return self.classifier(self.dropout(pooled))


def run_distilbert_block(layer: nn.Module, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Run one transformer block across supported transformers API variants."""
    params = inspect.signature(layer.forward).parameters
    if "hidden_states" in params:
        output = layer(hidden_states=hidden_states, attention_mask=attention_mask)
    elif "x" in params and "attn_mask" in params:
        output = layer(x=hidden_states, attn_mask=attention_mask, head_mask=None, output_attentions=False)
    else:
        output = layer(hidden_states, attention_mask)
    return output[0] if isinstance(output, tuple) else output


def load_model_and_tokenizer(model_name: str) -> Tuple[nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    return model, tokenizer


def build_partitions(model: nn.Module, split_idx: int, pad_token_id: int) -> Tuple[nn.Module, nn.Module, nn.Module, int]:
    total_layers = len(model.distilbert.transformer.layer)
    if not 0 < split_idx < total_layers:
        raise ValueError(f"split_idx must be in [1, {total_layers - 1}], got {split_idx}")
    return (
        CellA(model.distilbert, split_idx, pad_token_id).eval(),
        CellB(model.distilbert, split_idx).eval(),
        CellC(model).eval(),
        total_layers,
    )


def tokenize_text(tokenizer: Any, text: str, max_length: int) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    return {"input_ids": encoded["input_ids"].long(), "attention_mask": encoded["attention_mask"].long()}


def export_onnx(module: nn.Module, sample_input: torch.Tensor, out_path: Path, input_name: str, output_name: str, opset: int) -> None:
    dynamic_axes = {
        input_name: {0: "batch", 1: "seq"},
        output_name: {0: "batch", 1: "seq"},
    }
    if output_name == "logits":
        dynamic_axes[output_name] = {0: "batch"}
    torch.onnx.export(
        module,
        sample_input,
        out_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )


def verify_chain(model: nn.Module, cell_a: nn.Module, cell_b: nn.Module, cell_c: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        full_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        a_out = cell_a(input_ids)
        b_out = cell_b(a_out)
        c_out = cell_c(b_out)
    return {
        "pytorch_chain": {
            "max_abs_diff": float((full_logits - c_out).abs().max().item()),
            "mean_abs_diff": float((full_logits - c_out).abs().mean().item()),
            "input_ids_shape": list(input_ids.shape),
            "cell_a_output_shape": list(a_out.shape),
            "cell_b_output_shape": list(b_out.shape),
            "cell_c_output_shape": list(c_out.shape),
        },
        "tensors": {"a_out": a_out, "b_out": b_out, "full_logits": full_logits},
    }


def verify_onnx(output_dir: Path, input_ids: torch.Tensor, reference_logits: torch.Tensor) -> Dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception:
        return {"available": False, "note": "onnxruntime is not installed; skipped runtime verification"}
    providers = ["CPUExecutionProvider"]
    sessions = [
        ort.InferenceSession((output_dir / name).as_posix(), providers=providers)
        for name in ("cell_a.onnx", "cell_b.onnx", "cell_c.onnx")
    ]
    a_out = sessions[0].run(None, {"input_ids": input_ids.numpy().astype(np.int64)})[0]
    b_out = sessions[1].run(None, {"features_a": a_out.astype(np.float32)})[0]
    c_out = sessions[2].run(None, {"features_b": b_out.astype(np.float32)})[0]
    reference = reference_logits.numpy().astype(np.float32)
    return {
        "available": True,
        "max_abs_diff_vs_pytorch": float(np.max(np.abs(reference - c_out))),
        "mean_abs_diff_vs_pytorch": float(np.mean(np.abs(reference - c_out))),
        "cell_a_output_shape": list(a_out.shape),
        "cell_b_output_shape": list(b_out.shape),
        "cell_c_output_shape": list(c_out.shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--output-dir", type=Path, default=Path("tools/onnx_partitions_sentiment"))
    parser.add_argument("--split-idx", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--sample-text", default="I loved the app, the experience was smooth and fast.")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    cell_a, cell_b, cell_c, total_layers = build_partitions(model, args.split_idx, pad_token_id)
    tokenized = tokenize_text(tokenizer, args.sample_text, args.max_length)
    check = verify_chain(model, cell_a, cell_b, cell_c, tokenized["input_ids"], tokenized["attention_mask"])
    export_onnx(cell_a, tokenized["input_ids"], output_dir / "cell_a.onnx", "input_ids", "features_a", args.opset)
    export_onnx(cell_b, check["tensors"]["a_out"], output_dir / "cell_b.onnx", "features_a", "features_b", args.opset)
    export_onnx(cell_c, check["tensors"]["b_out"], output_dir / "cell_c.onnx", "features_b", "logits", args.opset)
    np.save(output_dir / "sample_input_ids.npy", tokenized["input_ids"].numpy().astype(np.int64))
    np.save(output_dir / "sample_attention_mask.npy", tokenized["attention_mask"].numpy().astype(np.int64))
    tokenizer.save_pretrained((output_dir / "tokenizer").as_posix())
    pred_idx = int(torch.argmax(check["tensors"]["full_logits"], dim=-1).item())
    id2label = getattr(model.config, "id2label", {}) or {}
    metadata = {
        "model": args.model_name,
        "opset": args.opset,
        "encoder_layers_total": total_layers,
        "split_idx": args.split_idx,
        "num_labels": int(getattr(model.config, "num_labels", 2)),
        "files": {"cell_a": "cell_a.onnx", "cell_b": "cell_b.onnx", "cell_c": "cell_c.onnx", "tokenizer_dir": "tokenizer"},
        "pipeline_contract": {
            "cell_a_input": {"name": "input_ids", "dtype": "int64", "shape": check["pytorch_chain"]["input_ids_shape"]},
            "cell_a_output": {"name": "features_a", "dtype": "float32", "shape": check["pytorch_chain"]["cell_a_output_shape"]},
            "cell_b_output": {"name": "features_b", "dtype": "float32", "shape": check["pytorch_chain"]["cell_b_output_shape"]},
            "cell_c_output": {"name": "logits", "dtype": "float32", "shape": check["pytorch_chain"]["cell_c_output_shape"]},
        },
        "sample": {"text": args.sample_text, "predicted_label": id2label.get(pred_idx, str(pred_idx)), "predicted_label_id": pred_idx, "id2label": id2label},
        "verification": {"pytorch_chain": check["pytorch_chain"]},
    }
    if not args.skip_verify:
        metadata["verification"]["onnxruntime"] = verify_onnx(output_dir, tokenized["input_ids"], check["tensors"]["full_logits"])
    (output_dir / "partition_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Export complete: {output_dir}")


if __name__ == "__main__":
    main()
