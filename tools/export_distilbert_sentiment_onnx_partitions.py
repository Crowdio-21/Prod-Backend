#!/usr/bin/env python3
"""
Export a pretrained sentiment model (DistilBERT SST-2) into 3 ONNX partitions:
- cell_a.onnx
- cell_b.onnx
- cell_c.onnx

Partition contract:
- cell_a input: input_ids (int64, [batch, seq_len])
- cell_a output: features_a (float32, [batch, seq_len, hidden+1])
    The last channel carries attention-mask values so stage B can reconstruct masking.
- cell_b input: features_a
- cell_b output: features_b (float32, [batch, seq_len, hidden])
- cell_c input: features_b
- cell_c output: logits (float32, [batch, num_labels])

Tokenizer files and metadata are also exported to the output directory.
"""

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
    def __init__(
        self, distilbert_model: nn.Module, split_idx: int, pad_token_id: int
    ) -> None:
        super().__init__()
        self.embeddings = distilbert_model.embeddings
        self.layers = nn.ModuleList(
            list(distilbert_model.transformer.layer)[:split_idx]
        )
        self.hidden_size = int(distilbert_model.config.dim)
        self.pad_token_id = int(pad_token_id)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = input_ids != self.pad_token_id

        hidden = self.embeddings(input_ids=input_ids)
        for layer in self.layers:
            hidden = run_distilbert_block(layer, hidden, attention_mask)

        packed = torch.cat(
            [hidden, attention_mask.unsqueeze(-1).to(hidden.dtype)],
            dim=-1,
        )
        return packed


class CellB(nn.Module):
    def __init__(self, distilbert_model: nn.Module, split_idx: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            list(distilbert_model.transformer.layer)[split_idx:]
        )
        self.hidden_size = int(distilbert_model.config.dim)

    def forward(self, features_a: torch.Tensor) -> torch.Tensor:
        hidden = features_a[..., : self.hidden_size]
        packed_mask = features_a[..., self.hidden_size]
        attention_mask = packed_mask > 0.5

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
        pooled = features_b[:, 0]
        pooled = self.pre_classifier(pooled)
        pooled = torch.relu(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def run_distilbert_block(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run one DistilBERT transformer block across transformers API variants."""
    params = inspect.signature(layer.forward).parameters

    if "hidden_states" in params:
        output = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
    elif "x" in params and "attn_mask" in params:
        output = layer(
            x=hidden_states,
            attn_mask=attention_mask,
            head_mask=None,
            output_attentions=False,
        )
    else:
        # Fallback for unknown variants: assume first arg is hidden states,
        # second arg is attention mask.
        output = layer(hidden_states, attention_mask)

    if isinstance(output, tuple):
        return output[0]
    return output


def load_model_and_tokenizer(model_name: str) -> Tuple[nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def build_partitions(
    model: nn.Module, split_idx: int, pad_token_id: int
) -> Tuple[nn.Module, nn.Module, nn.Module, int]:
    all_layers = list(model.distilbert.transformer.layer)
    total_layers = len(all_layers)

    if split_idx <= 0 or split_idx >= total_layers:
        raise ValueError(
            f"split_idx must be in [1, {total_layers - 1}], got {split_idx}"
        )

    cell_a = CellA(
        distilbert_model=model.distilbert,
        split_idx=split_idx,
        pad_token_id=pad_token_id,
    ).eval()
    cell_b = CellB(distilbert_model=model.distilbert, split_idx=split_idx).eval()
    cell_c = CellC(sequence_model=model).eval()

    return cell_a, cell_b, cell_c, total_layers


def tokenize_text(
    tokenizer: Any, text: str, max_length: int
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": encoded["input_ids"].long(),
        "attention_mask": encoded["attention_mask"].long(),
    }


def export_onnx(
    module: nn.Module,
    sample_input: torch.Tensor,
    out_path: Path,
    input_name: str,
    output_name: str,
    opset: int,
) -> None:
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
    )


def verify_chain_with_pytorch(
    model: nn.Module,
    cell_a: nn.Module,
    cell_b: nn.Module,
    cell_c: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, Any]:
    with torch.no_grad():
        full_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        a_out = cell_a(input_ids)
        b_out = cell_b(a_out)
        c_out = cell_c(b_out)

    max_abs_diff = float((full_logits - c_out).abs().max().item())
    mean_abs_diff = float((full_logits - c_out).abs().mean().item())

    return {
        "pytorch_chain_check": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "cell_a_output_shape": list(a_out.shape),
            "cell_b_output_shape": list(b_out.shape),
            "cell_c_output_shape": list(c_out.shape),
        },
        "tensors": {
            "a_out": a_out,
            "b_out": b_out,
            "c_out": c_out,
            "full_logits": full_logits,
        },
    }


def verify_chain_with_onnxruntime(
    output_dir: Path,
    input_ids: torch.Tensor,
    reference_logits: torch.Tensor,
) -> Dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception:
        return {
            "onnxruntime_check": {
                "available": False,
                "note": "onnxruntime not installed; skipping ONNX runtime verification",
            }
        }

    providers = ["CPUExecutionProvider"]

    sess_a = ort.InferenceSession(
        (output_dir / "cell_a.onnx").as_posix(), providers=providers
    )
    sess_b = ort.InferenceSession(
        (output_dir / "cell_b.onnx").as_posix(), providers=providers
    )
    sess_c = ort.InferenceSession(
        (output_dir / "cell_c.onnx").as_posix(), providers=providers
    )

    ids_np = input_ids.detach().cpu().numpy().astype(np.int64)
    a_out = sess_a.run(None, {"input_ids": ids_np})[0]
    b_out = sess_b.run(None, {"features_a": a_out.astype(np.float32)})[0]
    c_out = sess_c.run(None, {"features_b": b_out.astype(np.float32)})[0]

    ref = reference_logits.detach().cpu().numpy().astype(np.float32)
    max_abs_diff = float(np.max(np.abs(ref - c_out)))
    mean_abs_diff = float(np.mean(np.abs(ref - c_out)))

    return {
        "onnxruntime_check": {
            "available": True,
            "max_abs_diff_vs_pytorch": max_abs_diff,
            "mean_abs_diff_vs_pytorch": mean_abs_diff,
            "cell_a_output_shape": list(a_out.shape),
            "cell_b_output_shape": list(b_out.shape),
            "cell_c_output_shape": list(c_out.shape),
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DistilBERT sentiment model into cell_a/cell_b/cell_c ONNX partitions"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="HuggingFace model name for sequence classification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("onnx_partitions_sentiment_distilbert"),
        help="Directory where ONNX partitions and metadata will be written",
    )
    parser.add_argument(
        "--split-idx",
        type=int,
        default=3,
        help="DistilBERT encoder split index between cell_a and cell_b",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Token sequence length for sample input",
    )
    parser.add_argument(
        "--sample-text",
        type=str,
        default="I loved the app, the experience was smooth and fast.",
        help="Text used to generate a sample token input",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX Runtime verification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )

    cell_a, cell_b, cell_c, total_layers = build_partitions(
        model=model,
        split_idx=args.split_idx,
        pad_token_id=pad_token_id,
    )

    tokenized = tokenize_text(tokenizer, args.sample_text, max_length=args.max_length)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    check = verify_chain_with_pytorch(
        model=model,
        cell_a=cell_a,
        cell_b=cell_b,
        cell_c=cell_c,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    a_tensor = check["tensors"]["a_out"]
    b_tensor = check["tensors"]["b_out"]

    export_onnx(
        module=cell_a,
        sample_input=input_ids,
        out_path=output_dir / "cell_a.onnx",
        input_name="input_ids",
        output_name="features_a",
        opset=args.opset,
    )
    export_onnx(
        module=cell_b,
        sample_input=a_tensor,
        out_path=output_dir / "cell_b.onnx",
        input_name="features_a",
        output_name="features_b",
        opset=args.opset,
    )
    export_onnx(
        module=cell_c,
        sample_input=b_tensor,
        out_path=output_dir / "cell_c.onnx",
        input_name="features_b",
        output_name="logits",
        opset=args.opset,
    )

    np.save(
        output_dir / "sample_input_ids.npy",
        input_ids.detach().cpu().numpy().astype(np.int64),
    )
    np.save(
        output_dir / "sample_attention_mask.npy",
        attention_mask.detach().cpu().numpy().astype(np.int64),
    )

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir.as_posix())

    with torch.no_grad():
        full_logits = check["tensors"]["full_logits"]
        pred_idx = int(torch.argmax(full_logits, dim=-1).item())

    id2label = getattr(model.config, "id2label", {}) or {}
    predicted_label = id2label.get(pred_idx, str(pred_idx))

    metadata: Dict[str, Any] = {
        "model": args.model_name,
        "opset": args.opset,
        "encoder_layers_total": total_layers,
        "split_idx": args.split_idx,
        "num_labels": int(getattr(model.config, "num_labels", 2)),
        "files": {
            "cell_a": "cell_a.onnx",
            "cell_b": "cell_b.onnx",
            "cell_c": "cell_c.onnx",
            "sample_input_ids": "sample_input_ids.npy",
            "sample_attention_mask": "sample_attention_mask.npy",
            "tokenizer_dir": "tokenizer",
        },
        "pipeline_contract": {
            "cell_a_input": {
                "name": "input_ids",
                "dtype": "int64",
                "shape": check["pytorch_chain_check"]["input_ids_shape"],
                "description": "Token IDs from exported tokenizer",
            },
            "cell_a_output": {
                "name": "features_a",
                "dtype": "float32",
                "shape": check["pytorch_chain_check"]["cell_a_output_shape"],
                "description": "Hidden states with mask packed in last channel",
            },
            "cell_b_output": {
                "name": "features_b",
                "dtype": "float32",
                "shape": check["pytorch_chain_check"]["cell_b_output_shape"],
            },
            "cell_c_output": {
                "name": "logits",
                "dtype": "float32",
                "shape": check["pytorch_chain_check"]["cell_c_output_shape"],
                "description": "Sentiment logits",
            },
        },
        "sample": {
            "text": args.sample_text,
            "predicted_label": predicted_label,
            "predicted_label_id": pred_idx,
            "id2label": id2label,
        },
        "verification": {
            "pytorch_chain": check["pytorch_chain_check"],
        },
    }

    if not args.skip_verify:
        ort_check = verify_chain_with_onnxruntime(
            output_dir=output_dir,
            input_ids=input_ids,
            reference_logits=full_logits,
        )
        metadata["verification"].update(ort_check)

    (output_dir / "partition_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Export complete")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print(f"  - {output_dir / 'cell_a.onnx'}")
    print(f"  - {output_dir / 'cell_b.onnx'}")
    print(f"  - {output_dir / 'cell_c.onnx'}")
    print(f"  - {output_dir / 'sample_input_ids.npy'}")
    print(f"  - {output_dir / 'sample_attention_mask.npy'}")
    print(f"  - {output_dir / 'partition_metadata.json'}")
    print(f"  - {tokenizer_dir}")


if __name__ == "__main__":
    main()
