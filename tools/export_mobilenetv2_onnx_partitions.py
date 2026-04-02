#!/usr/bin/env python3
"""
Export a pretrained MobileNetV2 into three ONNX partitions:
- cell_a.onnx
- cell_b.onnx
- cell_c.onnx

The generated files are intended for staged/native execution pipelines where
stage outputs are passed as tensor payloads to the next stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    from torchvision import transforms
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "torchvision is required. Install with: pip install torchvision"
    ) from exc


class CellA(nn.Module):
    def __init__(self, features: nn.Sequential, split_idx: int) -> None:
        super().__init__()
        self.part = nn.Sequential(*list(features.children())[:split_idx])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.part(x)


class CellB(nn.Module):
    def __init__(self, features: nn.Sequential, split_idx: int) -> None:
        super().__init__()
        self.part = nn.Sequential(*list(features.children())[split_idx:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.part(x)


class CellC(nn.Module):
    def __init__(self, classifier: nn.Sequential) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_pretrained_mobilenet_v2() -> nn.Module:
    # Supports newer and older torchvision APIs.
    try:
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    except Exception:
        model = torchvision.models.mobilenet_v2(pretrained=True)

    model.eval()
    return model


def build_partitions(
    model: nn.Module, split_idx: int
) -> Tuple[nn.Module, nn.Module, nn.Module, int]:
    features = model.features
    total_feature_blocks = len(list(features.children()))

    if split_idx <= 0 or split_idx >= total_feature_blocks:
        raise ValueError(
            f"split_idx must be in [1, {total_feature_blocks - 1}], got {split_idx}"
        )

    cell_a = CellA(features=features, split_idx=split_idx).eval()
    cell_b = CellB(features=features, split_idx=split_idx).eval()
    cell_c = CellC(classifier=model.classifier).eval()

    return cell_a, cell_b, cell_c, total_feature_blocks


def preprocess_from_image(image_path: Path) -> torch.Tensor:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    pipeline = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor = pipeline(image).unsqueeze(0).float()
    return tensor


def create_sample_input(image_path: Optional[Path]) -> torch.Tensor:
    if image_path is not None:
        return preprocess_from_image(image_path)

    torch.manual_seed(42)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)


def export_onnx(
    module: nn.Module,
    sample_input: torch.Tensor,
    out_path: Path,
    input_name: str,
    output_name: str,
    opset: int,
) -> None:
    dynamic_axes = {
        input_name: {0: "batch"},
        output_name: {0: "batch"},
    }

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
    model_input: torch.Tensor,
) -> Dict[str, Any]:
    with torch.no_grad():
        full = model(model_input)
        a_out = cell_a(model_input)
        b_out = cell_b(a_out)
        c_out = cell_c(b_out)

    max_abs_diff = float((full - c_out).abs().max().item())
    mean_abs_diff = float((full - c_out).abs().mean().item())

    return {
        "pytorch_chain_check": {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "input_shape": list(model_input.shape),
            "cell_a_output_shape": list(a_out.shape),
            "cell_b_output_shape": list(b_out.shape),
            "cell_c_output_shape": list(c_out.shape),
        },
        "tensors": {
            "a_out": a_out,
            "b_out": b_out,
            "c_out": c_out,
            "full_out": full,
        },
    }


def verify_chain_with_onnxruntime(
    output_dir: Path,
    model_input: torch.Tensor,
    reference_output: torch.Tensor,
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

    inp = model_input.detach().cpu().numpy().astype(np.float32)
    a_out = sess_a.run(None, {"input": inp})[0]
    b_out = sess_b.run(None, {"features_a": a_out})[0]
    c_out = sess_c.run(None, {"features_b": b_out})[0]

    ref = reference_output.detach().cpu().numpy().astype(np.float32)
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
        description="Export pretrained MobileNetV2 into cell_a/cell_b/cell_c ONNX partitions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("onnx_partitions_mobilenetv2"),
        help="Directory where ONNX partitions and metadata will be written",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional input image path for sample export (RGB image)",
    )
    parser.add_argument(
        "--split-idx",
        type=int,
        default=7,
        help="MobileNetV2 feature split index for cell_a -> cell_b boundary",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
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

    model = load_pretrained_mobilenet_v2()
    cell_a, cell_b, cell_c, total_blocks = build_partitions(
        model, split_idx=args.split_idx
    )

    model_input = create_sample_input(args.image)

    check = verify_chain_with_pytorch(
        model=model,
        cell_a=cell_a,
        cell_b=cell_b,
        cell_c=cell_c,
        model_input=model_input,
    )

    a_tensor = check["tensors"]["a_out"]
    b_tensor = check["tensors"]["b_out"]

    export_onnx(
        module=cell_a,
        sample_input=model_input,
        out_path=output_dir / "cell_a.onnx",
        input_name="input",
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
        output_dir / "sample_input.npy",
        model_input.detach().cpu().numpy().astype(np.float32),
    )

    metadata: Dict[str, Any] = {
        "model": "torchvision.models.mobilenet_v2 (ImageNet pretrained)",
        "opset": args.opset,
        "feature_blocks_total": total_blocks,
        "split_idx": args.split_idx,
        "files": {
            "cell_a": "cell_a.onnx",
            "cell_b": "cell_b.onnx",
            "cell_c": "cell_c.onnx",
            "sample_input": "sample_input.npy",
        },
        "pipeline_contract": {
            "cell_a_input": {
                "name": "input",
                "dtype": "float32",
                "shape": check["pytorch_chain_check"]["input_shape"],
                "description": "NCHW image tensor normalized with ImageNet mean/std",
            },
            "cell_a_output": {
                "name": "features_a",
                "dtype": "float32",
                "shape": check["pytorch_chain_check"]["cell_a_output_shape"],
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
                "description": "1000-way ImageNet logits",
            },
        },
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "to_rgb": True,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "verification": {
            "pytorch_chain": check["pytorch_chain_check"],
        },
    }

    if not args.skip_verify:
        ort_check = verify_chain_with_onnxruntime(
            output_dir=output_dir,
            model_input=model_input,
            reference_output=check["tensors"]["full_out"],
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
    print(f"  - {output_dir / 'sample_input.npy'}")
    print(f"  - {output_dir / 'partition_metadata.json'}")


if __name__ == "__main__":
    main()
