from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Mapping
import torch

TensorMap = Mapping[str, torch.Tensor]
LayerPredicate = Callable[[str, torch.Tensor], bool]

"""Load a state dict from a path, a raw state dict, or a Lightning checkpoint."""
def _load_state_dict(checkpoint: str | Path | Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, Mapping):
        raise TypeError("A checkpoint must be a path or a mapping.")

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("The checkpoint state_dict must be a mapping.")

    tensors = {
        str(name).removeprefix("module."): value.detach().cpu()
        for name, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensors:
        raise ValueError("The checkpoint does not contain tensor parameters.")
    return tensors

"""Identify binary layers using the naming convention used by ReActNet."""
def _default_binary_layer(name: str, tensor: torch.Tensor) -> bool:
    del tensor
    name = name.lower()
    return any(token in name for token in ("binary", "bconv", "binaryactivation"))

def _latent_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float().reshape(-1)
    abs_values = values.abs()
    return {
        "numel": int(values.numel()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "mean_abs": float(abs_values.mean().item()),
        "median_abs": float(abs_values.median().item()),
        "zero_fraction": float((values == 0).float().mean().item()),
        "abs_values": abs_values.tolist(),
    }

"""Extract latent weights and sign-binarized weights layer by layer."""
def extract_weights(checkpoint: str | Path | Mapping[str, Any], binary_layer_predicate: LayerPredicate | None = None) -> dict[str, dict[str, torch.Tensor]]:
    state_dict = _load_state_dict(checkpoint)
    predicate = binary_layer_predicate or _default_binary_layer
    latent = {
        name: tensor.clone()
        for name, tensor in state_dict.items()
        if name.endswith(".weight") and tensor.is_floating_point()
    }
    binary = {
        name: torch.sign(tensor)
        for name, tensor in latent.items()
        if predicate(name, tensor)
    }
    return {"latent": latent, "binary": binary}

"""Compare two checkpoints layer by layer.
Binary weights are represented by ``sign(latent_weight)``. 
The returned ``layers`` entry contains latent distributions for every matching weight
tensor and bit-flip counts/percentages for detected binary layers.
"""
def compare_checkpoints(checkpoint_a: str | Path | Mapping[str, Any], checkpoint_b: str | Path | Mapping[str, Any], binary_layer_predicate: LayerPredicate | None = None) -> dict[str, Any]:
    state_a = _load_state_dict(checkpoint_a)
    state_b = _load_state_dict(checkpoint_b)
    predicate = binary_layer_predicate or _default_binary_layer

    weight_names = sorted(
        name
        for name in state_a.keys() & state_b.keys()
        if name.endswith(".weight")
        and state_a[name].shape == state_b[name].shape
        and state_a[name].is_floating_point()
        and state_b[name].is_floating_point()
    )
    if not weight_names:
        raise ValueError("No matching floating-point '*.weight' tensors were found.")

    layers: dict[str, dict[str, Any]] = {}
    total_flips = 0
    total_bits = 0
    for name in weight_names:
        tensor_a = state_a[name]
        tensor_b = state_b[name]
        layer_result: dict[str, Any] = {
            "shape": list(tensor_a.shape),
            "latent_a": _latent_stats(tensor_a),
            "latent_b": _latent_stats(tensor_b),
            "binary": bool(predicate(name, tensor_a)),
        }
        if layer_result["binary"]:
            binary_a = torch.sign(tensor_a)
            binary_b = torch.sign(tensor_b)
            flips = int((binary_a != binary_b).sum().item())
            bits = int(binary_a.numel())
            layer_result.update(
                {
                    "bit_flips": flips,
                    "total_bits": bits,
                    "bit_flip_percentage": 100.0 * flips / bits,
                }
            )
            total_flips += flips
            total_bits += bits
        layers[name] = layer_result

    return {
        "checkpoint_a": str(checkpoint_a) if isinstance(checkpoint_a, (str, Path)) else None,
        "checkpoint_b": str(checkpoint_b) if isinstance(checkpoint_b, (str, Path)) else None,
        "layers": layers,
        "binary_summary": {
            "bit_flips": total_flips,
            "total_bits": total_bits,
            "bit_flip_percentage": 100.0 * total_flips / total_bits
            if total_bits
            else 0.0,
        },
    }

"""Save a comparison result as JSON, including latent absolute-value distributions."""
def save_comparison(result: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

"""Print every comparison metric with percentages formatted as ``0.00%``.
The complete absolute-value distributions are already available in
``result`` and can be printed with ``include_abs_values=True``.
"""
def print_comparison(result: dict[str, Any], *, include_abs_values: bool = False) -> None:
    print(f"Checkpoint A: {result.get('checkpoint_a')}")
    print(f"Checkpoint B: {result.get('checkpoint_b')}")
    print("\nLAYER RESULTS")
    for layer_name, layer_data in result["layers"].items():
        print(f"\n[{layer_name}]")
        print(f"  shape: {layer_data['shape']}")
        print(f"  binary: {layer_data['binary']}")
        for checkpoint_name in ("latent_a", "latent_b"):
            stats = layer_data[checkpoint_name]
            print(f"  {checkpoint_name}:")
            for key, value in stats.items():
                if key == "abs_values" and not include_abs_values:
                    print(f"    {key}: {len(value)} values (available in result)")
                else:
                    print(f"    {key}: {value}")
        if layer_data["binary"]:
            print(f"  bit_flips: {layer_data['bit_flips']}")
            print(f"  total_bits: {layer_data['total_bits']}")
            print(
                f"  bit_flip_percentage: "
                f"{layer_data['bit_flip_percentage']:.2f}%"
            )

    summary = result["binary_summary"]
    print("\nGLOBAL BINARY SUMMARY")
    print(f"  bit_flips: {summary['bit_flips']}")
    print(f"  total_bits: {summary['total_bits']}")
    print(f"  bit_flip_percentage: {summary['bit_flip_percentage']:.2f}%")
