"""Layer-wise forgetting analysis for Split CIFAR-100 experiments.

For every training task from task 2 onward, this module restores one layer at
a time to its state after the previous task and evaluates all previously seen
tasks. Binary layers restore only the sign of their latent weights; ``fc``
restores its complete (non-binary) parameters.

Example:
    python cl/forgetting_analysis.py cl/results/cifar100_avalanche_naive_checkpoint_resneta_3seeds
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Network.ReActNet_A_Qaw import Reactnet

SEED_NAMES = ("seed42", "seed2023", "seed3047")
NUM_TASKS = 10
CLASSES_PER_TASK = 10
DEFAULT_DATASET_PATH = Path(r"F:/Dataset/CIFAR-100/")


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} does not contain a state dict.")
    return {
        str(name).removeprefix("module."): value.detach().cpu()
        for name, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def _task_datasets(dataset: Dataset) -> list[Dataset]:
    if not hasattr(dataset, "targets"):
        raise TypeError("The CIFAR-100 dataset must expose a 'targets' attribute.")
    targets = np.asarray(dataset.targets)
    return [
        torch.utils.data.Subset(
            dataset,
            np.flatnonzero(
                (targets >= task_id * CLASSES_PER_TASK)
                & (targets < (task_id + 1) * CLASSES_PER_TASK)
            ).tolist(),
        )
        for task_id in range(NUM_TASKS)
    ]


def _evaluate_task(
    model: nn.Module,
    task_dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    loader = DataLoader(
        task_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    allowed_classes = None
    if hasattr(task_dataset, "indices"):
        labels = [
            task_dataset.dataset.targets[index]  # type: ignore[attr-defined]
            for index in task_dataset.indices  # type: ignore[attr-defined]
        ]
        allowed_classes = torch.tensor(sorted(set(labels)), device=device)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            if allowed_classes is not None:
                logits = logits.index_select(1, allowed_classes)
                predictions = allowed_classes[logits.argmax(dim=1)]
            else:
                predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
    if was_training:
        model.train()
    return correct / total if total else 0.0


def _binary_layer_names(state_dict: dict[str, torch.Tensor]) -> list[str]:
    return sorted(
        name
        for name, tensor in state_dict.items()
        if name.endswith(".weight")
        and tensor.is_floating_point()
        and any(token in name.lower() for token in ("binary", "bconv"))
    )


def _restore_binary_bits(
    current: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor],
    layer_name: str,
) -> dict[str, torch.Tensor]:
    restored = {name: tensor.clone() for name, tensor in current.items()}
    current_weight = restored[layer_name]
    previous_weight = previous[layer_name]
    restored[layer_name] = current_weight.abs() * previous_weight.sign()
    return restored


def _restore_classifier(
    current: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    restored = {name: tensor.clone() for name, tensor in current.items()}
    classifier_names = [
        name for name in current if name.startswith("fc.") and name in previous
    ]
    if not classifier_names:
        raise ValueError("The checkpoints do not contain the final classifier (fc).")
    for name in classifier_names:
        if current[name].shape != previous[name].shape:
            raise ValueError(f"Classifier shape changed for parameter {name}.")
        restored[name] = previous[name].clone()
    return restored


def _rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def _statistics(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot calculate statistics for an empty sequence.")
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std(ddof=0)), 4),
    }


def _aggregate(
    per_seed: dict[str, Any],
    mode: str,
    layers: list[str],
) -> dict[str, Any]:
    per_task: dict[str, dict[str, dict[str, float]]] = {}
    metric_names = ("baseline_accuracy", "recovered_accuracy", "recovered_gain")
    all_values = {
        layer: {metric: [] for metric in metric_names} for layer in layers
    }
    for task_id in range(2, NUM_TASKS + 1):
        task_key = str(task_id)
        per_task[task_key] = {}
        for layer in layers:
            per_task[task_key][layer] = {
                metric: _statistics(
                    [run[mode][task_key][layer][metric] for run in per_seed.values()]
                )
                for metric in metric_names
            }
            for metric in metric_names:
                all_values[layer][metric].extend(
                    [run[mode][task_key][layer][metric] for run in per_seed.values()]
                )
    return {
        "per_training_task": per_task,
        "overall": {
            layer: {
                metric: _statistics(values)
                for metric, values in metric_values.items()
            }
            for layer, metric_values in all_values.items()
        },
    }


def analyze_forgetting(
    experiments_dir: str | Path,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    output_path: str | Path | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run the layer-wise restoration experiment and write ``forgetting.json``."""
    experiments_dir = Path(experiments_dir)
    dataset_path = Path(dataset_path)
    output_path = Path(output_path) if output_path else experiments_dir / "forgetting.json"
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    missing = [name for name in SEED_NAMES if not (experiments_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing seed directories in {experiments_dir}: {', '.join(missing)}"
        )
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"CIFAR-100 dataset directory not found: {dataset_path}")

    test_dataset = datasets.CIFAR100(
        dataset_path,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    )
    task_datasets = _task_datasets(test_dataset)
    per_seed: dict[str, Any] = {}
    layer_names: list[str] | None = None

    progress = tqdm(
        total=len(SEED_NAMES) * (NUM_TASKS - 1),
        desc="Forgetting analysis",
        unit="task",
        dynamic_ncols=True,
    )
    for seed_name in SEED_NAMES:
        checkpoint_dir = experiments_dir / seed_name / "checkpoints"
        per_seed[seed_name] = {"binary_layers": {}, "classifier": {}}
        for task_id in range(2, NUM_TASKS + 1):
            progress.set_postfix(seed=seed_name, task=f"{task_id}/{NUM_TASKS}")
            current_path = checkpoint_dir / f"after_task_{task_id:02d}.h5"
            previous_path = checkpoint_dir / f"after_task_{task_id - 1:02d}.h5"
            if not current_path.is_file() or not previous_path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoints for {seed_name}, task {task_id}."
                )
            current = _load_checkpoint(current_path)
            previous = _load_checkpoint(previous_path)
            current_layers = _binary_layer_names(current)
            if layer_names is None:
                layer_names = current_layers
            elif current_layers != layer_names:
                raise ValueError("Binary layer names differ between checkpoints.")

            model = Reactnet(num_classes=100, imagenet=False).to(device)
            model.load_state_dict(current, strict=True)
            old_task_ids = range(task_id - 1)
            baseline = {
                str(old_id + 1): _evaluate_task(
                    model, task_datasets[old_id], device, batch_size, num_workers
                )
                for old_id in old_task_ids
            }

            for layer in current_layers:
                model.load_state_dict(
                    _restore_binary_bits(current, previous, layer), strict=True
                )
                accuracies = [
                    _evaluate_task(
                        model, task_datasets[old_id], device, batch_size, num_workers
                    )
                    for old_id in old_task_ids
                ]
                restored = float(np.mean(accuracies))
                baseline_mean = float(np.mean(list(baseline.values())))
                per_seed[seed_name]["binary_layers"].setdefault(
                    str(task_id), {}
                )[layer] = {
                    "baseline_accuracy": baseline_mean,
                    "recovered_accuracy": restored,
                    "recovered_gain": restored - baseline_mean,
                    "old_task_accuracies": {
                        str(old_id + 1): accuracy
                        for old_id, accuracy in zip(old_task_ids, accuracies)
                    },
                }

            model.load_state_dict(_restore_classifier(current, previous), strict=True)
            accuracies = [
                _evaluate_task(
                    model, task_datasets[old_id], device, batch_size, num_workers
                )
                for old_id in old_task_ids
            ]
            restored = float(np.mean(accuracies))
            baseline_mean = float(np.mean(list(baseline.values())))
            per_seed[seed_name]["classifier"].setdefault(str(task_id), {})[
                "fc"
            ] = {
                "baseline_accuracy": baseline_mean,
                "recovered_accuracy": restored,
                "recovered_gain": restored - baseline_mean,
                "old_task_accuracies": {
                    str(old_id + 1): accuracy
                    for old_id, accuracy in zip(old_task_ids, accuracies)
                },
            }
            progress.update()
    progress.close()

    assert layer_names is not None
    result = {
        "schema_version": 1,
        "protocol": "Restore one layer after each task from task 2 onward",
        "metric": "task-incremental accuracy on the mean of previously seen tasks",
        "standard_deviation": "population standard deviation across the three seeds",
        "seeds": per_seed,
        "statistics": {
            "binary_layers": _aggregate(per_seed, "binary_layers", layer_names),
            "classifier": _aggregate(per_seed, "classifier", ["fc"]),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_rounded(result), indent=2), encoding="utf-8"
    )
    return result


def plot_forgetting_results(json_path: str | Path) -> list[Path]:
    """Create forgetting plots next to a previously generated JSON file.

    The first plot ranks individual binary layers by recovered accuracy
    (mean +/- standard deviation across seeds and training tasks). The second
    plot shows the same gain for every training task and binary layer. The
    final classifier is shown as a separate reference line.
    """
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Forgetting JSON not found: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("The JSON does not contain a valid 'statistics' object.")
    binary = statistics.get("binary_layers")
    classifier = statistics.get("classifier")
    if not isinstance(binary, dict) or not isinstance(classifier, dict):
        raise ValueError("The JSON must contain binary_layers and classifier statistics.")

    output_dir = json_path.parent
    overall = binary.get("overall", {})
    layer_names = _network_order(list(overall))
    if not layer_names:
        raise ValueError("No binary-layer statistics found in the JSON.")

    gains = np.asarray(
        [overall[name]["recovered_gain"]["mean"] * 100 for name in layer_names],
        dtype=float,
    )
    errors = np.asarray(
        [overall[name]["recovered_gain"]["std"] * 100 for name in layer_names],
        dtype=float,
    )
    order = np.arange(len(layer_names))
    labels = layer_names

    figure, axis = plt.subplots(figsize=(13, 8))
    y = np.arange(len(labels))
    axis.errorbar(
        gains[order],
        y,
        xerr=errors[order],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Recovered accuracy (percentage points)")
    axis.grid(axis="x", linestyle="--", alpha=0.4)
    figure.tight_layout()
    layer_plot = output_dir / "forgetting_gain_by_layer.png"
    figure.savefig(layer_plot, dpi=300)
    plt.close(figure)

    per_task = binary.get("per_training_task", {})
    task_ids = sorted(per_task, key=int)
    matrix = np.asarray(
        [
            [
                per_task[task_id][layer]["recovered_gain"]["mean"] * 100
                for layer in layer_names
            ]
            for task_id in task_ids
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(15, 5))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r")
    axis.set_xticks(np.arange(len(layer_names)))
    axis.set_xticklabels(layer_names, rotation=90, fontsize=7)
    axis.set_yticks(np.arange(len(task_ids)))
    axis.set_yticklabels([f"Task {task_id}" for task_id in task_ids])
    axis.set_xlabel("Restored binary layer")
    axis.set_ylabel("Task completed after training")
    figure.colorbar(image, ax=axis, label="Recovered accuracy (p.p.)")
    figure.tight_layout()
    heatmap_plot = output_dir / "forgetting_gain_task_layer.png"
    figure.savefig(heatmap_plot, dpi=300)
    plt.close(figure)

    classifier_tasks = classifier.get("per_training_task", {})
    classifier_gain = [
        classifier_tasks[task_id]["fc"]["recovered_gain"]["mean"] * 100
        for task_id in task_ids
    ]
    classifier_std = [
        classifier_tasks[task_id]["fc"]["recovered_gain"]["std"] * 100
        for task_id in task_ids
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(task_ids))
    axis.errorbar(
        x,
        classifier_gain,
        yerr=classifier_std,
        marker="o",
        capsize=4,
        color="#d62728",
        label="Final classifier (fc)",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x)
    axis.set_xticklabels([f"Task {task_id}" for task_id in task_ids])
    axis.set_xlabel("Task completed after training")
    axis.set_ylabel("Recovered accuracy (percentage points)")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    classifier_plot = output_dir / "forgetting_gain_classifier.png"
    figure.savefig(classifier_plot, dpi=300)
    plt.close(figure)

    return [layer_plot, heatmap_plot, classifier_plot]


def _network_order(
    layer_names: list[str],
) -> list[str]:
    """Return binary layer names in Reactnet execution order."""
    layer_pattern = re.compile(
        r"^feature\.(\d+)\.(binary_3x3|binary_pw|binary_pw_down1|binary_pw_down2)\.weight$"
    )
    block_order = {
        "binary_3x3": 0,
        "binary_pw": 1,
        "binary_pw_down1": 1,
        "binary_pw_down2": 2,
    }
    parsed = []
    for name in layer_names:
        match = layer_pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"Unrecognized binary layer name: {name}")
        feature_index = int(match.group(1))
        block_name = match.group(2)
        parsed.append((feature_index, block_order[block_name], name))

    ordered = [name for _, _, name in sorted(parsed)]
    if len(ordered) != len(set(ordered)):
        raise ValueError(
            "The JSON contains duplicate binary layer names."
        )
    return ordered


EXPERIMENTS_DIR = ROOT / "cl" / "results" / "cifar100_avalanche_naive_checkpoint_resneta_3seeds"
DATASET_PATH = DEFAULT_DATASET_PATH
OUTPUT_PATH: Path | None = None
BATCH_SIZE = 256
NUM_WORKERS = 0

if __name__ == "__main__":
    # analyze_forgetting(
    #     EXPERIMENTS_DIR,
    #     DATASET_PATH,
    #     output_path=OUTPUT_PATH,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    # )
    plot_forgetting_results(EXPERIMENTS_DIR / "forgetting.json")