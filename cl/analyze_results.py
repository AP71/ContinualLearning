"""Generate multi-seed experiment plots from an aggregated results JSON.

Usage:
    python cl/analyze_results.py path/to/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import (
    plot_accuracy_history,
    plot_accuracy_vs_classes,
    plot_matrix,
    plot_task_metric,
)


def _statistics(values: list[Any]) -> dict[str, Any]:
    arrays = [np.asarray(value, dtype=float) for value in values]
    shape = tuple(
        max((array.shape[axis] if axis < array.ndim else 0) for array in arrays)
        for axis in range(max((array.ndim for array in arrays), default=0))
    )
    data = np.full((len(arrays), *shape), np.nan)
    for index, array in enumerate(arrays):
        slices = (index,) + tuple(slice(0, size) for size in array.shape)
        data[slices] = array

    count = np.sum(~np.isnan(data), axis=0)
    total = np.nansum(data, axis=0)
    mean = np.divide(
        total,
        count,
        out=np.full_like(total, np.nan),
        where=count > 0,
    )
    centered = data - mean
    variance = np.divide(
        np.nansum(centered * centered, axis=0),
        count,
        out=np.full_like(total, np.nan),
        where=count > 0,
    )

    return {
        "mean": mean,
        "variance": variance,
        "count": count,
    }


def _save_curve(
    values: list[Any],
    path: Path,
    title: str,
    ylabel: str,
    *,
    x_values: list[float] | None = None,
    percentage: bool = False,
    variance_as_errorbar: bool = False,
) -> dict[str, Any]:
    stats = _statistics(values)
    mean = np.asarray(stats["mean"], dtype=float).reshape(-1)
    variance = np.asarray(stats["variance"], dtype=float).reshape(-1)
    spread = np.sqrt(np.maximum(variance, 0.0))
    x = np.arange(1, len(mean) + 1) if x_values is None else np.asarray(x_values)
    if x_values is not None:
        plot_accuracy_vs_classes(
            mean.tolist(),
            x.tolist(),
            path,
            variance=spread.tolist(),
        )
    else:
        plot_task_metric(
            mean.tolist(),
            ylabel,
            "",
            path,
            percentage_axis=percentage,
            variance=spread.tolist(),
            variance_as_errorbar=variance_as_errorbar,
        )
    return stats


def _save_history(values: list[Any], path: Path, title: str) -> dict[str, Any]:
    stats = _statistics(values)
    mean = np.asarray(stats["mean"], dtype=float)
    variance = np.asarray(stats["variance"], dtype=float)
    if mean.ndim == 1:
        mean = mean[None, :]
        variance = variance[None, :]
    plot_accuracy_history(
        mean.tolist(),
        path,
        title=None,
        variance_history=np.sqrt(np.maximum(variance, 0.0)).tolist(),
        show_markers=False,
    )
    return stats


def _save_matrix(values: list[Any], path: Path, title: str) -> dict[str, Any]:
    stats = _statistics(values)
    mean = np.asarray(stats["mean"], dtype=float)
    variance = np.asarray(stats["variance"], dtype=float)
    annotations = [
        [
            f"{mean[row, col]:.4f}\n±{variance[row, col]:.4f}"
            if not np.isnan(mean[row, col])
            else ""
            for col in range(mean.shape[1])
        ]
        for row in range(mean.shape[0])
    ]
    plot_matrix(
        mean,
        "",
        path,
        xlabel="Evaluated task",
        ylabel="Training task",
        annotations=annotations,
    )
    return stats


def analyze_results(results_path: Path) -> dict[str, Any]:
    """Create experiment-specific plots and update the aggregated JSON."""
    if results_path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON file, got: {results_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    seeds = results.get("seeds")
    if not isinstance(seeds, dict) or not seeds:
        raise ValueError("The JSON must contain a non-empty 'seeds' object.")
    runs = list(seeds.values())
    output_dir = results_path.parent
    experiment_type = str(results.get("experiment_type", "")).lower()
    def collect(key: str) -> list[Any]:
        values = [run.get(key) for run in runs]
        return values if all(value is not None for value in values) else []

    if "sequential" in experiment_type:
        for key, filename, title, ylabel in (
            ("bit_flips", "bitflips_per_task.png", "Bit flips per task", "Bit flips"),
            ("bwt", "bwt_per_task.png", "Backward transfer per task", "BWT"),
        ):
            values = collect(key)
            if values:
                _save_curve(
                    values,
                    output_dir / filename,
                    title,
                    ylabel,
                )

        for key, filename, title in (
            ("standard_class_incremental", "standard_r_matrix.png",
             "Standard class-incremental accuracy"),
            ("task_incremental", "task_incremental_r_matrix.png",
             "Task-incremental accuracy"),
        ):
            values = collect(key)
            if values:
                _save_matrix(values, output_dir / filename, title)

        values = collect("accuracy_vs_classes")
        if values:
            _save_curve(
                [item["accuracy_percent"] for item in values],
                output_dir / "accuracy_vs_classes.png",
                "Accuracy vs Seen Classes",
                "Accuracy (%)",
                x_values=values[0]["classes_seen"],
                percentage=True,
            )

        values = collect("accuracy_history")
        if values:
            _save_history(
                values, output_dir / "accuracy_per_epoch.png",
                "Accuracy per epoch",
            )
    elif "joint" in experiment_type:
        values = collect("final_accuracy")
        if values:
            _save_curve(
                values,
                output_dir / "final_accuracy.png",
                "Final joint-training accuracy",
                "Accuracy",
                percentage=True,
            )

    else:
        raise ValueError(
            "Unsupported experiment_type. Expected 'sequential' or 'joint'."
        )

    results.pop("statistics", None)
    results["analysis"] = {
        "num_seeds": len(runs),
        "seed_values": list(seeds),
        "plots_directory": str(output_dir),
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    analyze_results(Path("C:\\Users\\flypi\\Desktop\\TesiMagistrale\\ContinualLearning\\cl\\results\\cifar100_avalanche_naive_checkpoint_resneta_3seeds\\results.json"))
