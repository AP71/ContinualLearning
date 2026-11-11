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
        "std": np.sqrt(np.maximum(variance, 0.0)),
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
    standard_deviation_as_errorbar: bool = False,
) -> dict[str, Any]:
    stats = _statistics(values)
    mean = np.asarray(stats["mean"], dtype=float).reshape(-1)
    standard_deviation = np.asarray(stats["std"], dtype=float).reshape(-1)
    x = np.arange(1, len(mean) + 1) if x_values is None else np.asarray(x_values)
    if x_values is not None:
        plot_accuracy_vs_classes(
            mean.tolist(),
            x.tolist(),
            path,
            standard_deviation=standard_deviation.tolist(),
        )
    else:
        plot_task_metric(
            mean.tolist(),
            ylabel,
            "",
            path,
            percentage_axis=percentage,
            standard_deviation=standard_deviation.tolist(),
            standard_deviation_as_errorbar=standard_deviation_as_errorbar,
        )
    return stats


def _save_history(values: list[Any], path: Path, title: str) -> dict[str, Any]:
    stats = _statistics(values)
    mean = np.asarray(stats["mean"], dtype=float)
    standard_deviation = np.asarray(stats["std"], dtype=float)
    if mean.ndim == 1:
        mean = mean[None, :]
        standard_deviation = standard_deviation[None, :]
    plot_accuracy_history(
        mean.tolist(),
        path,
        title=None,
        standard_deviation_history=standard_deviation.tolist(),
        show_markers=False,
    )
    return stats


def _save_joint_history(values: list[Any], path: Path) -> dict[str, Any]:
    stats = _statistics(values)
    mean = np.asarray(stats["mean"], dtype=float).reshape(-1)
    standard_deviation = np.asarray(stats["std"], dtype=float).reshape(-1)
    plot_accuracy_history(
        [mean.tolist()],
        path,
        title=None,
        standard_deviation_history=[standard_deviation.tolist()],
        show_markers=False,
        legend_labels=["Mean ± SD"],
    )
    return stats


def _save_joint_point(
    values: list[Any],
    path: Path,
    ylabel: str,
    labels: list[Any],
    *,
    scale: float = 1.0,
    label_format: str = "{:.2f}%",
    standard_deviation_values: list[Any] | None = None,
) -> dict[str, Any]:
    stats = _statistics(values)
    mean = np.asarray(stats["mean"], dtype=float).reshape(-1) * scale
    standard_deviation = None
    if standard_deviation_values is not None:
        standard_deviation = (
            np.asarray(_statistics(standard_deviation_values)["std"], dtype=float).reshape(-1)
            * scale
        )
    plot_task_metric(
        mean.tolist(),
        ylabel,
        "",
        path,
        connect_points=False,
        point_labels=(np.asarray(labels, dtype=float).reshape(-1)).tolist(),
        label_format=label_format,
        standard_deviation=(
            None if standard_deviation is None else standard_deviation.tolist()
        ),
        standard_deviation_as_errorbar=True,
    )
    return stats


def _save_matrix(
    values: list[Any],
    path: Path,
    title: str,
    *,
    show_values: bool = True,
) -> dict[str, Any]:
    stats = _statistics(values)
    mean = np.asarray(stats["mean"], dtype=float)
    standard_deviation = np.asarray(stats["std"], dtype=float)
    annotations = [
        [
            f"{mean[row, col]:.4f}\n±{standard_deviation[row, col]:.4f}"
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
        show_values=show_values,
        show_tick_labels=show_values,
    )
    return stats


def _normalize_confusion_matrices(values: list[Any]) -> list[list[list[float]]]:
    """Convert per-class confusion counts into row-normalized accuracies."""
    normalized = []
    for value in values:
        matrix = np.asarray(value, dtype=float)
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix),
            where=row_totals > 0,
        )
        normalized.append(normalized_matrix.tolist())
    return normalized


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
        values = collect("accuracy_history")
        if values:
            _save_joint_history(
                values,
                output_dir / "accuracy_per_epoch.png",
            )

        values = collect("final_accuracy")
        if not values:
            values = collect("final_accuracy_on_test")
        if values:
            final_stats = _statistics(values)
            final_mean = float(np.asarray(final_stats["mean"]).reshape(-1)[0])
            _save_joint_point(
                values,
                output_dir / "final_accuracy.png",
                "Accuracy (%)",
                [final_mean * 100.0],
                scale=100.0,
                standard_deviation_values=values,
            )

        values = collect("confusion_matrix")
        if values:
            _save_matrix(
                _normalize_confusion_matrices(values),
                output_dir / "confusion_matrix.png",
                "",
                show_values=False,
            )

        values = collect("total_bit_flips")
        percentages = collect("bit_flip_percentage")
        if values and percentages:
            percentage_stats = _statistics(percentages)
            mean_percentage = float(
                np.asarray(percentage_stats["mean"]).reshape(-1)[0]
            )
            _save_joint_point(
                values,
                output_dir / "bitflips_total.png",
                "Flipped bits",
                [mean_percentage],
                label_format="{:.4f}%",
                standard_deviation_values=values,
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
    analyze_results(Path(r"F:\TesiMagistrale\ContinualLearning\cl\results\cifar100_avalanche_FreezingHead_checkpoint_resneta_3seeds\results.json"))