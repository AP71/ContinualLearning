"""Recreate experiment plots from a saved ``results.json`` file.

Usage:
    python utils/recreate_plots.py path/to/results.json

The generated images are written next to the input JSON file. Plots that
depend on checkpoint tensors, such as latent-distribution comparisons, cannot
be recreated from ``results.json`` alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.metrics import (  # noqa: E402
    calculate_bwt,
    plot_accuracy_history,
    plot_accuracy_vs_classes,
    plot_matrix,
    plot_task_metric,
)


def load_results(results_path: Path) -> dict[str, Any]:
    """Load and validate an experiment results JSON file."""
    if results_path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON file, got: {results_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    with results_path.open("r", encoding="utf-8") as file:
        results = json.load(file)
    if not isinstance(results, dict):
        raise ValueError("The JSON root must be an object.")
    return results


def recreate_sequential_plots(results: dict[str, Any], output_dir: Path) -> list[Path]:
    """Recreate plots produced by a sequential experiment."""
    standard_matrix = results.get("standard_class_incremental")
    task_matrix = results.get("task_incremental")
    history = results.get("history", [])
    accuracy_history = results.get("accuracy_history")
    bit_flips = results.get("bit_flips")

    if standard_matrix is None or task_matrix is None:
        raise ValueError("Sequential results must contain both accuracy matrices.")
    if not isinstance(history, list) or not isinstance(accuracy_history, list):
        raise ValueError("Sequential results contain invalid history data.")

    generated = []
    generated.append(
        _plot_matrix(
            standard_matrix,
            "Standard class-incremental accuracy",
            output_dir / "standard_r_matrix.png",
        )
    )
    generated.append(
        _plot_matrix(
            task_matrix,
            "Task-incremental accuracy (masked logits)",
            output_dir / "task_incremental_r_matrix.png",
        )
    )

    generated.append(
        _plot_accuracy_history(
            accuracy_history, output_dir / "accuracy_per_epoch.png"
        )
    )

    bwt_values = results.get("bwt")
    if bwt_values is None:
        bwt_values = [
            calculate_bwt(standard_matrix[: task_id + 1])
            for task_id in range(len(standard_matrix))
        ]
    generated.append(
        _plot_task_metric(
            bwt_values,
            "BWT (%)",
            "Backward transfer per task",
            output_dir / "bwt_per_task.png",
            connect_points=True,
            percentage_axis=True,
        )
    )

    if bit_flips is not None:
        generated.append(
            _plot_task_metric(
                bit_flips,
                "Bit flips",
                "Binary bit flips per task",
                output_dir / "bitflips_per_task.png",
                connect_points=True,
            )
        )

    accuracy_vs_classes = results.get("accuracy_vs_classes", {})
    classes_seen = accuracy_vs_classes.get("classes_seen")
    accuracy_percent = accuracy_vs_classes.get("accuracy_percent")
    if classes_seen is None or accuracy_percent is None:
        classes_seen = [(index + 1) * 10 for index in range(len(history))]
        accuracy_percent = [
            100.0 * item["standard_seen_accuracy"] for item in history
        ]
    generated.append(
        _plot_accuracy_vs_classes(
            accuracy_percent,
            classes_seen,
            output_dir / "accuracy_vs_classes.png",
        )
    )
    return generated


def recreate_joint_plots(results: dict[str, Any], output_dir: Path) -> list[Path]:
    """Recreate plots produced by a joint-training experiment."""
    generated = []
    accuracy_history = results.get("accuracy_history")
    if not isinstance(accuracy_history, list):
        raise ValueError("Joint results must contain accuracy_history.")

    accuracy_epochs = results.get("accuracy_epochs")
    if accuracy_epochs is None:
        accuracy_epochs = list(range(1, len(accuracy_history) + 1))
    generated.append(
        _plot_accuracy_history(
            [accuracy_history],
            output_dir / "accuracy_per_epoch.png",
            title="Joint-training accuracy per epoch",
            epoch_points=accuracy_epochs,
        )
    )

    standard_accuracy = results.get("standard_class_incremental_accuracy")
    if standard_accuracy is not None:
        generated.append(
            _plot_task_metric(
                [standard_accuracy],
                "Accuracy (%)",
                "Final class-incremental accuracy",
                output_dir / "final_accuracy.png",
                percentage_axis=True,
                point_labels=[100.0 * standard_accuracy],
                label_format="{:.2f}%",
            )
        )

    bit_flip_percentage = results.get("bit_flip_percentage")
    if bit_flip_percentage is not None:
        generated.append(
            _plot_task_metric(
                [bit_flip_percentage / 100.0],
                "Bit flips (%)",
                "Final binary bit-flip percentage",
                output_dir / "bitflip_percentage.png",
                percentage_axis=True,
                point_labels=[bit_flip_percentage],
                label_format="{:.4f}%",
            )
        )

    bit_flips = results.get("bit_flips")
    if bit_flips is not None:
        generated.append(
            _plot_task_metric(
                [bit_flips],
                "Bit flips",
                "Final binary bit flips",
                output_dir / "bitflips.png",
                point_labels=[bit_flips],
                label_format="{:.0f}",
            )
        )

    confusion_matrix = results.get("confusion_matrix")
    if confusion_matrix is not None:
        generated.append(
            _plot_matrix(
                confusion_matrix,
                "Joint-training confusion matrix",
                output_dir / "confusion_matrix.png",
                cmap="Blues",
                normalized=False,
                xlabel="Predicted class",
                ylabel="True class",
                show_ticks=False,
            )
        )

    for key, filename, title in (
        ("r_matrix", "r_matrix.png", "Joint-training R matrix"),
        (
            "task_incremental_r_matrix",
            "task_incremental_r_matrix.png",
            "Joint-training task-incremental R matrix",
        ),
    ):
        matrix = results.get(key)
        if matrix is not None:
            generated.append(
                _plot_matrix(
                    matrix,
                    title,
                    output_dir / filename,
                    xlabel="Evaluated task",
                    ylabel="Training task",
                )
            )
    return generated


def _plot_matrix(
    matrix: list[list[Any]],
    title: str,
    output_path: Path,
    *,
    xlabel: str = "Evaluated task",
    ylabel: str = "Training task",
    show_ticks: bool = True,
    cmap: str = "Blues",
    normalized: bool = True,
) -> Path:
    plot_matrix(
        matrix,
        title,
        output_path,
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        show_ticks=show_ticks,
        vmin=0.0 if normalized else None,
        vmax=1.0 if normalized else None,
    )
    return output_path


def _plot_accuracy_history(
    accuracy_history: list[list[float]],
    output_path: Path,
    *,
    title: str | None = None,
    epoch_points: list[int] | None = None,
) -> Path:
    plot_accuracy_history(
        accuracy_history,
        output_path,
        title=title,
        epoch_points=epoch_points,
    )
    return output_path


def _plot_task_metric(values: list[float], ylabel: str, title: str, output_path: Path, **kwargs: Any) -> Path:
    plot_task_metric(values, ylabel, title, output_path, **kwargs)
    return output_path


def _plot_accuracy_vs_classes(
    accuracy_percent: list[float], classes_seen: list[int], output_path: Path
) -> Path:
    plot_accuracy_vs_classes(accuracy_percent, classes_seen, output_path)
    return output_path


def recreate_plots(results_path: Path) -> list[Path]:
    """Recreate all plots available in a results file."""
    results = load_results(results_path)
    output_dir = results_path.parent
    protocol = str(results.get("protocol", "")).lower()
    if "joint" in protocol or "accuracy_epochs" in results:
        return recreate_joint_plots(results, output_dir)
    if "standard_class_incremental" in results:
        return recreate_sequential_plots(results, output_dir)
    raise ValueError("Could not determine the experiment type from the results file.")


if __name__ == "__main__":
    path = Path("C:\\Users\\flypi\\Desktop\\TesiMagistrale\\ContinualLearning\\cl\\results\\cifar100_joint_checkpoint_resneta_20261024_115152\\results.json")
    recreate_plots(path)
