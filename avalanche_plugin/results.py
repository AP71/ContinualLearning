from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from utils.metrics import (
    calculate_bwt,
    plot_accuracy_history,
    plot_accuracy_vs_classes,
    plot_matrix,
    plot_task_metric,
)


def save_avalanche_sequential_results(
    output_dir: Path,
    standard_matrix: list[list[float | None]],
    task_matrix: list[list[float | None]],
    accuracy_history: list[list[float]],
    loss_history: list[float],
    custom_plugin,
    num_tasks: int = 10,
    epochs_per_task: int = 10,
    model_name: str = "resneta",
    initialization: str = "checkpoint",
    confusion_matrix_arr: np.ndarray | None = None,
) -> None:
    """Salva tutti i plot e i file results.json e plot_points.json conformi a base_train_3.py."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. BWT per ogni task
    bwt_values = [
        calculate_bwt(standard_matrix[: task_id + 1])
        for task_id in range(num_tasks)
    ]

    # 2. Bit flips dal plugin
    bit_flips = getattr(custom_plugin, "bit_flips", [0] * num_tasks)
    bit_flip_percentages = getattr(custom_plugin, "bit_flip_percentages", [0.0] * num_tasks)
    total_binary_bits = getattr(custom_plugin, "total_binary_bits", None)

    # 3. Classi viste e cronologia per task
    classes_seen = [(task_id + 1) * 10 for task_id in range(num_tasks)]
    history = []
    for task_id in range(num_tasks):
        seen_accs = [
            standard_matrix[task_id][j]
            for j in range(task_id + 1)
            if standard_matrix[task_id][j] is not None
        ]
        mean_seen_acc = float(np.mean(seen_accs)) if seen_accs else 0.0

        task_seen_accs = [
            task_matrix[task_id][j]
            for j in range(task_id + 1)
            if task_matrix[task_id][j] is not None
        ]
        mean_task_acc = float(np.mean(task_seen_accs)) if task_seen_accs else 0.0

        task_loss = (
            loss_history[(task_id + 1) * epochs_per_task - 1]
            if len(loss_history) >= (task_id + 1) * epochs_per_task
            else (loss_history[-1] if loss_history else 0.0)
        )

        history.append({
            "task": task_id,
            "loss": task_loss,
            "standard_seen_accuracy": mean_seen_acc,
            "task_incremental_seen_accuracy": mean_task_acc,
        })

    class_incremental_accuracy = [100.0 * row["standard_seen_accuracy"] for row in history]

    if confusion_matrix_arr is None:
        confusion_matrix_arr = np.zeros((100, 100), dtype=int)

    # 4. plot_points.json
    plot_points = {
        "bitflips_per_task": {
            "x": list(range(len(bit_flips))),
            "y": bit_flips,
            "x_label": "Task",
            "y_label": "Bit flips",
        },
        "bwt_per_task": {
            "x": list(range(len(bwt_values))),
            "y": bwt_values,
            "x_label": "Task",
            "y_label": "BWT",
        },
        "accuracy_vs_classes": {
            "x": classes_seen,
            "y": class_incremental_accuracy,
            "x_label": "Number of seen classes",
            "y_label": "Accuracy (%)",
        },
        "loss_per_epoch": {
            "x": list(range(1, len(loss_history) + 1)),
            "y": loss_history,
            "x_label": "Epoch",
            "y_label": "Loss",
        },
    }

    # 5. results.json
    results = {
        "schema_version": 1,
        "protocol": "Split CIFAR-100 class-incremental",
        "experiment_type": "sequential",
        "initialization": initialization,
        "model_name": model_name,
        "num_tasks": num_tasks,
        "classes_per_task": 10,
        "epochs_completed": len(loss_history),
        "total_epochs": num_tasks * epochs_per_task,
        "loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "standard_class_incremental": standard_matrix,
        "task_incremental": task_matrix,
        "history": history,
        "accuracy_history": accuracy_history,
        "accuracy_epochs": list(range(1, len(loss_history) + 1)),
        "best_accuracy": None,
        "best_epoch": None,
        "early_stopping": {"enabled": False, "evaluation_interval": None, "patience": None, "min_delta": None},
        "standard_class_incremental_accuracy": (
            history[-1]["standard_seen_accuracy"] if history else None
        ),
        "task_incremental_accuracy": (
            float(np.mean(task_matrix[-1][: len(history)])) if history else None
        ),
        "final_accuracy_percent": (
            100.0 * history[-1]["standard_seen_accuracy"] if history else None
        ),
        "confusion_matrix": confusion_matrix_arr.tolist(),
        "r_matrix": standard_matrix,
        "task_incremental_r_matrix": task_matrix,
        "bit_flips": bit_flips,
        "bit_flip_percentages": bit_flip_percentages,
        "total_binary_bits": total_binary_bits,
        "bit_flip_percentage": (
            float(sum(bit_flips) / (total_binary_bits * len(bit_flips)) * 100.0)
            if total_binary_bits and bit_flips
            else None
        ),
        "bwt": bwt_values,
        "accuracy_vs_classes": {
            "classes_seen": classes_seen,
            "accuracy_percent": class_incremental_accuracy,
        },
    }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "plot_points.json").write_text(json.dumps(plot_points, indent=2), encoding="utf-8")

    # 6. Generazione e salvataggio di tutti i grafici PNG con le etichette corrette
    plot_matrix(
        standard_matrix,
        "Standard class-incremental accuracy",
        output_dir / "standard_r_matrix.png",
        xlabel="Evaluated task",
        ylabel="Training task",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    plot_matrix(
        task_matrix,
        "Task-incremental accuracy (masked logits)",
        output_dir / "task_incremental_r_matrix.png",
        xlabel="Evaluated task",
        ylabel="Training task",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    plot_accuracy_history(
        accuracy_history,
        output_dir / "accuracy_per_epoch.png",
        title="Accuracy per epoch",
    )
    plot_task_metric(
        bwt_values,
        "BWT (%)",
        "Backward transfer per task",
        output_dir / "bwt_per_task.png",
        connect_points=True,
        percentage_axis=True,
    )
    plot_task_metric(
        bit_flips,
        "Bit flips",
        "Binary bit flips per task",
        output_dir / "bitflips_per_task.png",
        connect_points=True,
    )
    plot_accuracy_vs_classes(
        class_incremental_accuracy,
        classes_seen,
        output_dir / "accuracy_vs_classes.png",
    )

