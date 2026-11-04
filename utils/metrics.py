from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def calculate_bwt(matrix: list[list[float | None]]) -> float:
    """Calcola il Backward Transfer medio (BWT) in percentuale [-100, 0]."""
    T = len(matrix)
    if T <= 1:
        return 0.0
    diffs = []
    for i in range(T - 1):
        if matrix[T - 1][i] is not None and matrix[i][i] is not None:
            diff = matrix[T - 1][i] - matrix[i][i]
            # Se la differenza e' espressa in scala [0, 1], convertila in percentuale [-100, 100]
            if abs(diff) <= 1.0:
                diff *= 100.0
            diffs.append(diff)

    return float(np.mean(diffs)) if diffs else 0.0


def plot_accuracy_history(
    accuracy_history,
    output_path,
    task_start_epochs=None,
    title=None,
    epoch_points=None,
    variance_history=None,
    show_markers=True,
    legend_labels=None,
):
    """Plot one accuracy curve for every task over global training epochs."""
    figure, axis = plt.subplots(figsize=(10, 6))
    markers = ["o", "^", "s"]
    for task_id, values in enumerate(accuracy_history):
        start_epoch = (
            task_start_epochs[task_id] if task_start_epochs is not None else 0
        )
        epochs = (
            epoch_points
            if epoch_points is not None
            else range(start_epoch + 1, start_epoch + len(values) + 1)
        )
        axis.plot(
            epochs,
            values,
            marker=markers[task_id % len(markers)] if show_markers else None,
            markersize=4 if show_markers else 0,
            label=(
                legend_labels[task_id]
                if legend_labels is not None
                else f"Task {task_id + 1}"
            ),
        )
        if variance_history is not None:
            variance = np.asarray(variance_history[task_id], dtype=float)
            axis.fill_between(
                epochs,
                np.asarray(values, dtype=float) - variance,
                np.asarray(values, dtype=float) + variance,
                alpha=0.15,
            )
    if title:
        axis.set_title(title, fontsize=13, pad=10)
    axis.set_xlabel("Epoch", fontsize=11)
    axis.set_ylabel("Accuracy", fontsize=11)
    axis.set_xlim(left=1)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, loc="lower right")
    figure.tight_layout()
    figure.savefig(Path(output_path), dpi=300)
    plt.close(figure)


def plot_task_metric(
    values: list[float],
    ylabel: str,
    title: str,
    output_path: Path | str,
    connect_points: bool = True,
    percentage_axis: bool = False,
    point_labels: list[float] | None = None,
    label_format: str = "{:.2f}",
    variance: list[float] | None = None,
    variance_as_errorbar: bool = False,
) -> None:
    """Disegna l'andamento di una metrica per task (BWT o Bit-flips)."""
    figure, axis = plt.subplots(figsize=(8, 5))
    x = list(range(1, len(values) + 1)) if len(values) > 1 else [1]

    if connect_points and len(values) > 1:
        axis.plot(x, values, marker="o", color="#1f77b4", linewidth=2, markersize=6)
    else:
        axis.scatter(x, values, color="#1f77b4", s=50)
    if variance is not None and variance_as_errorbar:
        axis.errorbar(
            x,
            values,
            yerr=np.asarray(variance, dtype=float),
            fmt="none",
            ecolor="#1f77b4",
            capsize=5,
            linewidth=1.5,
        )
    elif variance is not None:
        values_array = np.asarray(values, dtype=float)
        variance_array = np.asarray(variance, dtype=float)
        axis.fill_between(
            x,
            values_array - variance_array,
            values_array + variance_array,
            color="#1f77b4",
            alpha=0.2,
        )

    if point_labels:
        for xi, yi, lbl in zip(x, values, point_labels):
            axis.annotate(
                label_format.format(lbl),
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

    axis.set_title(title, fontsize=13)
    axis.set_xlabel("Task" if len(values) > 1 else "", fontsize=11)
    axis.set_ylabel(ylabel, fontsize=11)
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.set_xticks(x)

    if percentage_axis:
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    figure.tight_layout()
    figure.savefig(Path(output_path), dpi=300)
    plt.close(figure)


def plot_accuracy_vs_classes(accuracies, classes, output_path, variance=None):
    """Plot class-incremental accuracy as the number of seen classes grows."""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        classes,
        accuracies,
        marker="o",
        color="tab:blue",
        linewidth=2,
        label="Mean",
    )
    if variance is not None:
        mean = np.asarray(accuracies, dtype=float)
        spread = np.asarray(variance, dtype=float)
        axis.fill_between(
            classes,
            mean - spread,
            mean + spread,
            alpha=0.2,
            color="tab:blue",
        )
    axis.set_xlabel("Number of seen classes", fontsize=11)
    axis.set_ylabel("Accuracy (%)", fontsize=11)
    axis.set_ylim(0.0, 100.0)
    axis.set_xticks(classes)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(Path(output_path), dpi=300)
    plt.close(figure)


def plot_matrix(
    matrix,
    title,
    output_path,
    *,
    xlabel,
    ylabel,
    cmap="Blues",
    show_ticks=True,
    vmin=0.0,
    vmax=1.0,
    annotations=None,
    show_values=True,
    show_tick_labels=True,
):
    """Plot and save a matrix with values annotated in each valid cell."""
    values = np.asarray(matrix, dtype=float)
    figure, axis = plt.subplots(figsize=(9, 7))

    masked_values = np.ma.masked_invalid(values)
    image = axis.imshow(
        masked_values,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    figure.colorbar(image, ax=axis)

    # Scrive il valore numerico al centro di ogni cella valida (visibile)
    threshold = (vmin + vmax) / 2.0 if (vmin is not None and vmax is not None) else 0.5
    if show_values:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if not np.isnan(val):
                    # Contrasto del testo: bianco su sfondi scuri, nero su chiari
                    color = "white" if val > threshold else "black"
                    annotation = (
                        annotations[i][j]
                        if annotations is not None
                        else f"{val:.2f}"
                    )
                    axis.text(
                        j,
                        i,
                        annotation,
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                        linespacing=0.8,
                    )

    if title:
        axis.set_title(title, fontsize=13, pad=12)
    axis.set_xlabel(xlabel, fontsize=11)
    axis.set_ylabel(ylabel, fontsize=11)

    if show_ticks:
        axis.set_xticks(range(values.shape[1]))
        axis.set_yticks(range(values.shape[0]))
        if show_tick_labels:
            axis.set_xticklabels(range(1, values.shape[1] + 1))
            axis.set_yticklabels(range(1, values.shape[0] + 1))
        else:
            axis.set_xticklabels([])
            axis.set_yticklabels([])
    else:
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    figure.savefig(Path(output_path), dpi=300)
    plt.close(figure)