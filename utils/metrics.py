from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def calculate_bwt(R):
    T = len(R)
    if T <= 1:
        return 0.0
    bwt_sum = 0.0
    for i in range(T - 1):
        bwt_sum += (R[T - 1][i] - R[i][i])
    return bwt_sum / (T - 1)

def plot_accuracy_history(accuracy_history, output_path):
    """Plot one accuracy curve for every task over global training epochs."""
    figure, axis = plt.subplots(figsize=(10, 6))
    markers = ["o", "^", "s"]
    for task_id, values in enumerate(accuracy_history):
        epochs = range(1, len(values) + 1)
        axis.plot(
            epochs,
            values,
            marker=markers[task_id % len(markers)],
            markersize=4,
            label=f"Task {task_id}",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy per task during class-incremental training")
    axis.set_xlim(left=1)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(Path(output_path), dpi=160)
    plt.close(figure)


def plot_task_metric(
    values,
    ylabel,
    title,
    output_path,
    *,
    connect_points=False,
    point_labels=None,
    label_format="{:.2f}%",
    percentage_axis=False,
):
    """Plot a task-level metric with optional connecting lines and labels."""
    figure, axis = plt.subplots(figsize=(8, 5))
    task_ids = list(range(len(values)))
    if connect_points:
        axis.plot(task_ids, values, color="tab:blue", linewidth=1.2, zorder=1)
    axis.scatter(task_ids, values, color="tab:blue", zorder=2)
    if point_labels is not None:
        for task_id, value, label in zip(task_ids, values, point_labels):
            axis.annotate(
                label_format.format(label),
                (task_id, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_xlabel("Task")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if percentage_axis:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_xticks(task_ids)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(Path(output_path), dpi=160)
    plt.close(figure)