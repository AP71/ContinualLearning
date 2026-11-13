from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzer.analyze_results import _statistics
from cl.results_layout import RESULTS_ROOT, experiment_dirs, results_dir

# True: per ogni metodo (cartella di cl/results) solo la run con la Class-IL finale
# piu' alta; False: tutte le run.
BEST_RUN_PER_METHOD = True

# Riferimenti Class-IL (linee orizzontali tratteggiate): (etichetta, cartella, estrattore, stile)
_REFERENCE_STYLES = {
    "JOINT": {"color": "black", "linestyle": "--"},
    "Linear probe": {"color": "#555555", "linestyle": "-."},
}
_SLDA_STYLE = {"color": "#777777", "linestyle": "--", "marker": "x", "linewidth": 1.8, "markersize": 6}


_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#006d77",
    "#f4a261",
    "#2a9d8f",
    "#e9c46a",
    "#264653",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "H", "8", "<", ">"]


def _clean_experiment_name(folder_name: str) -> str:
    text = folder_name
    prefix = "cifar100_avalanche_"
    if text.startswith(prefix):
        text = text[len(prefix):]

    if text.startswith("metaplasticity"):
        if "paper" in text:
            return "Metaplasticity paper"

        match = re.search(r"v(\d+)(?:_m(\d+))?", text)
        if match:
            version = match.group(1)
            momentum = match.group(2)
            base = f"Metaplasticity v{version}"
            if momentum:
                return f"{base} m{momentum}"
            return base
        return "Metaplasticity"

    if text.startswith("naive_random"):
        return "Naive random"
    if text.startswith("naive_checkpoint") or text.startswith("naive"):
        return "Naive checkpoint"

    text = text.replace("_checkpoint_resneta_3seeds", "")
    text = text.replace("_random_resneta_3seeds", "")
    text = text.replace("_resneta_3seeds", "")
    text = text.replace("_checkpoint", "")
    text = text.replace("_random", "")
    text = text.replace("_v2", " v2")
    text = text.replace("_v3", " v3")
    text = text.replace("_v4", " v4")
    text = text.replace("_m3", " m3")
    text = text.replace("_m6", " m6")
    text = text.replace("_", " ")
    return text.strip()


def _is_sequential_result(results_path: Path) -> bool:
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    experiment_type = str(payload.get("experiment_type", "")).lower()
    if "joint" in experiment_type:
        return False
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict) or not seeds:
        return False
    return True


def _load_top_level_sequential_results() -> list[tuple[str, dict[str, Any], str]]:
    """(nome cartella, results.json, gruppo/metodo) di ogni run sequenziale."""
    results: list[tuple[str, dict[str, Any], str]] = []
    if not RESULTS_ROOT.exists():
        return results
    # cl/results/<gruppo>/<esperimento>/results.json
    for folder in experiment_dirs(RESULTS_ROOT):
        results_json = folder / "results.json"
        if not results_json.is_file():
            continue
        if not _is_sequential_result(results_json):
            continue
        try:
            payload = json.loads(results_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results.append((folder.name, payload, folder.parent.name))
    return results


def _final_class_incremental(series: dict[str, Any]) -> float:
    """Class-IL (%) a fine sequenza, media dei seed; -inf se manca."""
    curve = series.get("accuracy_vs_classes")
    if curve is None:
        return float("-inf")
    return float(np.asarray(curve[1], dtype=float)[-1])


def _best_run_per_method(
    runs: list[tuple[str, dict[str, Any], str]],
) -> list[tuple[str, dict[str, Any], str]]:
    """Per ogni gruppo tiene la run con la Class-IL finale piu' alta."""
    best: dict[str, tuple[float, tuple[str, dict[str, Any], str]]] = {}
    for run in runs:
        score = _final_class_incremental(_series_from_experiment(run[1]))
        if score == float("-inf"):  # nessuna curva Class-IL (es. linear probe)
            continue
        if run[2] not in best or score > best[run[2]][0]:
            best[run[2]] = (score, run)
    print("Miglior run per metodo (Class-IL finale):")
    for group, (score, run) in sorted(best.items()):
        print(f"  {group:15s} {score:6.2f}%  {run[0]}")
    return [run for _, run in best.values()]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_reference_lines() -> dict[str, float]:
    """Accuratezza finale (%) di JOINT e linear probe, se i risultati esistono."""
    references: dict[str, float] = {}

    joint = _read_json(results_dir("JOINT") / "cifar100_avalanche_JOINT_checkpoint_resneta_3seeds" / "results.json")
    if joint and isinstance(joint.get("seeds"), dict):
        values = [
            float(seed["final_accuracy_percent"])
            for seed in joint["seeds"].values()
            if isinstance(seed, dict) and "final_accuracy_percent" in seed
        ]
        if values:
            references["JOINT"] = float(np.mean(values))

    probe = _read_json(results_dir("linear_probe") / "cifar100_linear_probe_checkpoint_resneta_3seeds" / "results.json")
    if probe:
        value = probe.get("test_accuracy_at_best_val_percent", {}).get("mean")
        if value is not None:
            references["Linear probe"] = float(value)

    return references


def _load_slda_curve() -> tuple[np.ndarray, np.ndarray] | None:
    """Curva Class-IL dello SLDA sul backbone congelato (media dei seed), se esiste."""
    heads = _read_json(results_dir("statistical_heads") / "cifar100_statistical_heads_resneta_3seeds" / "results.json")
    if not heads:
        return None
    y = heads.get("heads", {}).get("slda", {}).get("accuracy_vs_classes_percent")
    if not isinstance(y, list) or not y:
        return None
    y_arr = np.asarray(y, dtype=float)
    return 10.0 * np.arange(1, len(y_arr) + 1), y_arr


def _mean_across_runs(values: list[Any]) -> np.ndarray:
    stats = _statistics(values)
    mean = np.asarray(stats["mean"], dtype=float)
    return mean.reshape(-1)


def _series_from_experiment(experiment: dict[str, Any]) -> dict[str, np.ndarray]:
    seeds = experiment.get("seeds")
    if not isinstance(seeds, dict):
        return {}

    series = {
        "accuracy_vs_classes": None,
        "task_incremental_accuracy_vs_classes": None,
        "bit_flips": None,
        "bwt": None,
    }

    task_incremental_samples: list[tuple[np.ndarray, np.ndarray]] = []
    acc_values = []
    classes_values = []
    for seed_data in seeds.values():
        if not isinstance(seed_data, dict):
            continue

        matrix = seed_data.get("task_incremental")
        if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
            arr = np.asarray(matrix, dtype=float)
            if arr.ndim == 2 and arr.size:
                row_mean = np.nanmean(arr, axis=1)
                classes_per_task = int(seed_data.get("classes_per_task", 10) or 10)
                classes_seen = np.asarray(
                    [classes_per_task * (idx + 1) for idx in range(len(row_mean))],
                    dtype=float,
                )
                task_incremental_samples.append((classes_seen, 100.0 * row_mean))

        acc_entry = seed_data.get("accuracy_vs_classes")
        if isinstance(acc_entry, dict):
            acc_values.append(acc_entry.get("accuracy_percent"))
            classes_values.append(acc_entry.get("classes_seen"))

    if task_incremental_samples:
        x_all = np.asarray([pair[0] for pair in task_incremental_samples], dtype=float)
        y_all = np.asarray([pair[1] for pair in task_incremental_samples], dtype=float)
        if x_all.size and y_all.size:
            x = x_all[0]
            y = np.nanmean(y_all, axis=0)
            series["task_incremental_accuracy_vs_classes"] = (x, y)

    if acc_values and classes_values and all(isinstance(values, list) for values in acc_values):
        all_acc = [np.asarray(values, dtype=float) for values in acc_values]
        x = np.asarray(classes_values[0], dtype=float)
        y = np.mean(np.vstack([arr for arr in all_acc]), axis=0)
        series["accuracy_vs_classes"] = (x, y)

    bitvalues = []
    for seed_data in seeds.values():
        if isinstance(seed_data, dict):
            value = seed_data.get("bit_flips")
            if isinstance(value, list):
                bitvalues.append(value)
    if bitvalues:
        series["bit_flips"] = _mean_across_runs(bitvalues)

    bwtvalues = []
    for seed_data in seeds.values():
        if isinstance(seed_data, dict):
            value = seed_data.get("bwt")
            if isinstance(value, list):
                bwtvalues.append(value)
    if bwtvalues:
        series["bwt"] = _mean_across_runs(bwtvalues)

    return {key: value for key, value in series.items() if value is not None}


def _save_comparison_plot(
    series_by_experiment: dict[str, dict[str, np.ndarray]],
    *,
    metric_key: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    y_formatter: Any | None = None,
    x_values: Any | None = None,
    reference_lines: dict[str, float] | None = None,
    slda_curve: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    for idx, (experiment_name, series) in enumerate(sorted(series_by_experiment.items())):
        color = _COLORS[idx % len(_COLORS)]
        marker = _MARKERS[idx % len(_MARKERS)]
        if metric_key == "accuracy_vs_classes" or metric_key == "task_incremental_accuracy_vs_classes":
            x, y = series[metric_key]
            axis.plot(x, y, marker=marker, linewidth=2, markersize=6, color=color, label=experiment_name)
        else:
            y = series[metric_key]
            x = np.arange(1, len(y) + 1) if x_values is None else x_values
            axis.plot(x, y, marker=marker, linewidth=2, markersize=6, color=color, label=experiment_name)

    if slda_curve is not None:
        x, y = slda_curve
        axis.plot(x, y, label=f"SLDA, no buffer ({y[-1]:.1f}%)", **_SLDA_STYLE)

    for name, value in (reference_lines or {}).items():
        style = _REFERENCE_STYLES.get(name, {"color": "gray", "linestyle": "--"})
        axis.axhline(value, linewidth=1.5, label=f"{name} ({value:.1f}%)", **style)

    axis.set_title(title, fontsize=14)
    axis.set_xlabel(xlabel, fontsize=11)
    axis.set_ylabel(ylabel, fontsize=11)
    axis.grid(True, alpha=0.35)
    handles, labels = axis.get_legend_handles_labels()
    # Legenda sotto l'asse x, su tutta la larghezza, piu' voci per riga
    axis.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.0, -0.12, 1.0, 0.0),
        mode="expand",
        ncol=min(4, max(1, len(labels))),
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
    )
    if y_formatter is not None:
        axis.yaxis.set_major_formatter(y_formatter)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_comparison_plots(output_dir: Path | None = None) -> list[Path]:
    output_dir = output_dir or (ROOT / "analyzer" / "comparison_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    sequential_results = _load_top_level_sequential_results()
    if not sequential_results:
        raise FileNotFoundError("No sequential experiment results were found under cl/results.")
    if BEST_RUN_PER_METHOD:
        sequential_results = _best_run_per_method(sequential_results)

    all_series_by_experiment: dict[str, dict[str, np.ndarray]] = {}
    bitflip_series_by_experiment: dict[str, dict[str, np.ndarray]] = {}
    for folder_name, payload, _group in sequential_results:
        label = _clean_experiment_name(folder_name)
        series = _series_from_experiment(payload)
        if not series:
            continue
        all_series_by_experiment[label] = series
        if label != "Naive random":
            bitflip_series_by_experiment[label] = series

    if not all_series_by_experiment:
        raise ValueError("No valid sequential runs with the expected metrics were found.")

    generated: list[Path] = []

    acc_path_png = output_dir / "accuracy_vs_classes_comparison.png"
    _save_comparison_plot(
        all_series_by_experiment,
        metric_key="accuracy_vs_classes",
        output_path=acc_path_png,
        title="",
        xlabel="Number of seen classes",
        ylabel="Accuracy (%)",
        reference_lines=_load_reference_lines(),
        slda_curve=_load_slda_curve(),
    )
    generated.append(acc_path_png)

    task_acc_path_png = output_dir / "task_incremental_r_matrix_comparison.png"
    _save_comparison_plot(
        all_series_by_experiment,
        metric_key="task_incremental_accuracy_vs_classes",
        output_path=task_acc_path_png,
        title="",
        xlabel="Number of seen classes",
        ylabel="Accuracy (%)",
    )
    generated.append(task_acc_path_png)

    bit_path = output_dir / "bitflips_per_task_comparison.png"
    _save_comparison_plot(
        bitflip_series_by_experiment,
        metric_key="bit_flips",
        output_path=bit_path,
        title="",
        xlabel="Task",
        ylabel="Bit flips",
    )
    generated.append(bit_path)

    bwt_path = output_dir / "bwt_per_task_comparison.png"
    _save_comparison_plot(
        all_series_by_experiment,
        metric_key="bwt",
        output_path=bwt_path,
        title="",
        xlabel="Task",
        ylabel="BWT (%)",
        y_formatter=matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:.0f}%"),
    )
    generated.append(bwt_path)

    return generated


if __name__ == "__main__":
    files = generate_comparison_plots()
    print("Created:")
    for file in files:
        print(file)
