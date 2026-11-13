"""Layout di cl/results: una cartella per metodo, dentro le cartelle degli esperimenti.

    cl/results/<gruppo>/<esperimento>/results.json
    es. cl/results/ATSLDA/cifar100_avalanche_AdaptThenSLDA_checkpoint_resneta_3seeds_a1_from12

Il gruppo di un metodo si legge da METHOD_GROUPS (nome passato a run_method -> cartella);
un metodo non elencato finisce in una cartella con il suo stesso nome.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "cl" / "results"

METHOD_GROUPS = {
    "AdaptThenSLDA": "ATSLDA",
    "BinaryLatentReplay": "BLR",
    "BinaryLatentReplaySLDA": "BLR_SLDA",
    "EWC": "EWC",
    "Freezing": "Freezing",
    "FreezingHead": "Freezing",
    "JOINT": "JOINT",
    "LwF": "LWF",
    "Metaplasticity": "Metaplasticity",
    "MetaSLDA": "Meta_SLDA",
    "naive": "naive",
    "Replay": "Replay",
    "linear_probe": "Linear",
    "statistical_heads": "Linear",
}

# Cartelle in cl/results che non contengono esperimenti
NON_EXPERIMENT_DIRS = {"reactnet_inspection"}


def group_name(method: str) -> str:
    return METHOD_GROUPS.get(method, method)


def results_dir(method: str, results_root: Path = RESULTS_ROOT) -> Path:
    """Cartella del metodo, es. results_dir("AdaptThenSLDA") -> cl/results/ATSLDA."""
    return results_root / group_name(method)


def experiment_dirs(results_root: Path = RESULTS_ROOT) -> list[Path]:
    """Tutte le cartelle di esperimento, gruppo per gruppo (ordinate per nome).

    Una cartella al primo livello con un proprio results.json (vecchio layout) viene
    trattata come esperimento.
    """
    found: list[Path] = []
    if not results_root.is_dir():
        return found
    for top in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if top.name in NON_EXPERIMENT_DIRS:
            continue
        if (top / "results.json").is_file():
            found.append(top)
            continue
        found.extend(sorted(p for p in top.iterdir() if p.is_dir()))
    return found


def find_experiment(folder_name: str, results_root: Path = RESULTS_ROOT) -> Path | None:
    """Cartella di esperimento con questo nome, in qualunque gruppo."""
    for path in experiment_dirs(results_root):
        if path.name == folder_name:
            return path
    return None
