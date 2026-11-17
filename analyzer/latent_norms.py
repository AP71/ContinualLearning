"""Crescita della norma dei pesi latenti binari lungo la sequenza di task.

Perche' serve: nei conv binari di A&B-BNN (HardBinaryScaledStdConv2d) il forward vede solo
z = (W - mu) / sigma, quindi conta la direzione di W e non la sua lunghezza. Il gradiente e'
ortogonale a (W - mu), per cui ||W - mu|| puo' solo crescere, e un passo di Adam ruota W di
circa lr / ||W - mu||. Se la norma cresce, la plasticita' cala da sola, qualunque sia il metodo.

Per il miglior run di ogni metodo (stesso criterio di results_table: Final Class-IL piu' alto)
e per il JOINT da checkpoint legge initial.h5, after_task_01.h5, ... di ogni seed e calcola,
per ogni layer binario, la norma media dei filtri centrati ||W - mu_filtro||.

Rapporti riportati (r = norma / norma iniziale):
  - r_start_last : norma all'inizio dell'ultimo task / norma all'inizio del task 1.
                   Con Adam il lr effettivo dell'ultimo task e' circa lr / r_start_last.
                   (JOINT: un solo task, si usa la norma finale.)
  - r_final      : norma dopo l'ultimo task / norma iniziale.

Il calcolo per esperimento finisce in <esperimento>/latent_norms.json e viene saltato se il
file esiste gia' (usa --force per rifarlo). Output comuni in analyzer/comparison_plots/:
  - latent_norms.csv               tabella riassuntiva (separatore ;)
  - latent_norms_per_task.png      norma globale relativa task per task, un metodo per linea
  - latent_norms_per_layer.png     r_final per layer, un metodo per linea

Uso (dalla root del repo):
    python -m analyzer.latent_norms                  # miglior run di ogni metodo + JOINT
    python -m analyzer.latent_norms --force          # ricalcola anche i JSON gia' presenti
    python -m analyzer.latent_norms --all-runs       # tutti i run, non solo il migliore
    python -m analyzer.latent_norms --include-random # aggiunge i run da inizializzazione random
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzer.results_table import collect_rows  # noqa: E402
from cl.results_layout import RESULTS_ROOT, find_experiment  # noqa: E402
from utils.checkpoint_analysis import _default_binary_layer, _load_state_dict  # noqa: E402

OUTPUT_DIR = ROOT / "analyzer" / "comparison_plots"
JSON_NAME = "latent_norms.json"
SCHEMA_VERSION = 1

# soglie sul calo del lr effettivo (Adam: lr_eff ~ lr / r)
NEGLIGIBLE, NOTABLE = 1.05, 1.30


# ----------------------------------------------------------------------------- calcolo
@torch.no_grad()
def checkpoint_norms(path: Path) -> tuple[dict[str, float], float]:
    """Norma media dei filtri centrati per layer binario, piu' la norma globale."""
    state = _load_state_dict(path)
    layers: dict[str, float] = {}
    total_sq = 0.0
    for name, w in state.items():
        if not (name.endswith(".weight") and w.ndim == 4 and _default_binary_layer(name, w)):
            continue
        wc = w.float() - w.float().mean(dim=(1, 2, 3), keepdim=True)
        per_filter = wc.flatten(1).norm(dim=1)
        layers[name] = float(per_filter.mean())
        total_sq += float(per_filter.pow(2).sum())
    if not layers:
        raise ValueError(f"nessun layer binario in {path}")
    return layers, total_sq ** 0.5


def checkpoint_chain(checkpoints: Path) -> list[Path]:
    initial = checkpoints / "initial.h5"
    tasks = sorted(checkpoints.glob("after_task_*.h5"))
    return [initial, *tasks] if initial.is_file() and tasks else []


def analyze_experiment(exp: Path, force: bool) -> dict[str, Any] | None:
    out_path = exp / JSON_NAME
    if out_path.is_file() and not force:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        if data.get("schema_version") == SCHEMA_VERSION:
            print("  -> gia' calcolato, letto da", JSON_NAME)
            return data

    seeds: dict[str, Any] = {}
    for seed_dir in sorted(d for d in exp.glob("seed*") if (d / "checkpoints").is_dir()):
        chain = checkpoint_chain(seed_dir / "checkpoints")
        if not chain:
            print(f"    {seed_dir.name}: checkpoint incompleti, saltato")
            continue
        per_layer: dict[str, list[float]] = {}
        global_norm: list[float] = []
        for ckpt in chain:
            layers, g = checkpoint_norms(ckpt)
            for name, value in layers.items():
                per_layer.setdefault(name, []).append(value)
            global_norm.append(g)
        seeds[seed_dir.name] = {
            "checkpoints": [c.stem for c in chain],
            "layers": per_layer,
            "global": global_norm,
        }
        print(f"    {seed_dir.name}: {len(chain)} checkpoint, "
              f"norma globale {global_norm[0]:.3f} -> {global_norm[-1]:.3f}")
    if not seeds:
        return None

    data = {"schema_version": SCHEMA_VERSION, "experiment": exp.name, "seeds": seeds}
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# ----------------------------------------------------------------------------- sintesi
def _ratios(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr / arr[0]


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    """Rapporti per seed, poi media e deviazione standard sui seed."""
    curves, start_last, final, layer_final = [], [], [], []
    layer_names: list[str] | None = None
    for run in data["seeds"].values():
        r = _ratios(run["global"])
        curves.append(r)
        # inizio ultimo task = checkpoint precedente all'ultimo; con un solo task si usa il finale
        start_last.append(r[-2] if len(r) > 2 else r[-1])
        final.append(r[-1])
        names = list(run["layers"])
        layer_names = layer_names or names
        layer_final.append([_ratios(run["layers"][n])[-1] for n in layer_names])

    n_steps = min(len(c) for c in curves)
    curves = np.stack([c[:n_steps] for c in curves])
    layer_final = np.asarray(layer_final)
    layer_mean = layer_final.mean(axis=0)
    worst = int(np.argmax(layer_mean))
    std = (lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)
    return {
        "seeds": len(curves),
        "curve_mean": curves.mean(axis=0),
        "curve_std": curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(n_steps),
        "r_start_last": (float(np.mean(start_last)), std(start_last)),
        "r_final": (float(np.mean(final)), std(final)),
        "layer_names": layer_names,
        "layer_final": layer_mean,
        "worst_layer": (layer_names[worst], float(layer_mean[worst])),
    }


def verdict(r: float) -> str:
    if r < NEGLIGIBLE:
        return "trascurabile"
    if r < NOTABLE:
        return "da citare"
    return "effetto reale"


def _label(row: dict[str, Any]) -> str:
    run, group = row["run"], row["group"]
    return run if run.lower().startswith(group.lower()) or group == "Reference" else f"{group} – {run}"


def _short_layer(name: str) -> str:
    return name.removeprefix("feature.").removesuffix(".weight").replace("binary_", "")


# ----------------------------------------------------------------------------- output
def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, delimiter=";")
        w.writerow(["Method", "Run", "Seeds",
                    "r_start_last", "r_start_last std", "Adam lr_eff ultimo task (%)",
                    "r_final", "r_final std", "Layer max r_final", "r_final layer max", "Verdetto"])
        for row in rows:
            s = row["summary"]
            r = s["r_start_last"][0]
            w.writerow([row["group"], row["run"], s["seeds"],
                        f"{r:.4f}", f"{s['r_start_last'][1]:.4f}", f"{(1 / r - 1) * 100:+.1f}",
                        f"{s['r_final'][0]:.4f}", f"{s['r_final'][1]:.4f}",
                        _short_layer(s["worst_layer"][0]), f"{s['worst_layer'][1]:.4f}",
                        verdict(r)])


def _legend_below(ax, offset: float) -> None:
    """Legenda sotto l'asse x su tutta la larghezza (come accuracy_vs_classes)."""
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.0, offset, 1.0, 0.0),
        mode="expand",
        ncol=min(4, max(1, len(labels))),
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
    )


def plot_per_task(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row in rows:
        s = row["summary"]
        label = _label(row)
        if row["is_joint"]:
            ax.axhline(s["r_final"][0], ls="--", lw=1.2, color="gray", label=f"{label} (final)")
            continue
        x = np.arange(len(s["curve_mean"]))
        ax.plot(x, s["curve_mean"], marker="o", ms=3, label=label)
        ax.fill_between(x, s["curve_mean"] - s["curve_std"], s["curve_mean"] + s["curve_std"], alpha=0.15)
    for y, ls in ((NEGLIGIBLE, ":"), (NOTABLE, "-.")):
        ax.axhline(y, ls=ls, lw=0.8, color="black", alpha=0.5)
    ax.set_xlabel("Checkpoint (0 = initial, t = after task t)")
    ax.set_ylabel(r"$\|W-\mu\|_t \,/\, \|W-\mu\|_0$  (all binary layers)")
    ax.set_title("Latent weight norm growth (Adam: effective lr ∝ 1 / ratio)")
    ax.grid(alpha=0.3)
    _legend_below(ax, -0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_per_layer(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    names = rows[0]["summary"]["layer_names"]
    for row in rows:
        s = row["summary"]
        if s["layer_names"] != names:
            print(f"  [plot layer] {row['run']}: layer diversi, saltato")
            continue
        ax.plot(range(len(names)), s["layer_final"], marker="o", ms=3,
                ls="--" if row["is_joint"] else "-", label=_label(row))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([_short_layer(n) for n in names], rotation=90, fontsize=7)
    ax.set_ylabel(r"$\|W-\mu\|_{final} \,/\, \|W-\mu\|_0$")
    ax.set_title("Latent norm growth per binary layer (end of sequence)")
    ax.grid(alpha=0.3)
    _legend_below(ax, -0.32)  # più in basso: sotto le etichette dei layer ruotate
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def select_rows(all_runs: bool, include_random: bool) -> list[dict[str, Any]]:
    selected = []
    for row in collect_rows(RESULTS_ROOT):
        folder = row["folder"].lower()
        is_joint = "joint" in folder
        if "_random_" in folder and not include_random:
            continue
        if is_joint or (row["kind"] == "run" and (all_runs or row.get("best"))):
            selected.append({**row, "is_joint": is_joint})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="ricalcola anche se latent_norms.json esiste")
    parser.add_argument("--all-runs", action="store_true", help="tutti i run, non solo il migliore per metodo")
    parser.add_argument("--include-random", action="store_true", help="includi i run da init random")
    args = parser.parse_args()

    rows = []
    for row in select_rows(args.all_runs, args.include_random):
        exp = find_experiment(row["folder"])
        print(f"[{row['group']}] {row['folder']}")
        if exp is None:
            print("  -> cartella non trovata, saltato")
            continue
        try:
            data = analyze_experiment(exp, args.force)
        except Exception as exc:  # continua con gli altri esperimenti
            print(f"  -> ERRORE: {exc!r}")
            continue
        if data is None:
            print("  -> nessun checkpoint, saltato")
            continue
        rows.append({**row, "summary": summarize(data)})

    if not rows:
        raise SystemExit("Nessun esperimento con checkpoint trovato.")
    rows.sort(key=lambda r: (r["is_joint"], r["group"].lower()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "latent_norms.csv"
    write_csv(rows, csv_path)
    plot_per_task(rows, OUTPUT_DIR / "latent_norms_per_task.png")
    plot_per_layer(rows, OUTPUT_DIR / "latent_norms_per_layer.png")

    print(f"\n{'Metodo':16s}{'Run':42s}{'r_start_last':>14s}{'lr_eff':>9s}{'r_final':>10s}  verdetto")
    for row in rows:
        s = row["summary"]
        r = s["r_start_last"][0]
        print(f"{row['group']:16s}{row['run'][:41]:42s}{r:14.4f}{(1 / r - 1) * 100:+8.1f}%"
              f"{s['r_final'][0]:10.4f}  {verdict(r)}")
    print(f"\n  {csv_path}\n  {OUTPUT_DIR / 'latent_norms_per_task.png'}\n  {OUTPUT_DIR / 'latent_norms_per_layer.png'}")


if __name__ == "__main__":
    main()
