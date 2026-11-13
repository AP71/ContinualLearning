"""Ricalcola i bit flip di tutti gli esperimenti dai checkpoint salvati, con i bit
effettivi di A&B-BNN (sign(W - media filtro), vedi utils.checkpoint_analysis.sws_bits).

Per ogni esperimento in cl/results e per ogni seed confronta in sequenza
initial -> after_task_01 -> after_task_02 -> ... (stesse coppie del plugin
CheckpointAndBitFlipPlugin), poi aggiorna:
  - seedXX/results.json      (bit_flips, bit_flip_percentages, total_bit_flips, ...)
  - seedXX/plot_points.json  (bitflips_per_task)
  - results.json aggregato   (seeds[...]) e rigenera i grafici con analyze_results
Al primo passaggio salva una copia *.pre_sws.json di ogni file modificato.
Gli esperimenti gia' ricalcolati (bit_flip_method == "sws") vengono saltati.

Uso (dalla root del repo):
    python -m analyzer.recompute_bitflips             # tutti
    python -m analyzer.recompute_bitflips LwF naive    # solo cartelle che contengono questi testi
    python -m analyzer.recompute_bitflips --force      # ricalcola anche se gia' fatto
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.checkpoint_analysis import compare_checkpoints  # noqa: E402
from cl.results_layout import RESULTS_ROOT, experiment_dirs  # noqa: E402

RESULTS_DIR = RESULTS_ROOT
METHOD = "sws"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    backup = path.with_name(path.stem + ".pre_sws.json")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _checkpoint_chain(checkpoints: Path) -> list[Path]:
    tasks = sorted(checkpoints.glob("after_task_*.h5"))
    initial = checkpoints / "initial.h5"
    return [initial, *tasks] if initial.is_file() and tasks else []


def _seed_flips(chain: list[Path]) -> dict:
    flips, pcts, total_bits = [], [], None
    for prev, curr in zip(chain, chain[1:]):
        summary = compare_checkpoints(prev, curr, distribution_sample_size=1)["binary_summary"]
        flips.append(int(summary["bit_flips"]))
        pcts.append(float(summary["bit_flip_percentage"]))
        total_bits = int(summary["total_bits"])
    return {
        "bit_flips": flips,
        "bit_flip_percentages": pcts,
        "total_binary_bits": total_bits,
        "bit_flip_percentage": (
            float(sum(flips) / (total_bits * len(flips)) * 100.0) if total_bits and flips else None
        ),
        "total_bit_flips": int(sum(flips)),
        "bit_flip_method": METHOD,
    }


def _apply(run: dict, new: dict) -> None:
    for key, value in new.items():
        if key == "total_bit_flips" and key not in run:
            continue  # chiave presente solo negli esperimenti JOINT
        run[key] = value


def process_experiment(exp: Path, force: bool) -> str:
    agg_path = exp / "results.json"
    seed_dirs = sorted(d for d in exp.glob("seed*") if (d / "checkpoints").is_dir())
    if not agg_path.is_file() or not seed_dirs:
        return "skip (niente results.json o checkpoint)"
    agg = _read(agg_path)
    seeds = agg.get("seeds", {})
    if not force and seeds and all(r.get("bit_flip_method") == METHOD for r in seeds.values()):
        return "skip (gia' ricalcolato)"

    for seed_dir in seed_dirs:
        chain = _checkpoint_chain(seed_dir / "checkpoints")
        if not chain:
            print(f"    {seed_dir.name}: checkpoint incompleti, saltato")
            continue
        new = _seed_flips(chain)
        old = None

        seed_json = seed_dir / "results.json"
        if seed_json.is_file():
            run = _read(seed_json)
            old = run.get("bit_flips")
            _apply(run, new)
            _write(seed_json, run)

        points_json = seed_dir / "plot_points.json"
        if points_json.is_file():
            points = _read(points_json)
            if "bitflips_per_task" in points:
                points["bitflips_per_task"]["x"] = list(range(len(new["bit_flips"])))
                points["bitflips_per_task"]["y"] = new["bit_flips"]
                _write(points_json, points)

        key = seed_dir.name.removeprefix("seed")
        key = key if key in seeds else seed_dir.name
        if key in seeds:
            _apply(seeds[key], new)

        old_tot = sum(old) if isinstance(old, list) else None
        print(f"    {seed_dir.name}: flip totali {old_tot} -> {new['total_bit_flips']}")

    _write(agg_path, agg)
    from analyzer.analyze_results import analyze_results
    analyze_results(agg_path)
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filters", nargs="*", help="testi da cercare nel nome della cartella")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiments = sorted(
        (p for p in experiment_dirs(RESULTS_DIR)  # cl/results/<gruppo>/<esperimento>
         if not args.filters or any(f.lower() in p.name.lower() for f in args.filters))
    )
    for exp in experiments:
        print(f"[{exp.name}]")
        try:
            print(f"  -> {process_experiment(exp, args.force)}")
        except Exception as exc:  # continua con gli altri esperimenti
            print(f"  -> ERRORE: {exc!r}")


if __name__ == "__main__":
    main()
