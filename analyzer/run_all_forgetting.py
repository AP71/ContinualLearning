"""Run the layer-wise forgetting analysis on every sequential experiment in results.

Scans ``cl/results/<group>/`` for experiment folders that contain all seed directories
(``seed42``, ``seed2023``, ``seed3047``) with their ``checkpoints``. JOINT runs
are skipped (no sequential checkpoints to compare). For each remaining folder
the analysis runs only if ``forgetting.json`` is missing; plots are then
regenerated (CIL_* and TIL_*).

Example:
    python analyzer/run_all_forgetting.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ANALYZER_DIR = Path(__file__).resolve().parent
for _path in (ANALYZER_DIR, ANALYZER_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from forgetting_analysis import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    NUM_TASKS,
    ROOT,
    SEED_NAMES,
    analyze_forgetting,
    plot_forgetting_results,
)
from cl.results_layout import RESULTS_ROOT, experiment_dirs  # noqa: E402

RESULTS_DIR = RESULTS_ROOT
DATASET_PATH = DEFAULT_DATASET_PATH
BATCH_SIZE = 256
NUM_WORKERS = 0
# True: re-run also folders whose forgetting.json has the old schema
# (TIL only, without CIL statistics). False: skip every folder with a JSON.
REANALYZE_OUTDATED = True


def _has_sequential_checkpoints(experiment_dir: Path) -> bool:
    for seed in SEED_NAMES:
        checkpoints = experiment_dir / seed / "checkpoints"
        for task_id in range(1, NUM_TASKS + 1):
            if not (checkpoints / f"after_task_{task_id:02d}.h5").is_file():
                return False
    return True


def _is_outdated(json_path: Path) -> bool:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return int(payload.get("schema_version", 1)) < 3  # < 3: pre-SWS restore


def find_experiments(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Sequential experiment folders with complete checkpoints, JOINT excluded."""
    return sorted(
        path
        for path in experiment_dirs(results_dir)  # cl/results/<gruppo>/<esperimento>
        if "joint" not in path.name.lower()
        and _has_sequential_checkpoints(path)
    )


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    experiments = find_experiments()
    print(f"Found {len(experiments)} sequential experiments in {RESULTS_DIR}")

    done, skipped, failed = [], [], []
    for experiment in experiments:
        json_path = experiment / "forgetting.json"
        if json_path.is_file() and not (REANALYZE_OUTDATED and _is_outdated(json_path)):
            print(f"[skip] {experiment.parent.name}/{experiment.name}: forgetting.json already present")
            skipped.append(experiment.name)
            continue

        reason = "outdated forgetting.json" if json_path.is_file() else "no forgetting.json"
        print(f"\n[run]  {experiment.name} ({reason})")
        try:
            analyze_forgetting(
                experiment,
                DATASET_PATH,
                output_path=json_path,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
            )
            plots = plot_forgetting_results(json_path)
            print(f"[done] {experiment.name}: {len(plots)} plots written")
            done.append(experiment.name)
        except Exception:  # keep going with the other experiments
            traceback.print_exc()
            print(f"[fail] {experiment.name}")
            failed.append(experiment.name)

    print("\n================ Summary ================")
    print(f"Analyzed: {len(done)}  Skipped: {len(skipped)}  Failed: {len(failed)}")
    for name in failed:
        print(f"  failed: {name}")


if __name__ == "__main__":
    main()
