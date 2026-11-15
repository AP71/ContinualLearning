"""Unisce in una cartella gli esperimenti degli stessi risultati lanciati su seed diversi.

Usage:
    python analyzer/merge_seeds.py DEST SRC [SRC ...] [--overwrite]

DEST e SRC sono cartelle di esperimento (quelle con results.json e seed<N>/), per esempio
    cl/results/BLR/cifar100_avalanche_BinaryLatentReplay_checkpoint_resneta_3seeds_14_wa
Le cartelle seed<N> di ogni SRC (checkpoint, grafici, log compresi) vengono copiate in DEST e il
results.json di DEST è rigenerato con tutti i seed presenti, nello stesso formato di
run_three_seeds. Se SRC non ha le cartelle seed<N> ma solo un results.json con "seeds", i
risultati di quei seed sono scritti in DEST/seed<N>/results.json. Un seed già presente in DEST
non è sovrascritto (--overwrite per farlo). SRC non viene modificata.
Il blocco "analysis" non viene copiato: rilancia analyzer/analyze_results.py sul nuovo results.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SEED_ORDER = (42, 2023, 3047)


def seed_dirs(folder: Path) -> dict[int, Path]:
    return {
        int(d.name.removeprefix("seed")): d
        for d in folder.glob("seed*")
        if d.is_dir() and d.name.removeprefix("seed").isdigit() and (d / "results.json").exists()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dest", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dest: Path = args.dest
    meta_path = dest / "results.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    existing = seed_dirs(dest)

    for src in args.sources:
        if src.resolve() == dest.resolve():
            sys.exit(f"{src} coincide con la destinazione.")
        found = seed_dirs(src)
        embedded = {}
        if not found and (src / "results.json").exists():
            src_json = json.loads((src / "results.json").read_text(encoding="utf-8"))
            embedded = {int(s): r for s, r in src_json.get("seeds", {}).items()}
            for key in ("strategy", "model_name", "initialization"):
                meta.setdefault(key, src_json.get(key))
        if not found and not embedded:
            sys.exit(f"{src}: nessun seed trovato.")

        for seed in sorted(set(found) | set(embedded)):
            if seed in existing and not args.overwrite:
                sys.exit(f"Il seed {seed} è già in {dest.name}: usa --overwrite per sostituirlo.")
            target = dest / f"seed{seed}"
            if seed in found:
                shutil.copytree(found[seed], target, dirs_exist_ok=True)
                print(f"copiato seed{seed} da {src.name}")
            else:
                target.mkdir(parents=True, exist_ok=True)
                (target / "results.json").write_text(json.dumps(embedded[seed], indent=2), encoding="utf-8")
                print(f"scritto seed{seed}/results.json da {src.name} (solo risultati)")
            existing[seed] = target

    order = sorted(existing, key=lambda s: (s not in SEED_ORDER, SEED_ORDER.index(s) if s in SEED_ORDER else s))
    by_seed = {}
    for seed in order:
        result = json.loads((existing[seed] / "results.json").read_text(encoding="utf-8"))
        result["seed"] = seed
        by_seed[str(seed)] = result

    aggregate = {
        "schema_version": 2,
        "experiment_type": "sequential",
        "strategy": meta.get("strategy"),
        "model_name": meta.get("model_name"),
        "initialization": meta.get("initialization"),
        "seeds": by_seed,
    }
    meta_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"[Aggregato] {dest.name}: seed presenti {order}")


if __name__ == "__main__":
    main()
