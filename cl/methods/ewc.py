"""Elastic Weight Consolidation.  python -m cl.methods.ewc"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import EWC

from cl.avalanche_trainer_methods import adam, run_method

SEEDS = None  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

EWC_LAMBDA = 1e5  # forza della penalità di Fisher


def build_strategy(common_args, cfg):
    return EWC(ewc_lambda=EWC_LAMBDA, **common_args)


if __name__ == "__main__":
    # Cartella con suffisso lambda: _100000, _04 (per 0.4)
    run_method("EWC", adam, build_strategy, suffix="_" + f"{EWC_LAMBDA:g}".replace(".", ""), seeds=SEEDS)
