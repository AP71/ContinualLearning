"""Freezing: backbone congelato, si addestra solo 'fc'.  python -m cl.methods.freezing"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cl.avalanche_trainer_methods import adam, run_method

SEEDS = None  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

CLASSIFIER = "fc"


def build_optimizer(model, cfg):
    for name, param in model.named_parameters():
        if not name.startswith(CLASSIFIER):
            param.requires_grad = False
    return adam(model, cfg)


if __name__ == "__main__":
    run_method("Freezing", build_optimizer, seeds=SEEDS)
