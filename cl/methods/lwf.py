"""Learning without Forgetting.  python -m cl.methods.lwf"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import LwF

from cl.avalanche_trainer_methods import adam, run_method

SEEDS = None  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

ALPHA = 1.0
TEMPERATURE = 2.0


def build_strategy(common_args, cfg):
    return LwF(alpha=ALPHA, temperature=TEMPERATURE, **common_args)


if __name__ == "__main__":
    run_method("LwF", adam, build_strategy, seeds=SEEDS)
