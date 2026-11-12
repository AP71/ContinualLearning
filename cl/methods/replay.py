"""Experience Replay con buffer di immagini.  python -m cl.methods.replay"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Replay

from cl.avalanche_trainer_methods import adam, run_method

SEEDS = None  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

MEM_SIZE = 2000


def build_strategy(common_args, cfg):
    return Replay(mem_size=MEM_SIZE, **common_args)


if __name__ == "__main__":
    run_method("Replay", adam, build_strategy, seeds=SEEDS)
