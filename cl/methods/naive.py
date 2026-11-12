"""Naive (fine-tuning sequenziale, lower bound).  python -m cl.methods.naive"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cl.avalanche_trainer_methods import run_method

SEEDS = None  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

INITIALIZATION = "checkpoint"

if __name__ == "__main__":
    run_method("naive", initialization=INITIALIZATION, seeds=SEEDS)
