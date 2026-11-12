"""FreezingHead: backbone congelato + protezioni del classificatore.
python -m cl.methods.freezing_head"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.classifier_protection import ClassifierProtectionPlugin
from cl.avalanche_trainer_methods import run_method
from cl.methods.freezing import CLASSIFIER, build_optimizer

SEEDS = None  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare


def build_strategy(common_args, cfg):
    plugin = ClassifierProtectionPlugin(classifier_name=CLASSIFIER)
    common_args["plugins"] = [plugin] + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("FreezingHead", build_optimizer, build_strategy, seeds=SEEDS)
