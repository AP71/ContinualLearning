"""CWR*: backbone congelato dopo il task 1, testa con pesi consolidati (cw) e temporanei (tw).
python -m cl.methods.cwr_star

Il primo task allena backbone (o solo feature[ADAPT_FROM_BLOCK:]) e fc, come negli altri
metodi con backbone congelato; dal secondo task si allena solo fc (tw) e a fine task cw
viene aggiornata (avalanche_plugin/cwr_star.py). Con classi disgiunte la media pesata per
numero di campioni si riduce alla copia di tw in cw.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.binary_latent_replay import freeze_lower_layers
from avalanche_plugin.classifier_protection import ClassifierProtectionPlugin
from avalanche_plugin.cwr_star import CWRStarPlugin
from cl.avalanche_trainer_methods import adam, run_method

SEEDS = [2023, 3047] # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

CLASSIFIER = "fc"
ADAPT_FROM_BLOCK = 10  # 0 = il task 1 adatta tutta la rete; es. 10 = solo feature[10:] + fc
ZERO_MEAN = False  # True = CWR+/CWR* (tw - media(tw)); False = CWR semplice

SUFFIX = (f"_from{ADAPT_FROM_BLOCK}" if ADAPT_FROM_BLOCK else "") + ("" if ZERO_MEAN else "_nozm")


def build_optimizer(model, cfg):
    if ADAPT_FROM_BLOCK:
        freeze_lower_layers(model, ADAPT_FROM_BLOCK)
    return adam(model, cfg)


def build_strategy(common_args, cfg):
    protection = ClassifierProtectionPlugin(classifier_name=CLASSIFIER, normalize_classifier=False)
    cwr = CWRStarPlugin(classifier_name=CLASSIFIER, adapt_tasks=1, zero_mean=ZERO_MEAN)
    common_args["plugins"] = [cwr, protection] + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("CWRStar", build_optimizer, build_strategy, suffix=SUFFIX, seeds=SEEDS)
