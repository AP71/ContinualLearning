"""AdaptThenSLDA: adatta il backbone sul primo task, poi lo congela e usa una testa SLDA.
python -m cl.methods.adapt_then_slda

- Task 1: addestramento normale (tutta la rete, o solo feature[ADAPT_FROM_BLOCK:] + fc).
- Dal task 2: backbone congelato, nessuna epoca di training; la testa fc viene riscritta
  come LDA con medie per classe e covarianza condivisa accumulate in streaming
  (avalanche_plugin/adapt_then_slda.py). Nessun buffer di immagini.
Riferimenti Class-IL al task 10: SLDA sul backbone ImageNet congelato 51.9%,
linear probe joint 57.6%, JOINT su tutta la rete 67.4%.
La cartella dei risultati prende il suffisso _a<ADAPT_TASKS>[_from<blocco>].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.adapt_then_slda import AdaptThenSLDAPlugin
from avalanche_plugin.binary_latent_replay import freeze_lower_layers
from cl.avalanche_trainer_methods import adam, run_method

SEEDS = [2023, 3047] # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

ADAPT_TASKS = 0  # 1 = adatta sul task 1; 0 = nessun adattamento (controllo: SLDA sul backbone ImageNet)
ADAPT_FROM_BLOCK = 0  # 0 = adatta tutta la rete; es. 12 = solo feature[12:] + fc
SHRINKAGE = 1e-2  # gamma della covarianza (scelto sul validation in cl/statistical_heads.py)

SUFFIX = f"_a{ADAPT_TASKS}" + (f"_from{ADAPT_FROM_BLOCK}" if ADAPT_TASKS and ADAPT_FROM_BLOCK else "")


def build_optimizer(model, cfg):
    if ADAPT_FROM_BLOCK:
        freeze_lower_layers(model, ADAPT_FROM_BLOCK)
    return adam(model, cfg)


def build_strategy(common_args, cfg):
    plugin = AdaptThenSLDAPlugin(
        adapt_tasks=ADAPT_TASKS,
        shrinkage=SHRINKAGE,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
    )
    common_args["plugins"] = [plugin] + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("AdaptThenSLDA", build_optimizer, build_strategy, suffix=SUFFIX, seeds=SEEDS)
