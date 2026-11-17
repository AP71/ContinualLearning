"""NullSpace: testa fc protetta nello spazio nullo delle feature vecchie + distillazione delle feature.
python -m cl.methods.null_space

Nessun buffer. Due strumenti contro il dimenticare, attivabili separatamente (ablazione):
  - proiezione nello spazio nullo sulla testa fc (avalanche_plugin/null_space_head.py): ENERGY, OLD_ROWS
  - distillazione delle feature su un teacher congelato (avalanche_plugin/feature_distill.py): DISTILL_WEIGHT
DISTILL_WEIGHT = 0 la spegne; OLD_ROWS = "free" spegne la proiezione.
NONBINARY_ONLY = True: i pesi binari restano fermi (zero bit flip); False: si allena tutta la rete.
Suffisso cartella: _e<ENERGY>_<OLD_ROWS>[_kd<peso>][_all] + manopole di training.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.feature_distill import FeatureDistillationPlugin
from avalanche_plugin.null_space_head import NullSpaceHeadPlugin
from cl import train_knobs
from cl.avalanche_trainer_methods import adam, run_method
from optimizer.adam_meta import binary_weight_params

SEEDS = [42]  # None = tutti (42, 2023, 3047)

NONBINARY_ONLY =  False
ENERGY = 0.99  # frazione di energia delle feature vecchie protetta (più bassa = più spazio libero)
OLD_ROWS = "project"  # "free" | "freeze" | "project" | "project_all"
DISTILL_WEIGHT = 1.0  # 0 = niente distillazione
EPOCHS = 40
LEARNING_RATE = 1e-3

SUFFIX = (
    f"_e{ENERGY:g}_{OLD_ROWS}"
    + (f"_kd{DISTILL_WEIGHT:g}" if DISTILL_WEIGHT else "")
    + ("" if NONBINARY_ONLY else "_all")
    + train_knobs.suffix(EPOCHS, LEARNING_RATE)
)


def build_optimizer(model, cfg):
    if NONBINARY_ONLY:
        for param in binary_weight_params(model):
            param.requires_grad = False
    return adam(model, cfg)


def build_strategy(common_args, cfg):
    loss = train_knobs.apply_loss(common_args, force=True)  # TrainLoss: serve per ristringere i logit
    plugins = [NullSpaceHeadPlugin(
        energy=ENERGY, old_rows=OLD_ROWS, loss=loss, batch_size=cfg.batch_size, num_workers=cfg.workers)]
    if DISTILL_WEIGHT:
        plugins.append(FeatureDistillationPlugin(DISTILL_WEIGHT))
    common_args["plugins"] = plugins + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("NullSpace", build_optimizer, build_strategy,
               cfg=train_knobs.make_config(EPOCHS, LEARNING_RATE), suffix=SUFFIX, seeds=SEEDS)
