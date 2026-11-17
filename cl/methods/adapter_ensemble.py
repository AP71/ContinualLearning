"""Adapter non binari per task su backbone binario congelato (zero bit flip) + RanPAC fuse.

Per ogni task si allenano solo bias, PReLU, gain e primo conv (~54k parametri, avalanche_plugin/
adapter_ensemble.py); i pesi binari restano quelli ImageNet. Ogni adapter ha la sua testa
RanPAC; i logit delle teste che conoscono una classe si mediano.
Fusione a due stadi: la testa 0 sceglie il task, poi si mediano i logit delle teste che conoscono la classe.
Suffisso cartella: _rp<RP_DIM>_2stage_nomask + manopole di training (cl/train_knobs.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.adapter_ensemble import AdapterEnsemblePlugin
from cl import train_knobs
from cl.avalanche_trainer_methods import adam, run_method
from optimizer.adam_meta import binary_weight_params

SEEDS = [2023, 3047]  # None = tutti (42, 2023, 3047)

RP_DIM = 5000  # per testa: la matrice G è RP_DIM^2 double per adapter (10 adapter a 20000 = 32 GB)
EPOCHS = 40
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.0

SUFFIX = f"_rp{RP_DIM}_2stage_nomask" + train_knobs.suffix(EPOCHS, LEARNING_RATE, label_smoothing=LABEL_SMOOTHING)


def build_optimizer(model, cfg):
    for param in binary_weight_params(model):
        param.requires_grad = False
    return adam(model, cfg)


def build_strategy(common_args, cfg):
    train_knobs.apply_loss(common_args, LABEL_SMOOTHING)
    plugin = AdapterEnsemblePlugin(
        rp_dim=RP_DIM, fusion="two_stage", batch_size=cfg.batch_size, num_workers=cfg.workers)
    common_args["plugins"] = [plugin] + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("AdapterEnsemble", build_optimizer, build_strategy,
               cfg=train_knobs.make_config(EPOCHS, LEARNING_RATE), suffix=SUFFIX, seeds=SEEDS)
