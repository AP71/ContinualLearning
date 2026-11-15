"""Adapter non binari per task su backbone binario congelato (zero bit flip) + RanPAC fuse.

Per ogni task si allenano solo bias, PReLU, gain e primo conv (~54k parametri, avalanche_plugin/
adapter_ensemble.py); i pesi binari restano quelli ImageNet. Ogni adapter ha la sua testa
RanPAC; i logit delle teste che conoscono una classe si mediano.
Suffisso cartella: _rp<RP_DIM>[_2stage][_prev][_nomask] + manopole di training (cl/train_knobs.py).
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

SEEDS = [42]  # None = tutti (42, 2023, 3047)

RP_DIM = 5000  # per testa: la matrice G è RP_DIM^2 double per adapter (10 adapter a 20000 = 32 GB)
RIDGE = None  # None = scelto su hold-out del primo task di ogni testa
RP_SEED = 0
FUSION = "mean"  # "mean": media dei logit delle teste che conoscono la classe; "two_stage": la testa 0 sceglie il task
INIT = "pretrained"  # "pretrained": ogni adapter parte dal backbone ImageNet; "previous": dal precedente
MASK_LOGITS = False  # loss sui soli logit delle classi del task (le altre righe di fc non c'entrano)
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
LABEL_SMOOTHING = 0.0
MIXUP_ALPHA = 0.0

SUFFIX = (
    f"_rp{RP_DIM}"
    + ("_2stage" if FUSION == "two_stage" else "")
    + ("_prev" if INIT == "previous" else "")
    + ("" if MASK_LOGITS else "_nomask")
    + train_knobs.suffix(EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING, MIXUP_ALPHA)
)


def build_optimizer(model, cfg):
    for param in binary_weight_params(model):
        param.requires_grad = False
    return adam(model, cfg)


def build_strategy(common_args, cfg):
    loss = train_knobs.apply_loss(common_args, LABEL_SMOOTHING, MIXUP_ALPHA, tasks=10**6, force=MASK_LOGITS)
    plugin = AdapterEnsemblePlugin(
        rp_dim=RP_DIM, ridge=RIDGE, rp_seed=RP_SEED, init=INIT, fusion=FUSION, loss=loss if MASK_LOGITS else None,
        batch_size=cfg.batch_size, num_workers=cfg.workers)
    common_args["plugins"] = [plugin] + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("AdapterEnsemble", build_optimizer, build_strategy,
               cfg=train_knobs.make_config(EPOCHS, LEARNING_RATE, WEIGHT_DECAY), suffix=SUFFIX, seeds=SEEDS)
