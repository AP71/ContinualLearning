"""Manopole di allenamento del backbone, condivise dai metodi con adattamento.

  - epoche per task di adattamento, learning rate, weight decay (TrainConfig);
    i milestone dello scheduler scalano con le epoche (come 20 e 25 su 40);
  - label smoothing e mixup (avalanche_plugin/mixup.py).
Con i valori di default (40 epoche, lr 1e-3, wd 0, niente smoothing/mixup) il training è
identico a quello di prima e il suffisso della cartella resta vuoto.
"""
from __future__ import annotations

from avalanche_plugin.mixup import MixupPlugin, TrainLoss
from cl.avalanche_trainer_methods import TrainConfig

DEFAULTS = dict(epochs=40, learning_rate=1e-3, weight_decay=0.0, label_smoothing=0.0, mixup_alpha=0.0)


def make_config(epochs: int = 40, learning_rate: float = 1e-3, weight_decay: float = 0.0) -> TrainConfig:
    milestones = (max(round(epochs * 0.5), 1), max(round(epochs * 0.625), 2))
    return TrainConfig(epochs_per_task=epochs, learning_rate=learning_rate,
                       weight_decay=weight_decay, lr_milestones=milestones)


def apply_loss(common_args: dict, label_smoothing: float = 0.0, mixup_alpha: float = 0.0,
               tasks: int = 1) -> TrainLoss | None:
    """Sostituisce la loss e, con mixup, aggiunge il plugin (solo sui primi ``tasks`` task)."""
    if not label_smoothing and not mixup_alpha:
        return None
    loss = TrainLoss(label_smoothing)
    common_args["criterion"] = loss
    if mixup_alpha:
        common_args["plugins"] = [MixupPlugin(loss, mixup_alpha, tasks)] + common_args["plugins"]
    return loss


def suffix(epochs: int = 40, learning_rate: float = 1e-3, weight_decay: float = 0.0,
           label_smoothing: float = 0.0, mixup_alpha: float = 0.0) -> str:
    return (
        (f"_e{epochs}" if epochs != DEFAULTS["epochs"] else "")
        + (f"_lr{learning_rate:g}" if learning_rate != DEFAULTS["learning_rate"] else "")
        + (f"_wd{weight_decay:g}" if weight_decay else "")
        + (f"_ls{label_smoothing:g}" if label_smoothing else "")
        + (f"_mix{mixup_alpha:g}" if mixup_alpha else "")
    )
