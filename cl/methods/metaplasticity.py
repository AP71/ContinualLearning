"""Metaplasticità (Laborieux et al., 2021).  python -m cl.methods.metaplasticity [--seed N]

In entrambe le versioni la regola metaplastica è dentro Adam (optimizer/adam_meta.py),
come nel paper, applicata a W - mean_filtro (bit effettivo di A&B-BNN).
- v1: solo la regola del paper.
- v3: regola del paper con m=6 + protezioni del classificatore e passo dei parametri
  reali del backbone al 5% dal secondo task (avalanche_plugin/classifier_protection.py).
La cartella dei risultati prende il suffisso _<versione>_m<m> (es. _v3_m6, _v1_m15).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.classifier_protection import ClassifierProtectionPlugin
from cl.avalanche_trainer_methods import run_method
from optimizer.adam_meta import AdamMeta, binary_weight_params

SEEDS = [2023, 3047]  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

VERSION = "v3"  # "v1" oppure "v3"
M = 20  # None = default della versione
CLASSIFIER = "fc"

VERSIONS = {
    "v1": {"m": 1.5, "protection": None},
    "v3": {
        "m": 6.0,
        "protection": dict(
            mask_unseen_logits=True,
            freeze_non_current_fc=True,
            normalize_classifier=True,
            reset_fc_moments=True,
            non_binary_lr_scale=0.05,
        ),
    },
}
m_value = VERSIONS[VERSION]["m"] if M is None else M


def build_optimizer(model, cfg):
    return AdamMeta(
        model.parameters(),
        binary_weight_params(model),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        meta=m_value,
    )


def build_strategy(common_args, cfg):
    protection = VERSIONS[VERSION]["protection"]
    if protection is not None:
        plugin = ClassifierProtectionPlugin(classifier_name=CLASSIFIER, **protection)
        common_args["plugins"] = [plugin] + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    suffix = f"_{VERSION}_m" + f"{m_value:g}".replace(".", "")
    run_method("Metaplasticity", build_optimizer, build_strategy, suffix=suffix, seeds=SEEDS)
