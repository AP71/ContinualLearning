"""MetaSLDA: backbone che continua ad adattarsi con la metaplasticità + testa SLDA con
compensazione del drift.  python -m cl.methods.meta_slda

- Ogni task (fino ad ADAPT_TASKS): training normale di feature[ADAPT_FROM_BLOCK:] + fc con
  Adam metaplastico (optimizer/adam_meta.py), che frena i bit flip dei pesi consolidati.
- A fine task le statistiche SLDA delle classi vecchie vengono trasportate nel nuovo
  spazio delle feature con una mappa affine stimata sui dati del task corrente
  (f_new ≈ f_old·W + b), poi si aggiungono le classi nuove e fc viene riscritta come LDA
  (avalanche_plugin/adapt_then_slda.py). Nessun buffer.
- La testa allenata con la loss serve solo a guidare il backbone: PROTECTION applica le
  protezioni di v3 (avalanche_plugin/classifier_protection.py); None = nessuna.
Ablazioni: DRIFT_COMPENSATION = "none" (statistiche vecchie non trasportate),
M = 0 (Adam standard), ADAPT_TASKS = 1 (= AdaptThenSLDA).
Riferimento: AdaptThenSLDA a1_from10 55.4% (backbone congelato dopo il task 1).
I risultati vanno in cl/results/Meta_SLDA/ (vedi cl/results_layout.py); la cartella
dell'esperimento prende il suffisso _m<M>_a<ADAPT_TASKS>_from<blocco>_<drift>.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.adapt_then_slda import AdaptThenSLDAPlugin
from avalanche_plugin.binary_latent_replay import freeze_lower_layers
from avalanche_plugin.classifier_protection import ClassifierProtectionPlugin
from cl.avalanche_trainer_methods import run_method
from optimizer.adam_meta import AdamMeta, binary_weight_params

SEEDS = [42]  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

M = 6.0  # forza della metaplasticità; 0 = Adam standard
ADAPT_TASKS = 10  # task con training del backbone (10 = tutti); dopo, backbone congelato
ADAPT_FROM_BLOCK = 10  # 0 = tutta la rete; es. 10 = solo feature[10:] + fc
DRIFT_COMPENSATION = "linear"  # "linear" = trasporta le statistiche vecchie; "none" = ablazione
DRIFT_RIDGE = 1e-2  # regolarizzazione della mappa verso l'identità
SHRINKAGE = 1e-2  # gamma della covarianza SLDA
CLASSIFIER = "fc"
PROTECTION = dict(  # protezioni della testa di training (come v3); None = nessuna
    mask_unseen_logits=True,
    freeze_non_current_fc=True,
    normalize_classifier=True,
    reset_fc_moments=True,
    non_binary_lr_scale=0.05,
)

SUFFIX = (
    f"_m{M:g}".replace(".", "")
    + f"_a{ADAPT_TASKS}"
    + (f"_from{ADAPT_FROM_BLOCK}" if ADAPT_FROM_BLOCK else "")
    + f"_{DRIFT_COMPENSATION}"
    + ("" if PROTECTION else "_noprot")
)


def build_optimizer(model, cfg):
    if ADAPT_FROM_BLOCK:
        freeze_lower_layers(model, ADAPT_FROM_BLOCK)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return AdamMeta(
        trainable,
        binary_weight_params(model),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        meta=M,
    )


def build_strategy(common_args, cfg):
    # Il plugin SLDA va per primo: ripristina la testa di training prima delle protezioni
    # e scrive la testa LDA prima del salvataggio del checkpoint.
    plugins = [
        AdaptThenSLDAPlugin(
            adapt_tasks=ADAPT_TASKS,
            shrinkage=SHRINKAGE,
            classifier_name=CLASSIFIER,
            batch_size=cfg.batch_size,
            num_workers=cfg.workers,
            drift_compensation=DRIFT_COMPENSATION,
            drift_ridge=DRIFT_RIDGE,
        )
    ]
    if PROTECTION is not None:
        plugins.append(ClassifierProtectionPlugin(classifier_name=CLASSIFIER, **PROTECTION))
    common_args["plugins"] = plugins + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("MetaSLDA", build_optimizer, build_strategy, suffix=SUFFIX, seeds=SEEDS)
