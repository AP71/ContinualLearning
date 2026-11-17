"""AdaptThenSLDA: adatta il backbone sul primo task, poi lo congela e usa una testa SLDA.
python -m cl.methods.adapt_then_slda

- Task 1: addestramento normale (tutta la rete, o solo feature[ADAPT_FROM_BLOCK:] + fc).
- Dal task 2: backbone congelato, nessuna epoca di training; la testa fc viene riscritta
  come LDA con medie per classe e covarianza condivisa accumulate in streaming
  (avalanche_plugin/adapt_then_slda.py). Nessun buffer di immagini.
Riferimenti Class-IL al task 10: SLDA sul backbone ImageNet congelato 51.9%,
linear probe joint 57.6%, JOINT su tutta la rete 67.4%.

HEAD = "ranpac": al posto di SLDA, testa RanPAC (proiezione casuale fissa + ReLU, poi
regressione ridge in forma chiusa su statistiche cumulate); RP_DIM = 0 dà la sola ridge
sulle feature. Stessa proprietà di SLDA: nessun buffer, nessun training dopo il task 1.
Richiede ADAPT_TASKS <= 1. Con RP_DIM = 10000 servono ~0.8 GB di RAM per la matrice G.
NONBINARY_ONLY = True: il task 1 non tocca i pesi binari (nessun bit flip, mai); False: si allena
tutta la rete (bit flip solo nei task di adattamento).
Suffisso cartella: ..._a<task>[_from<blocco>][_ranpac<dim>][_nb][_dlin][_kd<peso>].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.adapt_then_slda import AdaptThenSLDAPlugin
from avalanche_plugin.binary_latent_replay import freeze_lower_layers
from avalanche_plugin.feature_distill import FeatureDistillationPlugin
from cl import train_knobs
from cl.avalanche_trainer_methods import adam, run_method
from optimizer.adam_meta import binary_weight_params

SEEDS = [42]  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

ADAPT_TASKS = 1  # 1 = adatta sul task 1; 0 = nessun adattamento (controllo: SLDA sul backbone ImageNet)
ADAPT_FROM_BLOCK = 10  # 0 = adatta tutta la rete; es. 12 = solo feature[12:] + fc
NONBINARY_ONLY = True  # True: si allenano solo parametri non binari (zero bit flip); False: tutta la rete
HEAD = "ranpac"  # "slda" oppure "ranpac" (ranpac richiede ADAPT_TASKS <= 1)
RP_DIM = 20000  # RanPAC: dimensione della proiezione casuale (0 = ridge sulle sole feature)
DRIFT = "none"  # "none" | "linear": trasporto delle statistiche SLDA (solo con ADAPT_TASKS > 1)
DISTILL_WEIGHT = 0.0  # > 0: distillazione delle feature (solo con ADAPT_TASKS > 1)
EPOCHS = 40  # epoche del task di adattamento (i milestone dello scheduler scalano)
LEARNING_RATE = 1e-3

SUFFIX = (
    f"_a{ADAPT_TASKS}"
    + (f"_from{ADAPT_FROM_BLOCK}" if ADAPT_TASKS and ADAPT_FROM_BLOCK else "")
    + ((f"_ranpac{RP_DIM}" if RP_DIM else "_ridge") if HEAD == "ranpac" else "")
    + ("_nb" if NONBINARY_ONLY else "")
    + ("_dlin" if ADAPT_TASKS > 1 and DRIFT == "linear" else "")
    + (f"_kd{DISTILL_WEIGHT:g}" if ADAPT_TASKS > 1 and DISTILL_WEIGHT else "")
    + train_knobs.suffix(EPOCHS, LEARNING_RATE)
)


def build_optimizer(model, cfg):
    if ADAPT_FROM_BLOCK:
        freeze_lower_layers(model, ADAPT_FROM_BLOCK)
    if NONBINARY_ONLY:
        for param in binary_weight_params(model):
            param.requires_grad = False
    return adam(model, cfg)


def build_strategy(common_args, cfg):
    plugin = AdaptThenSLDAPlugin(
        adapt_tasks=ADAPT_TASKS,
        head=HEAD,
        rp_dim=RP_DIM,
        drift_compensation=DRIFT if ADAPT_TASKS > 1 else "none",
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
    )
    plugins = [plugin]
    if ADAPT_TASKS > 1 and DISTILL_WEIGHT:
        plugins.append(FeatureDistillationPlugin(DISTILL_WEIGHT))
    common_args["plugins"] = plugins + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("AdaptThenSLDA", build_optimizer, build_strategy,
               cfg=train_knobs.make_config(EPOCHS, LEARNING_RATE), suffix=SUFFIX, seeds=SEEDS)
