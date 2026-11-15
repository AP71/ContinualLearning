"""AdaptThenSLDA: adatta il backbone sul primo task, poi lo congela e usa una testa SLDA.
python -m cl.methods.adapt_then_slda

- Task 1: addestramento normale (tutta la rete, o solo feature[ADAPT_FROM_BLOCK:] + fc).
- Dal task 2: backbone congelato, nessuna epoca di training; la testa fc viene riscritta
  come LDA con medie per classe e covarianza condivisa accumulate in streaming
  (avalanche_plugin/adapt_then_slda.py). Nessun buffer di immagini.
Riferimenti Class-IL al task 10: SLDA sul backbone ImageNet congelato 51.9%,
linear probe joint 57.6%, JOINT su tutta la rete 67.4%.
La cartella dei risultati prende il suffisso _a<ADAPT_TASKS>[_from<blocco>][_ranpac<RP_DIM>].

HEAD = "ranpac": al posto di SLDA, testa RanPAC (proiezione casuale fissa + ReLU, poi
regressione ridge in forma chiusa su statistiche cumulate); RP_DIM = 0 dà la sola ridge
sulle feature. Stessa proprietà di SLDA: nessun buffer, nessun training dopo il task 1.
Richiede ADAPT_TASKS <= 1. Con RP_DIM = 10000 servono ~0.8 GB di RAM per la matrice G.
FEATURE_BLOCKS (solo RanPAC): concatena le uscite di più blocchi prima della proiezione.
NONBINARY_ONLY: il task 1 non tocca i pesi binari (nessun bit flip, mai).
Suffisso cartella: ..._ml<blocchi separati da ->[_nb].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.adapt_then_slda import AdaptThenSLDAPlugin
from avalanche_plugin.binary_latent_replay import freeze_lower_layers
from cl.avalanche_trainer_methods import adam, run_method
from cl import train_knobs
from optimizer.adam_meta import binary_weight_params

SEEDS = [2023, 3047] # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

ADAPT_TASKS = 1  # 1 = adatta sul task 1; 0 = nessun adattamento (controllo: SLDA sul backbone ImageNet)
ADAPT_FROM_BLOCK = 0  # 0 = adatta tutta la rete; es. 12 = solo feature[12:] + fc
SHRINKAGE = 1e-2  # gamma della covarianza (scelto sul validation in cl/statistical_heads.py)

HEAD = "ranpac"  # "slda" oppure "ranpac"
RP_DIM = 20000  # RanPAC: dimensione della proiezione casuale (0 = ridge sulle sole feature)
RIDGE = None  # RanPAC: r in lambda = r * mean(diag G); None = scelto su hold-out del primo task
RIDGE_RELATIVE = True  # True: lambda segue la scala di G a ogni task; False: assoluto dal task 1 (paper)
RP_SEED = 0
FEATURE_BLOCKS = ()  # RanPAC multi-livello: es. (8, 10, 12) concatena le uscite di questi blocchi (1-13) all'input di fc
EPOCHS = 40  # epoche del task di adattamento (i milestone dello scheduler scalano)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
LABEL_SMOOTHING = 0.0  # 0 = cross-entropy semplice
MIXUP_ALPHA = 0.0  # 0 = niente mixup; es. 0.2 (solo sui task di adattamento)
NONBINARY_ONLY = True  # True: il task 1 allena solo parametri non binari (bias, PReLU, gain, fc): zero bit flip anche al task 1

SUFFIX = (
    f"_a{ADAPT_TASKS}"
    + (f"_from{ADAPT_FROM_BLOCK}" if ADAPT_TASKS and ADAPT_FROM_BLOCK else "")
    + ((f"_ranpac{RP_DIM}" if RP_DIM else "_ridge") if HEAD == "ranpac" else "")
    + ("_ridgeabs" if HEAD == "ranpac" and not RIDGE_RELATIVE else "")
    + (f"_r{RIDGE:g}" if HEAD == "ranpac" and RIDGE is not None else "")
    + ("_ml" + "-".join(str(b) for b in FEATURE_BLOCKS) if FEATURE_BLOCKS else "")
    + ("_nb" if NONBINARY_ONLY else "")
    + train_knobs.suffix(EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING, MIXUP_ALPHA)
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
        shrinkage=SHRINKAGE,
        head=HEAD,
        rp_dim=RP_DIM,
        ridge=RIDGE,
        ridge_relative=RIDGE_RELATIVE,
        rp_seed=RP_SEED,
        feature_blocks=FEATURE_BLOCKS,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
    )
    common_args["plugins"] = [plugin] + common_args["plugins"]
    train_knobs.apply_loss(common_args, LABEL_SMOOTHING, MIXUP_ALPHA, tasks=max(ADAPT_TASKS, 1))
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("AdaptThenSLDA", build_optimizer, build_strategy,
               cfg=train_knobs.make_config(EPOCHS, LEARNING_RATE, WEIGHT_DECAY), suffix=SUFFIX, seeds=SEEDS)
