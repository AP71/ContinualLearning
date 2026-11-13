"""Binary latent replay con testa SLDA.  python -m cl.methods.blr_slda

Come cl/methods/binary_latent_replay.py (blocchi feature[0 .. CUT-1] congelati, buffer di
latenti binari, feature[CUT:] + fc allenati a ogni task con il replay), ma a fine task fc
viene riscritta come LDA: le statistiche delle classi vecchie si ricalcolano facendo
ripassare i latenti del buffer nei blocchi aggiornati, quelle delle classi nuove da tutti
i latenti del task corrente (avalanche_plugin/binary_latent_replay.py, slda_head=True).
La testa non ha recency bias per costruzione; la fc allenata serve solo al training.
META_M > 0 allena i blocchi sopra il taglio con Adam metaplastico, per frenarne il drift.
Riferimenti: BLR 12 balanced 44.9%, BLR 14 balanced 47.4%, AdaptThenSLDA a1_from10 55.4%.
I risultati vanno in cl/results/BLR_SLDA/ (vedi cl/results_layout.py); la cartella
dell'esperimento prende il suffisso _<CUT>[_fc][_m<M>][_<bit>bit].
Nota: le curve per epoca usano la fc allenata, le matrici finali la testa LDA.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import Naive

from avalanche_plugin.binary_latent_replay import (
    IMAGE_REPLAY_BYTES,
    BinaryLatentReplayPlugin,
    freeze_lower_layers,
)
from cl.avalanche_trainer_methods import adam, run_method
from optimizer.adam_meta import AdamMeta, binary_weight_params

SEEDS = [42]  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

CUT_BLOCK = 12  # 12 -> 512x4x4 = 1 KB per latente a 1 bit; 10 -> stessa dimensione, 4 blocchi allenati
LATENT_BITS = 1  # 1 = latenti binari, 32 = latenti float (ablazione)
MEMORY_BYTES = IMAGE_REPLAY_BYTES  # stessa memoria di Replay con 2000 immagini (~6.1 MB)
SLDA_HEAD = True  # False = BLR normale (testa fc)
SHRINKAGE = 1e-2  # gamma della covarianza SLDA
META_M = 0.0  # >0: Adam metaplastico sui blocchi sopra il taglio; 0 = Adam standard
BALANCED_FT_EPOCHS = 0  # correzioni della fc allenata: inutili con la testa SLDA
BALANCED_FT_LR = 1e-4
WEIGHT_ALIGNING = False
CLASSIFIER = "fc"

SUFFIX = (
    f"_{CUT_BLOCK}"
    + ("" if SLDA_HEAD else "_fc")
    + (f"_m{META_M:g}".replace(".", "") if META_M > 0 else "")
    + (f"_{LATENT_BITS}bit" if LATENT_BITS != 1 else "")
)


def build_optimizer(model, cfg):
    freeze_lower_layers(model, CUT_BLOCK)
    if META_M <= 0:
        return adam(model, cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return AdamMeta(
        trainable,
        binary_weight_params(model),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        meta=META_M,
    )


def build_strategy(common_args, cfg):
    latent_plugin = BinaryLatentReplayPlugin(
        cut_block=CUT_BLOCK,
        latent_bits=LATENT_BITS,
        memory_bytes=MEMORY_BYTES,
        classifier_name=CLASSIFIER,
        weight_aligning=WEIGHT_ALIGNING,
        balanced_ft_epochs=BALANCED_FT_EPOCHS,
        balanced_ft_lr=BALANCED_FT_LR,
        slda_head=SLDA_HEAD,
        slda_shrinkage=SHRINKAGE,
    )
    common_args["plugins"] = [latent_plugin] + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("BinaryLatentReplaySLDA", build_optimizer, build_strategy, suffix=SUFFIX, seeds=SEEDS)
