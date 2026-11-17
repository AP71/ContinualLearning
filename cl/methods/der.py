"""DER / DER++ (Buzzega et al., 2020): buffer di coppie (immagine, logit) e loss sui logit salvati.
python -m cl.methods.der

Sui campioni del buffer la loss aggiunge  ALPHA * MSE(logit attuali, logit salvati)  e, se
BETA > 0 (DER++), BETA * CE(logit attuali, etichetta vera). BETA = 0 è DER.
Differenze della strategia di Avalanche rispetto al paper: un solo batch di buffer per iterazione
(non due), forward congiunto batch nuovo + batch buffer, e i logit vengono salvati a fine task
sulle immagini senza augmentation (non al momento dell'inserimento).
NONBINARY_ONLY = True: i pesi binari restano fermi (zero bit flip), si allenano solo parametri reali e fc.
Suffisso cartella: _a<ALPHA>_b<BETA>[_nb].
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import DER

from cl.avalanche_trainer_methods import adam, run_method
from optimizer.adam_meta import binary_weight_params

SEEDS = [42]  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

NONBINARY_ONLY = False
ALPHA = 0.2  # peso della MSE sui logit salvati
BETA = 1.0 # peso della CE con etichette sul buffer (0 = DER, > 0 = DER++)

SUFFIX = f"_a{ALPHA:g}_b{BETA:g}" + ("_nb" if NONBINARY_ONLY else "")


def build_optimizer(model, cfg):
    if NONBINARY_ONLY:
        for param in binary_weight_params(model):
            param.requires_grad = False
    return adam(model, cfg)


def build_strategy(common_args, cfg):
    return DER(
        mem_size=2000,  # come Replay ed ER_ACE
        batch_size_mem=cfg.batch_size,
        alpha=ALPHA,
        beta=BETA,
        **common_args,
    )


if __name__ == "__main__":
    run_method("DER", build_optimizer, build_strategy, suffix=SUFFIX, seeds=SEEDS)
