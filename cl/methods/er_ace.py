"""ER-ACE (Caccia et al., 2022): Experience Replay con cross-entropy asimmetrica.
python -m cl.methods.er_ace

Stesso buffer di immagini di Replay, cambia la loss: sui campioni del task corrente la
cross-entropy usa solo i logit delle classi del task corrente, sui campioni del buffer tutti
i logit. I campioni nuovi quindi non spingono giù i logit delle classi vecchie, che è la
causa attesa dello scarto tra Task-IL (83%) e Class-IL (42%) di Replay.

Differenze della strategia ER_ACE di Avalanche rispetto alla nostra Replay, oltre alla loss:
  - il buffer è bilanciato per classe e contiene già le classi del task corrente
    (viene aggiornato prima del task, non dopo);
  - i due termini della loss sono mediati: (CE_buffer + CE_corrente) / 2.
Come in Replay, a ogni batch nuovo corrisponde un batch del buffer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import ER_ACE

from cl.avalanche_trainer_methods import adam, run_method

SEEDS = [2023, 3047]  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare


def build_strategy(common_args, cfg):
    return ER_ACE(
        mem_size=2000,  # come Replay
        batch_size_mem=cfg.batch_size,  # come il batch di training
        **common_args,
    )


if __name__ == "__main__":
    run_method("ER_ACE", adam, build_strategy, seeds=SEEDS)
