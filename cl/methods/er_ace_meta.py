"""ER-ACE con l'ottimizzatore metaplastico al posto di Adam.
python -m cl.methods.er_ace_meta

Unisce i due risultati: ER-ACE (cl/methods/er_ace.py) porta il Class-IL a ~52% ma cambia
~45% dei bit, come naive; la regola metaplastica (optimizer/adam_meta.py) riduce i flip in
modo selettivo ma da sola resta a ~25%. Qui il buffer e la loss sono quelli di ER-ACE e
cambia solo l'ottimizzatore: la domanda è se l'accuratezza di ER-ACE regge con meno flip.

Non usa le protezioni del classificatore di Metaplasticity v3/v4: congelano le righe di fc
delle classi vecchie e mascherano i logit, mentre ER-ACE deve allenarle sul buffer.

L'ottimizzatore è quello della v4 (distanza standardizzata z, norma dei filtri fissa), con m in
unità di z. I risultati vanno in cl/results/ER_ACE, con suffisso _meta_m<m>.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from avalanche.training import ER_ACE

from cl.avalanche_trainer_methods import run_method
from optimizer.adam_meta import AdamMeta, binary_weight_params

SEEDS = [42]  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

M = 3.0  # forza del consolidamento; più basso = più plasticità (0 = ER_ACE con Adam)


def build_optimizer(model, cfg):
    return AdamMeta(
        model.parameters(),
        binary_weight_params(model),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        meta=M,
        standardized=True,
        renorm=True,
    )


def build_strategy(common_args, cfg):
    return ER_ACE(
        mem_size=2000,  # come Replay ed ER_ACE
        batch_size_mem=cfg.batch_size,
        **common_args,
    )


if __name__ == "__main__":
    suffix = "_meta_m" + f"{M:g}".replace(".", "")
    run_method("ER_ACE", build_optimizer, build_strategy, suffix=suffix, seeds=SEEDS)
