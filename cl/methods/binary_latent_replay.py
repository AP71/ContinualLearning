"""Binary latent replay (Vorabbi et al.): blocchi feature[0 .. CUT-1] congelati, il buffer
salva le attivazioni all'uscita di feature[CUT-1].  python -m cl.methods.binary_latent_replay"""
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

SEEDS = [42]  # None = tutti (42, 2023, 3047); es. [42], poi [2023, 3047] per completare

CUT_BLOCK = 6  # 14 -> 1024x2x2 = 512 B per latente a 1 bit (solo fc addestrabile); 12 -> 512x4x4 = 1 KB
LATENT_BITS = 1  # 1 = latenti binari, 32 = latenti float (ablazione)
MEMORY_BYTES = IMAGE_REPLAY_BYTES  # stessa memoria di Replay con 2000 immagini (~6.1 MB)
WEIGHT_ALIGNING = False  # Weight Aligning delle righe di fc a fine task
BALANCED_FT_EPOCHS = 10  # >0: fine-tuning bilanciato di fc sul buffer a fine task
BALANCED_FT_LR = 1e-4
CLASSIFIER = "fc"


def build_optimizer(model, cfg):
    freeze_lower_layers(model, CUT_BLOCK)
    return adam(model, cfg)


def build_strategy(common_args, cfg):
    latent_plugin = BinaryLatentReplayPlugin(
        cut_block=CUT_BLOCK,
        latent_bits=LATENT_BITS,
        memory_bytes=MEMORY_BYTES,
        classifier_name=CLASSIFIER,
        weight_aligning=WEIGHT_ALIGNING,
        balanced_ft_epochs=BALANCED_FT_EPOCHS,
        balanced_ft_lr=BALANCED_FT_LR,
    )
    common_args["plugins"] = [latent_plugin] + common_args["plugins"]
    return Naive(**common_args)


if __name__ == "__main__":
    run_method("BinaryLatentReplay", build_optimizer, build_strategy, seeds=SEEDS)
