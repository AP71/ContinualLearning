import torch
from avalanche.core import SupervisedPlugin
from pathlib import Path

from utils.checkpoint_analysis import compare_checkpoints, plot_latent_distributions

class CheckpointAndBitFlipPlugin(SupervisedPlugin):
    """Isola il salvataggio dei pesi e l'analisi quantizzata/bit-flip ad ogni cambio task."""

    def __init__(self, checkpoint_dir: Path, compare_dir: Path, LOGGER=None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.compare_dir = compare_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.compare_dir.mkdir(parents=True, exist_ok=True)
        self.prev_checkpoint: Path | None = None
        self.bit_flips: list[int] = []
        self.LOGGER = LOGGER

    def before_training(self, strategy, **kwargs):
        """Salva lo stato iniziale (initial.h5) prima dell'inizio dello stream."""
        initial_path = self.checkpoint_dir / "initial.h5"
        torch.save(strategy.model.state_dict(), initial_path)
        self.prev_checkpoint = initial_path

    def after_training_exp(self, strategy, **kwargs):
        """Eseguito al termine di ogni task/esperienza."""
        exp_id = strategy.experience.current_experience
        curr_checkpoint = self.checkpoint_dir / f"after_task_{exp_id + 1:02d}.h5"
        torch.save(strategy.model.state_dict(), curr_checkpoint)

        # Esegue comparazione distribuzioni e bit-flips
        comp_dir = self.compare_dir / f"task_{exp_id + 1:02d}"
        comparison = compare_checkpoints(self.prev_checkpoint, curr_checkpoint)
        plot_latent_distributions(comparison, comp_dir / "latent_distributions.png")

        flips = comparison["binary_summary"]["bit_flips"]
        pct = comparison["binary_summary"]["bit_flip_percentage"]
        self.bit_flips.append(flips)
        if self.LOGGER:
            self.LOGGER.info("Task %d completed - Bit Flips: %d (%.4f%%)", exp_id + 1, flips, pct)

        self.prev_checkpoint = curr_checkpoint
