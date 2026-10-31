import torch
from avalanche.core import SupervisedPlugin

from cl.avalanche_trainer import evaluate_dataset

class EpochTrackerPlugin(SupervisedPlugin):
    """Traccia la loss e l'accuratezza di tutti i task visti alla fine di ogni singola epoca."""

    def __init__(self, test_stream, num_tasks: int, device: torch.device):
        super().__init__()
        self.test_stream = test_stream
        self.num_tasks = num_tasks
        self.device = device
        self.loss_history: list[float] = []
        self.accuracy_history: list[list[float]] = [[] for _ in range(num_tasks)]
        self._epoch_loss_sum = 0.0
        self._epoch_batches = 0

    def before_training_epoch(self, strategy, **kwargs):
        self._epoch_loss_sum = 0.0
        self._epoch_batches = 0

    def after_training_iteration(self, strategy, **kwargs):
        self._epoch_loss_sum += float(strategy.loss.item())
        self._epoch_batches += 1

    def after_training_epoch(self, strategy, **kwargs):
        # 1. Salva la loss media dell'epoca
        mean_loss = self._epoch_loss_sum / max(1, self._epoch_batches)
        self.loss_history.append(mean_loss)

        # 2. Valuta l'accuratezza di tutti i task visti finora
        curr_task = strategy.experience.current_experience
        for t_id in range(self.num_tasks):
            if t_id <= curr_task:
                test_exp_dataset = self.test_stream[t_id].dataset
                acc = evaluate_dataset(strategy.model, test_exp_dataset, self.device)
                self.accuracy_history[t_id].append(acc)
            else:
                self.accuracy_history[t_id].append(0.0)