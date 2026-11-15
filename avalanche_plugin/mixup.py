"""Loss e mixup per l'allenamento del backbone (task di adattamento).

``TrainLoss``: cross-entropy con label smoothing opzionale e, quando il plugin di mixup
ha preparato un batch mescolato, loss pesata  lam*CE(out, y) + (1-lam)*CE(out, y_perm).
``MixupPlugin``: dai batch dei primi ``tasks`` task mescola le immagini (lam ~ Beta(a, a)).
Il mixup agisce solo sul training: valutazione e statistiche usano immagini pulite.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from avalanche.core import SupervisedPlugin
from torch import nn


class TrainLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self._y2: torch.Tensor | None = None
        self._lam: float = 1.0
        self._classes: torch.Tensor | None = None
        self._map: torch.Tensor | None = None

    def set_classes(self, classes, num_classes: int = 100) -> None:
        """Loss ristretta ai logit delle ``classes`` (None = tutti): training intra-task."""
        if classes is None:
            self._classes = self._map = None
            return
        self._classes = torch.as_tensor(sorted(int(c) for c in classes))
        self._map = torch.full((num_classes,), -1, dtype=torch.long)
        self._map[self._classes] = torch.arange(len(self._classes))

    def _restrict(self, out, y):
        if self._classes is None:
            return out, y
        cls = self._classes.to(out.device)
        return out[:, cls], self._map.to(out.device)[y]

    def set_mix(self, y2: torch.Tensor, lam: float) -> None:
        self._y2, self._lam = y2, lam

    def clear(self) -> None:
        self._y2, self._lam = None, 1.0

    def forward(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out_r, y_r = self._restrict(out, y)
        loss = F.cross_entropy(out_r, y_r, label_smoothing=self.label_smoothing)
        if self._y2 is None:
            return loss
        y2_r = self._restrict(out, self._y2)[1]
        other = F.cross_entropy(out_r, y2_r, label_smoothing=self.label_smoothing)
        return self._lam * loss + (1.0 - self._lam) * other


class MixupPlugin(SupervisedPlugin):
    def __init__(self, loss: TrainLoss, alpha: float, tasks: int = 1):
        super().__init__()
        self.loss = loss
        self.alpha = alpha
        self.tasks = tasks
        self._beta = torch.distributions.Beta(alpha, alpha)

    def before_forward(self, strategy, **kwargs):
        self.loss.clear()
        if not strategy.is_training or strategy.experience.current_experience >= self.tasks:
            return
        x, y = strategy.mbatch[0], strategy.mbatch[1]
        lam = float(self._beta.sample())
        perm = torch.randperm(x.shape[0], device=x.device)
        strategy.mbatch[0] = lam * x + (1.0 - lam) * x[perm]
        self.loss.set_mix(y[perm], lam)

    def after_training_iteration(self, strategy, **kwargs):
        self.loss.clear()
