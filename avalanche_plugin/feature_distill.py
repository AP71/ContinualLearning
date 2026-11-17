"""Distillazione delle feature (senza buffer): tiene vicine le feature del backbone a quelle di un teacher.

Il teacher è una copia congelata del modello a fine task. Durante l'allenamento di un task
nuovo si aggiunge alla loss  weight * d(f_student, f_teacher)  dove f è l'ingresso di ``fc``
(le feature a 1024 dimensioni) calcolato sulle stesse immagini del minibatch corrente. Nessun
dato dei task passati: si limita il drift della rappresentazione, che è ciò che rende
obsolete le righe di ``fc`` / le statistiche SLDA delle classi vecchie.

Teacher = modello a fine del task precedente (LwF sulle feature). Distanza:
||f - ft||^2 / ||ft||^2, media sul batch (adimensionale).

Costo: un forward in più per iterazione (circa +30% di tempo).
"""
from __future__ import annotations

import copy

import torch
from avalanche.core import SupervisedPlugin


class FeatureDistillationPlugin(SupervisedPlugin):
    def __init__(self, weight: float = 1.0, classifier_name: str = "fc", eps: float = 1e-8):
        super().__init__()
        self.weight, self.classifier_name, self.eps = weight, classifier_name, eps
        self.teacher: torch.nn.Module | None = None
        self._feat: torch.Tensor | None = None
        self._t_feat: torch.Tensor | None = None
        self._handle = None
        self._sum, self._n = 0.0, 0

    # ------------------------------------------------------------------ hooks
    def _cap_student(self, module, inputs) -> None:
        if module.training:
            self._feat = inputs[0]

    def _cap_teacher(self, _module, inputs) -> None:
        self._t_feat = inputs[0]

    def _remove_hook(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def before_training_exp(self, strategy, **kwargs):
        self._remove_hook()
        classifier = getattr(strategy.model, self.classifier_name)
        self._handle = classifier.register_forward_pre_hook(self._cap_student)
        self._feat, self._sum, self._n = None, 0.0, 0

    def before_backward(self, strategy, **kwargs):
        if self.teacher is None or self._feat is None:
            return
        with torch.no_grad():
            self.teacher(strategy.mb_x)
        f, ft = self._feat.float(), self._t_feat.float()
        dist = (((f - ft) ** 2).sum(1) / (ft ** 2).sum(1).clamp_min(self.eps)).mean()
        strategy.loss = strategy.loss + self.weight * dist
        self._sum += float(dist.detach())
        self._n += 1
        self._feat = None

    def after_training_exp(self, strategy, **kwargs):
        self._remove_hook()  # prima della copia: un hook legato al plugin farebbe copiare anche il plugin
        if self._n:
            print(f"[FeatureDistillation] task {strategy.experience.current_experience + 1}: "
                  f"distanza media feature student-teacher = {self._sum / self._n:.4f} (peso {self.weight:g})")
        teacher = copy.deepcopy(strategy.model).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        getattr(teacher, self.classifier_name).register_forward_pre_hook(self._cap_teacher)
        self.teacher = teacher
