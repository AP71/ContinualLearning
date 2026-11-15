"""CWR* (Lomonaco, Maltoni, Pellegrini 2020; usato con le BNN da Vorabbi et al.).

Due insiemi di pesi per il classificatore ``fc``:
  - cw (consolidati): usati per inferenza, conservati dal plugin;
  - tw (temporanei): sono i pesi di ``fc`` durante il training del task.
Ogni task: tw delle classi del task = cw (zero per classi mai viste), le altre righe a 0;
si allena (dal secondo task solo ``fc``, backbone congelato); a fine task, per ogni classe j
del task:
    wpast_j = sqrt(past_j / cur_j)
    cw_j = (cw_j * wpast_j + (tw_j - media(tw))) / (wpast_j + 1)
    past_j += cur_j
e ``fc`` viene riscritta con cw (bias molto negativo sulle classi non viste) per
valutazione e checkpoint. Con classi disgiunte tra task (class-incremental su
Split-CIFAR-100) past_j = 0, quindi wpast_j = 0 e la media pesata si riduce alla copia
di tw - media(tw): è il comportamento atteso, non un errore.
``zero_mean=False`` disattiva la sottrazione della media (CWR senza +).

Il backbone viene addestrato solo nei primi ``adapt_tasks`` task (come negli altri
metodi con backbone congelato) e poi congelato: nessun bit cambia dopo.
"""
from __future__ import annotations

import torch
from avalanche.core import SupervisedPlugin
from torch import nn


class CWRStarPlugin(SupervisedPlugin):
    def __init__(self, classifier_name: str = "fc", adapt_tasks: int = 1,
                 zero_mean: bool = True, unseen_bias: float = -1e4):
        super().__init__()
        if adapt_tasks < 1:
            raise ValueError("adapt_tasks deve essere >= 1 (il primo task allena anche il backbone)")
        self.classifier_name = classifier_name
        self.adapt_tasks = adapt_tasks
        self.zero_mean = zero_mean
        self.unseen_bias = unseen_bias
        self.cw: torch.Tensor | None = None  # (C, D)
        self.cb: torch.Tensor | None = None  # (C,)
        self.past: torch.Tensor | None = None  # (C,)
        self.frozen = False

    def _fc(self, strategy) -> nn.Linear:
        return getattr(strategy.model, self.classifier_name)

    @torch.no_grad()
    def before_training_exp(self, strategy, **kwargs):
        fc = self._fc(strategy)
        if self.cw is None:
            self.cw = torch.zeros_like(fc.weight)
            self.cb = torch.zeros_like(fc.bias)
            self.past = torch.zeros(fc.out_features, device=fc.weight.device)
        task = strategy.experience.current_experience
        if task >= self.adapt_tasks:
            self._freeze(strategy.model)
        # tw: classi del task = cw (0 se nuove), le altre a 0; al task 1 si parte dalla fc iniziale
        if task > 0:
            classes = list(strategy.experience.classes_in_this_experience)
            fc.weight.zero_()
            fc.bias.zero_()
            fc.weight[classes] = self.cw[classes]
            fc.bias[classes] = self.cb[classes]

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        fc = self._fc(strategy)
        exp = strategy.experience
        classes = sorted(exp.classes_in_this_experience)
        targets = torch.as_tensor(list(exp.dataset.targets)).long()
        cur = torch.stack([(targets == c).sum() for c in classes]).to(fc.weight.device).float()
        tw, tb = fc.weight[classes].clone(), fc.bias[classes].clone()
        if self.zero_mean:
            tw = tw - tw.mean(dim=0, keepdim=True)
            tb = tb - tb.mean()
        past = self.past[classes]
        wpast = torch.sqrt(past / cur.clamp(min=1))
        self.cw[classes] = (self.cw[classes] * wpast[:, None] + tw) / (wpast[:, None] + 1)
        self.cb[classes] = (self.cb[classes] * wpast + tb) / (wpast + 1)
        self.past[classes] += cur
        seen = self.past > 0
        fc.weight.copy_(self.cw)
        fc.bias.copy_(torch.where(seen, self.cb, torch.full_like(self.cb, self.unseen_bias)))
        print(f"[CWR*] consolidate {len(classes)} classi, {int(seen.sum())} viste, zero_mean={self.zero_mean}")
        if task_done := (exp.current_experience + 1 >= self.adapt_tasks):
            self._freeze(strategy.model, quiet=task_done and exp.current_experience + 1 > self.adapt_tasks)

    def _freeze(self, model: nn.Module, quiet: bool = False) -> None:
        if self.frozen:
            return
        for name, param in model.named_parameters():
            if not name.startswith(self.classifier_name):
                param.requires_grad = False
        self.frozen = True
        if not quiet:
            print("[CWR*] backbone congelato: dal task successivo si allena solo fc (tw)")
