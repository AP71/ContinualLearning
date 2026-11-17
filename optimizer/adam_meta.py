"""Adam metaplastico (Laborieux et al., 2021) adattato ad A&B-BNN.

Nel paper la metaplasticità è dentro l'ottimizzatore (Adam_meta): per ogni peso
binario, se il momento di Adam spinge verso il cambio di segno (W_b * m_t > 0, cioè
il passo -lr * m_t / denom va verso la soglia), il passo viene moltiplicato per
f_meta = 1 - tanh^2(meta * |W_h|). Gli altri parametri si aggiornano con Adam standard.

Adattamento ad A&B-BNN: la binarizzazione avviene dopo la Scaled Weight
Standardization, quindi il bit è sign(W - mean_filtro). W_b e |W_h| sono calcolati
su W - mean_filtro invece che su W.

Due opzioni (spente di default: senza, l'ottimizzatore è identico a prima), usate dalla v4:
- standardized: f_meta usa z = (W - mean_filtro) / std_filtro al posto di |W - mean_filtro|.
  Il forward vede solo z, quindi z è la distanza dal flip nelle unità della rete e non
  dipende dalla scala dei pesi latenti.
- renorm: dopo ogni passo ogni filtro binario viene riportato alla norma ||W - mean_filtro||
  che aveva al primo passo. L'uscita della rete non cambia (il forward è invariante alla
  scala del filtro); cambia la plasticità, che non cala più con la crescita della norma.
  Nei run v3 la norma raddoppiava in 10 task (analyzer/latent_norms.py): quel rallentamento
  è globale, non selettivo, e si sommava alla saturazione di f_meta.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def binary_weight_params(model: nn.Module) -> list[nn.Parameter]:
    """Pesi latenti delle convoluzioni binarie (HardBinaryScaledStdConv2d)."""
    return [
        param
        for module in model.modules()
        if module.__class__.__name__ == "HardBinaryScaledStdConv2d"
        for param in module.parameters(recurse=False)
        if param.ndim == 4 and param.shape[1] > 1
    ]


class AdamMeta(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable,
        binary_params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        meta: float = 1.5,
        standardized: bool = False,
        renorm: bool = False,
        std_eps: float = 1e-5,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, meta=meta)
        super().__init__(params, defaults)
        # Identificati per oggetto: robusto ai riordinamenti dei param_groups fatti da Avalanche
        self._binary_ids = {id(p) for p in binary_params}
        self.standardized = standardized
        self.renorm = renorm
        self.std_eps = std_eps  # stesso eps della standardizzazione in HardBinaryScaledStdConv2d

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                is_binary = id(p) in self._binary_ids
                dims = tuple(range(1, p.ndim))
                if self.renorm and is_binary and "filter_norm" not in state:
                    centered = p - p.mean(dim=dims, keepdim=True)
                    state["filter_norm"] = centered.pow(2).sum(dim=dims, keepdim=True).sqrt()

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # Stessa formula di torch.optim.Adam (usato dalle baseline): con meta=0
                # l'ottimizzatore coincide con Adam standard.
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                if is_binary and group["meta"] > 0:
                    centered = p - p.mean(dim=dims, keepdim=True)
                    toward_flip = torch.sign(centered) * exp_avg > 0
                    distance = centered.abs()
                    if self.standardized:
                        std = centered.pow(2).mean(dim=dims, keepdim=True).sqrt()
                        distance = distance / (std + self.std_eps)
                    f_meta = 1.0 - torch.tanh(group["meta"] * distance) ** 2
                    update = torch.where(toward_flip, f_meta * exp_avg, exp_avg)
                else:
                    update = exp_avg

                p.addcdiv_(update, denom, value=-step_size)

                if self.renorm and is_binary:
                    mean = p.mean(dim=dims, keepdim=True)
                    centered = p - mean
                    norm = centered.pow(2).sum(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)
                    p.copy_(mean + centered * (state["filter_norm"] / norm))

        return loss
