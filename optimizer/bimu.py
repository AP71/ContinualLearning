"""Ottimizzatore BiMU (Cottart et al., 2026) per i pesi binari di A&B-BNN.

Ogni peso binario ha un posterior Bernoulli mean-field con parametro naturale λ
(P(ω=+1) = σ(2λ)). Il tensore latente della convoluzione binaria È λ (vedi
avalanche_plugin/bimu.py). Con g = dL/dλ (stimata con il rilassamento concreto,
mediata sui campioni MC), l'aggiornamento dell'Algoritmo 1 / Eq. 61-62 è:

    η   = 1 / ( βKL (1 - tanh²λ) + 2 βL (tanh λ · g + |g|) + 1/αmax )
    λ  <- λ - η · γ · ( βL g + βKL/N · (λ - λprior) · (1 - tanh²λ) )

Il passo è limitato da αmax; i passi che spingono contro il segno corrente di λ sono
più piccoli (metaplasticità); il secondo termine rilassa verso il prior solo i pesi
incerti. Gli altri parametri (gain, bias, RPReLU, fc, ...) usano Adam standard.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


class BiMU(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable,
        binary_params: Iterable[nn.Parameter],
        lr: float = 1e-3,            # lr di Adam per i parametri non binari
        gamma: float = 1.0,          # γ: scala del gradiente sui pesi binari
        alpha_max: float = 0.06,     # passo massimo
        beta_l: float = 100.0,       # βL: peso del termine dati
        beta_kl: float = 1.0,        # βKL: peso dei termini KL
        memory_n: float = 7000.0,    # N: finestra di memoria (in aggiornamenti)
        lambda_prior: float = 0.0,   # λ_prior (0 = prior massimamente incerto)
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, gamma=gamma, alpha_max=alpha_max,
                        beta_l=beta_l, beta_kl=beta_kl, memory_n=memory_n,
                        lambda_prior=lambda_prior)
        super().__init__(params, defaults)
        self._binary_ids = {id(p) for p in binary_params}

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
                state = self.state[p]

                if id(p) in self._binary_ids:
                    g = grad
                    t = torch.tanh(p)
                    unc = 1.0 - t * t                      # = cosh^-2(λ) = Var(ω)
                    bl, bkl = group["beta_l"], group["beta_kl"]
                    inv_eta = bkl * unc + 2.0 * bl * (t * g + g.abs()) + 1.0 / group["alpha_max"]
                    relax = (bkl / group["memory_n"]) * (p - group["lambda_prior"]) * unc
                    p.sub_(group["gamma"] * (bl * g + relax) / inv_eta)
                    continue

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-group["lr"] / bc1)

        return loss
