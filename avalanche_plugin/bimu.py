"""BiMU (Cottart et al., ICML 2026) su Reactnet_A_Qaw: posterior Bernoulli sui pesi binari.

Adattamento ad A&B-BNN (dove il paper usa MLP e un classificatore lineare):
- Il latente delle HardBinaryScaledStdConv2d è il parametro naturale λ del posterior
  Bernoulli. All'inizio λ è ricavato dal latente preaddestrato:
      λ = lambda_init · (W - media_filtro) / media(|W - media_filtro|)
  (il segno, cioè il bit di A&B-BNN, resta quello del checkpoint; |λ| medio = lambda_init).
- In training ogni peso è campionato col rilassamento concreto del paper (Alg. 1):
      ω = tanh((λc + δ)/T),  δ = ½(log ε - log(1-ε)),  ε ~ U(0,1),  λc = λ - media_filtro
  e il gradiente arriva a λ per backprop attraverso il rilassamento (un campione per passo,
  K=1; K>1 si ottiene con più passi). Il fattore di scala per filtro è quello di A&B-BNN.
- In valutazione i pesi sono la moda del posterior, sign(λ - media_filtro): lo stesso bit
  usato dal calcolo dei bit flip. (Il paper usa inferenza MC; qui no, per costo.)
- Query attive opzionali (ActiveQueryPlugin): K passate MC, variation ratio VR = 1 - f_mode/K;
  solo i campioni con VR >= τ contribuiscono alla loss.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from avalanche.training.plugins import SupervisedPlugin

from optimizer.adam_meta import binary_weight_params


def _bimu_get_weight(self):
    lam = self.weight
    std, mean = torch.std_mean(lam, dim=[1, 2, 3], keepdim=True, unbiased=False)
    standardized = self.scale * (lam - mean) / (std + self.eps)
    scaling = standardized.abs().mean(dim=(1, 2, 3), keepdim=True).detach()
    z = lam - mean.detach()
    if self.training and self.bimu_sampling:
        eps = torch.rand_like(z).clamp_(1e-6, 1 - 1e-6)
        delta = 0.5 * (torch.log(eps) - torch.log1p(-eps))
        omega = torch.tanh((z + delta) / self.bimu_temperature)
    else:
        omega = torch.sign(z)
    return self.gain * scaling * omega


def convert_to_bimu(model, lambda_init: float = 2.0, temperature: float = 1.0, adapt_from_block: int = 0):
    """Trasforma i latenti binari in λ e sostituisce il campionamento dei pesi.

    I blocchi feature[:adapt_from_block] (congelati) restano deterministici: sign(λ - media).
    """
    n = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "HardBinaryScaledStdConv2d":
            continue
        frozen = name.startswith("feature.") and int(name.split(".")[1]) < adapt_from_block
        with torch.no_grad():
            centered = module.weight - module.weight.mean(dim=(1, 2, 3), keepdim=True)
            module.weight.copy_(lambda_init * centered / centered.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12))
        module.bimu_temperature = temperature
        module.bimu_sampling = not frozen
        module.get_weight = types.MethodType(_bimu_get_weight, module)
        n += 1
    print(f"[BiMU] {n} convoluzioni binarie convertite (λ medio |λ|={lambda_init}, T={temperature})")
    return binary_weight_params(model)


class ActiveQueryPlugin(SupervisedPlugin):
    """Query attive di BiMU: la loss usa solo i campioni con disaccordo MC >= threshold."""

    def __init__(self, threshold: float, mc_samples: int = 5):
        super().__init__()
        self.threshold = threshold
        self.mc_samples = mc_samples
        self.queried = 0
        self.seen = 0

    @torch.no_grad()
    def _variation_ratio(self, strategy):
        model, x = strategy.model, strategy.mbatch[0]
        votes = torch.stack([model(x).argmax(1) for _ in range(self.mc_samples)])  # K×B
        counts = F.one_hot(votes, strategy.mb_output.shape[1]).sum(0)               # B×C
        return 1.0 - counts.max(1).values.float() / self.mc_samples

    def before_backward(self, strategy, **kwargs):
        vr = self._variation_ratio(strategy)
        mask = vr >= self.threshold
        self.queried += int(mask.sum())
        self.seen += int(mask.numel())
        if mask.any():
            strategy.loss = strategy._criterion(strategy.mb_output[mask], strategy.mb_y[mask])
        else:
            strategy.loss = strategy.mb_output.sum() * 0.0

    def after_training_exp(self, strategy, **kwargs):
        if self.seen:
            print(f"[BiMU] query attive: {self.queried}/{self.seen} campioni ({100 * self.queried / self.seen:.1f}%)")
        self.queried = self.seen = 0
