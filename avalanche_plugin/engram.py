"""Engram-mBNN (Aguilar et al., 2025) su Reactnet_A_Qaw.

Nel paper una mBNN a 2 layer nascosti riceve, per ogni layer, un vettore di gate stocastici
prodotto da un "Engram Gating Block" che vede l'input. Qui:
- Contesto: uscita del primo blocco (feature[0], 32 canali), media spaziale, detach.
- Encoder 32→128→64 (reale, ReLU) come spazio latente; poi lineare 64→somma dei canali
  dei blocchi gated; i logit alimentano gate_b ~ Bernoulli(σ(z_b)) con valori {0,1}.
- Il gate del blocco b moltiplica per canale l'uscita del blocco (B×C×1×1).
- Gradiente: straight-through (forward campione 0/1, backward di σ). Il paper non lo
  specifica. In valutazione il gate è deterministico (σ>0.5) o campionato (gate_eval).
- La metaplasticità sui pesi binari è AdamMeta (Laborieux), come nel paper; l'engram è
  addestrato con Adam standard (il paper non dice nulla sulla sua precisione).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StraightThroughBernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prob):
        return (torch.rand_like(prob) < prob).to(prob.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad


class EngramGate(nn.Module):
    def __init__(self, context_dim: int, gate_channels: list[int], hidden=(128, 64), gate_eval: str = "deterministic"):
        super().__init__()
        assert gate_eval in ("deterministic", "sample")
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU()
        )
        self.to_gates = nn.Linear(hidden[1], sum(gate_channels))
        self.gate_channels = list(gate_channels)
        self.gate_eval = gate_eval
        self.current: list[torch.Tensor] | None = None

    def compute(self, context: torch.Tensor):
        prob = torch.sigmoid(self.to_gates(self.encoder(context)))
        if self.training or self.gate_eval == "sample":
            gates = StraightThroughBernoulli.apply(prob)
        else:
            gates = (prob > 0.5).to(prob.dtype)
        self.current = [g[:, :, None, None] for g in torch.split(gates, self.gate_channels, dim=1)]


def attach_engram(model, gate_blocks, hidden=(128, 64), gate_eval="deterministic", input_size=32):
    """Aggiunge model.engram e gli hook; va chiamata prima di creare l'ottimizzatore."""
    blocks = list(model.feature)
    device = next(model.parameters()).device
    channels, ctx_dim = [], None
    was_training = model.training
    model.eval()
    with torch.no_grad():  # una passata a vuoto per leggere i canali (gli hook non ci sono ancora)
        x = torch.zeros(1, 3, input_size, input_size, device=device)
        for i, block in enumerate(blocks):
            x = block(x)
            if i == 0:
                ctx_dim = x.shape[1]
            if i in gate_blocks:
                channels.append(x.shape[1])
    model.train(was_training)

    engram = EngramGate(ctx_dim, channels, hidden, gate_eval).to(device)
    model.engram = engram

    blocks[0].register_forward_hook(lambda m, inp, out: engram.compute(out.mean(dim=(2, 3)).detach()))
    for k, i in enumerate(gate_blocks):
        blocks[i].register_forward_hook(lambda m, inp, out, k=k: out * engram.current[k])
    print(f"[Engram] gate sui blocchi {list(gate_blocks)} ({sum(channels)} gate), "
          f"contesto {ctx_dim}->{hidden}, gate_eval={gate_eval}")
    return engram
