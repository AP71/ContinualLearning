"""Adapter non binari per task su backbone binario congelato + teste RanPAC fuse.

Ogni task allena SOLO i parametri non binari del backbone (LearnableBias, Q_PReLU, gain,
primo conv: ~54k valori); i pesi binari non cambiano mai (zero bit flip). Dopo il task t
l'adapter t viene congelato e salvato. A ogni task, i dati nuovi passano da TUTTI gli
adapter 0..t: l'adapter j alimenta una testa RanPAC propria (statistiche G_j, C_j sulle
classi dei task >= j). In inferenza ogni adapter produce i suoi logit e la classe c, vista
dagli adapter 0..task(c), prende la media dei logit delle teste che la conoscono (fusion="mean").
fusion="two_stage": la testa 0, che conosce tutte le classi, sceglie il task; la classe si decide
solo tra quelle del task scelto con la media di cui sopra (le altre classi ricevono un offset
costante di -1e3, quindi l'ordine dentro ogni task, usato dal TIL, non cambia).

NOTA: le classi dei task vecchi non hanno feature con gli adapter nuovi (i loro dati non ci
sono più), quindi le feature non si possono concatenare tra adapter: si fondono i logit.
"""
from __future__ import annotations

import functools
import types

import torch
import torch.nn as nn
from avalanche.core import SupervisedPlugin
from torch.utils.data import DataLoader

from avalanche_plugin.adapt_then_slda import EVAL_TRANSFORM, StreamingRidge, clean_experience_dataset
from optimizer.adam_meta import binary_weight_params


class AdapterEnsembleHead(nn.Module):
    """Sostituisce fc: ricarica uno a uno gli adapter, rilegge le feature e fonde i logit."""

    def __init__(self, base: nn.Linear, model: nn.Module, names, adapters, projs, readouts, valid,
                 fusion: str = "mean", class_task: torch.Tensor | None = None):
        super().__init__()
        self.fusion, self.class_task = fusion, class_task
        self.weight = base.weight  # restano (inutilizzati): l'ottimizzatore non resta senza parametri
        self.bias = base.bias
        self.in_features, self.out_features = base.in_features, base.out_features
        self._m = [model]  # lista: il modello non diventa sotto-modulo (niente ricorsione)
        self.names = list(names)
        self.adapters, self.projs, self.readouts, self.valid = adapters, projs, readouts, valid
        self._inner = False
        self._img: torch.Tensor | None = None
        self._feat: torch.Tensor | None = None
        self.handle = model.register_forward_pre_hook(functools.partial(self._grab_image))

    def _grab_image(self, _module, inputs) -> None:
        if not self._inner:
            self._img = inputs[0]

    def close(self) -> None:
        self.handle.remove()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._inner:  # passaggio interno: serve solo l'input di fc
            self._feat = x.float()
            return x.new_zeros(x.shape[0], self.out_features)
        model, img = self._m[0], self._img
        params = dict(model.named_parameters())
        total = torch.zeros(x.shape[0], self.out_features, device=x.device)
        logits0 = None
        count = torch.zeros(self.out_features, device=x.device)
        for j, adapter in enumerate(self.adapters):
            for n, v in adapter.items():
                params[n].copy_(v)
            self._inner = True
            try:
                model(img)
            finally:
                self._inner = False
            logits = torch.relu(self._feat @ self.projs[j]) @ self.readouts[j]
            total += logits * self.valid[j]
            count += self.valid[j]
            if j == 0:
                logits0 = torch.where(self.valid[0] > 0, logits, torch.full_like(logits, -1e4))
        fused = torch.where(count > 0, total / count.clamp(min=1), torch.full_like(total, -1e4))
        if self.fusion == "two_stage":
            pred_task = self.class_task[logits0.argmax(dim=1)]  # task scelto dalla testa 0
            fused = fused - 1e3 * (self.class_task[None, :] != pred_task[:, None]).float()
        return fused


class AdapterEnsemblePlugin(SupervisedPlugin):
    def __init__(self, rp_dim: int = 5000, ridge: float | None = None, rp_seed: int = 0,
                 init: str = "pretrained", fusion: str = "mean", loss=None, classifier_name: str = "fc",
                 eval_transform=EVAL_TRANSFORM, batch_size: int = 256, num_workers: int = 0,
                 unseen_bias: float = -1e4):
        super().__init__()
        if init not in ("pretrained", "previous"):
            raise ValueError("init deve essere 'pretrained' oppure 'previous'")
        if fusion not in ("mean", "two_stage"):
            raise ValueError("fusion deve essere 'mean' oppure 'two_stage'")
        self.fusion = fusion
        self.class_task: torch.Tensor | None = None
        self.rp_dim, self.ridge, self.rp_seed, self.init = rp_dim, ridge, rp_seed, init
        self.loss, self.classifier_name = loss, classifier_name
        self.eval_transform, self.batch_size, self.num_workers = eval_transform, batch_size, num_workers
        self.unseen_bias = unseen_bias
        self.names: list[str] = []
        self.fc0: nn.Module | None = None
        self.fc0_init: dict[str, torch.Tensor] | None = None
        self.pretrained: dict[str, torch.Tensor] = {}
        self.adapters: list[dict[str, torch.Tensor]] = []
        self.ridges: list[StreamingRidge] = []
        self.head: AdapterEnsembleHead | None = None
        self.binary_ref: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------ hooks
    def before_training_exp(self, strategy, **kwargs):
        model, task = strategy.model, strategy.experience.current_experience
        if task == 0:
            self.fc0 = getattr(model, self.classifier_name)
            self.fc0_init = {k: v.detach().clone() for k, v in self.fc0.state_dict().items()}
            skip = {id(p) for p in binary_weight_params(model)} | {id(p) for p in self.fc0.parameters()}
            self.names = [n for n, p in model.named_parameters() if id(p) not in skip]
            params = dict(model.named_parameters())
            self.pretrained = {n: params[n].detach().clone() for n in self.names}
            self.binary_ref = [p.detach().clone() for p in binary_weight_params(model)]
            size = sum(v.numel() for v in self.pretrained.values())
            print(f"[AdapterEnsemble] adapter non binario: {len(self.names)} tensori, {size} valori "
                  f"({size * 4 / 1024:.0f} KB fp32); pesi binari congelati")
        else:  # fc torna quella allenabile (non l'ensemble) e l'adapter riparte da zero o dal precedente
            if self.head is not None:
                self.head.close()
                self.head = None
            setattr(model, self.classifier_name, self.fc0)
            self.fc0.load_state_dict(self.fc0_init)
            params = dict(model.named_parameters())
            start = self.pretrained if self.init == "pretrained" else self.adapters[-1]
            with torch.no_grad():
                for n in self.names:
                    params[n].copy_(start[n])
        for p in binary_weight_params(model):
            p.requires_grad = False
        params = dict(model.named_parameters())
        for n in self.names:
            params[n].requires_grad = True
        for p in self.fc0.parameters():
            p.requires_grad = True
        if self.loss is not None:
            self.loss.set_classes(strategy.experience.classes_in_this_experience)

    def after_training_exp(self, strategy, **kwargs):
        model, task = strategy.model, strategy.experience.current_experience
        params = dict(model.named_parameters())
        self.adapters.append({n: params[n].detach().clone() for n in self.names})
        for p in model.parameters():
            p.requires_grad = False
        dataset = clean_experience_dataset(strategy.experience, self.eval_transform, "AdapterEnsemble")
        num_classes = self.fc0.out_features
        for j in range(task + 1):  # i dati nuovi passano da tutti gli adapter
            feats, labels = self._extract(strategy, j, dataset)
            if j == len(self.ridges):  # primo uso della testa j: proiezione propria, r scelto sul suo primo task
                ridge = StreamingRidge(feats.shape[1], num_classes, self.rp_dim, self.rp_seed + j,
                                       strategy.device, self.batch_size, True)
                ridge.ridge = self.ridge if self.ridge is not None else ridge.select_ridge(
                    feats, labels, seed=self.rp_seed + j)
                self.ridges.append(ridge)
            self.ridges[j].update(feats, labels)
        if self.head is not None:
            self.head.close()
        projs, readouts, valid = [], [], []
        for ridge in self.ridges:
            dummy = types.SimpleNamespace(readout=torch.zeros(ridge.gram.shape[0], num_classes),
                                          bias=torch.zeros(num_classes))
            ridge.write(dummy, self.unseen_bias)
            projs.append(ridge.proj.to(strategy.device))
            readouts.append(dummy.readout.float().to(strategy.device))
            valid.append((dummy.bias == 0).float().to(strategy.device))
        adapters = [{n: v.to(strategy.device) for n, v in a.items()} for a in self.adapters]
        if self.class_task is None:
            self.class_task = torch.full((num_classes,), -1, dtype=torch.long)
        self.class_task[torch.as_tensor(sorted(int(c) for c in strategy.experience.classes_in_this_experience))] = task
        self.head = AdapterEnsembleHead(self.fc0, model, self.names, adapters, projs, readouts, valid,
                                        self.fusion, self.class_task.to(strategy.device))
        setattr(model, self.classifier_name, self.head)
        self._check_binary(model)
        covered = [int(v.sum()) for v in valid]
        print(f"[AdapterEnsemble] task {task + 1}: {len(self.adapters)} adapter, classi per testa {covered}, "
              f"rp_dim={self.rp_dim}, bit flip totali = 0")

    # ---------------------------------------------------------------- helpers
    def _check_binary(self, model) -> None:
        for ref, p in zip(self.binary_ref, binary_weight_params(model)):
            if not torch.equal(ref.to(p.device), p.detach()):
                raise RuntimeError("i pesi binari sono cambiati: l'adapter non dovrebbe toccarli")

    @torch.no_grad()
    def _extract(self, strategy, j: int, dataset) -> tuple[torch.Tensor, torch.Tensor]:
        model = strategy.model
        classifier = getattr(model, self.classifier_name)  # fc0 (semplice) durante il task
        params = dict(model.named_parameters())
        for n, v in self.adapters[j].items():
            params[n].copy_(v)
        captured: list[torch.Tensor] = []
        hook = classifier.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0]))
        was_training = model.training
        model.eval()
        feats, labels = [], []
        try:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            for batch in loader:
                model(batch[0].to(strategy.device))
                feats.append(captured.pop().float().double().cpu())
                labels.append(torch.as_tensor(batch[1]).long().cpu())
        finally:
            hook.remove()
            model.train(was_training)
        return torch.cat(feats), torch.cat(labels)
