from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.training.plugins import SupervisedPlugin


class CosineLinear(nn.Module):
    """
    Sostituisce 'fc' in 'v4' normalizzando L2 sia le feature x (da pool1)
    sia i pesi W, eliminando lo squilibrio di norma tra i primi task e gli ultimi.
    """

    def __init__(self, orig_linear: nn.Linear, scale: float = 16.0):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.weight = orig_linear.weight
        self.bias = orig_linear.bias
        if self.bias is not None:
            self.bias.data.zero_()
            self.bias.requires_grad_(False)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w_norm) * self.scale


class MetaplasticityPlugin(SupervisedPlugin):
    """
    Plugin Avalanche per la Metaplasticità Sinaptica (Laborieux et al., 2021)
    su Reactnet_A_Qaw (A&B-BNN).

    Versioni selezionabili tramite la stringa `version`:
    - "v1" (o "paper") : Versione base fedele al paper originale.
    - "v2"             : Step effettivo + masking classi non viste + norm L2 di 'fc' (default m=1.5).
    - "v3"             : Come v2 + freno sul drift dei moduli non-binari (default m=6.0).
    - "v4"             : Come v3 + CosineLinear (norm anche sulle feature x) +
                         Allineamento Prototipale (NCM Centroid Calibration) di 'fc' a fine task.
    """

    PRESETS = {
        "v1": {
            "default_m": 1.5,
            "mask_unseen_logits": False,
            "normalize_classifier": False,
            "freeze_future_fc": False,
            "reset_fc_moments": False,
            "non_binary_lr_scale": 1.0,
            "use_cosine_fc": False,
            "prototype_alignment": False,
            "proto_alpha": 0.0,
        },
        "v2": {
            "default_m": 1.5,
            "mask_unseen_logits": True,
            "normalize_classifier": True,
            "freeze_future_fc": True,
            "reset_fc_moments": True,
            "non_binary_lr_scale": 1.0,
            "use_cosine_fc": False,
            "prototype_alignment": False,
            "proto_alpha": 0.0,
        },
        "v3": {
            "default_m": 6.0,
            "mask_unseen_logits": True,
            "normalize_classifier": True,
            "freeze_future_fc": True,
            "reset_fc_moments": True,
            "non_binary_lr_scale": 0.05,
            "use_cosine_fc": False,
            "prototype_alignment": False,
            "proto_alpha": 0.0,
        },
        "v4": {
            "default_m": 5.0,
            "mask_unseen_logits": True,        # Mantiene la protezione totale di v3!
            "normalize_classifier": True,
            "freeze_future_fc": True,
            "reset_fc_moments": True,
            "non_binary_lr_scale": 0.03,
            "use_cosine_fc": True,             # Normalizza L2 anche le feature x
            "prototype_alignment": True,       # Calibra i pesi di fc col centroide di classe
            "proto_alpha": 0.5,                # 50% peso ottimizzato + 50% centroide NCM puro
        },
    }
    PRESETS["paper"] = PRESETS["v1"]

    def __init__(
        self,
        version: str = "v3",
        m: float | None = None,
        classifier_name: str = "fc",
        cosine_scale: float = 16.0,
        **overrides,
    ):
        super().__init__()
        version_key = version.lower().strip()
        if version_key not in self.PRESETS:
            valid = "'v1' (o 'paper'), 'v2', 'v3', 'v4'"
            raise ValueError(f"Versione '{version}' non valida. Scegli tra: {valid}")

        cfg = dict(self.PRESETS[version_key])
        cfg.update(overrides)

        self.version = version_key
        self.m = float(cfg["default_m"] if m is None else m)
        self.classifier_name = classifier_name
        self.cosine_scale = cosine_scale

        self.mask_unseen_logits: bool = cfg["mask_unseen_logits"]
        self.normalize_classifier: bool = cfg["normalize_classifier"]
        self.freeze_future_fc: bool = cfg["freeze_future_fc"]
        self.reset_fc_moments: bool = cfg["reset_fc_moments"]
        self.non_binary_lr_scale: float = cfg["non_binary_lr_scale"]
        self.use_cosine_fc: bool = cfg["use_cosine_fc"]
        self.prototype_alignment: bool = cfg["prototype_alignment"]
        self.proto_alpha: float = cfg["proto_alpha"]

        self.seen_classes: set[int] = set()
        self.curr_classes: set[int] = set()
        self.past_classes: list[int] = []
        self.binary_weight_names: set[str] = set()
        self.non_binary_backbone_names: set[str] = set()
        self.old_params: dict[str, torch.Tensor] = {}

        # Hook per raccogliere i centroidi delle feature nell'ultima epoca (v4)
        self._last_features: torch.Tensor | None = None
        self._hook_handle = None
        self._class_feat_sums: dict[int, torch.Tensor] = {}
        self._class_counts: dict[int, int] = {}

        print(
            f"[MetaplasticityPlugin] Configurato: version='{self.version}', m={self.m}, "
            f"mask_unseen={self.mask_unseen_logits}, cosine_fc={self.use_cosine_fc}, "
            f"proto_align={self.prototype_alignment}, non_binary_lr_scale={self.non_binary_lr_scale}"
        )

    def _identify_weights(self, model: nn.Module):
        self.binary_weight_names.clear()
        self.non_binary_backbone_names.clear()

        for mod_name, mod in model.named_modules():
            if mod.__class__.__name__ == "HardBinaryScaledStdConv2d":
                for p_name, param in mod.named_parameters(recurse=False):
                    full_name = f"{mod_name}.{p_name}"
                    if param.ndim == 4 and param.shape[1] > 1:
                        self.binary_weight_names.add(full_name)
                    else:
                        self.non_binary_backbone_names.add(full_name)

        for name, _ in model.named_parameters():
            if name not in self.binary_weight_names and self.classifier_name not in name:
                self.non_binary_backbone_names.add(name)

    @torch.no_grad()
    def before_training_exp(self, strategy, **kwargs):
        if self.use_cosine_fc and hasattr(strategy.model, self.classifier_name):
            fc_layer = getattr(strategy.model, self.classifier_name)
            if not isinstance(fc_layer, CosineLinear):
                setattr(
                    strategy.model,
                    self.classifier_name,
                    CosineLinear(fc_layer, scale=self.cosine_scale).to(
                        fc_layer.weight.device
                    ),
                )

        if self.prototype_alignment and self._hook_handle is None:
            fc_layer = getattr(strategy.model, self.classifier_name)

            def _capture_input(module, inputs, output):
                self._last_features = inputs[0].detach()

            self._hook_handle = fc_layer.register_forward_hook(_capture_input)

        self._identify_weights(strategy.model)

        self.curr_classes = set(strategy.experience.classes_in_this_experience)
        self.past_classes = sorted(list(self.seen_classes - self.curr_classes))
        self.seen_classes.update(self.curr_classes)

        self._class_feat_sums.clear()
        self._class_counts.clear()

        if self.reset_fc_moments:
            for group in strategy.optimizer.param_groups:
                for p in group["params"]:
                    if p in strategy.optimizer.state and p.ndim <= 2:
                        strategy.optimizer.state[p].clear()

    def before_training_epoch(self, strategy, **kwargs):
        # Raccoglie le statistiche dei centroidi solo nelle ultime 3 epoche del task
        # quando il backbone ha ormai stabilizzato le feature del task corrente
        self._class_feat_sums.clear()
        self._class_counts.clear()

    @torch.no_grad()
    def after_forward(self, strategy, **kwargs):
        if strategy.is_training:
            # Accumula le feature normalizzate per calcolare i prototipi di classe (v4)
            if self.prototype_alignment and self._last_features is not None:
                feats = F.normalize(self._last_features, p=2, dim=1)
                targets = strategy.mb_y
                for c in self.curr_classes:
                    mask = targets == c
                    if mask.any():
                        c_sum = feats[mask].sum(dim=0)
                        if c not in self._class_feat_sums:
                            self._class_feat_sums[c] = c_sum.clone()
                            self._class_counts[c] = int(mask.sum().item())
                        else:
                            self._class_feat_sums[c] += c_sum
                            self._class_counts[c] += int(mask.sum().item())

            if self.mask_unseen_logits:
                num_classes = strategy.mb_output.shape[1]
                unseen_classes = [c for c in range(num_classes) if c not in self.curr_classes]
                if unseen_classes:
                    strategy.mb_output[:, unseen_classes] = -1e9

    @torch.no_grad()
    def before_update(self, strategy, **kwargs):
        curr_task = strategy.experience.current_experience
        self.old_params.clear()

        for name, param in strategy.model.named_parameters():
            if not param.requires_grad:
                continue
            self.old_params[name] = param.data.clone()

            if self.classifier_name in name and param.grad is not None:
                if self.freeze_future_fc:
                    num_classes = param.shape[0]
                    locked_classes = [c for c in range(num_classes) if c not in self.curr_classes]
                else:
                    locked_classes = self.past_classes if curr_task > 0 else []

                if locked_classes:
                    if param.dim() > 1:
                        param.grad.data[locked_classes, :] = 0.0
                    elif param.dim() == 1:
                        param.grad.data[locked_classes] = 0.0

    @torch.no_grad()
    def after_update(self, strategy, **kwargs):
        curr_task = strategy.experience.current_experience

        for name, param in strategy.model.named_parameters():
            if name not in self.old_params:
                continue

            w_old = self.old_params[name]

            # 1. Classificatore finale 'fc'
            if self.classifier_name in name:
                if self.freeze_future_fc:
                    num_classes = param.shape[0]
                    locked_classes = [c for c in range(num_classes) if c not in self.curr_classes]
                else:
                    locked_classes = self.past_classes if curr_task > 0 else []

                if locked_classes:
                    if param.dim() > 1:
                        param.data[locked_classes, :] = w_old[locked_classes, :]
                    elif param.dim() == 1:
                        param.data[locked_classes] = w_old[locked_classes]

                if self.normalize_classifier:
                    if param.dim() > 1:
                        seen_list = list(self.seen_classes)
                        scale_factor = 1.0 if self.use_cosine_fc else 5.0
                        param.data[seen_list] = (
                            F.normalize(param.data[seen_list], p=2, dim=1) * scale_factor
                        )
                    elif param.dim() == 1:
                        param.data.zero_()
                continue

            # 2. Pesi binari 4D (HardBinaryScaledStdConv2d)
            if name in self.binary_weight_names:
                delta_w = param.data - w_old
                w_b = torch.sign(w_old)
                is_opposite = (delta_w * w_b) < 0

                f_meta = 1.0 - torch.tanh(self.m * w_old) ** 2
                effective_delta = torch.where(is_opposite, delta_w * f_meta, delta_w)
                param.data.copy_(w_old + effective_delta)
                continue

            # 3. Moduli non-binari del backbone
            if (
                name in self.non_binary_backbone_names
                and curr_task > 0
                and self.non_binary_lr_scale < 1.0
            ):
                delta_w = param.data - w_old
                param.data.copy_(w_old + self.non_binary_lr_scale * delta_w)

        self.old_params.clear()

    @torch.no_grad()
    def after_training_exp(self, strategy, **kwargs):
        """
        Allineamento Prototipale (v4) a fine task:
        Combina il peso appreso da SGD/Adam con il vero centroide normalizzato (NCM)
        delle feature della classe, garantendo che tutti i task abbiano lo stesso
        allineamento geometrico senza mai sbilanciare il backbone.
        """
        if not self.prototype_alignment or not self._class_feat_sums:
            return

        fc_layer = getattr(strategy.model, self.classifier_name)
        w = fc_layer.weight.data

        for c, feat_sum in self._class_feat_sums.items():
            if self._class_counts.get(c, 0) > 0:
                proto = F.normalize(feat_sum, p=2, dim=0)
                w_curr = F.normalize(w[c], p=2, dim=0)
                w_aligned = F.normalize(
                    (1.0 - self.proto_alpha) * w_curr + self.proto_alpha * proto,
                    p=2,
                    dim=0,
                )
                w[c].copy_(w_aligned)