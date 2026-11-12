from __future__ import annotations

import torch
import torch.nn.functional as F
from avalanche.training.plugins import SupervisedPlugin


class ClassifierProtectionPlugin(SupervisedPlugin):
    """
    Protezioni per il class-incremental, indipendenti dalla regola metaplastica
    (che sta in optimizer/adam_meta.py):

    - mask_unseen_logits: in training i logit delle classi fuori dal task corrente
      valgono -1e9, quindi il task nuovo non spinge giù i logit delle classi vecchie.
    - freeze_non_current_fc: le righe di 'fc' fuori dal task corrente non si aggiornano.
    - normalize_classifier: dopo ogni passo le righe delle classi viste hanno norma 5
      e il bias è zero (contro lo sbilanciamento verso le classi recenti).
    - reset_fc_moments: a inizio task si azzerano i momenti dell'ottimizzatore dei
      parametri con <= 2 dimensioni ('fc', pendenze RPReLU).
    - non_binary_lr_scale: dal secondo task il passo dei parametri reali del backbone
      (gain, bias, RPReLU, primo layer) è moltiplicato per questo fattore.
      I pesi binari e 'fc' non sono toccati.
    """

    def __init__(
        self,
        classifier_name: str = "fc",
        mask_unseen_logits: bool = True,
        freeze_non_current_fc: bool = True,
        normalize_classifier: bool = True,
        reset_fc_moments: bool = True,
        non_binary_lr_scale: float = 1.0,
    ):
        super().__init__()
        self.classifier_name = classifier_name
        self.mask_unseen_logits = mask_unseen_logits
        self.freeze_non_current_fc = freeze_non_current_fc
        self.normalize_classifier = normalize_classifier
        self.reset_fc_moments = reset_fc_moments
        self.non_binary_lr_scale = non_binary_lr_scale

        self.seen_classes: set[int] = set()
        self.curr_classes: set[int] = set()
        self.binary_weight_names: set[str] = set()
        self.non_binary_backbone_names: set[str] = set()
        self.old_params: dict[str, torch.Tensor] = {}

        print(
            f"[ClassifierProtectionPlugin] mask_unseen={mask_unseen_logits}, "
            f"freeze_fc={freeze_non_current_fc}, normalize_fc={normalize_classifier}, "
            f"reset_fc_moments={reset_fc_moments}, non_binary_lr_scale={non_binary_lr_scale}"
        )

    def _identify_weights(self, model: torch.nn.Module):
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

    def _locked_fc_rows(self, num_classes: int) -> list[int]:
        if not self.freeze_non_current_fc:
            return []
        return [c for c in range(num_classes) if c not in self.curr_classes]

    @torch.no_grad()
    def before_training_exp(self, strategy, **kwargs):
        self._identify_weights(strategy.model)

        self.curr_classes = set(strategy.experience.classes_in_this_experience)
        self.seen_classes.update(self.curr_classes)

        if self.reset_fc_moments:
            for group in strategy.optimizer.param_groups:
                for p in group["params"]:
                    if p in strategy.optimizer.state and p.ndim <= 2:
                        strategy.optimizer.state[p].clear()

    @torch.no_grad()
    def after_forward(self, strategy, **kwargs):
        if strategy.is_training and self.mask_unseen_logits:
            num_classes = strategy.mb_output.shape[1]
            unseen_classes = [c for c in range(num_classes) if c not in self.curr_classes]
            if unseen_classes:
                strategy.mb_output[:, unseen_classes] = -1e9

    @torch.no_grad()
    def before_update(self, strategy, **kwargs):
        self.old_params.clear()

        for name, param in strategy.model.named_parameters():
            if not param.requires_grad:
                continue
            self.old_params[name] = param.data.clone()

            if self.classifier_name in name and param.grad is not None:
                locked = self._locked_fc_rows(param.shape[0])
                if locked:
                    param.grad.data[locked] = 0.0

    @torch.no_grad()
    def after_update(self, strategy, **kwargs):
        curr_task = strategy.experience.current_experience

        for name, param in strategy.model.named_parameters():
            if name not in self.old_params:
                continue

            w_old = self.old_params[name]

            if self.classifier_name in name:
                locked = self._locked_fc_rows(param.shape[0])
                if locked:
                    param.data[locked] = w_old[locked]

                if self.normalize_classifier:
                    if param.dim() > 1:
                        seen_list = list(self.seen_classes)
                        param.data[seen_list] = F.normalize(param.data[seen_list], p=2, dim=1) * 5.0
                    elif param.dim() == 1:
                        param.data.zero_()
                continue

            if (
                name in self.non_binary_backbone_names
                and curr_task > 0
                and self.non_binary_lr_scale < 1.0
            ):
                delta_w = param.data - w_old
                param.data.copy_(w_old + self.non_binary_lr_scale * delta_w)

        self.old_params.clear()
