import torch
import torch.nn.functional as F
from avalanche.training.plugins import SupervisedPlugin


class MetaplasticityPlugin(SupervisedPlugin):
    """
    Plugin Avalanche per la tesi:
    - Modalità Paper Originale (Laborieux et al., 2021):
      normalize_classifier=False, rms_norm=False, depth_scaling=False, non_binary_lr_scale=1.0
    - Modalità Migliorata per Reactnet_A_Qaw (A&B-BNN):
      attiva i miglioramenti per risolvere i limiti dei layer non-binari e del Fan-In SWS.
    """

    def __init__(
        self,
        m: float = 1.5,
        classifier_name: str = "fc",
        # --- Flag per Ablation Study nella Tesi ---
        mask_unseen_logits: bool = False,       # Fix 1a: evita l'affossamento dei Task 4-10
        normalize_classifier: bool = False,     # Fix 1b: simula l'equilibrio di norma di una testa binaria
        rms_norm: bool = False,                # Fix 2a: pareggia la scala tra kernel 3x3 e 1x1 (pw)
        depth_scaling: bool = False,           # Fix 2b: protegge i blocchi profondi (feature.7 -> 13)
        non_binary_lr_scale: float = 1.0,      # Fix 3: se < 1.0 (es. 0.05), frena il drift dei LearnableBias
    ):
        super().__init__()
        self.m = m
        self.classifier_name = classifier_name
        self.mask_unseen_logits = mask_unseen_logits
        self.normalize_classifier = normalize_classifier
        self.rms_norm = rms_norm
        self.depth_scaling = depth_scaling
        self.non_binary_lr_scale = non_binary_lr_scale

        self.seen_classes = set()
        self.curr_classes = set()
        self.binary_weight_names = set()
        self.non_binary_backbone_names = set()
        self.layer_rms_scale = {}
        self.layer_m = {}
        self.old_params = {}

    def _get_block_index(self, param_name: str) -> int:
        parts = param_name.split(".")
        if len(parts) >= 2 and parts[0] == "feature" and parts[1].isdigit():
            return int(parts[1])
        return 1

    def _identify_weights(self, model: torch.nn.Module):
        self.binary_weight_names.clear()
        self.non_binary_backbone_names.clear()

        for mod_name, mod in model.named_modules():
            if mod.__class__.__name__ == "HardBinaryScaledStdConv2d":
                for p_name, param in mod.named_parameters(recurse=False):
                    full_name = f"{mod_name}.{p_name}"
                    if param.ndim == 4 and param.shape[1] > 1:
                        self.binary_weight_names.add(full_name)
                        if self.depth_scaling:
                            block_idx = self._get_block_index(full_name)
                            depth_factor = 0.5 + 0.1 * block_idx
                            kernel_boost = 1.25 if "binary_3x3" in full_name else 1.0
                            self.layer_m[full_name] = self.m * depth_factor * kernel_boost
                        else:
                            self.layer_m[full_name] = self.m
                    else:
                        self.non_binary_backbone_names.add(full_name)

        for name, _ in model.named_parameters():
            if name not in self.binary_weight_names and self.classifier_name not in name:
                self.non_binary_backbone_names.add(name)

    @torch.no_grad()
    def before_training_exp(self, strategy, **kwargs):
        self._identify_weights(strategy.model)
        self.curr_classes = set(strategy.experience.classes_in_this_experience)
        self.seen_classes.update(self.curr_classes)

        if self.rms_norm:
            for name, param in strategy.model.named_parameters():
                if name in self.binary_weight_names and name not in self.layer_rms_scale:
                    rms = torch.sqrt(torch.mean(param.data ** 2)).clamp(min=1e-6).item()
                    self.layer_rms_scale[name] = rms

        for group in strategy.optimizer.param_groups:
            for p in group["params"]:
                if p in strategy.optimizer.state and p.ndim <= 2:
                    strategy.optimizer.state[p].clear()

    def after_forward(self, strategy, **kwargs):
        if self.mask_unseen_logits and strategy.is_training:
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
                num_classes = param.shape[0]
                not_curr = [c for c in range(num_classes) if c not in self.curr_classes]
                if not_curr:
                    if param.dim() > 1:
                        param.grad.data[not_curr, :] = 0.0
                    elif param.dim() == 1:
                        param.grad.data[not_curr] = 0.0

    @torch.no_grad()
    def after_update(self, strategy, **kwargs):
        curr_task = strategy.experience.current_experience

        for name, param in strategy.model.named_parameters():
            if name not in self.old_params:
                continue

            w_old = self.old_params[name]

            # 1. Protezione Classificatore fc
            if self.classifier_name in name:
                num_classes = param.shape[0]
                not_curr = [c for c in range(num_classes) if c not in self.curr_classes]
                if not_curr:
                    if param.dim() > 1:
                        param.data[not_curr, :] = w_old[not_curr, :]
                    elif param.dim() == 1:
                        param.data[not_curr] = w_old[not_curr]

                if self.normalize_classifier:
                    if param.dim() > 1:
                        seen_list = list(self.seen_classes)
                        param.data[seen_list] = F.normalize(param.data[seen_list], p=2, dim=1) * 5.0
                    elif param.dim() == 1:
                        param.data.zero_()
                continue

            # 2. Algoritmo 1 di Laborieux et al. (2021) sullo step effettivo Uw
            if name in self.binary_weight_names:
                delta_w = param.data - w_old
                w_b = torch.sign(w_old)
                is_opposite = (delta_w * w_b) < 0

                rms_scale = self.layer_rms_scale.get(name, 1.0) if self.rms_norm else 1.0
                m_layer = self.layer_m.get(name, self.m)

                f_meta = 1.0 - torch.tanh(m_layer * (w_old / rms_scale)) ** 2
                effective_delta = torch.where(is_opposite, delta_w * f_meta, delta_w)
                param.data.copy_(w_old + effective_delta)
                continue

            # 3. Controllo opzionale del drift sui parametri non-binari di Reactnet_A_Qaw
            if name in self.non_binary_backbone_names and curr_task > 0 and self.non_binary_lr_scale < 1.0:
                delta_w = param.data - w_old
                param.data.copy_(w_old + self.non_binary_lr_scale * delta_w)

        self.old_params.clear()