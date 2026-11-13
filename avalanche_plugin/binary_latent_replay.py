"""Binary latent replay for ReActNet (Vorabbi et al., "Enabling On-device Continual
Learning with Binary Neural Networks").

The network is split at block ``cut_block``:
- lower part ``feature[0 .. cut_block-1]``: frozen (must be frozen before building the
  optimizer, see ``freeze_lower_layers``);
- upper part ``feature[cut_block ..] + pool + fc``: trainable on every task.

The replay buffer stores the activations at the cut point instead of images. With
``latent_bits=1`` they are binarized (1 bit per value). ReActNet blocks also use the
real-valued input on the residual path, so to keep training, replay and inference
consistent a *binary bottleneck* is inserted at the cut for every input:

    h_hat = sign(h) * scale_c        (scale_c: per-channel mean |h|, computed on task 1, then fixed)

``latent_bits=32`` stores the real activations (no bottleneck), for an iso-memory ablation.

The buffer is class balanced: after each task every seen class keeps
``capacity // seen_classes`` latents (random subset). The number of latents follows from a
memory budget, by default the one of the image Replay (2000 x 3 x 32 x 32 bytes).

During training on task t >= 2, each step forwards a batch of buffer latents through the
upper part only and combines its loss with the current-task loss, weighted by batch size
(the same 50/50 mixing of Avalanche's Replay with ``batch_size_mem = train_mb_size``).

Optional head corrections against the bias toward the newest task, applied at the end of
every task from task 2 (both off by default, they can be combined):
- ``balanced_ft_epochs > 0``: class-balanced fine-tuning of ``fc`` only, on the updated
  buffer (same number of latents for every seen class, as in EEIL);
- ``weight_aligning=True``: Weight Aligning (Zhao et al., 2020), the rows of ``fc`` of the
  new classes are rescaled so that their mean norm equals that of the old classes.
If both are on, fine-tuning runs first and Weight Aligning is applied afterwards.

Optional statistical head (``slda_head=True``): at the end of every task ``fc`` is
rewritten as an LDA classifier (class means + shared covariance, see
avalanche_plugin/adapt_then_slda.py) computed on the features of the *current* upper
blocks: new classes use the latents of all the current-task samples, old classes the
latents in the buffer, so the statistics are recomputed exactly after every update of
the upper blocks (no drift approximation) and the head has no bias toward the newest
task by construction. The ``fc`` trained with the loss is saved before being
overwritten and restored at the start of the next task (it is only needed to train the
upper blocks); evaluation and checkpoints use the LDA head.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from avalanche.core import SupervisedPlugin
from torch import nn
from torch.utils.data import DataLoader

from avalanche_plugin.adapt_then_slda import StreamingLDA, restore_head, save_head

IMAGE_REPLAY_BYTES = 2000 * 3 * 32 * 32  # memory of the image Replay with mem_size=2000


def freeze_lower_layers(model: nn.Module, cut_block: int) -> None:
    """Freeze ``feature[0 .. cut_block-1]``; everything above stays trainable."""
    for name, param in model.named_parameters():
        if name.startswith("feature."):
            block = int(name.split(".")[1])
            param.requires_grad = block >= cut_block


class BinaryLatentReplayPlugin(SupervisedPlugin):
    def __init__(
        self,
        cut_block: int = 12,
        latent_bits: int = 1,
        memory_bytes: int = IMAGE_REPLAY_BYTES,
        batch_size_mem: int | None = None,
        classifier_name: str = "fc",
        weight_aligning: bool = False,
        balanced_ft_epochs: int = 0,
        balanced_ft_lr: float = 1e-4,
        slda_head: bool = False,
        slda_shrinkage: float = 1e-2,
    ):
        super().__init__()
        if latent_bits not in (1, 32):
            raise ValueError("latent_bits must be 1 or 32")
        self.cut_block = cut_block
        self.latent_bits = latent_bits
        self.memory_bytes = memory_bytes
        self.batch_size_mem = batch_size_mem
        self.classifier_name = classifier_name
        self.weight_aligning = weight_aligning
        self.balanced_ft_epochs = balanced_ft_epochs
        self.balanced_ft_lr = balanced_ft_lr
        self.slda_head = slda_head
        self.slda_shrinkage = slda_shrinkage
        self._train_head: dict[str, torch.Tensor] | None = None  # fc trained with the loss

        self.scale: torch.Tensor | None = None  # per-channel scale of the bottleneck
        self.capacity: int | None = None  # number of latents that fit the budget
        self.buffer_x: torch.Tensor | None = None  # bool (1 bit) or float (32 bit), on CPU
        self.buffer_y: torch.Tensor | None = None
        self._hook = None
        self._seen_classes: set[int] = set()

    # ------------------------------------------------------------------ helpers
    def _lower(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        for block in model.feature[: self.cut_block]:
            x = block(x)
        return x

    def _upper(self, model: nn.Module, h: torch.Tensor) -> torch.Tensor:
        for block in model.feature[self.cut_block:]:
            h = block(h)
        h = model.pool1(h).flatten(1)
        return getattr(model, self.classifier_name)(h)

    def _decode(self, stored: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.latent_bits == 1:
            signs = stored.to(device).float() * 2 - 1  # bool -> {-1, +1}
            return signs * self.scale.to(device)
        return stored.to(device)

    def _encode(self, h: torch.Tensor) -> torch.Tensor:
        """Value stored in the buffer for the (bottleneck-free) activation ``h``."""
        if self.latent_bits == 1:
            return (h >= 0).cpu()  # 1 bit per value
        return h.detach().float().cpu()

    def _bottleneck_hook(self, _module, _inputs, output):
        return torch.where(output >= 0, 1.0, -1.0) * self.scale.to(output.device)

    @staticmethod
    def _loader(dataset, batch_size: int):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------ setup (task 1)
    @torch.no_grad()
    def _setup(self, strategy) -> None:
        model = strategy.model
        device = strategy.device
        was_training = model.training
        model.eval()
        abs_sum, count, shape = None, 0, None
        for batch in self._loader(strategy.experience.dataset, strategy.train_mb_size):
            h = self._lower(model, batch[0].to(device))
            shape = h.shape[1:]
            batch_sum = h.abs().sum(dim=(0, 2, 3))
            abs_sum = batch_sum if abs_sum is None else abs_sum + batch_sum
            count += h.shape[0] * h.shape[2] * h.shape[3]
        if was_training:
            model.train()

        values_per_latent = math.prod(shape)
        bytes_per_latent = values_per_latent * self.latent_bits / 8
        self.capacity = int(self.memory_bytes // bytes_per_latent)
        if self.latent_bits == 1:
            self.scale = (abs_sum / count).view(1, -1, 1, 1).cpu()
            self._hook = model.feature[self.cut_block - 1].register_forward_hook(self._bottleneck_hook)
        print(
            f"[BinaryLatentReplay] cut after feature.{self.cut_block - 1}, latent shape "
            f"{tuple(shape)}, {self.latent_bits} bit -> {bytes_per_latent:.0f} B/latent, "
            f"capacity {self.capacity} latents ({self.memory_bytes / 1e6:.2f} MB)"
        )

    def before_training_exp(self, strategy, *args, **kwargs):
        if self.capacity is None:
            self._setup(strategy)
        if self._train_head is not None:  # fc back to the trained head, not the LDA one
            restore_head(getattr(strategy.model, self.classifier_name), self._train_head)

    # ------------------------------------------------------------------ replay loss
    def before_backward(self, strategy, *args, **kwargs):
        if self.buffer_x is None or len(self.buffer_y) == 0:
            return
        batch_mem = self.batch_size_mem or strategy.train_mb_size
        idx = torch.randint(len(self.buffer_y), (batch_mem,))
        h = self._decode(self.buffer_x[idx], strategy.device)
        y = self.buffer_y[idx].to(strategy.device)
        loss_mem = F.cross_entropy(self._upper(strategy.model, h), y)
        batch_cur = strategy.mb_y.shape[0]
        strategy.loss = (strategy.loss * batch_cur + loss_mem * batch_mem) / (batch_cur + batch_mem)

    # ------------------------------------------------------------------ buffer update
    @torch.no_grad()
    def after_training_exp(self, strategy, *args, **kwargs):
        model = strategy.model
        device = strategy.device
        was_training = model.training
        model.eval()
        latents, labels = [], []
        # The lower part is frozen. With 1 bit the hook already returns sign(h) * scale,
        # whose sign is the stored bit; with 32 bit there is no hook and h is stored as is.
        for batch in self._loader(strategy.experience.dataset, strategy.train_mb_size):
            h = self._lower(model, batch[0].to(device))
            latents.append(self._encode(h))
            labels.append(batch[1].cpu())
        if was_training:
            model.train()
        new_x, new_y = torch.cat(latents), torch.cat(labels)
        new_classes = sorted(int(c) for c in new_y.unique())
        old_classes = sorted(self._seen_classes)
        self._seen_classes.update(new_classes)

        all_x = new_x if self.buffer_x is None else torch.cat([self.buffer_x, new_x])
        all_y = new_y if self.buffer_y is None else torch.cat([self.buffer_y, new_y])
        quota = self.capacity // len(self._seen_classes)
        keep = []
        for c in sorted(self._seen_classes):
            c_idx = (all_y == c).nonzero(as_tuple=True)[0]
            keep.append(c_idx[torch.randperm(len(c_idx))[:quota]])
        keep = torch.cat(keep)
        self.buffer_x, self.buffer_y = all_x[keep], all_y[keep]
        print(
            f"[BinaryLatentReplay] buffer: {len(self.buffer_y)} latents, "
            f"{quota} per class over {len(self._seen_classes)} classes"
        )

        if old_classes:  # head corrections only from task 2 onward
            if self.balanced_ft_epochs > 0:
                self._balanced_finetune(strategy)
            if self.weight_aligning:
                self._weight_align(strategy, old_classes, new_classes)
        if self.slda_head:
            self._write_slda_head(strategy, new_x, new_y, old_classes)

    # ------------------------------------------------------------------ head corrections
    @torch.no_grad()
    def _pooled_features(self, model: nn.Module, stored: torch.Tensor, device: torch.device, batch_size: int):
        """Input of ``fc`` for stored latents, through the current upper blocks."""
        feats = []
        for start in range(0, len(stored), batch_size):
            h = self._decode(stored[start:start + batch_size], device)
            for block in model.feature[self.cut_block:]:
                h = block(h)
            feats.append(model.pool1(h).flatten(1))
        return torch.cat(feats)

    def _pooled_buffer_features(self, model: nn.Module, device: torch.device, batch_size: int):
        """Input of ``fc`` for every buffer latent (upper blocks fixed during fine-tuning)."""
        return self._pooled_features(model, self.buffer_x, device, batch_size)

    @torch.no_grad()
    def _write_slda_head(self, strategy, new_x: torch.Tensor, new_y: torch.Tensor, old_classes: list[int]) -> None:
        model = strategy.model
        device = strategy.device
        head = getattr(model, self.classifier_name)
        self._train_head = save_head(head)
        was_training = model.training
        model.eval()
        lda = StreamingLDA()  # recomputed from scratch: the upper blocks have changed
        lda.update(self._pooled_features(model, new_x, device, strategy.train_mb_size), new_y)
        if old_classes:
            old = torch.isin(self.buffer_y, torch.tensor(old_classes))
            lda.update(
                self._pooled_features(model, self.buffer_x[old], device, strategy.train_mb_size),
                self.buffer_y[old],
            )
        n_total = lda.write(head, self.slda_shrinkage)
        if was_training:
            model.train()
        print(f"[BinaryLatentReplay] SLDA head: {len(lda.means)} classes, {n_total} latents "
              f"({len(new_y)} current + {n_total - len(new_y)} buffer), gamma={self.slda_shrinkage:g}")

    def _balanced_finetune(self, strategy) -> None:
        model = strategy.model
        device = strategy.device
        head = getattr(model, self.classifier_name)
        was_training = model.training
        model.eval()
        feats = self._pooled_buffer_features(model, device, strategy.train_mb_size)
        labels = self.buffer_y.to(device)
        seen = torch.zeros(head.out_features, dtype=torch.bool, device=device)
        seen[sorted(self._seen_classes)] = True

        optimizer = torch.optim.Adam(head.parameters(), lr=self.balanced_ft_lr)
        for _ in range(self.balanced_ft_epochs):
            order = torch.randperm(len(labels), device=device)
            for start in range(0, len(order), strategy.train_mb_size):
                idx = order[start:start + strategy.train_mb_size]
                with torch.enable_grad():
                    logits = head(feats[idx]).masked_fill(~seen, float("-inf"))
                    loss = F.cross_entropy(logits, labels[idx])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
        if was_training:
            model.train()
        print(f"[BinaryLatentReplay] balanced fine-tuning of {self.classifier_name}: "
              f"{self.balanced_ft_epochs} epochs on {len(labels)} latents")

    @torch.no_grad()
    def _weight_align(self, strategy, old_classes: list[int], new_classes: list[int]) -> None:
        head = getattr(strategy.model, self.classifier_name)
        norms = head.weight.norm(dim=1)
        gamma = norms[old_classes].mean() / norms[new_classes].mean()
        head.weight[new_classes] *= gamma
        if head.bias is not None:
            head.bias[new_classes] *= gamma  # rescale the whole logit of the new classes
        print(f"[BinaryLatentReplay] weight aligning: new-class rows scaled by {float(gamma):.4f}")
