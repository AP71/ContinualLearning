"""Adatta il backbone una sola volta, poi lo congela e classifica con SLDA.

Schema (Hayes & Kanan, Deep SLDA 2020; Vorabbi et al. per le BNN):
  - task 1 (``adapt_tasks=1``): addestramento normale della rete sui dati del primo
    task, così il backbone ImageNet si adatta alle immagini 32x32 di CIFAR;
  - fine del task 1: tutti i parametri vengono congelati (nessun bit cambia più);
  - ogni task: le feature a 1024 dimensioni (input di ``fc``) dei dati del task,
    senza augmentation, aggiornano le medie per classe e la scatter within-class
    condivisa; il classificatore ``fc`` viene riscritto come LDA:
        w_c = Λ μ_c,   b_c = -1/2 μ_cᵀ Λ μ_c,   Λ = (Σ + γ·mean(diag Σ)·I)⁻¹
    Le classi non ancora viste ricevono bias molto negativo.
Dal task 2 in poi non c'è training a gradiente (``train_epochs = 0``): le statistiche
si accumulano in streaming, senza conservare dati dei task passati. Con
``adapt_tasks=0`` il backbone ImageNet resta congelato da subito (controllo: SLDA
sul backbone originale).

Adattamento continuo (``adapt_tasks > 1``, es. 10 = tutti i task): il backbone continua
ad allenarsi, quindi le statistiche delle classi vecchie diventano obsolete. Con
``drift_compensation="linear"`` vengono trasportate nel nuovo spazio delle feature:
prima e dopo il training del task si estraggono le feature degli stessi dati (quelli
del task corrente), si stima la mappa affine  f_new ≈ f_old·W + b  (ridge verso
l'identità) e si aggiornano  μ_c ← μ_c·W + b,  S ← Wᵀ·S·W.  Nessun dato dei task
passati viene conservato. ``drift_compensation="none"`` è l'ablazione (statistiche
vecchie lasciate come sono). Pensato per essere usato con un ottimizzatore che limita
il drift (metaplasticità, cl/methods/meta_slda.py).

Durante i task di training ``fc`` è la testa allenata con la loss; a fine task viene
salvata e sostituita dalla testa LDA (usata per valutazione e checkpoint), poi
ripristinata all'inizio del task successivo. Il plugin va messo per primo nella lista.

Testa alternativa ``head="ranpac"`` (RanPAC, McDonnell et al. NeurIPS 2023; con
``rp_dim=0`` è una semplice ridge regression): le feature f (1024) del backbone congelato
vengono proiettate in modo casuale e fisso, h = ReLU(f·W_rp) con W_rp ~ N(0,1) di
dimensione ``rp_dim`` (default 10000), e il classificatore è una regressione ridge sui
target one-hot, risolta in forma chiusa dalle sole statistiche cumulate
G = Σ hhᵀ e C = Σ h yᵀ:   W_o = (G + λI)⁻¹ C.   Nessun dato passato viene conservato e il
risultato non dipende dall'ordine dei task. λ = r·mean(diag G): r si sceglie sul primo
task con un hold-out (``ridge=None``) oppure si fissa a mano; con ``ridge_relative=False``
λ resta quello assoluto calcolato al primo task (come nel paper). ``fc`` viene sostituita
da ``RanPACHead`` (conserva weight/bias originali non usati; bias = -1e4 sulle classi non
viste). Richiede ``adapt_tasks <= 1`` (nessun trasporto delle statistiche).

Feature multi-livello (``feature_blocks=(8, 10, 12)``, solo RanPAC): prima della proiezione
casuale si concatenano le uscite (global average pooling) dei blocchi indicati e l'input di
``fc``; ogni livello è riscalato a RMS unitario per coordinata (scala fissata al task 1,
salvata nella testa). Il backbone e il numero di bit flip non cambiano.

``StreamingLDA`` e le funzioni di questo modulo sono riusate dal binary latent replay
con testa SLDA (avalanche_plugin/binary_latent_replay.py).
"""
from __future__ import annotations

import copy
import functools
import logging

import torch
import torch.nn.functional as F
from avalanche.core import SupervisedPlugin
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

LOGGER = logging.getLogger("split_cifar100")

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def _find_subset(dataset) -> Subset | None:
    """Trova il Subset PyTorch (SubsetWithTargets) dentro i wrapper di Avalanche."""
    if isinstance(dataset, Subset):
        return dataset
    for attr in ("_datasets", "datasets"):
        for child in getattr(dataset, attr, None) or []:
            found = _find_subset(child)
            if found is not None:
                return found
    return None


def _shrink(cov: torch.Tensor, gamma: float) -> torch.Tensor:
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype)
    return cov + gamma * cov.diagonal().mean() * eye


def clean_experience_dataset(experience, eval_transform=EVAL_TRANSFORM, tag: str = "AdaptThenSLDA"):
    """Dati del task corrente senza augmentation (stesse immagini del training)."""
    subset = _find_subset(experience.dataset)
    if subset is None or eval_transform is None or not hasattr(subset.dataset, "transform"):
        print(f"[{tag}] ATTENZIONE: dataset senza augmentation non trovato, uso quello di training")
        return experience.dataset
    indices = experience.origin_stream.benchmark.train_exps_patterns_assignment[
        experience.current_experience
    ]
    base = copy.copy(subset.dataset)  # copia superficiale: cambia solo la transform
    base.transform = eval_transform
    clean = Subset(base, [subset.indices[i] for i in indices])
    expected = set(int(c) for c in experience.classes_in_this_experience)
    found = {int(base.targets[subset.indices[i]]) for i in indices}
    if found != expected:
        raise RuntimeError(f"classi del dataset pulito {sorted(found)} != {sorted(expected)}")
    return clean


def save_head(classifier: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in classifier.named_parameters()}


@torch.no_grad()
def restore_head(classifier: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Ripristino in place: i parametri restano gli stessi oggetti visti dall'ottimizzatore."""
    for name, param in classifier.named_parameters():
        param.copy_(state[name])


def fit_linear_drift(old: torch.Tensor, new: torch.Tensor, ridge: float):
    """Mappa affine  new ≈ old·W + b  con ridge verso l'identità (ridge relativo alla
    varianza media delle feature). Restituisce anche due diagnostiche:
    drift = ‖new − old‖ / ‖old‖,  residuo = ‖old·W + b − new‖ / ‖new − old‖."""
    mean_old, mean_new = old.mean(0), new.mean(0)
    x, y = old - mean_old, new - mean_new
    gram = x.T @ x
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    lam = ridge * gram.diagonal().mean()
    weight = torch.linalg.solve(gram + lam * eye, x.T @ y + lam * eye)
    bias = mean_new - mean_old @ weight
    moved = (new - old).norm()
    drift = float(moved / old.norm().clamp_min(1e-12))
    residual = float((old @ weight + bias - new).norm() / moved.clamp_min(1e-12))
    return weight, bias, drift, residual


class StreamingLDA:
    """Medie per classe + scatter within-class condivisa, e scrittura della testa LDA."""

    def __init__(self):
        self.means: dict[int, torch.Tensor] = {}  # classe -> media (float64, cpu)
        self.counts: dict[int, int] = {}
        self.scatter: torch.Tensor | None = None  # somma delle scatter within-class

    def reset(self) -> None:
        self.means.clear()
        self.counts.clear()
        self.scatter = None

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.double().cpu()
        y = y.cpu()
        if self.scatter is None:
            self.scatter = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
        for c in y.unique().tolist():
            xc = x[y == c]
            mu = xc.mean(0)
            centered = xc - mu
            self.scatter += centered.T @ centered
            self.means[c] = mu
            self.counts[c] = xc.shape[0]

    def transport(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """Porta le statistiche nel nuovo spazio delle feature:  f_new = f_old·W + b."""
        for c in self.means:
            self.means[c] = self.means[c] @ weight + bias
        if self.scatter is not None:
            self.scatter = weight.T @ self.scatter @ weight

    @torch.no_grad()
    def write(self, classifier: nn.Module, shrinkage: float, unseen_bias: float = -1e4) -> int:
        """Riscrive il classificatore come LDA; restituisce il numero di campioni usati."""
        classes = sorted(self.means)
        n_total = sum(self.counts.values())
        cov = self.scatter / max(n_total - len(classes), 1)
        mu = torch.stack([self.means[c] for c in classes])  # [C, D]
        weights = torch.linalg.solve(_shrink(cov, shrinkage), mu.T).T  # Λ μ_c (Λ simmetrica)
        bias = -0.5 * (weights * mu).sum(1)

        idx = torch.tensor(classes, device=classifier.weight.device)
        classifier.weight.data.zero_()
        classifier.bias.data.fill_(unseen_bias)
        classifier.weight.data[idx] = weights.to(classifier.weight)
        classifier.bias.data[idx] = bias.to(classifier.bias)
        return n_total


def pool_block(output: torch.Tensor) -> torch.Tensor:
    """Uscita di un blocco (B, C, H, W) -> (B, C) con global average pooling."""
    return output.float().mean(dim=(2, 3))


class RanPACHead(nn.Module):
    """Testa RanPAC: logits = ReLU(z·W_rp)·W_o + bias  (proj=None: logits = z·W_o + bias).

    z = x (input di fc, 1024) oppure, con ``blocks``, la concatenazione
    [pool(feature[b]) per b in blocks, x] riscalata da ``feat_scale``: le uscite dei blocchi
    vengono lette con forward hook sul modello (registrati qui, quindi la testa funziona da
    sola in valutazione).
    """

    def __init__(self, base: nn.Linear, proj: torch.Tensor | None, readout: torch.Tensor,
                 model: nn.Module | None = None, blocks: tuple[int, ...] = (),
                 feat_scale: torch.Tensor | None = None):
        super().__init__()
        # i parametri originali restano (inutilizzati, congelati): l'ottimizzatore di
        # Avalanche non si ritrova con una lista vuota
        self.weight = base.weight
        self.bias = base.bias
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.blocks = tuple(blocks)
        self.register_buffer("proj", proj)
        self.register_buffer("readout", readout)
        self.register_buffer("feat_scale", feat_scale)
        self._cap: dict[int, torch.Tensor] = {}
        if self.blocks:
            for b in self.blocks:  # hook = metodo legato: sopravvive a deepcopy del modello
                model.feature[b].register_forward_hook(functools.partial(self._capture, b))

    def _capture(self, block: int, _module, _inputs, output) -> None:
        self._cap[block] = pool_block(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.blocks:
            x = torch.cat([self._cap[b] for b in self.blocks] + [x.float()], dim=1) * self.feat_scale
        h = x if self.proj is None else torch.relu(x @ self.proj)
        return h @ self.readout + self.bias


class StreamingRidge:
    """Regressione ridge in streaming su feature (random-)proiettate: statistiche G e C."""

    GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)  # r in lambda = r * mean(diag G)

    def __init__(self, in_dim: int, num_classes: int, rp_dim: int = 10000, seed: int = 0,
                 device="cpu", batch_size: int = 256, ridge_relative: bool = True):
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.ridge_relative = ridge_relative
        if rp_dim > 0:
            gen = torch.Generator().manual_seed(seed)
            self.proj = torch.randn(in_dim, rp_dim, generator=gen).to(self.device)
        else:
            self.proj = None
        dim = rp_dim if rp_dim > 0 else in_dim
        self.gram = torch.zeros(dim, dim, dtype=torch.float64)
        self.cross = torch.zeros(dim, num_classes, dtype=torch.float64)
        self.seen: set[int] = set()
        self.ridge: float | None = None  # r scelto/fissato
        self.lam_abs: float | None = None

    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().to(self.device)
        return x if self.proj is None else torch.relu(x @ self.proj)

    @torch.no_grad()
    def _accumulate(self, features: torch.Tensor, labels: torch.Tensor):
        dim = self.gram.shape[0]
        g = torch.zeros(dim, dim, device=self.device)
        c = torch.zeros(dim, self.num_classes, device=self.device)
        for i in range(0, features.shape[0], self.batch_size):
            h = self._expand(features[i:i + self.batch_size])
            y = F.one_hot(labels[i:i + self.batch_size].to(self.device), self.num_classes).float()
            g += h.T @ h
            c += h.T @ y
        return g.double().cpu(), c.double().cpu()

    @staticmethod
    def _solve(gram: torch.Tensor, cross: torch.Tensor, lam: float) -> torch.Tensor:
        a = gram + lam * torch.eye(gram.shape[0], dtype=gram.dtype)
        try:
            return torch.cholesky_solve(cross, torch.linalg.cholesky(a))
        except RuntimeError:  # non definita positiva per errori numerici: fallback generico
            return torch.linalg.solve(a, cross)

    @torch.no_grad()
    def select_ridge(self, features: torch.Tensor, labels: torch.Tensor,
                     val_fraction: float = 0.2, seed: int = 0) -> float:
        """Sceglie r sul primo task: fit su (1-val) dei dati, MSE sui target one-hot del resto."""
        perm = torch.randperm(features.shape[0], generator=torch.Generator().manual_seed(seed))
        n_val = max(int(val_fraction * len(perm)), 1)
        val, tr = perm[:n_val], perm[n_val:]
        gram, cross = self._accumulate(features[tr], labels[tr])
        h_val = self._expand(features[val]).double().cpu()
        y_val = F.one_hot(labels[val], self.num_classes).double()
        base = float(gram.diagonal().mean())
        best, best_mse = None, float("inf")
        for r in self.GRID:
            mse = float(((h_val @ self._solve(gram, cross, r * base) - y_val) ** 2).mean())
            print(f"[AdaptThenSLDA/RanPAC]   r={r:g}  MSE hold-out = {mse:.6f}")
            if mse < best_mse:
                best, best_mse = r, mse
        print(f"[AdaptThenSLDA/RanPAC] r scelto sul primo task: {best:g}")
        return best

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        gram, cross = self._accumulate(features, labels)
        self.gram += gram
        self.cross += cross
        self.seen |= set(int(c) for c in labels.unique().tolist())

    @torch.no_grad()
    def write(self, head: RanPACHead, unseen_bias: float = -1e4) -> int:
        base = float(self.gram.diagonal().mean())
        if self.lam_abs is None:
            self.lam_abs = self.ridge * base
        lam = self.ridge * base if self.ridge_relative else self.lam_abs
        readout = self._solve(self.gram, self.cross, lam)
        head.readout.copy_(readout.to(head.readout))
        head.bias.fill_(unseen_bias)
        head.bias[torch.tensor(sorted(self.seen), device=head.bias.device)] = 0.0
        return len(self.seen)


class AdaptThenSLDAPlugin(SupervisedPlugin):
    def __init__(
        self,
        adapt_tasks: int = 1,
        shrinkage: float = 1e-2,
        classifier_name: str = "fc",
        eval_transform=EVAL_TRANSFORM,
        batch_size: int = 256,
        num_workers: int = 0,
        unseen_bias: float = -1e4,
        drift_compensation: str = "none",
        drift_ridge: float = 1e-2,
        head: str = "slda",
        rp_dim: int = 10000,
        ridge: float | None = None,
        ridge_relative: bool = True,
        rp_seed: int = 0,
        feature_blocks: tuple[int, ...] = (),
    ):
        super().__init__()
        if head not in ("slda", "ranpac"):
            raise ValueError("head deve essere 'slda' oppure 'ranpac'")
        if head == "ranpac" and adapt_tasks > 1:
            raise ValueError("head='ranpac' richiede adapt_tasks <= 1 (statistiche non trasportabili)")
        if adapt_tasks < 0:
            raise ValueError("adapt_tasks deve essere >= 0")
        if feature_blocks and head != "ranpac":
            raise ValueError("feature_blocks è supportato solo con head='ranpac'")
        if drift_compensation not in ("none", "linear"):
            raise ValueError("drift_compensation deve essere 'none' oppure 'linear'")
        if adapt_tasks > 1 and drift_compensation == "none":
            print("[AdaptThenSLDA] ATTENZIONE: adapt_tasks > 1 senza compensazione del drift, "
                  "le statistiche delle classi vecchie restano quelle del vecchio backbone")
        self.adapt_tasks = adapt_tasks
        self.shrinkage = shrinkage
        self.classifier_name = classifier_name
        self.eval_transform = eval_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unseen_bias = unseen_bias
        self.drift_compensation = drift_compensation
        self.drift_ridge = drift_ridge
        self.head = head
        self.rp_dim = rp_dim
        self.ridge = ridge
        self.ridge_relative = ridge_relative
        self.rp_seed = rp_seed
        self.feature_blocks = tuple(feature_blocks)
        self.feat_scale: torch.Tensor | None = None  # scala per coordinata, fissata al task 1
        self.readout: StreamingRidge | None = None
        self._part_dims: list[int] = []

        self.frozen = False
        self.lda = StreamingLDA()
        self.drift_log: list[dict[str, float]] = []  # una voce per task compensato
        self._train_head: dict[str, torch.Tensor] | None = None
        self._old_features: torch.Tensor | None = None

    # ------------------------------------------------------------------ hooks
    def before_training_exp(self, strategy, **kwargs):
        self._old_features = None
        if strategy.experience.current_experience >= self.adapt_tasks:
            self._freeze(strategy.model)
            strategy.train_epochs = 0  # nessun training a gradiente: solo statistiche
            return
        if self._train_head is not None:  # fc torna la testa allenata, non quella LDA
            restore_head(getattr(strategy.model, self.classifier_name), self._train_head)
        if self.drift_compensation != "none" and self.lda.means:
            # feature dei dati del task corrente con il backbone PRIMA del training
            self._old_features, _ = self._extract(strategy, self._clean_dataset(strategy.experience))

    def after_training_exp(self, strategy, **kwargs):
        task = strategy.experience.current_experience
        classifier = getattr(strategy.model, self.classifier_name)
        if task < self.adapt_tasks:
            self._train_head = save_head(classifier)
        if task + 1 >= self.adapt_tasks:
            self._freeze(strategy.model)  # dopo l'ultimo task di adattamento il backbone è definitivo
        features, labels = self._extract(strategy, self._clean_dataset(strategy.experience))
        if self._old_features is not None:
            weight, bias, drift, residual = fit_linear_drift(self._old_features, features, self.drift_ridge)
            self.lda.transport(weight, bias)
            self.drift_log.append({"task": task + 1, "drift": drift, "residual": residual})
            print(f"[AdaptThenSLDA] drift delle feature {drift * 100:.1f}%, residuo dopo la mappa "
                  f"lineare {residual * 100:.1f}% del drift: statistiche di {len(self.lda.means)} classi trasportate")
            self._old_features = None
        if self.head == "ranpac":
            self._update_ranpac(strategy, classifier, features, labels)
            return
        self.lda.update(features, labels)
        n_total = self.lda.write(classifier, self.shrinkage, self.unseen_bias)
        print(f"[AdaptThenSLDA] testa SLDA: {len(self.lda.means)} classi, {n_total} campioni, "
              f"gamma={self.shrinkage:g}")

    def _update_ranpac(self, strategy, classifier, features, labels) -> None:
        if self.feature_blocks:
            if self.feat_scale is None:  # task 1: ogni livello a RMS unitario per coordinata
                self.feat_scale = self._block_scale(features, strategy.device)
            features = features * self.feat_scale.cpu().double()
        if self.readout is None:  # primo task: crea proiezione, sceglie r, sostituisce fc
            self.readout = StreamingRidge(
                features.shape[1], classifier.out_features, self.rp_dim, self.rp_seed,
                strategy.device, self.batch_size, self.ridge_relative)
            self.readout.ridge = self.ridge if self.ridge is not None else self.readout.select_ridge(
                features, labels, seed=self.rp_seed)
            dim = self.readout.gram.shape[0]
            head = RanPACHead(classifier, self.readout.proj,
                              torch.zeros(dim, classifier.out_features),
                              model=strategy.model, blocks=self.feature_blocks,
                              feat_scale=self.feat_scale).to(strategy.device)
            setattr(strategy.model, self.classifier_name, head)
            classifier = head
        self.readout.update(features, labels)
        n_classes = self.readout.write(classifier, self.unseen_bias)
        print(f"[AdaptThenSLDA] testa RanPAC: {n_classes} classi, rp_dim={self.rp_dim}, "
              f"r={self.readout.ridge:g}, dim feature={features.shape[1]}")

    def _block_scale(self, features: torch.Tensor, device) -> torch.Tensor:
        """Una scala per livello (1/RMS delle sue coordinate sul task 1), espansa per coordinata."""
        dims = self._part_dims
        scale = []
        start = 0
        for d in dims:
            rms = features[:, start:start + d].pow(2).mean().sqrt().clamp(min=1e-8)
            scale.append(torch.full((d,), 1.0 / float(rms)))
            start += d
        out = torch.cat(scale).float().to(device)
        print(f"[AdaptThenSLDA] feature multi-livello: blocchi {list(self.feature_blocks)} + fc, "
              f"dimensioni {dims}, scala RMS {[round(float(1 / sc[0]), 4) for sc in scale]}")
        return out

    # ---------------------------------------------------------------- helpers
    def _freeze(self, model: nn.Module) -> None:
        if self.frozen:
            return
        for param in model.parameters():
            param.requires_grad = False
        self.frozen = True
        print("[AdaptThenSLDA] backbone congelato: da qui in poi nessun bit cambia")

    def _clean_dataset(self, experience):
        return clean_experience_dataset(experience, self.eval_transform)

    @torch.no_grad()
    def _extract(self, strategy, dataset) -> tuple[torch.Tensor, torch.Tensor]:
        model = strategy.model
        classifier = getattr(model, self.classifier_name)
        captured: list[torch.Tensor] = []
        pooled: dict[int, torch.Tensor] = {}
        hooks = [classifier.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0]))]
        for b in self.feature_blocks:
            hooks.append(model.feature[b].register_forward_hook(
                lambda _m, _i, out, b=b: pooled.__setitem__(b, pool_block(out))))
        was_training = model.training
        model.eval()
        feats, labels = [], []
        try:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            for batch in loader:
                model(batch[0].to(strategy.device))
                parts = [pooled[b] for b in self.feature_blocks] + [captured.pop().float()]
                self._part_dims = [p.shape[1] for p in parts]
                feats.append(torch.cat(parts, dim=1).double().cpu())
                labels.append(torch.as_tensor(batch[1]).long().cpu())
        finally:
            for hook in hooks:
                hook.remove()
            model.train(was_training)
        return torch.cat(feats), torch.cat(labels)
