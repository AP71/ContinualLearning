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

``StreamingLDA`` e le funzioni di questo modulo sono riusate dal binary latent replay
con testa SLDA (avalanche_plugin/binary_latent_replay.py).
"""
from __future__ import annotations

import copy
import logging

import torch
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
    ):
        super().__init__()
        if adapt_tasks < 0:
            raise ValueError("adapt_tasks deve essere >= 0")
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
        self.lda.update(features, labels)
        n_total = self.lda.write(classifier, self.shrinkage, self.unseen_bias)
        print(f"[AdaptThenSLDA] testa SLDA: {len(self.lda.means)} classi, {n_total} campioni, "
              f"gamma={self.shrinkage:g}")

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
        hook = classifier.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0]))
        was_training = model.training
        model.eval()
        feats, labels = [], []
        try:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            for batch in loader:
                model(batch[0].to(strategy.device))
                feats.append(captured.pop().double().cpu())
                labels.append(torch.as_tensor(batch[1]).long().cpu())
        finally:
            hook.remove()
            model.train(was_training)
        return torch.cat(feats), torch.cat(labels)
