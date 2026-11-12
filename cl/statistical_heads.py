"""Training-free statistical heads (NCM, SLDA, FeCAM-style) on the FreezingHead backbone.

The backbone of FreezingHead is frozen, so its features never change: the class
statistics computed when a class arrives stay exact for the whole stream and no
image needs to be stored. Only per-class means (and covariances) are kept.

Heads compared on the same features (pooled 1024-d vector, input of ``fc``):
- ``fc``     : FreezingHead's own trained classifier (sanity check, ~33.4%);
- ``ncm``    : nearest class mean (cosine similarity);
- ``slda``   : streaming LDA, one shared covariance updated task by task;
- ``fecam``  : FeCAM-style Mahalanobis distance with one covariance per class.

Class-incremental protocol: after task t the prediction is restricted to the
classes seen so far (0 .. 10t-1). The covariance shrinkage of SLDA / FeCAM is
selected on the validation split (50 images per class), never on the test set.

Simplifications w.r.t. FeCAM (Goswami et al., 2023): per-class shrinkage
Sigma_c + gamma * mean(diag(Sigma_c)) * I; optional sign-preserving power
transform instead of Tukey's transform (ReActNet features can be negative).

Results go to ``cl/results/statistical_heads/``.

Example:
    python cl/statistical_heads.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avalanche_plugin.utils import create_model, set_seed  # noqa: E402

DATASET_PATH = Path(r"F:/Dataset/CIFAR-100/")
EXPERIMENT_DIR = ROOT / "cl" / "results" / "cifar100_avalanche_FreezingHead_checkpoint_resneta_3seeds"
CHECKPOINT_NAME = "after_task_10.h5"
MODEL_NAME = "resneta"
OUTPUT_ROOT = ROOT / "cl" / "results" / "statistical_heads"
BATCH_SIZE = 256
WORKERS = 0
SEEDS = (42, 2023, 3047)
NUM_TASKS = 10
CLASSES_PER_TASK = 10
NUM_CLASSES = NUM_TASKS * CLASSES_PER_TASK
SHRINKAGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0)
POWER_TRANSFORM: float | None = None  # e.g. 0.5 -> sign(x) * |x| ** 0.5
FLIP_AVERAGE = False  # average features of image and horizontal flip
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger("statistical_heads")
torch.set_float32_matmul_precision("high")

# Class-incremental accuracy after task 10 (mean of 3 seeds), for comparison.
REFERENCES = {"Linear probe (joint)": 57.6, "Replay (2000 img)": 41.7, "FreezingHead": 33.4}

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# --------------------------------------------------------------------------- data
def _splits_without_augmentation(seed: int):
    """Train / validation split identical to ``create_benchmark`` (450 / 50 per class),
    but with the test transform: statistics must come from clean, deterministic features."""
    base = datasets.CIFAR100(DATASET_PATH, train=True, download=False, transform=TEST_TRANSFORM)
    test = datasets.CIFAR100(DATASET_PATH, train=False, download=False, transform=TEST_TRANSFORM)
    targets = np.array(base.targets)
    rng = np.random.default_rng(seed)  # same RNG and order as create_benchmark
    train_idx, val_idx = [], []
    for c in range(NUM_CLASSES):
        c_idx = np.where(targets == c)[0]
        rng.shuffle(c_idx)
        val_idx.extend(c_idx[:50])
        train_idx.extend(c_idx[50:])
    return Subset(base, train_idx), Subset(base, val_idx), test


def _load_backbone(seed: int) -> nn.Module:
    checkpoint = EXPERIMENT_DIR / f"seed{seed}" / "checkpoints" / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FreezingHead checkpoint not found: {checkpoint}")
    model = create_model("random", DEVICE, LOGGER, None, ROOT)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def _extract(model: nn.Module, dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled 1024-d features (input of ``fc``) and labels."""
    captured: list[torch.Tensor] = []
    hook = model.fc.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0]))
    feats, labels = [], []
    try:
        for images, targets in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS):
            images = images.to(DEVICE)
            model(images)
            x = captured.pop()
            if FLIP_AVERAGE:
                model(torch.flip(images, dims=[3]))
                x = 0.5 * (x + captured.pop())
            feats.append(x.double().cpu())
            labels.append(torch.as_tensor(targets))
    finally:
        hook.remove()
    x = torch.cat(feats)
    if POWER_TRANSFORM is not None:
        x = torch.sign(x) * x.abs().pow(POWER_TRANSFORM)
    return x, torch.cat(labels)


# --------------------------------------------------------------------------- heads
def _class_stats(x: torch.Tensor, y: torch.Tensor):
    means = torch.stack([x[y == c].mean(0) for c in range(NUM_CLASSES)])
    centered = x - means[y]
    counts = torch.bincount(y, minlength=NUM_CLASSES).double()
    return means, centered, counts


def _shrink(cov: torch.Tensor, gamma: float) -> torch.Tensor:
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype)
    scale = cov.diagonal(dim1=-2, dim2=-1).mean(-1)
    return cov + gamma * scale[..., None, None] * eye


def _scores_ncm(x: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    return nn.functional.normalize(x, dim=1) @ nn.functional.normalize(means, dim=1).T


def _scores_fc(x: torch.Tensor, fc_weight: torch.Tensor, fc_bias: torch.Tensor) -> torch.Tensor:
    return x @ fc_weight.T + fc_bias


def _scores_slda(x, means, centered, y, seen: int, gamma: float) -> torch.Tensor:
    """LDA with the covariance pooled over the classes seen so far (streaming = batch)."""
    mask = y < seen
    cov = centered[mask].T @ centered[mask] / max(int(mask.sum()) - 1, 1)
    precision = torch.linalg.inv(_shrink(cov, gamma))
    mu = means[:seen]
    weights = mu @ precision
    bias = -0.5 * (weights * mu).sum(1)
    return x @ weights.T + bias


def _fecam_precisions(centered, y, counts, gamma: float) -> torch.Tensor:
    covs = torch.stack([
        centered[y == c].T @ centered[y == c] / max(int(counts[c]) - 1, 1)
        for c in range(NUM_CLASSES)
    ])
    return torch.linalg.inv(_shrink(covs, gamma))


@torch.no_grad()
def _scores_fecam(x, means, precisions) -> torch.Tensor:
    """Negative Mahalanobis distance to each class (per-class covariance).

    Covariances are inverted in float64; distances are computed in float32 on DEVICE.
    """
    x_dev = x.float().to(DEVICE)
    scores = torch.empty(x.shape[0], NUM_CLASSES, dtype=torch.float32, device=DEVICE)
    for c in range(NUM_CLASSES):
        diff = x_dev - means[c].float().to(DEVICE)
        scores[:, c] = -((diff @ precisions[c].float().to(DEVICE)) * diff).sum(1)
    return scores.double().cpu()


# --------------------------------------------------------------------------- protocol
def _accuracy(scores: torch.Tensor, y: torch.Tensor, seen: int) -> float:
    return float((scores[:, :seen].argmax(1) == y).double().mean())


def _r_matrix(score_fn, x_test, y_test) -> np.ndarray:
    """R[t, i]: accuracy on task i after training task t, prediction among seen classes."""
    r = np.full((NUM_TASKS, NUM_TASKS), np.nan)
    for t in range(NUM_TASKS):
        seen = (t + 1) * CLASSES_PER_TASK
        scores = score_fn(seen)
        for i in range(t + 1):
            mask = (y_test >= i * CLASSES_PER_TASK) & (y_test < (i + 1) * CLASSES_PER_TASK)
            r[t, i] = _accuracy(scores[mask], y_test[mask], seen)
    return r


def _summary(r: np.ndarray) -> dict:
    final = r[-1]
    return {
        "final_accuracy": float(np.mean(final)),
        "bwt": float(np.mean([final[i] - r[i, i] for i in range(NUM_TASKS - 1)])),
        "accuracy_vs_classes": [float(np.nanmean(r[t, : t + 1])) for t in range(NUM_TASKS)],
        "per_task_final": final.tolist(),
        "r_matrix": r.tolist(),
    }


def _select_gamma(name: str, score_builder, x_val, y_val) -> tuple[float, dict[str, float]]:
    """Shrinkage chosen on the validation split, all classes seen (end of stream)."""
    val_acc = {}
    for gamma in SHRINKAGE_GRID:
        val_acc[str(gamma)] = _accuracy(score_builder(gamma)(x_val), y_val, NUM_CLASSES)
        print(f"    {name:6s} gamma={gamma:<7g} val {100 * val_acc[str(gamma)]:.2f}%")
    best = max(SHRINKAGE_GRID, key=lambda g: val_acc[str(g)])
    return best, val_acc


def run_statistical_heads(seed: int) -> dict:
    set_seed(seed)
    output_dir = OUTPUT_ROOT / f"cifar100_statistical_heads_{MODEL_NAME}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Statistical heads, seed {seed}: FreezingHead backbone ({CHECKPOINT_NAME}) ===")
    model = _load_backbone(seed)
    train_set, val_set, test_set = _splits_without_augmentation(seed)
    x_tr, y_tr = _extract(model, train_set)
    x_val, y_val = _extract(model, val_set)
    x_te, y_te = _extract(model, test_set)
    fc_w = model.fc.weight.detach().double().cpu()
    fc_b = model.fc.bias.detach().double().cpu()
    means, centered, counts = _class_stats(x_tr, y_tr)

    # Shrinkage selection on validation.
    gamma_slda, val_slda = _select_gamma(
        "slda", lambda g: (lambda x: _scores_slda(x, means, centered, y_tr, NUM_CLASSES, g)), x_val, y_val
    )
    gamma_fecam, val_fecam = _select_gamma(
        "fecam", lambda g: (lambda x, p=_fecam_precisions(centered, y_tr, counts, g): _scores_fecam(x, means, p)),
        x_val, y_val,
    )
    fecam_precisions = _fecam_precisions(centered, y_tr, counts, gamma_fecam)

    # Scores on test are computed once per head (features and class statistics never change);
    # SLDA is recomputed per task because its shared covariance grows with the seen classes.
    fc_scores = _scores_fc(x_te, fc_w, fc_b)
    ncm_scores = _scores_ncm(x_te, means)
    fecam_scores = _scores_fecam(x_te, means, fecam_precisions)
    heads = {
        "fc": _summary(_r_matrix(lambda seen: fc_scores, x_te, y_te)),
        "ncm": _summary(_r_matrix(lambda seen: ncm_scores, x_te, y_te)),
        "slda": _summary(_r_matrix(
            lambda seen: _scores_slda(x_te, means, centered, y_tr, seen, gamma_slda), x_te, y_te
        )),
        "fecam": _summary(_r_matrix(lambda seen: fecam_scores, x_te, y_te)),
    }
    heads["slda"]["shrinkage"] = gamma_slda
    heads["slda"]["validation_accuracy"] = val_slda
    heads["fecam"]["shrinkage"] = gamma_fecam
    heads["fecam"]["validation_accuracy"] = val_fecam

    result = {
        "experiment_type": "statistical_heads_frozen_backbone",
        "backbone_checkpoint": str(EXPERIMENT_DIR / f"seed{seed}" / "checkpoints" / CHECKPOINT_NAME),
        "model_name": MODEL_NAME,
        "seed": seed,
        "protocol": "class-incremental, prediction among classes seen so far",
        "power_transform": POWER_TRANSFORM,
        "flip_average": FLIP_AVERAGE,
        "heads": heads,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, head in heads.items():
        print(f"  -> {name:6s} final {100 * head['final_accuracy']:.2f}%  BWT {100 * head['bwt']:+.2f}")
    return result


def run_three_seeds() -> Path:
    results = {str(seed): run_statistical_heads(seed) for seed in SEEDS}
    head_names = list(next(iter(results.values()))["heads"])

    def stats(values) -> dict[str, float]:
        values = 100 * np.asarray(values, dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}

    aggregate_heads = {}
    for name in head_names:
        per_seed = [r["heads"][name] for r in results.values()]
        r_stack = np.array([h["r_matrix"] for h in per_seed], dtype=float) * 100
        aggregate_heads[name] = {
            "final_accuracy_percent": stats([h["final_accuracy"] for h in per_seed]),
            "bwt_percent": stats([h["bwt"] for h in per_seed]),
            "accuracy_vs_classes_percent": (100 * np.mean([h["accuracy_vs_classes"] for h in per_seed], 0)).tolist(),
            "r_matrix_mean_percent": np.nanmean(r_stack, 0).tolist(),
            "r_matrix_std_percent": np.nanstd(r_stack, 0, ddof=1).tolist(),
        }

    aggregate = {
        "experiment_type": "statistical_heads_frozen_backbone",
        "standard_deviation": "sample standard deviation (ddof=1) across seeds",
        "protocol": "class-incremental, prediction among classes seen so far",
        "heads": aggregate_heads,
        "references_class_incremental_percent": REFERENCES,
        "seeds": results,
    }
    aggregate_dir = OUTPUT_ROOT / f"cifar100_statistical_heads_{MODEL_NAME}_3seeds"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("\n================ Statistical heads (class-incremental, task 10) ================")
    for name, head in aggregate_heads.items():
        acc, bwt = head["final_accuracy_percent"], head["bwt_percent"]
        print(f"{name:6s}  acc {acc['mean']:6.2f} ± {acc['std']:.2f} %   BWT {bwt['mean']:+6.2f} ± {bwt['std']:.2f}")
    for ref, value in REFERENCES.items():
        print(f"  reference {ref:<22s} {value:5.1f} %")
    print(f"Saved to {aggregate_dir}")
    return aggregate_dir


if __name__ == "__main__":
    run_three_seeds()
