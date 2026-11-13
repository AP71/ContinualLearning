"""Linear-probe upper bound for exemplar-free heads on the frozen backbone.

Backbone frozen (as in Freezing / FreezingHead), only ``fc`` trained on all
100 classes together (joint, no task sequence). The resulting class-incremental
accuracy is the ceiling for any linear head built on these frozen features
without storing data (e.g. analytic / RLS heads reach exactly this value).

Same data split, augmentation, optimizer, learning rate and per-task schedule
as the sequential experiments (Adam 1e-3, 40 epochs, MultiStepLR [20, 25]).
The epoch is selected on the validation split (50 images per class), never on
the test set; the last-epoch test accuracy is reported as well.

Results go to ``cl/results/Linear/`` (outside the top-level folders that
the comparison and forgetting scripts scan).

Example:
    python cl/linear_probe_upper_bound.py
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avalanche_plugin.utils import (  # noqa: E402
    create_benchmark,
    create_model,
    evaluate_dataset,
    set_seed,
)

DATASET_PATH = Path(r"F:/Dataset/CIFAR-100/")
CHECKPOINT_PATH: Path | None = None
INITIALIZATION = "checkpoint"
MODEL_NAME = "resneta"
from cl.results_layout import results_dir  # noqa: E402

OUTPUT_ROOT = results_dir("linear_probe")  # cl/results/Linear
BATCH_SIZE = 256
WORKERS = 0
EPOCHS = 40
MILESTONES = [20, 25]
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEEDS = (42, 2023, 3047)
NUM_TASKS = 10
CLASSES_PER_TASK = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger("linear_probe")
torch.set_float32_matmul_precision("high")

# Reference values (class-incremental accuracy after task 10, mean of 3 seeds).
REFERENCES = {"Replay (2000 img)": 41.7, "FreezingHead": 33.4}


def _freeze_backbone(model: nn.Module) -> list[nn.Parameter]:
    """Same rule as the Freezing strategy: everything except ``fc`` is frozen."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc")
    return [param for param in model.parameters() if param.requires_grad]


def _per_task_accuracy(model: nn.Module, test_dataset) -> list[float]:
    """Class-incremental accuracy (all 100 logits) on each block of 10 classes."""
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    correct = np.zeros(NUM_TASKS)
    total = np.zeros(NUM_TASKS)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            preds = model(images).argmax(1)
            tasks = (labels // CLASSES_PER_TASK).cpu().numpy()
            hits = (preds == labels).cpu().numpy()
            np.add.at(correct, tasks, hits)
            np.add.at(total, tasks, 1)
    return (correct / np.maximum(total, 1)).tolist()


def run_linear_probe(seed: int) -> dict:
    set_seed(seed)
    output_dir = OUTPUT_ROOT / (
        f"cifar100_linear_probe_{INITIALIZATION}_{MODEL_NAME}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # is_joint=True: one experience with all 100 classes, same 450/50 split per class.
    scenario, val_dataset = create_benchmark(DATASET_PATH, seed=seed, is_joint=True)
    train_dataset = scenario.train_stream[0].dataset
    test_dataset = scenario.test_stream[0].dataset

    model = create_model(INITIALIZATION, DEVICE, LOGGER, CHECKPOINT_PATH, ROOT)
    trainable = _freeze_backbone(model)
    optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    print(f"\n=== Linear probe, seed {seed}: training only fc on 100 classes ===")
    val_history: list[float] = []
    best_val, best_epoch, best_fc = -1.0, 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for batch in loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            seen += labels.numel()
        scheduler.step()

        val_acc = evaluate_dataset(model, val_dataset, DEVICE, batch_size=BATCH_SIZE, num_workers=WORKERS)
        val_history.append(val_acc)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_fc = copy.deepcopy(model.fc.state_dict())
        print(f"  epoch {epoch:2d}/{EPOCHS}  loss {running_loss / seen:.4f}  val {100 * val_acc:.2f}%")

    last_test = evaluate_dataset(model, test_dataset, DEVICE, batch_size=BATCH_SIZE, num_workers=WORKERS)
    model.fc.load_state_dict(best_fc)
    best_test = evaluate_dataset(model, test_dataset, DEVICE, batch_size=BATCH_SIZE, num_workers=WORKERS)
    per_task = _per_task_accuracy(model, test_dataset)

    result = {
        "experiment_type": "linear_probe_frozen_backbone",
        "initialization": INITIALIZATION,
        "model_name": MODEL_NAME,
        "seed": seed,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "milestones": MILESTONES,
        "best_epoch_on_val": best_epoch,
        "best_val_accuracy": best_val,
        "test_accuracy_at_best_val": best_test,
        "test_accuracy_last_epoch": last_test,
        "test_accuracy_per_task_at_best_val": per_task,
        "val_accuracy_history": val_history,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  -> test {100 * best_test:.2f}% (epoch {best_epoch} by val), last epoch {100 * last_test:.2f}%")
    return result


def run_three_seeds() -> Path:
    results = {str(seed): run_linear_probe(seed) for seed in SEEDS}

    def stats(key: str) -> dict[str, float]:
        values = np.array([100 * r[key] for r in results.values()])
        return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}

    per_task = np.array([r["test_accuracy_per_task_at_best_val"] for r in results.values()]) * 100
    aggregate = {
        "experiment_type": "linear_probe_frozen_backbone",
        "standard_deviation": "sample standard deviation (ddof=1) across seeds",
        "test_accuracy_at_best_val_percent": stats("test_accuracy_at_best_val"),
        "test_accuracy_last_epoch_percent": stats("test_accuracy_last_epoch"),
        "test_accuracy_per_task_percent": {
            "mean": per_task.mean(0).tolist(),
            "std": per_task.std(0, ddof=1).tolist(),
        },
        "references_class_incremental_percent": REFERENCES,
        "seeds": results,
    }
    aggregate_dir = OUTPUT_ROOT / f"cifar100_linear_probe_{INITIALIZATION}_{MODEL_NAME}_3seeds"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    best = aggregate["test_accuracy_at_best_val_percent"]
    print("\n================ Linear-probe upper bound ================")
    print(f"Test accuracy (epoch by val): {best['mean']:.2f} ± {best['std']:.2f} %")
    last = aggregate["test_accuracy_last_epoch_percent"]
    print(f"Test accuracy (last epoch):   {last['mean']:.2f} ± {last['std']:.2f} %")
    for name, value in REFERENCES.items():
        print(f"  vs {name:<18s} {value:5.1f} %  ->  {best['mean'] - value:+.2f} p.p.")
    print("Per task:", " ".join(f"{v:.1f}" for v in aggregate["test_accuracy_per_task_percent"]["mean"]))
    print(f"Saved to {aggregate_dir}")
    return aggregate_dir


if __name__ == "__main__":
    run_three_seeds()
