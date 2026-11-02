"""Joint Training CIFAR-100 experiment with Avalanche and LR Scheduler."""

from __future__ import annotations

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Avalanche imports
from avalanche.benchmarks import nc_benchmark
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import (
    EarlyStoppingPlugin,
    EvaluationPlugin,
    LRSchedulerPlugin,
)
from avalanche.training.supervised import JointTraining
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics

from avalanche_plugin.utils import evaluate_confusion_matrix, create_benchmark, create_model, set_seed

# --- PATCH PER AVALANCHE MultiStepLR (Rimuove 'verbose') ---
_orig_multistep_init = torch.optim.lr_scheduler.MultiStepLR.__init__


def _patched_multistep_init(self, *args, **kwargs):
    kwargs.pop('verbose', None)
    _orig_multistep_init(self, *args, **kwargs)


torch.optim.lr_scheduler.MultiStepLR.__init__ = _patched_multistep_init
# -----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Network.ReActNet_A_Qaw import Reactnet
from avalanche_plugin.checkpoint_and_BitFlip import CheckpointAndBitFlipPlugin

DATASET_PATH = Path(r"F:/Dataset/CIFAR-100/")
CHECKPOINT_PATH: Path | None = None
MODEL_NAME = "resneta"
OUTPUT_ROOT = ROOT / "cl" / "results"
BATCH_SIZE = 256
WORKERS = 0

JOINT_EPOCHS = 400
JOINT_EARLY_STOPPING_ENABLED = True
JOINT_EARLY_STOPPING_PATIENCE = 5
JOINT_EARLY_STOPPING_MIN_DELTA = 1e-3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEEDS = (42, 2023, 3047)
SEED = SEEDS[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger("joint_cifar100")
torch.set_float32_matmul_precision("high")


def extract_validation_accuracy(metrics: dict[str, float]) -> float:
    """Return the test-stream top-1 accuracy reported by Avalanche."""
    metric_key = "Top1_Acc_Stream/eval_phase/test_stream"
    matching_keys = [key for key in metrics if key.startswith(metric_key)]
    if not matching_keys:
        raise KeyError(
            "Avalanche did not report Top1_Acc_Stream for test_stream. "
            f"Available metrics: {sorted(metrics)}"
        )
    return float(metrics[matching_keys[-1]])


def run_joint_avalanche(initialization: str = "checkpoint"):
    set_seed(SEED)
    output_dir = OUTPUT_ROOT / f"cifar100_avalanche_JOINT_{initialization}_{MODEL_NAME}_seed{SEED}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = create_benchmark(DATASET_PATH, SEED)
    model = create_model(initialization, DEVICE, LOGGER, CHECKPOINT_PATH, ROOT)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 320], gamma=0.1)

    # 3. Wrapping dello Scheduler nel plugin di Avalanche
    scheduler_plugin = LRSchedulerPlugin(
        scheduler,
        step_granularity='epoch'
    )

    # Plugin per Bit-Flip
    custom_plugin = CheckpointAndBitFlipPlugin(
        checkpoint_dir=output_dir / "checkpoints",
        compare_dir=output_dir / "compare",
        LOGGER=LOGGER,
    )

    # Loggers e metriche nativi di Avalanche
    loggers = [
        InteractiveLogger(),
        TextLogger(open(output_dir / "training_log.txt", "w", encoding="utf-8")),
    ]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, stream=True),
        loss_metrics(epoch=True, stream=True),
        loggers=loggers
    )

    # Inizializzazione della strategia JointTraining
    print(f"\n=======================================================")
    print(f" Inizializzazione Avalanche Joint Training ({JOINT_EPOCHS} Epoche)")
    print(f"=======================================================")

    early_stopping_plugin = EarlyStoppingPlugin(
        patience=JOINT_EARLY_STOPPING_PATIENCE,
        val_stream_name="test_stream",
        metric_name="Top1_Acc_Stream",
        mode="max",
        peval_mode="epoch",
        margin=JOINT_EARLY_STOPPING_MIN_DELTA,
        verbose=True,
    )

    strategy = JointTraining(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=BATCH_SIZE,
        train_epochs=JOINT_EPOCHS,
        eval_mb_size=BATCH_SIZE,
        device=DEVICE,
        plugins=[scheduler_plugin, custom_plugin, early_stopping_plugin],
        evaluator=eval_plugin,
        eval_every=1,
    )

    if not JOINT_EARLY_STOPPING_ENABLED:
        raise ValueError("EarlyStoppingPlugin must be enabled for joint training.")

    # EvaluationPlugin evaluates test_stream every epoch; EarlyStoppingPlugin
    # restores the best model and stops the strategy after the patience limit.
    strategy.train(
        scenario.train_stream,
        eval_streams=[scenario.test_stream],
        num_workers=WORKERS,
    )
    epochs_completed = strategy.clock.train_exp_epochs
    best_accuracy = early_stopping_plugin.best_val
    best_epoch = early_stopping_plugin.best_step
    stopped_early = epochs_completed < JOINT_EPOCHS
    epochs_without_improvement = (
        epochs_completed - best_epoch if best_epoch is not None else None
    )

    # Valutazione Finale sull'intero test set
    print("\n>> Addestramento completato. Avvio valutazione finale...")
    final_metrics = strategy.eval(scenario.test_stream)

    print("\n>> Calcolo Confusion Matrix finale...")
    conf_matrix = evaluate_confusion_matrix(strategy.model, scenario.test_stream[0].dataset, DEVICE, batch_size=BATCH_SIZE, num_workers=WORKERS)

    # Salvataggio metriche di base
    results = {
        "schema_version": 2,
        "experiment_type": "joint_training",
        "initialization": initialization,
        "model_name": MODEL_NAME,
        "seed": SEED,
        "total_epochs": JOINT_EPOCHS,
        "epochs_completed": epochs_completed,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "early_stopping": {
            "enabled": JOINT_EARLY_STOPPING_ENABLED,
            "monitor": "validation_accuracy",
            "patience": JOINT_EARLY_STOPPING_PATIENCE,
            "min_delta": JOINT_EARLY_STOPPING_MIN_DELTA,
            "stopped_early": stopped_early,
            "epochs_without_improvement": epochs_without_improvement,
        },
        "final_accuracy": extract_validation_accuracy(final_metrics),
        "confusion_matrix": conf_matrix.tolist()
    }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Completato] Joint Training terminato. Risultati in: {output_dir}")
    return output_dir


def run_three_seeds(initialization: str = "checkpoint") -> Path:
    """Esegue il Joint Training per tutti i seed configurati."""
    global SEED
    results_by_seed = {}
    for seed in SEEDS:
        SEED = seed
        output_dir = run_joint_avalanche(initialization)
        results_path = output_dir / "results.json"
        result = json.loads(results_path.read_text(encoding="utf-8"))
        results_by_seed[str(seed)] = result

    aggregate_dir = OUTPUT_ROOT / (
        f"cifar100_avalanche_JOINT_{initialization}_{MODEL_NAME}_3seeds"
    )
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "schema_version": 2,
        "experiment_type": "joint_training",
        "model_name": MODEL_NAME,
        "initialization": initialization,
        "seeds": results_by_seed,
    }
    (aggregate_dir / "results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate_dir


if __name__ == "__main__":
    run_three_seeds("checkpoint")