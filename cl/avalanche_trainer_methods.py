"""Motore comune per gli esperimenti Split CIFAR-100 class-incremental con Avalanche.

Ogni metodo (cl/methods/*.py) fornisce solo:
  - build_optimizer(model, cfg) -> optimizer   (default: Adam sui parametri addestrabili)
  - build_strategy(common_args, cfg) -> strategy (default: Naive)
Tutto il resto (benchmark, scheduler, checkpoint e bit flip, valutazione CIL/TIL,
salvataggio dei risultati, aggregazione dei 3 seed) è identico per tutti i metodi.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from torch import nn
from torchvision import datasets, transforms

from avalanche_plugin.checkpoint_and_BitFlip import CheckpointAndBitFlipPlugin
from avalanche_plugin.epoch_tracker import EpochTrackerPlugin
from avalanche_plugin.results import save_avalanche_sequential_results
from avalanche_plugin.utils import (
    create_benchmark,
    create_model,
    evaluate_confusion_matrix,
    evaluate_dataset,
    set_seed,
)

# Compatibilità: alcune versioni di Avalanche passano 'verbose' a MultiStepLR
_orig_multistep_init = torch.optim.lr_scheduler.MultiStepLR.__init__


def _patched_multistep_init(self, *args, **kwargs):
    kwargs.pop("verbose", None)
    _orig_multistep_init(self, *args, **kwargs)


torch.optim.lr_scheduler.MultiStepLR.__init__ = _patched_multistep_init

LOGGER = logging.getLogger("split_cifar100")
torch.set_float32_matmul_precision("high")


@dataclass
class TrainConfig:
    dataset_path: Path = Path(r"F:/Dataset/CIFAR-100/")
    checkpoint_path: Path | None = None
    model_name: str = "resneta"
    output_root: Path = ROOT / "cl" / "results"
    batch_size: int = 256
    workers: int = 0
    epochs_per_task: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_milestones: tuple[int, ...] = (20, 25)
    lr_gamma: float = 0.1
    seeds: tuple[int, ...] = (42, 2023, 3047)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OptimizerBuilder = Callable[[nn.Module, TrainConfig], torch.optim.Optimizer]
StrategyBuilder = Callable[[dict, TrainConfig], object]


def adam(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Adam sui soli parametri addestrabili (tutti, se nulla è congelato)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def naive(common_args: dict, cfg: TrainConfig):
    return Naive(**common_args)


def run_sequential(
    name: str,
    build_optimizer: OptimizerBuilder = adam,
    build_strategy: StrategyBuilder = naive,
    cfg: TrainConfig | None = None,
    seed: int = 42,
    initialization: str = "checkpoint",
    output_dir: Path | None = None,
    suffix: str = "",
) -> Path:
    """Un'esecuzione completa (10 task) per un seed."""
    cfg = cfg or TrainConfig()
    set_seed(seed=seed)
    if output_dir is None:
        output_dir = cfg.output_root / (
            f"cifar100_avalanche_{name}_{initialization}_{cfg.model_name}_"
            f"seed{seed}_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario, _ = create_benchmark(cfg.dataset_path, seed=seed, is_joint=False)
    num_tasks = scenario.n_experiences
    model = create_model(initialization, cfg.device, LOGGER, cfg.checkpoint_path, ROOT)
    optimizer = build_optimizer(model, cfg)

    # Scheduler per il singolo task (resettato a ogni nuovo task)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma
    )
    lr_scheduler_plugin = LRSchedulerPlugin(
        scheduler, step_granularity="epoch", reset_scheduler=True, reset_lr=True
    )
    # Salvataggio dei pesi e calcolo dei bit flip (codice nostro)
    bitflip_plugin = CheckpointAndBitFlipPlugin(
        checkpoint_dir=output_dir / "checkpoints",
        compare_dir=output_dir / "compare",
        LOGGER=LOGGER,
    )
    epoch_tracker = EpochTrackerPlugin(
        test_stream=scenario.test_stream,
        num_tasks=num_tasks,
        device=cfg.device,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
    )
    loggers = [
        InteractiveLogger(),
        TextLogger(open(output_dir / "training_log.txt", "w", encoding="utf-8")),
    ]

    common_args = {
        "model": model,
        "optimizer": optimizer,
        "criterion": nn.CrossEntropyLoss(),
        "train_mb_size": cfg.batch_size,
        "train_epochs": cfg.epochs_per_task,
        "eval_mb_size": cfg.batch_size,
        "device": cfg.device,
        "plugins": [bitflip_plugin, epoch_tracker, lr_scheduler_plugin],
        "evaluator": EvaluationPlugin(loggers=loggers),
    }
    print(f"\n=======================================================")
    print(f" Strategia: {name}{suffix}  |  seed {seed}")
    print(f"=======================================================")
    strategy = build_strategy(common_args, cfg)

    standard_matrix: list[list[float | None]] = [[None] * num_tasks for _ in range(num_tasks)]
    task_matrix: list[list[float | None]] = [[None] * num_tasks for _ in range(num_tasks)]
    # Split CIFAR-100 a blocchi ordinati di 10 classi
    task_classes = [torch.arange(t * 10, (t + 1) * 10, device=cfg.device) for t in range(num_tasks)]

    for experience in scenario.train_stream:
        task_id = experience.current_experience
        print(f"\n=======================================================")
        print(f"  Training Task {task_id + 1}/{num_tasks}")
        print(f"=======================================================")
        strategy.train(experience, num_workers=cfg.workers)

        print(f"\n>> Valutazione Cross-Task a fine Task {task_id + 1}...")
        for evaluated_id in range(task_id + 1):
            test_dataset = scenario.test_stream[evaluated_id].dataset
            # Class-incremental: tutti i logit
            std_acc = evaluate_dataset(strategy.model, test_dataset, cfg.device, allowed_classes=None)
            # Task-incremental: logit ristretti alle classi del task
            task_acc = evaluate_dataset(
                strategy.model,
                test_dataset,
                cfg.device,
                allowed_classes=task_classes[evaluated_id],
                batch_size=cfg.batch_size,
                num_workers=cfg.workers,
            )
            standard_matrix[task_id][evaluated_id] = std_acc
            task_matrix[task_id][evaluated_id] = task_acc
            print(
                f"  -> Task {evaluated_id + 1}: Standard Acc = {std_acc * 100:.2f}%, "
                f"Masked Acc = {task_acc * 100:.2f}%"
            )

    print("\n>> Calcolo Confusion Matrix finale...")
    full_test_dataset = datasets.CIFAR100(
        cfg.dataset_path,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
    )
    conf_matrix = evaluate_confusion_matrix(
        strategy.model, full_test_dataset, cfg.device, batch_size=cfg.batch_size, num_workers=cfg.workers
    )

    save_avalanche_sequential_results(
        output_dir=output_dir,
        standard_matrix=standard_matrix,
        task_matrix=task_matrix,
        accuracy_history=epoch_tracker.accuracy_history,
        loss_history=epoch_tracker.loss_history,
        custom_plugin=bitflip_plugin,
        num_tasks=num_tasks,
        epochs_per_task=cfg.epochs_per_task,
        model_name=cfg.model_name,
        initialization=initialization,
        confusion_matrix_arr=conf_matrix,
    )
    print(f"\n[Completato] JSON salvato in: {output_dir}")
    return output_dir


def run_three_seeds(
    name: str,
    build_optimizer: OptimizerBuilder = adam,
    build_strategy: StrategyBuilder = naive,
    cfg: TrainConfig | None = None,
    initialization: str = "checkpoint",
    suffix: str = "",
    overwrite: bool = False,
    seeds: Sequence[int] | None = None,
) -> Path:
    """Esegue i seed richiesti nella cartella dell'esperimento e rigenera il results.json
    aggregato con tutti i seed presenti nella cartella.

    seeds=None esegue tutti i seed di cfg; una lista (es. [42]) solo quelli, così i
    mancanti si aggiungono in seguito lanciando di nuovo con la lista dei seed restanti.
    Un seed che ha già risultati non viene rieseguito, a meno di overwrite=True.
    """
    cfg = cfg or TrainConfig()
    seeds = list(cfg.seeds if seeds is None else seeds)
    aggregate_dir = cfg.output_root / (
        f"cifar100_avalanche_{name}_{initialization}_{cfg.model_name}_3seeds{suffix}"
    )
    done = [s for s in seeds if (aggregate_dir / f"seed{s}" / "results.json").exists()]
    if done and not overwrite:
        raise FileExistsError(
            f"{aggregate_dir.name}: i seed {done} hanno già risultati. "
            "Toglili dalla lista o usa overwrite=True."
        )
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_sequential(
            name, build_optimizer, build_strategy, cfg,
            seed=seed, initialization=initialization, output_dir=aggregate_dir / f"seed{seed}",
        )

    # Aggregato con tutti i seed presenti (anche quelli di lanci precedenti)
    present = sorted(
        (int(d.name.removeprefix("seed")) for d in aggregate_dir.glob("seed*")
         if (d / "results.json").exists()),
        key=lambda s: (s not in cfg.seeds, cfg.seeds.index(s) if s in cfg.seeds else s),
    )
    results_by_seed = {}
    for seed in present:
        result = json.loads((aggregate_dir / f"seed{seed}" / "results.json").read_text(encoding="utf-8"))
        result["seed"] = seed
        results_by_seed[str(seed)] = result

    aggregate = {
        "schema_version": 2,
        "experiment_type": "sequential",
        "strategy": name,
        "model_name": cfg.model_name,
        "initialization": initialization,
        "seeds": results_by_seed,
    }
    (aggregate_dir / "results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\n[Aggregato] {aggregate_dir.name}: seed presenti {present}")
    return aggregate_dir


def run_method(
    name: str,
    build_optimizer: OptimizerBuilder = adam,
    build_strategy: StrategyBuilder = naive,
    cfg: TrainConfig | None = None,
    initialization: str = "checkpoint",
    suffix: str = "",
    seeds: Sequence[int] | None = None,
) -> Path:
    """Punto di ingresso dei file in cl/methods (seeds=None: tutti i seed)."""
    return run_three_seeds(
        name, build_optimizer, build_strategy, cfg, initialization, suffix, seeds=seeds
    )
