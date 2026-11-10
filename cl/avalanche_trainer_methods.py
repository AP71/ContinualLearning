"""Split CIFAR-100 experiments refactored with Avalanche (Multiple CL Strategies)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms

# Avalanche imports per le strategie di Continual Learning
from avalanche.training import Naive, Replay, LwF, EWC
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin

# Custom imports dal tuo progetto
from avalanche_plugin.epoch_tracker import EpochTrackerPlugin
from avalanche_plugin.utils import evaluate_dataset, set_seed, create_benchmark, create_model, evaluate_confusion_matrix

_orig_multistep_init = torch.optim.lr_scheduler.MultiStepLR.__init__


def _patched_multistep_init(self, *args, **kwargs):
    kwargs.pop('verbose', None)
    _orig_multistep_init(self, *args, **kwargs)


torch.optim.lr_scheduler.MultiStepLR.__init__ = _patched_multistep_init

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avalanche_plugin.checkpoint_and_BitFlip import CheckpointAndBitFlipPlugin
from avalanche_plugin.metaplasticity import MetaplasticityPlugin
from avalanche_plugin.results import save_avalanche_sequential_results

# ==============================================================================
# STRATEGY CONFIGURATION - Scegli qui l'esperimento da eseguire
# ==============================================================================
# Opzioni supportate: "Freezing", "Replay", "LwF", "EWC", "Metaplasticity"
STRATEGY_NAME = "Metaplasticity"

# Hyper-parameters specifici per le strategie CL (da tunare per la BNN)
REPLAY_MEM_SIZE = 2000  # Numero totale di immagini salvate nel buffer
LWF_ALPHA = 1.0  # Peso della loss di distillazione (abbassare se i bit non flippano)
LWF_TEMPERATURE = 2.0  # Temperatura per ammorbidire i logit
EWC_LAMBDA = 0.4  # Forza della penalità basata sull'informazione di Fisher
GEM_PATTERNS_PER_EXP = 200  # Pattern salvati per esperienza per GEM/AGEM
AGEM_SAMPLE_SIZE = 64  # Batch size del replay in AGEM
METAPLASTICITY_M = 1.5  # Forza di consolidamento metaplastico m (Laborieux et al.)
METAPLASTICITY_CLASSIFIER = "fc"  # Nome del classificatore finale da proteggere
# ==============================================================================

DATASET_PATH = Path(r"F:/Dataset/CIFAR-100/")
CHECKPOINT_PATH: Path | None = None
MODEL_NAME = "resneta"
OUTPUT_ROOT = ROOT / "cl" / "results"
BATCH_SIZE = 256
WORKERS = 0
EPOCHS_PER_TASK = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEEDS = (42, 2023, 3047)
SEED = SEEDS[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger("split_cifar100")
torch.set_float32_matmul_precision("high")


def run_sequential_avalanche(
        initialization: str = "checkpoint",
        output_dir: Path | None = None,
):
    set_seed(seed=SEED)
    if output_dir is None:
        output_dir = OUTPUT_ROOT / (
            f"cifar100_avalanche_{STRATEGY_NAME}_{initialization}_{MODEL_NAME}_"
            f"seed{SEED}_{datetime.now():%Y%m%d_%H%M%S}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario, val_dataset = create_benchmark(DATASET_PATH, seed=SEED, is_joint=False)
    num_tasks = scenario.n_experiences
    model = create_model(initialization, DEVICE, LOGGER, CHECKPOINT_PATH, ROOT)

    # --------------------------------------------------------------------------
    # GESTIONE OTTIMIZZATORE: Seleziona quali pesi addestrare
    # --------------------------------------------------------------------------
    if STRATEGY_NAME == "Freezing":
        print("\n>>> STRATEGIA FREEZING: Blocco tutti i layer tranne il classificatore finale ('fc')...")
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

        # Passiamo all'ottimizzatore SOLO i parametri che richiedono il gradiente
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        # Per tutte le altre strategie (Replay, LwF, EWC, Metaplasticity...), addestriamo tutta la rete
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # --------------------------------------------------------------------------

    # Scheduler settato per il singolo Task (Reset a ogni nuovo task)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 25],
        gamma=0.1
    )

    lr_scheduler_plugin = LRSchedulerPlugin(
        scheduler,
        step_granularity='epoch',
        reset_scheduler=True,
        reset_lr=True
    )

    # Plugin per salvare i pesi e calcolare i bit-flips
    custom_plugin = CheckpointAndBitFlipPlugin(
        checkpoint_dir=output_dir / "checkpoints",
        compare_dir=output_dir / "compare",
        LOGGER=LOGGER,
    )

    epoch_tracker = EpochTrackerPlugin(
        test_stream=scenario.test_stream,
        num_tasks=num_tasks,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
    )

    loggers = [
        InteractiveLogger(),
        TextLogger(open(output_dir / "training_log.txt", "w", encoding="utf-8")),
    ]
    eval_plugin = EvaluationPlugin(loggers=loggers)

    # --------------------------------------------------------------------------
    # INIZIALIZZAZIONE DELLA STRATEGIA AVALANCHE SCELTA
    # --------------------------------------------------------------------------
    common_args = {
        "model": model,
        "optimizer": optimizer,
        "criterion": nn.CrossEntropyLoss(),
        "train_mb_size": BATCH_SIZE,
        "train_epochs": EPOCHS_PER_TASK,
        "eval_mb_size": BATCH_SIZE,
        "device": DEVICE,
        "plugins": [custom_plugin, epoch_tracker, lr_scheduler_plugin],
        "evaluator": eval_plugin,
    }

    print(f"\n=======================================================")
    print(f" Inizializzazione Strategia Continual Learning: {STRATEGY_NAME}")
    print(f"=======================================================")

    if STRATEGY_NAME == "Freezing":
        strategy = Naive(**common_args)
    elif STRATEGY_NAME == "Replay":
        strategy = Replay(mem_size=REPLAY_MEM_SIZE, **common_args)
    elif STRATEGY_NAME == "LwF":
        strategy = LwF(alpha=LWF_ALPHA, temperature=LWF_TEMPERATURE, **common_args)
    elif STRATEGY_NAME == "EWC":
        strategy = EWC(ewc_lambda=EWC_LAMBDA, **common_args)
    elif STRATEGY_NAME == "Metaplasticity":
        metaplasticity_plugin = MetaplasticityPlugin(
            m=METAPLASTICITY_M,
            classifier_name=METAPLASTICITY_CLASSIFIER,
        )
        common_args["plugins"] = [metaplasticity_plugin] + common_args["plugins"]
        strategy = Naive(**common_args)
    else:
        raise ValueError(f"Strategia sconosciuta: {STRATEGY_NAME}")
    # --------------------------------------------------------------------------

    standard_matrix: list[list[float | None]] = [[None] * num_tasks for _ in range(num_tasks)]
    task_matrix: list[list[float | None]] = [[None] * num_tasks for _ in range(num_tasks)]

    # Mappa le classi per ciascun task (Split-CIFAR-100 a blocchi ordinati di 10 classi)
    task_classes_tensors = [
        torch.arange(t * 10, (t + 1) * 10, device=DEVICE) for t in range(num_tasks)
    ]

    # Training Sequenziale Continual Learning
    for experience in scenario.train_stream:
        task_id = experience.current_experience
        print(f"\n=======================================================")
        print(f"  Training Task {task_id + 1}/{num_tasks}")
        print(f"=======================================================")
        strategy.train(experience, num_workers=WORKERS)

        print(f"\n>> Valutazione Cross-Task a fine Task {task_id + 1}...")
        # Valuta tutti i task passati (0 .. task_id) per calcolare la riga task_id delle due matrici
        for evaluated_id in range(task_id + 1):
            test_dataset = scenario.test_stream[evaluated_id].dataset

            # 1. Valutazione Class-Incremental (standard, senza logit masking)
            std_acc = evaluate_dataset(strategy.model, test_dataset, DEVICE, allowed_classes=None)
            standard_matrix[task_id][evaluated_id] = std_acc

            # 2. Valutazione Task-Incremental (con logit masking ristretto alle classi del task)
            task_acc = evaluate_dataset(
                strategy.model,
                test_dataset,
                DEVICE,
                allowed_classes=task_classes_tensors[evaluated_id],
                batch_size=BATCH_SIZE,
                num_workers=WORKERS,
            )
            task_matrix[task_id][evaluated_id] = task_acc

            print(
                f"  -> Task {evaluated_id + 1}: Standard Acc = {std_acc * 100:.2f}%, Masked Acc = {task_acc * 100:.2f}%")

    # Calcolo della matrice di confusione finale 100x100
    print("\n>> Calcolo Confusion Matrix finale...")
    full_test_dataset = datasets.CIFAR100(
        DATASET_PATH,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
    )
    conf_matrix = evaluate_confusion_matrix(strategy.model, full_test_dataset, DEVICE, batch_size=BATCH_SIZE,
                                            num_workers=WORKERS)

    save_avalanche_sequential_results(
        output_dir=output_dir,
        standard_matrix=standard_matrix,
        task_matrix=task_matrix,
        accuracy_history=epoch_tracker.accuracy_history,
        loss_history=epoch_tracker.loss_history,
        custom_plugin=custom_plugin,
        num_tasks=num_tasks,
        epochs_per_task=EPOCHS_PER_TASK,
        model_name=MODEL_NAME,
        initialization=initialization,
        confusion_matrix_arr=conf_matrix,
    )
    print(f"\n[Completato] JSON salvato in: {output_dir}")
    return output_dir


def run_three_seeds(initialization: str = "checkpoint") -> Path:
    """Run Avalanche for all seeds inside one parent experiment directory."""
    global SEED
    aggregate_dir = OUTPUT_ROOT / (
        f"cifar100_avalanche_{STRATEGY_NAME}_{initialization}_{MODEL_NAME}_3seeds"
    )
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    results_by_seed = {}
    for seed in SEEDS:
        SEED = seed
        output_dir = aggregate_dir / f"seed{seed}"
        run_sequential_avalanche(initialization, output_dir=output_dir)
        results_path = output_dir / "results.json"
        result = json.loads(results_path.read_text(encoding="utf-8"))
        result["seed"] = seed
        results_by_seed[str(seed)] = result

    aggregate = {
        "schema_version": 2,
        "experiment_type": "sequential",
        "strategy": STRATEGY_NAME,
        "model_name": MODEL_NAME,
        "initialization": initialization,
        "seeds": results_by_seed,
    }
    (aggregate_dir / "results.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate_dir


if __name__ == "__main__":
    run_three_seeds("checkpoint")
