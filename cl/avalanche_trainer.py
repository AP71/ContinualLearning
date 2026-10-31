"""Split CIFAR-100 experiments refactored with Avalanche."""

from __future__ import annotations

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
from avalanche.training.plugins import EvaluationPlugin, SupervisedPlugin
from avalanche.training.supervised import Naive

from avalanche_plugin.epoch_tracker import EpochTrackerPlugin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Network.ReActNet_18_Qaw import ResNet18
from Network.ReActNet_A_Qaw import Reactnet
from avalanche_plugin.checkpoint_and_BitFlip import CheckpointAndBitFlipPlugin
from avalanche_plugin.results import save_avalanche_sequential_results

DATASET_PATH = Path(r"F:/Dataset/CIFAR-100/")
CHECKPOINT_PATH: Path | None = None
MODEL_NAME = "resneta"
OUTPUT_ROOT = ROOT / "cl" / "results"
BATCH_SIZE = 256
WORKERS = 0
EPOCHS_PER_TASK = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
SEED = 2023
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger("split_cifar100")
torch.set_float32_matmul_precision("high")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(initialization: str) -> nn.Module:
    if MODEL_NAME == "resneta":
        model = Reactnet(num_classes=100, imagenet=False)
        checkpoint_name = "imagenet_reactneta_step2.h5"
    elif MODEL_NAME == "resnet18":
        model = ResNet18(num_classes=100, imagenet=False)
        checkpoint_name = "imagenet_resnet18_step2.h5"
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    if initialization == "checkpoint":
        checkpoint = CHECKPOINT_PATH or ROOT / "inference" / "models" / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        state_dict = {
            name.removeprefix("module."): tensor for name, tensor in state_dict.items()
        }
        model_state = model.state_dict()
        compatible_state = {
            k: v for k, v in state_dict.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        model.load_state_dict(compatible_state, strict=False)
        LOGGER.info("Loaded %d compatible tensors from %s", len(compatible_state), checkpoint)
    return model.to(DEVICE)


def create_benchmark():
    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_data = datasets.CIFAR100(DATASET_PATH, train=True, download=False, transform=train_transform)
    test_data = datasets.CIFAR100(DATASET_PATH, train=False, download=False, transform=test_transform)

    return nc_benchmark(
        train_dataset=train_data,
        test_dataset=test_data,
        n_experiences=10,
        task_labels=False,  # Class-incremental
        shuffle=False,
        seed=SEED,
    )


def evaluate_dataset(
    model: nn.Module,
    dataset,
    device: torch.device,
    allowed_classes: torch.Tensor | None = None,
) -> float:
    """Valuta l'accuratezza su un dataset, con opzione di logit masking per task-incremental."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            # Gestisce sia batch a 2 elementi (PyTorch standard) sia a 3 elementi (Avalanche con task_id)
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            if allowed_classes is not None:
                logits = logits.index_select(1, allowed_classes)
                preds = allowed_classes[logits.argmax(1)]
            else:
                preds = logits.argmax(1)
            correct += int((preds == labels).sum().item())
            total += labels.numel()
    return correct / total if total else 0.0


def evaluate_confusion_matrix(model: nn.Module, full_test_dataset, device: torch.device) -> np.ndarray:
    """Calcola la matrice di confusione 100x100 su tutto il test set CIFAR-100."""
    loader = DataLoader(full_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    matrix = np.zeros((100, 100), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            preds = model(images).argmax(1).cpu()
            indices = labels * 100 + preds
            matrix += (
                torch.bincount(indices, minlength=10000)
                .reshape(100, 100)
                .numpy()
            )
    return matrix


def run_sequential_avalanche(initialization: str = "checkpoint"):
    set_seed()
    output_dir = OUTPUT_ROOT / f"cifar100_avalanche_{initialization}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = create_benchmark()
    num_tasks = scenario.n_experiences
    model = create_model(initialization)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Plugin per salvare i pesi e calcolare i bit-flips
    custom_plugin = CheckpointAndBitFlipPlugin(
        checkpoint_dir=output_dir / "checkpoints",
        compare_dir=output_dir / "compare",
        LOGGER=LOGGER,
    )

    # Plugin che monitora epoca per epoca la loss e l'accuratezza di ogni task visto
    epoch_tracker = EpochTrackerPlugin(
        test_stream=scenario.test_stream,
        num_tasks=num_tasks,
        device=DEVICE,
    )

    loggers = [
        InteractiveLogger(),
        TextLogger(open(output_dir / "training_log.txt", "w", encoding="utf-8")),
    ]
    eval_plugin = EvaluationPlugin(loggers=loggers)

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=BATCH_SIZE,
        train_epochs=EPOCHS_PER_TASK,
        eval_mb_size=BATCH_SIZE,
        device=DEVICE,
        plugins=[custom_plugin, epoch_tracker],
        evaluator=eval_plugin,
    )

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
            )
            task_matrix[task_id][evaluated_id] = task_acc

            print(f"  -> Task {evaluated_id + 1}: Standard Acc = {std_acc * 100:.2f}%, Masked Acc = {task_acc * 100:.2f}%")

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
    conf_matrix = evaluate_confusion_matrix(strategy.model, full_test_dataset, DEVICE)

    # Salvataggio di tutti i plot e dei file JSON identici a base_train_3.py
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
    print(f"\n[Completato] Grafici e JSON salvati con successo in: {output_dir}")


if __name__ == "__main__":
    run_sequential_avalanche("checkpoint")