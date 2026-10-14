"""Class-incremental Split CIFAR-100 experiment."""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Network.ReActNet_18_Qaw import ResNet18
from Network.ReActNet_A_Qaw import Reactnet
from datasets.split_cifar100 import SplitCIFAR100
from utils.checkpoint_analysis import (
    compare_checkpoints,
    plot_latent_distributions,
)
from utils.metrics import calculate_bwt, plot_accuracy_history, plot_task_metric


DATASET_PATH = Path(r"F:/Dataset/CIFAR-100/")
CHECKPOINT_PATH: Path | None = None
MODEL_NAME = "resneta"
OUTPUT_ROOT = ROOT / "cl" / "results"
BATCH_SIZE = 512
WORKERS = 0
EPOCHS_PER_TASK = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
SEED = 2023
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGGER = logging.getLogger("split_cifar100")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Pad(4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, test_transform

def create_datasets() -> tuple[SplitCIFAR100, SplitCIFAR100]:
    train_transform, test_transform = create_transforms()
    train_data = datasets.CIFAR100(
        DATASET_PATH, train=True, download=False, transform=train_transform
    )
    test_data = datasets.CIFAR100(
        DATASET_PATH, train=False, download=False, transform=test_transform
    )
    LOGGER.info("Loaded CIFAR-100 from %s without downloading.", DATASET_PATH)
    return SplitCIFAR100(train_data), SplitCIFAR100(test_data)

def create_model() -> nn.Module:
    if MODEL_NAME == "resneta":
        model = Reactnet(num_classes=100, imagenet=False)
        checkpoint_name = "cifar100_reactneta_step2.h5"
    elif MODEL_NAME == "resnet18":
        model = ResNet18(num_classes=100, imagenet=False)
        checkpoint_name = "cifar100_reactnet18_step2.h5"
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    checkpoint = CHECKPOINT_PATH or ROOT / "inference" / "models" / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state_dict = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if DEVICE.type == "cuda":
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    LOGGER.info("Loaded %s checkpoint from %s.", MODEL_NAME, checkpoint)
    return model.to(DEVICE)

def create_loader(dataset: SplitCIFAR100, shuffle: bool, num_workers: int = WORKERS) -> DataLoader:
    loader_options = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
    return DataLoader(
        dataset,
        **loader_options,
    )

def evaluate_task(model: nn.Module, loader: DataLoader, task_id: int) -> float:
    started_at = time.perf_counter()
    LOGGER.info(
        "Accuracy evaluation started for task %d (%d samples).",
        task_id + 1,
        len(loader.dataset),
    )
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            correct += int((model(images).argmax(1) == labels).sum().item())
            total += labels.numel()
    accuracy = correct / total if total else 0.0
    LOGGER.info(
        "Accuracy evaluation finished for task %d: accuracy=%.4f, samples=%d, elapsed=%.2fs.",
        task_id + 1,
        accuracy,
        total,
        time.perf_counter() - started_at,
    )
    return accuracy

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, task_id: int, epoch: int) -> float:
    started_at = time.perf_counter()
    LOGGER.info(
        "Training started for task %d, epoch %d/%d, batches=%d.",
        task_id + 1,
        epoch + 1,
        EPOCHS_PER_TASK,
        len(loader),
    )
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch_id, (images, labels) in enumerate(
        tqdm(
        loader,
        desc=f"Task {task_id + 1}/10, epoch {epoch + 1}/{EPOCHS_PER_TASK}",
        ),
        start=1,
    ):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        sample_count += labels.size(0)
        if batch_id == 1 or batch_id == len(loader) or batch_id % 10 == 0:
            LOGGER.info(
                "Training progress: task=%d, epoch=%d, batch=%d/%d, loss=%.4f.",
                task_id + 1,
                epoch + 1,
                batch_id,
                len(loader),
                loss.item(),
            )
    average_loss = loss_sum / sample_count if sample_count else 0.0
    LOGGER.info(
        "Training finished for task %d, epoch %d: loss=%.4f, samples=%d, elapsed=%.2fs.",
        task_id + 1,
        epoch + 1,
        average_loss,
        sample_count,
        time.perf_counter() - started_at,
    )
    return average_loss

def evaluate_all_tasks(model: nn.Module, test_loaders: list[DataLoader], accuracy_history: list[list[float]]) -> None:
    started_at = time.perf_counter()
    LOGGER.info(
        "Evaluation of all tasks started after global epoch %d.",
        len(accuracy_history[0]) + 1,
    )
    for evaluated_task, test_loader in enumerate(test_loaders):
        LOGGER.info(
            "Evaluating task %d/%d after global epoch %d.",
            evaluated_task + 1,
            len(test_loaders),
            len(accuracy_history[0]) + 1,
        )
        accuracy = evaluate_task(model, test_loader, evaluated_task)
        accuracy_history[evaluated_task].append(accuracy)
    LOGGER.info(
        "Evaluation of all tasks finished after global epoch %d: elapsed=%.2fs.",
        len(accuracy_history[0]),
        time.perf_counter() - started_at,
    )

def copy_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

def create_task_row(task_id: int, train_tasks: SplitCIFAR100, accuracy_matrix: list[list[float | None]], loss_value: float, bit_flip_result: dict) -> dict:
    bwt = calculate_bwt(accuracy_matrix[: task_id + 1])
    binary_summary = bit_flip_result["binary_summary"]
    row = {
        "task": task_id,
        "classes": "-".join(map(str, train_tasks.classes)),
        "train_loss": loss_value,
        "accuracy_current_task": accuracy_matrix[task_id][task_id],
        "accuracy_seen_tasks": float(
            np.mean(accuracy_matrix[task_id][: task_id + 1])
        ),
        "bwt": bwt,
        "bit_flips": binary_summary["bit_flips"],
        "total_binary_bits": binary_summary["total_bits"],
        "bit_flip_percentage": binary_summary["bit_flip_percentage"],
    }
    LOGGER.info(
        "Task %d completed: loss=%.4f, ACC=%.4f, BWT=%.4f, bit flips=%d (%.4f%%).",
        task_id + 1,
        loss_value,
        row["accuracy_current_task"],
        bwt,
        row["bit_flips"],
        row["bit_flip_percentage"],
    )
    return row

def save_results(output_dir: Path, results: dict, task_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    with (output_dir / "tasks.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=task_rows[0].keys())
        writer.writeheader()
        writer.writerows(task_rows)

    plot_accuracy_history(results["accuracy_history"], output_dir / "accuracy_per_epoch.png")
    plot_task_metric(
        [row["bwt"] for row in task_rows],
        "BWT (%)",
        "Backward transfer per task",
        output_dir / "bwt_per_task.png",
        connect_points=True,
        percentage_axis=True,
    )
    plot_task_metric(
        [row["bit_flips"] for row in task_rows],
        "Bit flips",
        "Binary bit flips per task",
        output_dir / "bitflips_per_task.png",
        point_labels=[row["bit_flip_percentage"] for row in task_rows],
        label_format="{:.4f}%",
    )
    LOGGER.info("Saved metrics, plots, and logs to %s.", output_dir)

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    LOGGER.info("Starting Split CIFAR-100 class-incremental experiment.")
    LOGGER.info(
        "Configuration: model=%s, device=%s, epochs per task=%d.",
        MODEL_NAME,
        DEVICE,
        EPOCHS_PER_TASK,
    )
    set_seed()
    train_tasks, test_tasks = create_datasets()
    test_loaders = [
        create_loader(
            SplitCIFAR100(test_tasks.dataset, task_id),
            shuffle=False,
            num_workers=0,
        )
        for task_id in range(test_tasks.NUM_TASKS)
    ]
    LOGGER.info(
        "Created %d persistent test loaders once; worker startup will not repeat.",
        len(test_loaders),
    )
    model = create_model()
    accuracy_matrix = [
        [None] * train_tasks.NUM_TASKS for _ in range(train_tasks.NUM_TASKS)
    ]
    accuracy_history = [[] for _ in range(train_tasks.NUM_TASKS)]
    task_rows = []
    previous_state = copy_model_state(model)
    output_dir = OUTPUT_ROOT / f"cifar100_{MODEL_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir = output_dir / "sequence"
    sequence_dir.mkdir(parents=True, exist_ok=True)

    for task_id in range(train_tasks.NUM_TASKS):
        task_started_at = time.perf_counter()
        train_tasks.load_task(task_id)
        LOGGER.info(
            "Starting task %d/%d with classes %s (%d samples).",
            task_id + 1,
            train_tasks.NUM_TASKS,
            train_tasks.classes,
            len(train_tasks),
        )
        train_loader = create_loader(train_tasks, shuffle=True)
        LOGGER.info(
            "Training loader created for task %d: batches=%d, batch_size=%d.",
            task_id + 1,
            len(train_loader),
            BATCH_SIZE,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        LOGGER.info(
            "Optimizer created for task %d: lr=%g, weight_decay=%g.",
            task_id + 1,
            LEARNING_RATE,
            WEIGHT_DECAY,
        )
        loss_value = 0.0
        for epoch in range(EPOCHS_PER_TASK):
            LOGGER.info(
                "Starting task %d epoch %d/%d.",
                task_id + 1,
                epoch + 1,
                EPOCHS_PER_TASK,
            )
            loss_value = train_epoch(
                model, train_loader, criterion, optimizer, task_id, epoch
            )
            LOGGER.info(
                "Task %d epoch %d training phase completed with loss %.4f.",
                task_id + 1,
                epoch + 1,
                loss_value,
            )
            evaluate_all_tasks(model, test_loaders, accuracy_history)

        LOGGER.info("Copying model state after task %d.", task_id + 1)
        current_state = copy_model_state(model)
        LOGGER.info("Calculating bit flips for task %d.", task_id + 1)
        bitflip_started_at = time.perf_counter()
        bit_flip_result = compare_checkpoints(previous_state, current_state)
        LOGGER.info(
            "Bit flip calculation finished for task %d: flips=%d, total_bits=%d, elapsed=%.2fs.",
            task_id + 1,
            bit_flip_result["binary_summary"]["bit_flips"],
            bit_flip_result["binary_summary"]["total_bits"],
            time.perf_counter() - bitflip_started_at,
        )
        previous_state = current_state
        LOGGER.info("Updating accuracy matrix for task %d.", task_id + 1)
        for evaluated_task in range(task_id + 1):
            accuracy_matrix[task_id][evaluated_task] = accuracy_history[evaluated_task][-1]
        LOGGER.info("Calculating BWT for task %d.", task_id + 1)
        task_rows.append(
            create_task_row(
                task_id,
                train_tasks,
                accuracy_matrix,
                loss_value,
                bit_flip_result,
            )
        )
        LOGGER.info(
            "Task %d completed: elapsed=%.2fs.",
            task_id + 1,
            time.perf_counter() - task_started_at,
        )

        plot_latent_distributions(
            bit_flip_result,
            sequence_dir / f"task_{task_id + 1:02d}.png",
        )
        LOGGER.info(
            "Saved task %d checkpoint comparison plot to %s.",
            task_id + 1,
            sequence_dir,
        )

    save_results(
        output_dir,
        {
            "model": MODEL_NAME,
            "device": str(DEVICE),
            "seed": SEED,
            "accuracy_matrix": accuracy_matrix,
            "accuracy_history": accuracy_history,
            "tasks": task_rows,
            "final_bwt": task_rows[-1]["bwt"],
        },
        task_rows,
    )
    LOGGER.info("Experiment completed successfully.")


if __name__ == "__main__":
    main()
