"""Reusable standard and task-incremental Split CIFAR-100 experiments."""

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
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Network.ReActNet_18_Qaw import ResNet18
from Network.ReActNet_A_Qaw import Reactnet
from datasets.CIFAR100 import CIFAR100
from utils.checkpoint_analysis import (
    compare_checkpoints,
    plot_latent_distributions,
)
from utils.metrics import (
    calculate_bwt,
    plot_accuracy_history,
    plot_accuracy_vs_classes,
    plot_matrix,
    plot_task_metric,
)

DATASET_PATH = Path(r"F:/Dataset/CIFAR-100/")
CHECKPOINT_PATH: Path | None = None
MODEL_NAME = "resneta"
OUTPUT_ROOT = ROOT / "cl" / "results"
BATCH_SIZE = 256
WORKERS = 0
EPOCHS_PER_TASK = 10
JOINT_EPOCHS = 100
JOINT_EVAL_INTERVAL = 5
JOINT_EARLY_STOPPING_PATIENCE = 5
JOINT_EARLY_STOPPING_MIN_DELTA = 1e-3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
SEED = 2023
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger("split_cifar100")
torch.set_float32_matmul_precision("high")
_CLASS_TENSOR_CACHE: dict[tuple[int, ...], torch.Tensor] = {}


# Reproducibility and data preparation

def set_seed() -> None:
    """Configure deterministic random generators for reproducible experiments."""
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



def create_datasets() -> tuple[CIFAR100, CIFAR100]:
    """Load CIFAR-100 and expose it through the split-task dataset wrapper."""
    train_transform, test_transform = create_transforms()
    train_data = datasets.CIFAR100(
        DATASET_PATH, train=True, download=False, transform=train_transform
    )
    test_data = datasets.CIFAR100(
        DATASET_PATH, train=False, download=False, transform=test_transform
    )
    return CIFAR100(train_data), CIFAR100(test_data)


def create_model(initialization: str) -> nn.Module:
    """Create the selected network and optionally load compatible ImageNet weights."""
    if MODEL_NAME == "resneta":
        model = Reactnet(num_classes=100, imagenet=False)
        checkpoint_name = "imagenet_reactneta_step2.h5"
    elif MODEL_NAME == "resnet18":
        model = ResNet18(num_classes=100, imagenet=False)
        checkpoint_name = "imagenet_resnet18_step2.h5"
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    if initialization not in {"checkpoint", "random"}:
        raise ValueError("initialization must be 'checkpoint' or 'random'.")
    if initialization == "checkpoint":
        checkpoint = CHECKPOINT_PATH or ROOT / "inference" / "models" / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {
            name.removeprefix("module."): tensor
            for name, tensor in state_dict.items()
        }
        model_state = model.state_dict()
        compatible_state = {}
        skipped_keys = []
        for name, tensor in state_dict.items():
            if name not in model_state or model_state[name].shape != tensor.shape:
                skipped_keys.append(name)
            else:
                compatible_state[name] = tensor
        model.load_state_dict(compatible_state, strict=False)
        LOGGER.info(
            "Loaded %d compatible tensors from %s; skipped %d tensors "
            "(including the ImageNet classifier when dimensions differ).",
            len(compatible_state),
            checkpoint,
            len(skipped_keys),
        )
    return model.to(DEVICE)


def create_loader(dataset, shuffle: bool) -> DataLoader:
    """Create a DataLoader with the experiment-wide data-loading settings."""
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=WORKERS,
        pin_memory=DEVICE.type == "cuda",
        drop_last=False,
    )


# Model creation and training

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    description: str,
    scaler: torch.amp.GradScaler,
) -> float:
    """Run one cross-entropy training epoch and return its mean loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    samples = 0
    progress = tqdm(loader, desc=description)
    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda",
        ):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * labels.size(0)
        samples += labels.size(0)
        progress.set_postfix(loss=f"{loss_sum / samples:.4f}")
    return loss_sum / samples if samples else 0.0


def copy_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU copy of all model parameters."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    **state: object,
) -> None:
    """Save model, optimizer, and experiment state for resumable training."""
    checkpoint = {
        "model_state": copy_model_state(model),
        "optimizer_state": optimizer.state_dict(),
        **state,
    }
    torch.save(checkpoint, path)


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, object]:
    """Restore a resumable checkpoint and return its experiment state."""
    if not path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid training checkpoint: {path}")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


# Evaluation and result serialization

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    allowed_classes: torch.Tensor | tuple[int, ...] | None = None,
) -> float:
    """Measure top-1 accuracy, optionally restricting predictions to task classes."""
    model.eval()
    correct = 0
    total = 0
    if isinstance(allowed_classes, torch.Tensor):
        class_tensor = allowed_classes
    elif allowed_classes is not None:
        class_tensor = _CLASS_TENSOR_CACHE.get(allowed_classes)
        if class_tensor is None:
            class_tensor = torch.tensor(allowed_classes, device=DEVICE)
            _CLASS_TENSOR_CACHE[allowed_classes] = class_tensor
    else:
        class_tensor = None
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            if class_tensor is not None:
                logits = logits.index_select(1, class_tensor)
                predictions = class_tensor[logits.argmax(1)]
            else:
                predictions = logits.argmax(1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
    return correct / total if total else 0.0


def evaluate_task_accuracies(
    model: nn.Module,
    loader: DataLoader,
    class_order: tuple[int, ...],
) -> list[float]:
    """Measure unrestricted accuracy for every task in one test-set pass."""
    model.eval()
    num_tasks = len(class_order) // CIFAR100.CLASSES_PER_TASK
    class_to_task = torch.empty(len(class_order), dtype=torch.long, device=DEVICE)
    for task_id in range(num_tasks):
        start = task_id * CIFAR100.CLASSES_PER_TASK
        task_classes = class_order[start : start + CIFAR100.CLASSES_PER_TASK]
        class_to_task[list(task_classes)] = task_id

    correct_by_task = torch.zeros(num_tasks, dtype=torch.long, device=DEVICE)
    total_by_task = torch.zeros(num_tasks, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            predictions = model(images).argmax(1)
            task_ids = class_to_task[labels]
            correct = predictions == labels
            total_by_task += torch.bincount(task_ids, minlength=num_tasks)
            correct_by_task += torch.bincount(
                task_ids[correct], minlength=num_tasks
            )
    return [
        (
            correct_by_task[task_id].item() / total_by_task[task_id].item()
            if total_by_task[task_id].item()
            else 0.0
        )
        for task_id in range(num_tasks)
    ]


def evaluate_confusion_matrix(
    model: nn.Module, loader: DataLoader, num_classes: int
) -> np.ndarray:
    """Return the confusion matrix for an unrestricted 100-class evaluation."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            predictions = model(images).argmax(1).cpu()
            indices = labels * num_classes + predictions
            matrix += (
                torch.bincount(indices, minlength=num_classes * num_classes)
                .reshape(num_classes, num_classes)
                .numpy()
            )
    return matrix


def save_sequential_results(
    output_dir: Path,
    initialization: str,
    standard_matrix: list[list[float | None]],
    task_matrix: list[list[float | None]],
    history: list[dict],
    accuracy_history: list[list[float]],
    bit_flips: list[int],
    bit_flip_percentages: list[float],
    total_binary_bits: int | None,
    loss_history: list[float],
    confusion_matrix: np.ndarray,
) -> None:
    """Persist sequential metrics and generate all sequential experiment plots."""
    bwt_values = [
        calculate_bwt(standard_matrix[: task_id + 1])
        for task_id in range(len(standard_matrix))
    ]
    classes_seen = [
        (task_id + 1) * 10 for task_id in range(len(history))
    ]
    class_incremental_accuracy = [
        100.0 * row["standard_seen_accuracy"] for row in history
    ]
    plot_points = {
        "bitflips_per_task": {
            "x": list(range(len(bit_flips))),
            "y": bit_flips,
            "x_label": "Task",
            "y_label": "Bit flips",
        },
        "bwt_per_task": {
            "x": list(range(len(bwt_values))),
            "y": bwt_values,
            "x_label": "Task",
            "y_label": "BWT",
        },
        "accuracy_vs_classes": {
            "x": classes_seen,
            "y": class_incremental_accuracy,
            "x_label": "Number of seen classes",
            "y_label": "Accuracy (%)",
        },
        "loss_per_epoch": {
            "x": list(range(1, len(loss_history) + 1)),
            "y": loss_history,
            "x_label": "Epoch",
            "y_label": "Loss",
        },
    }
    results = {
        "schema_version": 1,
        "protocol": "Split CIFAR-100 class-incremental",
        "experiment_type": "sequential",
        "initialization": initialization,
        "model_name": MODEL_NAME,
        "num_tasks": 10,
        "classes_per_task": 10,
        "epochs_completed": len(loss_history),
        "total_epochs": 10 * EPOCHS_PER_TASK,
        "loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "standard_class_incremental": standard_matrix,
        "task_incremental": task_matrix,
        "history": history,
        "accuracy_history": accuracy_history,
        "accuracy_epochs": list(range(1, len(loss_history) + 1)),
        "best_accuracy": None,
        "best_epoch": None,
        "early_stopping": {"enabled": False, "evaluation_interval": None, "patience": None, "min_delta": None},
        "standard_class_incremental_accuracy": (
            history[-1]["standard_seen_accuracy"] if history else None
        ),
        "task_incremental_accuracy": (
            float(np.mean(task_matrix[-1][: len(history)]))
            if history
            else None
        ),
        "final_accuracy_percent": (
            100.0 * history[-1]["standard_seen_accuracy"] if history else None
        ),
        "confusion_matrix": confusion_matrix.tolist(),
        "r_matrix": standard_matrix,
        "task_incremental_r_matrix": task_matrix,
        "bit_flips": bit_flips,
        "bit_flip_percentages": bit_flip_percentages,
        "total_binary_bits": total_binary_bits,
        "bit_flip_percentage": (
            float(sum(bit_flips) / (total_binary_bits * len(bit_flips)) * 100.0)
            if total_binary_bits and bit_flips
            else None
        ),
        "bwt": bwt_values,
        "accuracy_vs_classes": {
            "classes_seen": classes_seen,
            "accuracy_percent": class_incremental_accuracy,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "plot_points.json").write_text(
        json.dumps(plot_points, indent=2), encoding="utf-8"
    )
    plot_matrix(
        standard_matrix,
        "Standard class-incremental accuracy",
        output_dir / "standard_r_matrix.png",
        xlabel="Evaluated task",
        ylabel="Training task",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    plot_matrix(
        task_matrix,
        "Task-incremental accuracy (masked logits)",
        output_dir / "task_incremental_r_matrix.png",
        xlabel="Evaluated task",
        ylabel="Training task",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    plot_accuracy_history(
        accuracy_history,
        output_dir / "accuracy_per_epoch.png",
        task_start_epochs=None,
    )
    plot_task_metric(
        bwt_values,
        "BWT (%)",
        "Backward transfer per task",
        output_dir / "bwt_per_task.png",
        connect_points=True,
        percentage_axis=True,
    )
    plot_task_metric(
        bit_flips,
        "Bit flips",
        "Binary bit flips per task",
        output_dir / "bitflips_per_task.png",
        connect_points=True,
    )
    plot_accuracy_vs_classes(
        class_incremental_accuracy,
        classes_seen,
        output_dir / "accuracy_vs_classes.png",
    )


def save_checkpoint_comparison(
    checkpoint_a: Path, checkpoint_b: Path, output_dir: Path
) -> dict:
    """Compare two checkpoints and save the latent-distribution plot."""
    result = compare_checkpoints(checkpoint_a, checkpoint_b)
    plot_latent_distributions(result, output_dir / "latent_distributions.png")
    return result


# Experiment entry points

def run_sequential(
    initialization: str = "checkpoint",
    resume_from: Path | str | None = None,
) -> Path:
    """Run sequential fine-tuning, optionally resuming from ``last.h5``."""
    set_seed()
    train_tasks, test_tasks = create_datasets()
    test_task_loaders = [
        create_loader(CIFAR100(test_tasks.dataset, task_id), False)
        for task_id in range(train_tasks.NUM_TASKS)
    ]
    model = create_model(initialization)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    standard_matrix = [[None] * 10 for _ in range(10)]
    task_matrix = [[None] * 10 for _ in range(10)]
    history = []
    accuracy_history = [[] for _ in range(10)]
    bit_flips = []
    bit_flip_percentages = []
    total_binary_bits = None
    loss_history = []
    resume_path = Path(resume_from) if resume_from is not None else None
    output_dir = (
        resume_path.parent.parent
        if resume_path is not None
        else OUTPUT_ROOT / (
            f"cifar100_sequential_{initialization}_{MODEL_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
        )
    )
    checkpoint_dir = output_dir / "checkpoints"
    compare_dir = output_dir / "compare"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    initial_checkpoint = checkpoint_dir / "initial.h5"
    last_checkpoint = checkpoint_dir / "last.h5"
    start_task = 0
    start_epoch = 0
    previous_checkpoint = initial_checkpoint
    if resume_path is None:
        torch.save(copy_model_state(model), initial_checkpoint)
    else:
        state = load_training_checkpoint(resume_path, model, optimizer, scaler)
        start_task = int(state["next_task"])
        start_epoch = int(state["next_epoch"])
        previous_checkpoint = Path(state["previous_checkpoint"])
        standard_matrix = state["standard_matrix"]
        task_matrix = state["task_matrix"]
        history = state["history"]
        accuracy_history = state["accuracy_history"]
        bit_flips = state["bit_flips"]
        bit_flip_percentages = state.get("bit_flip_percentages", [])
        total_binary_bits = state.get("total_binary_bits")
        loss_history = state.get("loss_history", [])
    for task_id in range(start_task, train_tasks.NUM_TASKS):
        train_tasks.load_task(task_id)
        train_loader = create_loader(train_tasks, True)
        loss = 0.0
        first_epoch = start_epoch if task_id == start_task else 0
        for epoch in range(first_epoch, EPOCHS_PER_TASK):
            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                f"Task {task_id + 1}/10, epoch {epoch + 1}/{EPOCHS_PER_TASK}",
                scaler,
            )
            loss_history.append(loss)
            current_task_accuracy = evaluate(
                model,
                test_task_loaders[task_id],
                tuple(test_tasks.class_order[task_id * 10 : (task_id + 1) * 10]),
            )
            for evaluated_id in range(train_tasks.NUM_TASKS):
                accuracy_history[evaluated_id].append(
                    current_task_accuracy if evaluated_id == task_id else 0.0
                )
            save_training_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                next_task=task_id,
                next_epoch=epoch + 1,
                initialization=initialization,
                previous_checkpoint=str(previous_checkpoint),
                standard_matrix=standard_matrix,
                task_matrix=task_matrix,
                history=history,
                accuracy_history=accuracy_history,
                bit_flips=bit_flips,
                bit_flip_percentages=bit_flip_percentages,
                total_binary_bits=total_binary_bits,
                loss_history=loss_history,
                scaler_state=scaler.state_dict(),
            )
        current_checkpoint = checkpoint_dir / f"after_task_{task_id + 1:02d}.h5"
        torch.save(copy_model_state(model), current_checkpoint)
        comparison = save_checkpoint_comparison(
            previous_checkpoint,
            current_checkpoint,
            compare_dir / f"task_{task_id + 1:02d}",
        )
        bit_flips.append(comparison["binary_summary"]["bit_flips"])
        bit_flip_percentages.append(comparison["binary_summary"]["bit_flip_percentage"])
        total_binary_bits = comparison["binary_summary"]["total_bits"]
        previous_checkpoint = current_checkpoint
        seen_tasks = [
            CIFAR100(test_tasks.dataset, evaluated_id)
            for evaluated_id in range(task_id + 1)
        ]
        seen_loader = create_loader(ConcatDataset(seen_tasks), False)
        standard_accuracy = evaluate(model, seen_loader)
        for evaluated_id in range(task_id + 1):
            standard_matrix[task_id][evaluated_id] = evaluate(
                model, test_task_loaders[evaluated_id]
            )
            task_matrix[task_id][evaluated_id] = evaluate(
                model,
                test_task_loaders[evaluated_id],
                tuple(test_tasks.class_order[evaluated_id * 10 : (evaluated_id + 1) * 10]),
            )
        history.append(
            {
                "task": task_id,
                "loss": loss,
                "standard_seen_accuracy": standard_accuracy,
                "task_incremental_seen_accuracy": float(
                    np.mean(task_matrix[task_id][: task_id + 1])
                ),
            }
        )
        save_training_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            next_task=task_id + 1,
            next_epoch=0,
            initialization=initialization,
            previous_checkpoint=str(previous_checkpoint),
            standard_matrix=standard_matrix,
            task_matrix=task_matrix,
            history=history,
            accuracy_history=accuracy_history,
            bit_flips=bit_flips,
            bit_flip_percentages=bit_flip_percentages,
            total_binary_bits=total_binary_bits,
            loss_history=loss_history,
            scaler_state=scaler.state_dict(),
        )
        LOGGER.info(
            "Task %d: standard=%.4f, task-incremental=%.4f.",
            task_id + 1,
            standard_accuracy,
            history[-1]["task_incremental_seen_accuracy"],
        )
    save_sequential_results(
        output_dir,
        initialization,
        standard_matrix,
        task_matrix,
        history,
        accuracy_history,
        bit_flips,
        bit_flip_percentages,
        total_binary_bits,
        loss_history,
        evaluate_confusion_matrix(model, create_loader(test_tasks.dataset, False), 100),
    )
    return output_dir


def run_joint(
    initialization: str = "checkpoint",
    resume_from: Path | str | None = None,
) -> Path:
    """Run joint fine-tuning, optionally resuming from ``last.h5``."""
    set_seed()
    train_tasks, test_tasks = create_datasets()
    model = create_model(initialization)
    initial_state = copy_model_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    train_loader = create_loader(train_tasks.dataset, True)
    test_loader = create_loader(test_tasks.dataset, False)
    resume_path = Path(resume_from) if resume_from is not None else None
    output_dir = (
        resume_path.parent.parent
        if resume_path is not None
        else OUTPUT_ROOT / (
            f"cifar100_joint_{initialization}_{MODEL_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
        )
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint_dir / "last.h5"
    loss = 0.0
    loss_history = []
    accuracy_history = []
    accuracy_epochs = []
    best_accuracy = -float("inf")
    best_state = copy_model_state(model)
    epochs_without_improvement = 0
    best_epoch = 0
    epochs_completed = 0
    start_epoch = 0
    if resume_path is not None:
        state = load_training_checkpoint(resume_path, model, optimizer, scaler)
        start_epoch = int(state["next_epoch"])
        loss = float(state["loss"])
        loss_history = state.get("loss_history", [])
        accuracy_history = state["accuracy_history"]
        accuracy_epochs = state["accuracy_epochs"]
        best_accuracy = float(state["best_accuracy"])
        best_state = state["best_state"]
        epochs_without_improvement = int(state["epochs_without_improvement"])
        best_epoch = int(state["best_epoch"])
        epochs_completed = int(state["epochs_completed"])
        initial_state = state["initial_state"]
    for epoch in range(start_epoch, JOINT_EPOCHS):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            f"Joint epoch {epoch + 1}/{JOINT_EPOCHS}",
            scaler,
        )
        loss_history.append(loss)
        epoch_number = epoch + 1
        epochs_completed = epoch_number
        should_evaluate = (
            epoch_number == 1
            or epoch_number % JOINT_EVAL_INTERVAL == 0
            or epoch_number == JOINT_EPOCHS
        )
        if not should_evaluate:
            save_training_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                next_epoch=epoch_number,
                loss=loss,
                initialization=initialization,
                accuracy_history=accuracy_history,
                accuracy_epochs=accuracy_epochs,
                loss_history=loss_history,
                best_accuracy=best_accuracy,
                best_state=best_state,
                epochs_without_improvement=epochs_without_improvement,
                best_epoch=best_epoch,
                epochs_completed=epochs_completed,
                initial_state=initial_state,
                scaler_state=scaler.state_dict(),
            )
            continue

        validation_accuracy = evaluate(model, test_loader)
        accuracy_history.append(validation_accuracy)
        accuracy_epochs.append(epoch_number)
        if validation_accuracy > best_accuracy + JOINT_EARLY_STOPPING_MIN_DELTA:
            best_accuracy = validation_accuracy
            best_state = copy_model_state(model)
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        LOGGER.info(
            "Joint epoch %d/%d: loss=%.6f, accuracy=%.4f, best=%.4f, "
            "patience=%d/%d.",
            epoch + 1,
            JOINT_EPOCHS,
            loss,
            validation_accuracy,
            best_accuracy,
            epochs_without_improvement,
            JOINT_EARLY_STOPPING_PATIENCE,
        )
        save_training_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            next_epoch=epoch_number,
            loss=loss,
            initialization=initialization,
            accuracy_history=accuracy_history,
            accuracy_epochs=accuracy_epochs,
            loss_history=loss_history,
            best_accuracy=best_accuracy,
            best_state=best_state,
            epochs_without_improvement=epochs_without_improvement,
            best_epoch=best_epoch,
            epochs_completed=epochs_completed,
            initial_state=initial_state,
            scaler_state=scaler.state_dict(),
        )
        if epochs_without_improvement >= JOINT_EARLY_STOPPING_PATIENCE:
            LOGGER.info(
                "Early stopping joint training at epoch %d; best epoch was %d.",
                epoch_number,
                best_epoch,
            )
            break
    model.load_state_dict(best_state)
    standard_accuracy = evaluate(model, test_loader)
    task_incremental_accuracy = float(
        np.mean(
            [
                evaluate(
                    model,
                    create_loader(CIFAR100(test_tasks.dataset, task_id), False),
                    tuple(test_tasks.class_order[task_id * 10 : (task_id + 1) * 10]),
                )
                for task_id in range(train_tasks.NUM_TASKS)
            ]
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_checkpoint = checkpoint_dir / "initial.h5"
    final_checkpoint = checkpoint_dir / "final.h5"
    torch.save(initial_state, initial_checkpoint)
    torch.save(copy_model_state(model), final_checkpoint)
    comparison = save_checkpoint_comparison(
        initial_checkpoint, final_checkpoint, output_dir / "compare"
    )
    confusion_matrix = evaluate_confusion_matrix(model, test_loader, 100)
    standard_r_matrix = [[standard_accuracy]]
    task_r_matrix = [[task_incremental_accuracy]]
    accuracy_percent = [100.0 * value for value in accuracy_history]
    plot_accuracy_history(
        [accuracy_history],
        output_dir / "accuracy_per_epoch.png",
        title="Joint-training accuracy per epoch",
        epoch_points=accuracy_epochs,
    )
    plot_task_metric(
        [standard_accuracy],
        "Accuracy (%)",
        "Final class-incremental accuracy",
        output_dir / "final_accuracy.png",
        percentage_axis=True,
        point_labels=[100.0 * standard_accuracy],
        label_format="{:.2f}%",
    )
    plot_task_metric(
        [comparison["binary_summary"]["bit_flip_percentage"] / 100.0],
        "Bit flips (%)",
        "Final binary bit-flip percentage",
        output_dir / "bitflip_percentage.png",
        percentage_axis=True,
        point_labels=[comparison["binary_summary"]["bit_flip_percentage"]],
        label_format="{:.4f}%",
    )
    plot_task_metric(
        [comparison["binary_summary"]["bit_flips"]],
        "Bit flips",
        "Final binary bit flips",
        output_dir / "bitflips.png",
        point_labels=[comparison["binary_summary"]["bit_flips"]],
        label_format="{:.0f}",
    )
    plot_matrix(
        confusion_matrix,
        "Joint-training confusion matrix",
        output_dir / "confusion_matrix.png",
        xlabel="Predicted class",
        ylabel="True class",
        show_ticks=False,
        cmap="Blues",
    )
    plot_matrix(
        standard_r_matrix,
        "Joint-training R matrix",
        output_dir / "r_matrix.png",
        xlabel="Evaluated task",
        ylabel="Training task",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    plot_matrix(
        task_r_matrix,
        "Joint-training task-incremental R matrix",
        output_dir / "task_incremental_r_matrix.png",
        xlabel="Evaluated task",
        ylabel="Training task",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    plot_points = {
        "accuracy_per_epoch": {
            "x": accuracy_epochs,
            "y": accuracy_percent,
            "x_label": "Epoch",
            "y_label": "Accuracy (%)",
        },
        "final_accuracy": {
            "x": [100],
            "y": [100.0 * standard_accuracy],
            "x_label": "Number of classes",
            "y_label": "Accuracy (%)",
        },
        "bitflips": {
            "x": [0],
            "y": [comparison["binary_summary"]["bit_flips"]],
            "x_label": "Checkpoint transition",
            "y_label": "Bit flips",
        },
        "bitflip_percentage": {
            "x": [0],
            "y": [comparison["binary_summary"]["bit_flip_percentage"]],
            "x_label": "Checkpoint transition",
            "y_label": "Bit flips (%)",
        },
        "loss_per_epoch": {
            "x": list(range(1, len(loss_history) + 1)),
            "y": loss_history,
            "x_label": "Epoch",
            "y_label": "Loss",
        },
        "accuracy_vs_classes": {
            "x": [100],
            "y": [100.0 * standard_accuracy],
            "x_label": "Number of seen classes",
            "y_label": "Accuracy (%)",
        },
    }
    results = {
        "schema_version": 1,
        "protocol": "Split CIFAR-100 joint-training upper bound",
        "experiment_type": "joint",
        "initialization": initialization,
        "model_name": MODEL_NAME,
        "num_tasks": 10,
        "classes_per_task": 10,
        "total_epochs": JOINT_EPOCHS,
        "loss": loss,
        "loss_history": loss_history,
        "standard_class_incremental_accuracy": standard_accuracy,
        "task_incremental_accuracy": task_incremental_accuracy,
        "accuracy_history": accuracy_history,
        "accuracy_epochs": accuracy_epochs,
        "history": [],
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "epochs_completed": epochs_completed,
        "early_stopping": {
            "enabled": True,
            "evaluation_interval": JOINT_EVAL_INTERVAL,
            "patience": JOINT_EARLY_STOPPING_PATIENCE,
            "min_delta": JOINT_EARLY_STOPPING_MIN_DELTA,
        },
        "final_accuracy_percent": 100.0 * standard_accuracy,
        "bit_flips": [comparison["binary_summary"]["bit_flips"]],
        "total_binary_bits": comparison["binary_summary"]["total_bits"],
        "bit_flip_percentage": comparison["binary_summary"]["bit_flip_percentage"],
        "bit_flip_percentages": [
            comparison["binary_summary"]["bit_flip_percentage"]
        ],
        "bwt": [0.0],
        "standard_class_incremental": standard_r_matrix,
        "task_incremental": task_r_matrix,
        "r_matrix": standard_r_matrix,
        "task_incremental_r_matrix": task_r_matrix,
        "confusion_matrix": confusion_matrix.tolist(),
        "accuracy_vs_classes": {
            "classes_seen": [100],
            "accuracy_percent": [100.0 * standard_accuracy],
        },
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "plot_points.json").write_text(
        json.dumps(plot_points, indent=2), encoding="utf-8"
    )
    return output_dir


def resume_sequential(last_checkpoint: Path | str) -> Path:
    """Resume a sequential run from its ``checkpoints/last.h5`` file."""
    checkpoint = Path(last_checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoints" / "last.h5"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return run_sequential(
        initialization=state.get("initialization", "checkpoint"),
        resume_from=checkpoint,
    )


def resume_joint(last_checkpoint: Path | str) -> Path:
    """Resume a joint run from its ``checkpoints/last.h5`` file."""
    checkpoint = Path(last_checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoints" / "last.h5"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return run_joint(
        initialization=state.get("initialization", "checkpoint"),
        resume_from=checkpoint,
    )

if __name__ == "__main__":
    run_sequential("checkpoint")
    # run_sequential("random")
    # run_joint("checkpoint")
    # run_joint("random")