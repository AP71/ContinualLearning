import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from avalanche.benchmarks import nc_benchmark
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from Network.ReActNet_A_Qaw import Reactnet


def evaluate_dataset(model: nn.Module, dataset, device: torch.device, allowed_classes: torch.Tensor | None = None, batch_size: int = 256, num_workers: int = 0) -> float:
    """Valuta l'accuratezza su un dataset, con opzione di logit masking per task-incremental."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    is_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            # Gestisce sia batch a 2 elementi (PyTorch standard) sia a 3 elementi (Avalanche con task_id)
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # FIX per iCaRL: Bypass del classificatore NCM durante le epoche
            # Utilizza il layer lineare di addestramento invece delle medie di classe non ancora calcolate
            if hasattr(model, 'feature_extractor') and hasattr(model, 'train_classifier'):
                features = model.feature_extractor(images)
                logits = model.train_classifier(features)
            else:
                logits = model(images)

            if allowed_classes is not None:
                logits = logits.index_select(1, allowed_classes)
                preds = allowed_classes[logits.argmax(1)]
            else:
                preds = logits.argmax(1)

            correct += int((preds == labels).sum().item())
            total += labels.numel()
    if is_training:
        model.train()

    return correct / total if total else 0.0

def evaluate_confusion_matrix(model: nn.Module, full_test_dataset, device: torch.device, batch_size: int = 256, num_workers: int = 0) -> np.ndarray:
    """Calcola la matrice di confusione 100x100 su tutto il test set CIFAR-100."""
    loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    matrix = np.zeros((100, 100), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            preds = model(images).argmax(1).cpu()
            indices = labels * 100 + preds
            matrix += (
                torch.bincount(indices, minlength=10000)
                .reshape(100, 100)
                .numpy()
            )
    return matrix

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_model(initialization: str, device: torch.device, log: logging.Logger, checkpoint_path: Path, root: Path) -> nn.Module:
    model = Reactnet(num_classes=100, imagenet=False)
    checkpoint_name = "imagenet_reactneta_step2.h5"

    if initialization == "checkpoint":
        checkpoint = checkpoint_path or root / "inference" / "models" / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
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
        log.info("Loaded %d compatible tensors from %s", len(compatible_state), checkpoint)
    return model.to(device)

class SubsetWithTargets(Subset):
    """
    Wrapper per Subset di PyTorch.
    Avalanche richiede esplicitamente che il dataset esponga l'attributo '.targets'
    per poter dividere correttamente le classi nelle experience.
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = [dataset.targets[i] for i in indices]


def create_benchmark(dataset_path: Path, seed: int, is_joint: bool = False):
    """
    Crea il benchmark CIFAR-100 per Avalanche, estraendo un Validation Set
    perfettamente bilanciato (50 immagini per classe).

    Args:
        dataset_path: Percorso del dataset CIFAR-100.
        seed: Seme per la riproducibilità dello split.
        is_joint: Se True, crea 1 singola experience (Joint). Se False, 10 experience (Sequential).
    """
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

    # 1. Carica i dataset originali
    # IMPORTANTE: Il Validation Set deve usare il test_transform (nessuna augmentation)
    full_train_data = datasets.CIFAR100(dataset_path, train=True, download=False, transform=train_transform)
    full_val_data_base = datasets.CIFAR100(dataset_path, train=True, download=False, transform=test_transform)
    test_data = datasets.CIFAR100(dataset_path, train=False, download=False, transform=test_transform)

    # 2. Split stratificato: esattamente 50 immagini per classe per il Validation
    targets = np.array(full_train_data.targets)
    train_indices = []
    val_indices = []

    rng = np.random.default_rng(seed)
    for c in range(100):
        c_idx = np.where(targets == c)[0]
        rng.shuffle(c_idx)  # Usa il seed per garantire riproducibilità
        val_indices.extend(c_idx[:50])  # Le prime 50 alla validazione
        train_indices.extend(c_idx[50:])  # Le restanti 450 al training

    # 3. Applica i wrapper per mantenere i targets visibili ad Avalanche
    train_subset = SubsetWithTargets(full_train_data, train_indices)
    val_subset = SubsetWithTargets(full_val_data_base, val_indices)

    # 4. Creazione del benchmark Avalanche
    n_experiences = 1 if is_joint else 10

    benchmark = nc_benchmark(
        train_dataset=train_subset,
        test_dataset=test_data,
        n_experiences=n_experiences,
        task_labels=False,  # Class-incremental
        shuffle=False,  # False mantiene l'ordine fisso dei task (Split CIFAR-100)
        seed=seed,
    )

    return benchmark, val_subset