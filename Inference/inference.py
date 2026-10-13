import sys
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from Network.ReActNet_18_Qaw import ResNet18, ResNet34
from Network.ReActNet_A_Qaw import Reactnet

sys.path.append('../')

# Select the experiment here. The remaining settings are shared by all datasets.
DATASET = "imagenet" # "cifar100" or "imagenet"
MODEL = "resneta"  # CIFAR: "resnet18" or "resneta"; ImageNet: "resnet18", "resnet34" or "resneta"
DATASET_PATHS = {
    "cifar100": r"F:/Dataset/CIFAR-100/",
    "imagenet": r"F:/Dataset/ImageNet/",
}
DATASET_PATH = DATASET_PATHS[DATASET]
BATCH_SIZE = 256
WORKERS = 8
SEED = 2023
MODEL_PATH = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_CONFIG = {
    "cifar100": {
        "classes": 100,
        "imagenet": False,
        "checkpoint": {
            "resnet18": "cifar100_reactnet18_step2.h5",
            "resneta": "cifar100_reactneta_step2.h5",
        },
    },
    "imagenet": {
        "classes": 1000,
        "imagenet": True,
        "checkpoint": {
            "resnet18": "imagenet_reactnet18_step2.h5",
            "resnet34": "imagenet_reactnet34_step2.h5",
            "resneta": "imagenet_reactneta_step2.h5",
        },
    },
}

def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_test_transform():
    if DATASET_CONFIG[DATASET]["imagenet"]:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )

def create_test_dataset():
    test_transform = get_test_transform()
    if DATASET == "cifar100":
        return datasets.CIFAR100(
            root=DATASET_PATH,
            train=False,
            download=False,
            transform=test_transform,
        )
    return datasets.ImageNet(
        root=DATASET_PATH,
        split="val",
        transform=test_transform,
    )

def create_model():
    config = DATASET_CONFIG[DATASET]
    classes = config["classes"]

    if MODEL == "resnet18":
        model = ResNet18(num_classes=classes, imagenet=config["imagenet"])
    elif MODEL == "resnet34" and DATASET == "imagenet":
        model = ResNet34(num_classes=classes, imagenet=True)
    elif MODEL == "resneta":
        model = Reactnet(num_classes=classes, imagenet=config["imagenet"])
    else:
        raise ValueError(f"Unsupported model '{MODEL}' for dataset '{DATASET}'.")

    checkpoint_name = config["checkpoint"].get(MODEL)
    if MODEL_PATH:
        checkpoint_path = Path(MODEL_PATH)
    else:
        checkpoint_path = SCRIPT_DIR / "models" / checkpoint_name

    # Path.is_file() verifica che il checkpoint esista davvero.
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if DEVICE.type == "cuda":
        model = torch.nn.DataParallel(model)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    return model.to(DEVICE)


def evaluate(model, data_loader):
    classes = DATASET_CONFIG[DATASET]["classes"]
    confusion_matrix = torch.zeros((classes, classes), dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            predictions = model(images).argmax(1)
            indices = labels * classes + predictions
            confusion_matrix += torch.bincount(
                indices, minlength=classes * classes
            ).reshape(classes, classes).cpu()

    accuracy = confusion_matrix.diag().sum().float() / confusion_matrix.sum()
    print(f"Confusion_Matrix = \n{confusion_matrix}")
    print(f"acc = {accuracy.item()}")


def main():
    if DATASET not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{DATASET}'.")
    if MODEL not in DATASET_CONFIG[DATASET]["checkpoint"]:
        raise ValueError(f"Unsupported model '{MODEL}' for dataset '{DATASET}'.")

    set_seed()
    test_data = create_test_dataset()
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=DEVICE.type == "cuda",
        drop_last=False,
    )
    model = create_model()
    evaluate(model, test_data_loader)


if __name__ == "__main__":
    main()
