from copy import deepcopy
from pathlib import Path

import torch


SOURCE = Path(r"../Inference/models/imagenet_reactneta_step2.h5")
DESTINATION = Path(r"./checkpoint_modificato.h5")

FLIP_PERCENTAGE = 0.50
SEED = 1234


checkpoint = torch.load(
    SOURCE,
    map_location="cpu",
    weights_only=False,
)

modified_checkpoint = deepcopy(checkpoint)

if isinstance(modified_checkpoint, dict) and "state_dict" in modified_checkpoint:
    state_dict = modified_checkpoint["state_dict"]
else:
    state_dict = modified_checkpoint

generator = torch.Generator().manual_seed(SEED)

for name, tensor in state_dict.items():
    if not isinstance(tensor, torch.Tensor):
        continue

    if not name.endswith(".weight") or not tensor.is_floating_point():
        continue

    nonzero_indices = torch.nonzero(tensor.reshape(-1) != 0).flatten()

    if nonzero_indices.numel() == 0:
        continue

    number_of_flips = max(
        1,
        int(nonzero_indices.numel() * FLIP_PERCENTAGE),
    )
    number_of_flips = min(number_of_flips, nonzero_indices.numel())

    selected = nonzero_indices[
        torch.randperm(
            nonzero_indices.numel(),
            generator=generator,
        )[:number_of_flips]
    ]

    flattened_tensor = tensor.reshape(-1).clone()
    flattened_tensor[selected] *= -1
    state_dict[name] = flattened_tensor.reshape(tensor.shape)

    print(f"{name}: modificati {number_of_flips} pesi")

torch.save(modified_checkpoint, DESTINATION)

print(f"\nCheckpoint di test salvato in:\n{DESTINATION}")