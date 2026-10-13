import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.checkpoint_analysis import (
    compare_checkpoints,
    print_comparison,
)

result = compare_checkpoints(
    ROOT_DIR / "Inference" / "models" / "imagenet_reactneta_step2.h5",
    Path(__file__).resolve().parent / "checkpoint_modificato.h5",
)

print_comparison(result)