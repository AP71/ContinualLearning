import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.checkpoint_analysis import (
    compare_checkpoints,
    plot_latent_distributions,
    print_comparison,
    save_comparison,
)

def resolve_checkpoint(path: Path) -> Path:
    """Resolve repository-relative paths written with a leading slash."""
    if path.exists():
        return path

    path_text = str(path)
    if path_text.startswith(("/", "\\")):
        repository_relative = ROOT_DIR / path_text.lstrip("/\\")
        if repository_relative.exists():
            return repository_relative

    raise FileNotFoundError(
        f"Checkpoint not found: {path}\n"
        f"Use an absolute Windows path or a path relative to {ROOT_DIR}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two checkpoints layer by layer."
    )
    parser.add_argument(
        "checkpoint_a",
        nargs="?",
        type=Path,
        default=ROOT_DIR / "inference" / "models" / "imagenet_reactneta_step2.h5",
    )
    parser.add_argument(
        "checkpoint_b",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint_modificato.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("risultati"),
    )
    parser.add_argument(
        "--distribution-sample-size",
        type=int,
        default=10_000,
        help="Maximum number of latent magnitudes retained per checkpoint and layer.",
    )
    args = parser.parse_args()

    checkpoint_a = resolve_checkpoint(args.checkpoint_a)
    checkpoint_b = resolve_checkpoint(args.checkpoint_b)
    if args.distribution_sample_size <= 0:
        parser.error("--distribution-sample-size must be positive")
    result = compare_checkpoints(
        checkpoint_a,
        checkpoint_b,
        distribution_sample_size=args.distribution_sample_size,
    )
    print_comparison(result)
    args.output.mkdir(parents=True, exist_ok=True)
    save_comparison(result, args.output / "comparison.json")
    plot_latent_distributions(result, args.output / "latent_distributions.png")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()