"""Joint-training upper bound for Split CIFAR-100."""

from split_cifar100_runner import run_joint


if __name__ == "__main__":
    run_joint(initialization="checkpoint")
