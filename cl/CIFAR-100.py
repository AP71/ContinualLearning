"""Checkpoint-initialized sequential Split CIFAR-100 experiment."""

from split_cifar100_runner import run_sequential


if __name__ == "__main__":
    run_sequential(initialization="checkpoint")
