"""Random-initialization Split CIFAR-100 reference runs."""

from split_cifar100_runner import run_sequential, run_joint


if __name__ == "__main__":
    run_sequential(initialization="random")
