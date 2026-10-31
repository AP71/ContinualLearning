from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from torch.utils.data import Dataset

"""Dataset view containing the classes of one CIFAR-100 task.

    The CIFAR-100 dataset must be created by the caller, so this class never
    downloads or reloads the data from disk.  Call :meth:`load_task` after a
    training phase to switch this view to the next task, then create a new
    ``DataLoader`` for the updated dataset.
"""
class CIFAR100(Dataset):
    NUM_TASKS = 10
    CLASSES_PER_TASK = 10

    def __init__(self, dataset: Dataset, task_id: int = 0, class_order: Sequence[int] | None = None) -> None:
        if not hasattr(dataset, "targets"):
            raise TypeError("dataset must expose a 'targets' attribute.")

        self.dataset = dataset
        self.class_order = tuple(
            range(self.NUM_TASKS * self.CLASSES_PER_TASK)
            if class_order is None
            else class_order
        )
        if len(self.class_order) != 100 or set(self.class_order) != set(range(100)):
            raise ValueError("class_order must be a permutation of 0..99.")

        self.indices: list[int] = []
        self.classes: tuple[int, ...] = ()
        self.task_id = -1
        self.load_task(task_id)

    """Switch to ``task_id`` and return this dataset view.
        A new ``DataLoader`` should be created after this call, especially
        when it uses worker processes.
    """
    def load_task(self, task_id: int) -> CIFAR100:
        if not 0 <= task_id < self.NUM_TASKS:
            raise ValueError(f"task_id must be between 0 and 9, got {task_id}.")

        start = task_id * self.CLASSES_PER_TASK
        self.classes = self.class_order[start : start + self.CLASSES_PER_TASK]
        class_ids = set(self.classes)
        self.indices = [index for index, target in enumerate(self.dataset.targets) if target in class_ids]
        self.task_id = task_id
        return self

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    @classmethod
    def from_classes(
        cls,
        dataset: Dataset,
        classes: Sequence[int],
        class_order: Sequence[int] | None = None,
    ) -> CIFAR100:
        """Create a dataset view containing an arbitrary set of CIFAR-100 classes."""
        view = cls.__new__(cls)
        if not hasattr(dataset, "targets"):
            raise TypeError("dataset must expose a 'targets' attribute.")
        view.dataset = dataset
        view.class_order = tuple(
            range(cls.NUM_TASKS * cls.CLASSES_PER_TASK)
            if class_order is None
            else class_order
        )
        view.indices = [
            index for index, target in enumerate(dataset.targets) if target in set(classes)
        ]
        view.classes = tuple(classes)
        view.task_id = -1
        return view
