"""CIFAR-100 loaders that return BOTH labels: (image, coarse, fine).

Contrast with the upstream repo: no PNG dump, no CSV, no cv2, no dead resize.
torchvision downloads once to ./data, and we attach the coarse label via the
canonical map in hierarchy.py.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from hierarchy import SPARSE2COARSE

# Per-channel mean and standard deviation of the CIFAR-100 training set.
_MEAN = (0.5071, 0.4866, 0.4409)
_STD = (0.2673, 0.2564, 0.2762)


def _train_tf() -> transforms.Compose:
    """Augmentation + standardization for training.

    Normalize applies (x - mean) / std per channel, i.e. an affine map with a
    diagonal matrix. Diagonal matters: each channel is rescaled independently, so
    no colour mixing happens -- we are only recentring the input cloud on the
    origin and making its per-axis spread ~1. Gradient descent cares because the
    curvature of the loss along an input direction scales with that input's
    magnitude; leaving channels on different scales gives the Hessian a large
    condition number, and the step size that is stable for the widest direction
    then crawls along the others.

    Crop and flip come BEFORE ToTensor because they are PIL-level ops, and the
    order is deliberate: augment the raw image, then convert, then standardize
    with the statistics that describe the raw data.
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _eval_tf() -> transforms.Compose:
    """Standardization only -- augmentation at test time would make the number noisy.

    Uses the TRAINING set's mean/std, not the test set's. Computing statistics
    from the test data would leak information about it into the pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class HierCIFAR100(Dataset):
    """Wraps CIFAR100 so each item is (image, coarse_label, fine_label)."""

    def __init__(self, root: str = "./data", train: bool = True, download: bool = True):
        tf = _train_tf() if train else _eval_tf()
        self.base = CIFAR100(root=root, train=train, download=download, transform=tf)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, fine = self.base[idx]
        coarse = SPARSE2COARSE[fine]
        return image, coarse, fine


def make_loaders(
    root: str = "./data",
    batch_size: int = 256,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader). Small batches keep MPS memory calm.

    num_workers defaults to 0 deliberately. On macOS, DataLoader workers use the
    `spawn` start method, and combining them with MPS reliably deadlocks here:
    the workers park asleep and the main process blocks forever on the first
    batch. CIFAR-100 is 32x32 and lives in RAM, so in-process loading is fast
    enough that workers buy nothing anyway. Override at your own risk on Linux.
    """
    train_ds = HierCIFAR100(root=root, train=True, download=download)
    test_ds = HierCIFAR100(root=root, train=False, download=download)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader
