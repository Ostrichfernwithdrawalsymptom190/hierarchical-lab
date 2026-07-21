"""Integration: validate our hardcoded taxonomy against the REAL CIFAR-100
labels, and that the loader returns (image, coarse, fine) correctly.

Skipped automatically until the dataset is extracted under ./data. To enable:
    curl -fsSL -o data/cifar-100-python.tar.gz \\
        https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    uv run python -c "from data import make_loaders; make_loaders()"  # extracts
"""

import os
import pickle

import pytest

from hierarchy import COARSE_NAMES, FINE_NAMES, SPARSE2COARSE

_ROOT = "data/cifar-100-python"
_needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(_ROOT, "train")),
    reason="CIFAR-100 not extracted under ./data",
)


def _unpickle(name):
    with open(os.path.join(_ROOT, name), "rb") as f:
        return pickle.load(f, encoding="bytes")


@_needs_data
def test_our_map_matches_real_cifar100_labels():
    """The ground-truth test: for every training image, our SPARSE2COARSE[fine]
    must equal CIFAR-100's own coarse label."""
    d = _unpickle("train")
    fine = d[b"fine_labels"]
    coarse = d[b"coarse_labels"]
    mismatches = [i for i in range(len(fine)) if SPARSE2COARSE[fine[i]] != coarse[i]]
    assert mismatches == []


@_needs_data
def test_our_names_match_the_meta_file():
    meta = _unpickle("meta")
    fine_names = [n.decode() for n in meta[b"fine_label_names"]]
    coarse_names = [n.decode() for n in meta[b"coarse_label_names"]]
    assert fine_names == FINE_NAMES
    assert coarse_names == COARSE_NAMES


@_needs_data
def test_loader_returns_both_labels_consistently():
    from data import HierCIFAR100

    ds = HierCIFAR100(train=False, download=False)
    image, coarse, fine = ds[0]
    assert image.shape == (3, 32, 32)
    assert coarse == SPARSE2COARSE[fine]
