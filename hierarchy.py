"""The CIFAR-100 taxonomy — 100 fine classes nested under 20 coarse superclasses.

This replaces the upstream repo's whole `process_cifar100.py` + `level_dict.py`
dance (which unpickled a meta file, dumped 60k PNGs, and re-read them with cv2).
We instead hard-code the canonical `fine_index -> coarse_index` map that
torchvision's CIFAR100 fine labels obey, and derive everything else from it.

Everything here is plain data + pure functions so `viz.py` can draw the tree
without downloading the dataset.
"""

from __future__ import annotations

import torch

# Canonical CIFAR-100 "sparse2coarse" map: SPARSE2COARSE[fine_label] -> coarse_label.
# Indices match torchvision.datasets.CIFAR100's fine label ordering.
SPARSE2COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

NUM_FINE = 100
NUM_COARSE = 20

COARSE_NAMES = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores", "medium_mammals", "non-insect_invertebrates",
    "people", "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2",
]

FINE_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

# coarse_index -> sorted list of its fine indices (the "children").
COARSE_TO_FINE: dict[int, list[int]] = {c: [] for c in range(NUM_COARSE)}
for _fine, _coarse in enumerate(SPARSE2COARSE):
    COARSE_TO_FINE[_coarse].append(_fine)

# Order fine labels by (coarse, fine) so confusion matrices form clean superclass
# blocks along the diagonal. `FINE_BLOCK_ORDER[k]` is the fine id at block row k.
FINE_BLOCK_ORDER: list[int] = [f for c in range(NUM_COARSE) for f in COARSE_TO_FINE[c]]


def coarse_of(fine_labels: torch.Tensor) -> torch.Tensor:
    """Map a batch of fine labels (long) to their coarse labels, on the same device."""
    lut = torch.as_tensor(SPARSE2COARSE, dtype=torch.long, device=fine_labels.device)
    return lut[fine_labels]


def parent_matrix(device: torch.device | str = "cpu") -> torch.Tensor:
    """(NUM_COARSE x NUM_FINE) 0/1 membership matrix M[c, f] = 1 iff fine f is under coarse c.

    Used by the *differentiable* soft-hierarchy loss to marginalize fine
    probabilities up to their parent: p_coarse = fine_probs @ M.T .
    """
    m = torch.zeros(NUM_COARSE, NUM_FINE, device=device)
    for f, c in enumerate(SPARSE2COARSE):
        m[c, f] = 1.0
    return m
