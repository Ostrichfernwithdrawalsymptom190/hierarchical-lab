"""The taxonomy is load-bearing for every other module, so pin it hard."""

import torch

import hierarchy as H


def test_map_shape_and_range():
    assert len(H.SPARSE2COARSE) == H.NUM_FINE == 100
    assert min(H.SPARSE2COARSE) == 0
    assert max(H.SPARSE2COARSE) == H.NUM_COARSE - 1 == 19


def test_names_lengths():
    assert len(H.FINE_NAMES) == 100
    assert len(H.COARSE_NAMES) == 20


def test_coarse_to_fine_is_a_partition():
    # 20 superclasses, exactly 5 children each, covering all 100 fine ids once.
    assert len(H.COARSE_TO_FINE) == 20
    assert all(len(v) == 5 for v in H.COARSE_TO_FINE.values())
    union = [f for children in H.COARSE_TO_FINE.values() for f in children]
    assert sorted(union) == list(range(100))  # covers all, no duplicates


def test_block_order_is_a_permutation_grouped_by_coarse():
    assert sorted(H.FINE_BLOCK_ORDER) == list(range(100))
    # Each consecutive run of 5 fine ids shares one parent -> clean diagonal blocks.
    for start in range(0, 100, 5):
        block = H.FINE_BLOCK_ORDER[start:start + 5]
        parents = {H.SPARSE2COARSE[f] for f in block}
        assert len(parents) == 1


def test_coarse_of_matches_the_table():
    fine = torch.arange(100)
    coarse = H.coarse_of(fine)
    assert coarse.tolist() == H.SPARSE2COARSE


def test_known_membership():
    # Spot-check against the published CIFAR-100 taxonomy.
    aquatic = H.COARSE_NAMES.index("aquatic_mammals")
    names = {H.FINE_NAMES[f] for f in H.COARSE_TO_FINE[aquatic]}
    assert names == {"beaver", "dolphin", "otter", "seal", "whale"}


def test_parent_matrix_is_one_hot_over_parents():
    m = H.parent_matrix()
    assert m.shape == (20, 100)
    # Every fine class has exactly one parent -> each column sums to 1.
    assert torch.allclose(m.sum(0), torch.ones(100))
    # Row c has 1s exactly at c's children.
    for c in range(20):
        ones = torch.nonzero(m[c]).flatten().tolist()
        assert ones == H.COARSE_TO_FINE[c]
