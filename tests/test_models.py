"""Tests for the three heads — shapes, the shared feature vector, and the claim
that the coarse-to-fine head actually conditions the fine output on the coarse."""

import torch

from models import FEATURE_DIM, Backbone, CoarseToFineModel, FlatModel, TwoHeadModel


def _x(batch=4):
    torch.manual_seed(0)
    return torch.randn(batch, 3, 32, 32)


def test_backbone_pools_to_feature_dim():
    z = Backbone()(_x())
    assert z.shape == (4, FEATURE_DIM)


def test_flat_model_outputs_100():
    assert FlatModel()(_x()).shape == (4, 100)


def test_two_head_shapes_and_features():
    m = TwoHeadModel()
    c, f = m(_x())
    assert c.shape == (4, 20)
    assert f.shape == (4, 100)
    assert m.features(_x()).shape == (4, FEATURE_DIM)


def test_coarse_to_fine_shapes():
    c, f = CoarseToFineModel()(_x())
    assert c.shape == (4, 20)
    assert f.shape == (4, 100)


def test_fine_head_actually_depends_on_coarse_logits():
    """The whole point of Stage 3: perturbing the coarse branch must move the
    fine output. We freeze the trunk output and nudge softmax_reg1's bias; if the
    fine head ignored the coarse logits, the fine output wouldn't budge."""
    torch.manual_seed(0)
    m = CoarseToFineModel().eval()
    x = _x()
    with torch.no_grad():
        _, f_before = m(x)
        m.softmax_reg1.bias += 5.0   # shift the coarse logits fed into the fine head
        _, f_after = m(x)
    assert not torch.allclose(f_before, f_after)
