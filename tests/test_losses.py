"""Tests for the loss functions — especially the headline claim of the whole
project: their dependency loss has NO gradient, ours does."""

import pytest
import torch
import torch.nn.functional as F

from hierarchy import SPARSE2COARSE
from losses import (
    SoftHierarchyLoss,
    accuracy,
    faithful_dloss,
    flat_violation_rate,
    layer_loss,
    violation_rate,
)
from models import CoarseToFineModel


def _logits_requiring_grad(batch=8):
    torch.manual_seed(0)
    c = torch.randn(batch, 20, requires_grad=True)
    f = torch.randn(batch, 100, requires_grad=True)
    coarse = torch.randint(0, 20, (batch,))
    fine = torch.randint(0, 100, (batch,))
    return c, f, coarse, fine


# --- The cautionary tale: their dloss is dead -------------------------------
def test_faithful_dloss_has_no_gradient_path():
    c, f, coarse, fine = _logits_requiring_grad()
    d = faithful_dloss(c, f, coarse, fine)
    assert d.requires_grad is False
    assert d.grad_fn is None
    # It literally cannot backpropagate.
    with pytest.raises(RuntimeError):
        d.backward()


def test_faithful_dloss_adds_zero_gradient_to_the_model():
    """Behavioral proof: lloss and lloss+dloss produce identical parameter grads."""
    torch.manual_seed(0)
    x = torch.randn(16, 3, 32, 32)
    coarse = torch.randint(0, 20, (16,))
    fine = torch.randint(0, 100, (16,))

    def grads_for(add_dloss):
        torch.manual_seed(1)
        model = CoarseToFineModel()
        cl, fl = model(x)
        loss = layer_loss(cl, fl, coarse, fine)
        if add_dloss:
            loss = loss + faithful_dloss(cl, fl, coarse, fine)
        model.zero_grad()
        loss.backward()
        return [p.grad.clone() for p in model.parameters()]

    for g_plain, g_dloss in zip(grads_for(False), grads_for(True)):
        assert torch.allclose(g_plain, g_dloss)


# --- Our differentiable replacement -----------------------------------------
def test_soft_loss_is_differentiable_into_fine_logits():
    _, f, _, _ = _logits_requiring_grad()
    coarse = torch.randint(0, 20, (f.shape[0],))
    s = SoftHierarchyLoss()(f, coarse)
    assert s.requires_grad
    s.backward()
    assert f.grad is not None
    assert f.grad.abs().sum() > 0


def test_soft_loss_reaches_the_backbone():
    torch.manual_seed(0)
    model = CoarseToFineModel()
    x = torch.randn(8, 3, 32, 32)
    coarse = torch.randint(0, 20, (8,))
    _, fl = model(x)
    SoftHierarchyLoss()(fl, coarse).backward()
    trunk = [p.grad for n, p in model.named_parameters()
             if n.startswith("backbone") and p.grad is not None]
    assert trunk and sum(g.abs().sum() for g in trunk) > 0


def test_soft_loss_low_when_mass_in_true_parent_high_when_not():
    fine_id = 0                       # apple
    parent = SPARSE2COARSE[fine_id]   # -> fruit_and_vegetables
    logits = torch.full((1, 100), -10.0)
    logits[0, fine_id] = 10.0         # nearly all probability on `apple`
    loss_fn = SoftHierarchyLoss()

    good = loss_fn(logits, torch.tensor([parent]))
    bad = loss_fn(logits, torch.tensor([(parent + 1) % 20]))
    assert good.item() < 1e-2
    assert bad.item() > 5.0


def test_soft_loss_equals_neg_log_mass_in_parent():
    torch.manual_seed(3)
    logits = torch.randn(4, 100)
    coarse = torch.randint(0, 20, (4,))
    probs = F.softmax(logits, dim=1)
    expected = 0.0
    for b in range(4):
        mass = sum(probs[b, f] for f in range(100) if SPARSE2COARSE[f] == coarse[b].item())
        expected += -torch.log(mass + 1e-8)
    expected /= 4
    got = SoftHierarchyLoss()(logits, coarse)
    assert torch.allclose(got, expected, atol=1e-5)


# --- Layer loss + metrics ---------------------------------------------------
def test_layer_loss_is_sum_of_cross_entropies():
    torch.manual_seed(0)
    cl, fl = torch.randn(8, 20), torch.randn(8, 100)
    coarse, fine = torch.randint(0, 20, (8,)), torch.randint(0, 100, (8,))
    expected = F.cross_entropy(cl, coarse) + F.cross_entropy(fl, fine)
    assert torch.allclose(layer_loss(cl, fl, coarse, fine), expected)


def test_violation_rate_extremes():
    fine_id = 0
    parent = SPARSE2COARSE[fine_id]
    fl = torch.full((1, 100), -10.0)
    fl[0, fine_id] = 10.0

    consistent = torch.full((1, 20), -10.0)
    consistent[0, parent] = 10.0
    assert violation_rate(consistent, fl) == 0.0

    inconsistent = torch.full((1, 20), -10.0)
    inconsistent[0, (parent + 1) % 20] = 10.0
    assert violation_rate(inconsistent, fl) == 100.0


def test_flat_violation_rate_uses_true_coarse():
    fine_id = 0
    parent = SPARSE2COARSE[fine_id]
    fl = torch.full((2, 100), -10.0)
    fl[0, fine_id] = 10.0   # predicts apple -> lands in true parent
    fl[1, fine_id] = 10.0   # predicts apple -> lands in wrong parent
    coarse_true = torch.tensor([parent, (parent + 1) % 20])
    assert flat_violation_rate(fl, coarse_true) == 50.0


def test_accuracy():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    target = torch.tensor([1, 0, 0])  # 2 of 3 correct
    assert accuracy(logits, target) == pytest.approx(200 / 3)
