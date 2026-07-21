"""The hierarchy losses — the heart of the whole exercise.

Three things live here:

  1. layer_loss           — their `lloss`: per-level cross-entropy. Fine and healthy.
  2. faithful_dloss       — their `dloss`, ported verbatim in spirit. We keep it
                            ONLY to prove (Stage 4) that it has no gradient path.
  3. SoftHierarchyLoss    — my differentiable replacement: push fine probability
                            mass into the true parent's children via marginalization.

Plus small metrics (accuracy, hierarchy-violation rate) used by every stage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hierarchy import coarse_of, parent_matrix


# ----------------------------------------------------------------------------
# 1. Layer loss (their lloss) — deep supervision, one CE per level.
# ----------------------------------------------------------------------------
def layer_loss(coarse_logits, fine_logits, coarse_true, fine_true, alpha: float = 1.0):
    lloss = F.cross_entropy(coarse_logits, coarse_true) + F.cross_entropy(fine_logits, fine_true)
    return alpha * lloss


# ----------------------------------------------------------------------------
# 2. Their dependency loss, ported faithfully — the cautionary tale.
#    Built from argmax + comparisons, so it is CONSTANT w.r.t. the weights:
#    every tensor it touches is detached by argmax/`==`. Kept to be dissected.
# ----------------------------------------------------------------------------
def faithful_dloss(coarse_logits, fine_logits, coarse_true, fine_true,
                   beta: float = 0.8, p_loss: float = 3.0):
    coarse_pred = torch.argmax(F.softmax(coarse_logits, dim=1), dim=1)   # <- detaches
    fine_pred = torch.argmax(F.softmax(fine_logits, dim=1), dim=1)       # <- detaches

    # D_l = 1 when the fine prediction is NOT a child of the coarse prediction.
    # (Vectorized version of their per-sample Python loop over numeric_hierarchy.)
    D_l = (coarse_of(fine_pred) != coarse_pred).float()

    l_prev = torch.where(coarse_pred == coarse_true, 0.0, 1.0)
    l_curr = torch.where(fine_pred == fine_true, 0.0, 1.0)

    dloss = torch.sum(torch.pow(p_loss, D_l * l_prev) * torch.pow(p_loss, D_l * l_curr) - 1.0)
    return beta * dloss


# ----------------------------------------------------------------------------
# 3. The differentiable fix. Marginalize fine softmax up to the parent
#    (p_coarse = fine_probs @ Mᵀ) and take NLL of the TRUE coarse label:
#    -log(probability mass the fine head placed inside the correct superclass).
#    Gradient flows straight into the fine logits — this one actually teaches.
# ----------------------------------------------------------------------------
class SoftHierarchyLoss(nn.Module):
    def __init__(self, beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.beta = beta
        self.eps = eps
        # Registered so .to(device) moves it with the model.
        self.register_buffer("M", parent_matrix())

    def forward(self, fine_logits, coarse_true):
        fine_probs = F.softmax(fine_logits, dim=1)          # (B, 100), differentiable
        coarse_from_fine = fine_probs @ self.M.t()          # (B, 20) marginalized
        mass_in_parent = coarse_from_fine.gather(1, coarse_true.unsqueeze(1)).squeeze(1)
        return self.beta * (-torch.log(mass_in_parent + self.eps)).mean()


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
@torch.no_grad()
def accuracy(logits, target) -> float:
    return (logits.argmax(1) == target).float().mean().item() * 100.0


@torch.no_grad()
def violation_rate(coarse_logits, fine_logits) -> float:
    """% of samples where the fine prediction's parent != the coarse prediction.

    This is exactly what the dependency loss is *supposed* to drive down; we plot
    it across stages to see which mechanisms actually move it.
    """
    coarse_pred = coarse_logits.argmax(1)
    fine_pred = fine_logits.argmax(1)
    return (coarse_of(fine_pred) != coarse_pred).float().mean().item() * 100.0


@torch.no_grad()
def flat_violation_rate(fine_logits, coarse_true) -> float:
    """For the flat model (no coarse head): does the fine prediction land in the
    TRUE superclass? Lets Stage 1 report a comparable number."""
    fine_pred = fine_logits.argmax(1)
    return (coarse_of(fine_pred) != coarse_true).float().mean().item() * 100.0
