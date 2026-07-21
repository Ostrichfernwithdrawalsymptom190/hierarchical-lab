"""The takeaway toolkit — dataset-agnostic structured-output helpers.

Everything in the pathway that's worth keeping, distilled into one file with no
CIFAR/torchvision dependency. Point it at ANY two-level taxonomy: pass a
`child_to_parent` list (child_to_parent[fine_id] = coarse_id) and you get a
differentiable "respect the tree" loss and a violation metric.

Where this applies in your future work
---------------------------------------
Any time you know structure among your outputs and want the model to honor it:
  * LLM classification/routing over a label taxonomy (intent -> sub-intent).
  * Cascaded / coarse-to-fine decoders (predict the region, then the point).
  * Product / document categorization, ICD codes, any ontology.
The pattern: (1) share a trunk, (2) supervise every level, (3) enforce
consistency by MARGINALIZING the fine distribution up to the parent and making
that agree with the coarse target — differentiably. Never enforce it with
argmax; argmax has no gradient (see faithful_dloss in losses.py for the trap).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_parent_matrix(child_to_parent: list[int]) -> torch.Tensor:
    """(num_parents x num_children) 0/1 membership matrix from a child->parent map."""
    num_children = len(child_to_parent)
    num_parents = max(child_to_parent) + 1
    m = torch.zeros(num_parents, num_children)
    for child, parent in enumerate(child_to_parent):
        m[parent, child] = 1.0
    return m


class SoftHierarchyLoss(nn.Module):
    """Differentiable "child prediction must respect its parent" loss.

    loss = -log( sum of fine-class probability the model put inside the TRUE parent )

    Marginalizes fine softmax up to the parent (probs @ Mᵀ) and takes the NLL of
    the true parent. Gradient flows into the fine logits, so the model actually
    learns to keep its probability mass in the right superclass.
    """

    def __init__(self, child_to_parent: list[int], beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.register_buffer("M", build_parent_matrix(child_to_parent))
        self.register_buffer("child_to_parent",
                             torch.tensor(child_to_parent, dtype=torch.long))

    def forward(self, fine_logits: torch.Tensor, parent_target: torch.Tensor) -> torch.Tensor:
        fine_probs = F.softmax(fine_logits, dim=1)
        parent_probs = fine_probs @ self.M.t()
        mass = parent_probs.gather(1, parent_target.unsqueeze(1)).squeeze(1)
        return self.beta * (-torch.log(mass + self.eps)).mean()

    @torch.no_grad()
    def violation_rate(self, fine_logits: torch.Tensor,
                       parent_pred_or_target: torch.Tensor) -> float:
        """% of samples whose argmax fine class sits under a different parent."""
        fine_pred = fine_logits.argmax(1)
        implied_parent = self.child_to_parent[fine_pred]
        return (implied_parent != parent_pred_or_target).float().mean().item() * 100.0
