"""Stage 4 — the intellectual core.

Part A (the proof): their dependency loss `dloss` is built from argmax + `==` +
where. We run it once and show it has NO gradient path — `requires_grad` is
False, so it cannot even .backward(). It's a monitoring number wearing a loss's
clothes. Then we show SoftHierarchyLoss DOES push gradient into the trunk.

Part B (the fix): train three identical CoarseToFineModels and compare the
hierarchy-violation curve:
    1. lloss only            (no consistency pressure)
    2. lloss + their dloss   (should behave ~identically to #1 — because it's dead)
    3. lloss + soft loss     (violation rate actually drops)

    uv run python s4_dependency_loss.py --epochs 6

Writes the money shot: runs/violation_comparison.png
"""

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from data import make_loaders
from device import pick_device, set_seed
from losses import (
    SoftHierarchyLoss,
    accuracy,
    faithful_dloss,
    layer_loss,
    violation_rate,
)
from models import CoarseToFineModel
from viz import plot_violation_comparison


def grad_probe(device) -> None:
    """Part A — demonstrate the dead gradient, then a live one."""
    print("\n=== Part A: does the dependency loss have a gradient? ===")
    model = CoarseToFineModel().to(device)
    train_loader, _ = make_loaders(batch_size=128)
    x, coarse, fine = next(iter(train_loader))
    x, coarse, fine = x.to(device), coarse.to(device), fine.to(device)
    c_logits, f_logits = model(x)

    d = faithful_dloss(c_logits, f_logits, coarse, fine)
    print(f"their dloss value        : {d.item():.4f}")
    print(f"their dloss requires_grad: {d.requires_grad}   grad_fn: {d.grad_fn}")
    if not d.requires_grad:
        print("  -> no graph -> .backward() is impossible -> ZERO gradient to every weight.")

    soft = SoftHierarchyLoss().to(device)
    s = soft(f_logits, coarse)
    model.zero_grad()
    s.backward()
    trunk_grad = sum(
        p.grad.norm().item()
        for n, p in model.named_parameters()
        if n.startswith("backbone") and p.grad is not None
    )
    print(f"soft loss  requires_grad : {s.requires_grad}   grad_fn: {type(s.grad_fn).__name__}")
    print(f"soft loss  trunk grad-norm: {trunk_grad:.4f}   -> gradient reaches the backbone. ")


def train_variant(name, use_dloss, use_soft, args, device):
    set_seed(args.seed)  # identical init/data order per variant for a fair A/B
    train_loader, test_loader = make_loaders(batch_size=args.batch_size)
    model = CoarseToFineModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    soft = SoftHierarchyLoss(beta=1.0).to(device)

    viol_hist, fine_hist = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, coarse, fine in tqdm(train_loader, desc=f"{name} e{epoch}", leave=False):
            x, coarse, fine = x.to(device), coarse.to(device), fine.to(device)
            opt.zero_grad()
            c_logits, f_logits = model(x)
            loss = layer_loss(c_logits, f_logits, coarse, fine)
            if use_dloss:
                loss = loss + faithful_dloss(c_logits, f_logits, coarse, fine)
            if use_soft:
                loss = loss + soft(f_logits, coarse)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            cl, fl, ct, ft = [], [], [], []
            for x, coarse, fine in test_loader:
                c_logits, f_logits = model(x.to(device))
                cl.append(c_logits.cpu())
                fl.append(f_logits.cpu())
                ct.append(coarse)
                ft.append(fine)
            cl, fl, ct, ft = torch.cat(cl), torch.cat(fl), torch.cat(ct), torch.cat(ft)
        viol_hist.append(violation_rate(cl, fl))
        fine_hist.append(accuracy(fl, ft))
        print(f"  {name:<16} epoch {epoch}: fine acc {fine_hist[-1]:.2f}%  viol {viol_hist[-1]:.2f}%")
    return viol_hist


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    grad_probe(device)

    print("\n=== Part B: violation rate under three losses ===")
    series = {
        "lloss only": train_variant("lloss only", False, False, args, device),
        "lloss + their dloss": train_variant("their dloss", True, False, args, device),
        "lloss + soft (differentiable)": train_variant("soft loss", False, True, args, device),
    }
    path = plot_violation_comparison(series)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
