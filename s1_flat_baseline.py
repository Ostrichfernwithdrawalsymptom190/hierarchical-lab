"""Stage 1 — the problem: a flat 100-way classifier that ignores the hierarchy.

We still measure the hierarchy it's implicitly violating: map each fine
prediction to its superclass and check whether it lands in the TRUE superclass.
That "free" coarse accuracy / violation rate is the baseline every later stage
must beat.

    uv run python s1_flat_baseline.py --epochs 8

Writes runs/s1_confusion.png (block-structured — watch errors leak across
superclass blocks).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import make_loaders
from device import pick_device, set_seed
from hierarchy import coarse_of
from losses import accuracy, flat_violation_rate
from models import FlatModel
from viz import block_confusion, plot_curves


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, fine_all, coarse_all = [], [], []
    for x, coarse, fine in loader:
        logits = model(x.to(device))
        logits_all.append(logits.cpu())
        fine_all.append(fine)
        coarse_all.append(coarse)
    return torch.cat(logits_all), torch.cat(fine_all), torch.cat(coarse_all)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    train_loader, test_loader = make_loaders(batch_size=args.batch_size)
    model = FlatModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = {"fine acc": [], "coarse acc (implied)": [], "violation %": []}
    print(f"{'epoch':>5} | {'train loss':>10} | {'fine acc':>8} | {'coarse acc':>10} | {'viol %':>7}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, _coarse, fine in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, fine = x.to(device), fine.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), fine)
            loss.backward()
            opt.step()
            running += loss.item()
        running /= len(train_loader)

        logits, fine_t, coarse_t = evaluate(model, test_loader, device)
        fine_acc = accuracy(logits, fine_t)
        viol = flat_violation_rate(logits, coarse_t)
        coarse_acc = (coarse_of(logits.argmax(1)) == coarse_t).float().mean().item() * 100.0
        hist["fine acc"].append(fine_acc)
        hist["coarse acc (implied)"].append(coarse_acc)
        hist["violation %"].append(viol)
        print(f"{epoch:>5} | {running:>10.3f} | {fine_acc:>7.2f}% | {coarse_acc:>9.2f}% | {viol:>6.2f}%")

    logits, fine_t, _ = evaluate(model, test_loader, device)
    cm_path = block_confusion(
        fine_t.numpy(), logits.argmax(1).numpy(),
        "Stage 1 — flat baseline (fine confusion, grouped by superclass)",
        "runs/s1_confusion.png",
    )
    curve_path = plot_curves(hist, "Stage 1 — flat baseline", "runs/s1_curves.png")
    print(f"wrote {cm_path}\nwrote {curve_path}")


if __name__ == "__main__":
    main()
