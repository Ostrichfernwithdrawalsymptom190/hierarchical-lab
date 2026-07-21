"""Stage 3 — conditioning: the repo's actual architecture. The coarse logits are
concatenated into the fine head (cat(20, 100) -> 100), so the fine decision is
conditioned on the coarse one.

Same trunk and same layer loss as Stage 2 — the ONLY change is the head wiring —
so any delta isolates what conditioning alone buys. This is the controlled A/B
the upstream README never ran.

    uv run python s3_coarse_to_fine.py --epochs 8
"""

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from data import make_loaders
from device import pick_device, set_seed
from losses import accuracy, layer_loss, violation_rate
from models import CoarseToFineModel
from viz import block_confusion, plot_curves


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cl, fl, ct, ft = [], [], [], []
    for x, coarse, fine in loader:
        c_logits, f_logits = model(x.to(device))
        cl.append(c_logits.cpu())
        fl.append(f_logits.cpu())
        ct.append(coarse)
        ft.append(fine)
    return torch.cat(cl), torch.cat(fl), torch.cat(ct), torch.cat(ft)


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
    model = CoarseToFineModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = {"fine acc": [], "coarse acc": [], "violation %": []}
    print(f"{'epoch':>5} | {'train loss':>10} | {'fine acc':>8} | {'coarse acc':>10} | {'viol %':>7}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, coarse, fine in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, coarse, fine = x.to(device), coarse.to(device), fine.to(device)
            opt.zero_grad()
            c_logits, f_logits = model(x)
            loss = layer_loss(c_logits, f_logits, coarse, fine)
            loss.backward()
            opt.step()
            running += loss.item()
        running /= len(train_loader)

        cl, fl, ct, ft = evaluate(model, test_loader, device)
        hist["fine acc"].append(accuracy(fl, ft))
        hist["coarse acc"].append(accuracy(cl, ct))
        hist["violation %"].append(violation_rate(cl, fl))
        print(f"{epoch:>5} | {running:>10.3f} | {hist['fine acc'][-1]:>7.2f}% | "
              f"{hist['coarse acc'][-1]:>9.2f}% | {hist['violation %'][-1]:>6.2f}%")

    cl, fl, ct, ft = evaluate(model, test_loader, device)
    block_confusion(ft.numpy(), fl.argmax(1).numpy(),
                    "Stage 3 — coarse-to-fine conditioning", "runs/s3_confusion.png")
    plot_curves(hist, "Stage 3 — coarse-to-fine conditioning", "runs/s3_curves.png")
    print("wrote runs/s3_confusion.png, runs/s3_curves.png")


if __name__ == "__main__":
    main()
