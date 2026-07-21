"""Stage 2 — deep supervision: a coarse head + a fine head on a shared trunk,
trained with the layer loss (their `lloss` = CE at each level).

Now the model has an explicit coarse opinion, so "violation %" becomes the
disagreement between the two heads (fine prediction's parent vs coarse
prediction). Compare against Stage 1.

    uv run python s2_two_head_lloss.py --epochs 8
"""

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from data import make_loaders
from device import pick_device, set_seed
from losses import accuracy, layer_loss, violation_rate
from models import TwoHeadModel
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
    model = TwoHeadModel().to(device)
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
        fine_acc = accuracy(fl, ft)
        coarse_acc = accuracy(cl, ct)
        viol = violation_rate(cl, fl)
        hist["fine acc"].append(fine_acc)
        hist["coarse acc"].append(coarse_acc)
        hist["violation %"].append(viol)
        print(f"{epoch:>5} | {running:>10.3f} | {fine_acc:>7.2f}% | {coarse_acc:>9.2f}% | {viol:>6.2f}%")

    cl, fl, ct, ft = evaluate(model, test_loader, device)
    block_confusion(ft.numpy(), fl.argmax(1).numpy(),
                    "Stage 2 — two-head + layer loss", "runs/s2_confusion.png")
    plot_curves(hist, "Stage 2 — two-head + layer loss", "runs/s2_curves.png")
    print("wrote runs/s2_confusion.png, runs/s2_curves.png")


if __name__ == "__main__":
    main()
