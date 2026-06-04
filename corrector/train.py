"""
Train a corrector for one rat.

Usage:
    python -m corrector.train --rat R1 --model linear
    python -m corrector.train --rat R1 --model mlp --hidden 128

Saves to corrector/checkpoints/<rat>_<model>.pt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
from config import EDGES

from corrector.data import make_loaders
from corrector.models import build_model

CKPT_DIR = _THIS.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def bone_length_loss(pred, target, edges):
    """Penalize the model for changing bone lengths relative to the target.

    pred, target : (B, 23, 3)
    edges        : list of (i, j)
    """
    e = torch.as_tensor(edges, dtype=torch.long, device=pred.device)
    pred_d = (pred[:, e[:, 0], :] - pred[:, e[:, 1], :]).norm(dim=-1)
    tgt_d  = (target[:, e[:, 0], :] - target[:, e[:, 1], :]).norm(dim=-1)
    return ((pred_d - tgt_d) ** 2).mean()


def evaluate(model, loader, device):
    """Return mean per-frame MSE in keypoint space."""
    model.eval()
    sse, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            sse += ((pred - y) ** 2).sum().item()
            n += y.numel()
    return sse / n


def train_one(rat: str, model_name: str, hidden: int, n_hidden_layers: int,
              epochs: int, batch_size: int, lr: float, weight_decay: float,
              bone_weight: float, num_workers: int, device: torch.device,
              seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader, val_loader, test_loader, splits = make_loaders(
        rat, batch_size=batch_size, num_workers=num_workers, seed=seed)

    model = build_model(model_name, hidden=hidden, n_hidden_layers=n_hidden_layers)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: {model_name}  params: {n_params:,}  device: {device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    history = []
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_sse, train_n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss_mse = ((pred - y) ** 2).mean()
            loss = loss_mse
            if bone_weight > 0:
                loss = loss + bone_weight * bone_length_loss(pred, y, EDGES)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sse += ((pred.detach() - y) ** 2).sum().item()
            train_n += y.numel()
        train_mse = train_sse / train_n
        val_mse = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
                        "elapsed_s": time.time() - t0})
        print(f"epoch {epoch:3d}  train_mse={train_mse:.3f}  val_mse={val_mse:.3f}  "
              f"({time.time()-t0:.1f}s)")
        if val_mse < best_val - 1e-4:
            best_val = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 8:
                print(f"early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mse = evaluate(model, test_loader, device)
    print(f"\nBEST val_mse={best_val:.3f}  test_mse={test_mse:.3f}")

    ckpt = CKPT_DIR / f"{rat}_{model_name}.pt"
    torch.save({
        "model_name": model_name,
        "hidden": hidden,
        "n_hidden_layers": n_hidden_layers,
        "rat": rat,
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "best_val_mse": best_val,
        "test_mse": test_mse,
        "history": history,
        "splits": splits,
    }, ckpt)
    print(f"saved {ckpt}")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat", required=True, choices=["R1", "R2", "R3"])
    ap.add_argument("--model", default="mlp", choices=["linear", "mlp"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_hidden_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--bone_weight", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA available")
    args = ap.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    train_one(args.rat, args.model, args.hidden, args.n_hidden_layers,
              args.epochs, args.batch_size, args.lr, args.weight_decay,
              args.bone_weight, args.num_workers, device, args.seed)


if __name__ == "__main__":
    main()
