"""
Train a world-space corrector on one or more rats combined.

Usage:
    python -m corrector.train_world --rats R2 R3 --model mlp --tag R2R3
    python -m corrector.train_world --rats R3 --model mlp --tag R3only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from config import EDGES

from corrector.data_world import (WindowedWorldDataset, WorldPairedDataset,
                                    session_split_multi)
from corrector.models import build_model

CKPT_DIR = _THIS.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def bone_length_loss(pred, target, edges):
    e = torch.as_tensor(edges, dtype=torch.long, device=pred.device)
    pred_d = (pred[:, e[:, 0], :] - pred[:, e[:, 1], :]).norm(dim=-1)
    tgt_d = (target[:, e[:, 0], :] - target[:, e[:, 1], :]).norm(dim=-1)
    return ((pred_d - tgt_d) ** 2).mean()


def evaluate(model, loader, device):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rats", nargs="+", required=True,
                    choices=["R1", "R2", "R3"])
    ap.add_argument("--tag", default=None,
                    help="checkpoint tag; defaults to rats joined with '_'")
    ap.add_argument("--model", default="mlp",
                    choices=["linear", "mlp", "temporal_mlp"])
    ap.add_argument("--ctx", type=int, default=5,
                    help="frames of causal context (only for temporal_mlp)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_hidden_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--bone_weight", type=float, default=0.1)
    ap.add_argument("--max_residual", type=float, default=60.0)
    ap.add_argument("--early_stop_patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = args.tag or "_".join(args.rats)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = session_split_multi(args.rats, seed=args.seed)
    print("Split sizes:")
    for which in ("train", "val", "test"):
        sizes = {r: len(splits[which][r]) for r in args.rats}
        print(f"  {which}: {sizes}", flush=True)

    print("\nBuilding train dataset...", flush=True)
    if args.model == "temporal_mlp":
        train_ds = WindowedWorldDataset(splits["train"], ctx=args.ctx,
                                         max_residual=args.max_residual)
        val_ds = WindowedWorldDataset(splits["val"], ctx=args.ctx,
                                       max_residual=args.max_residual)
    else:
        train_ds = WorldPairedDataset(splits["train"], max_residual=args.max_residual)
        print("Building val dataset...", flush=True)
        val_ds = WorldPairedDataset(splits["val"], max_residual=args.max_residual)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    if args.model == "temporal_mlp":
        model = build_model(args.model, ctx=args.ctx, hidden=args.hidden,
                            n_hidden_layers=args.n_hidden_layers).to(device)
    else:
        model = build_model(args.model, hidden=args.hidden,
                            n_hidden_layers=args.n_hidden_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: {args.model}  params: {n_params:,}  device: {device}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best_val = float("inf"); best_state = None; history = []
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_sse, train_n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss_mse = ((pred - y) ** 2).mean()
            loss = loss_mse
            if args.bone_weight > 0:
                loss = loss + args.bone_weight * bone_length_loss(pred, y, EDGES)
            opt.zero_grad(); loss.backward(); opt.step()
            train_sse += ((pred.detach() - y) ** 2).sum().item()
            train_n += y.numel()
        train_mse = train_sse / train_n
        val_mse = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "val_mse": val_mse, "elapsed_s": elapsed})
        print(f"epoch {epoch:3d}  train_mse={train_mse:.3f}  val_mse={val_mse:.3f}  "
              f"({elapsed:.1f}s)", flush=True)
        if val_mse < best_val - 1e-4:
            best_val = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = CKPT_DIR / f"{tag}_world_{args.model}.pt"
    torch.save({
        "model_name": args.model, "hidden": args.hidden,
        "n_hidden_layers": args.n_hidden_layers,
        "ctx": args.ctx if args.model == "temporal_mlp" else 1,
        "rats": args.rats, "tag": tag,
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "best_val_mse": best_val, "history": history,
        "splits": splits,
        "max_residual": args.max_residual,
        "session_residuals_train": train_ds.session_residuals,
        "session_residuals_val": val_ds.session_residuals,
    }, ckpt)
    print(f"\nsaved {ckpt}  best_val_mse={best_val:.3f}", flush=True)


if __name__ == "__main__":
    main()
