"""
v2 trainer for world-space correctors. Handles:

  * Single-frame MLP (baseline / linear)
  * Temporal MLP (causal context window)
  * Velocity/acceleration MLP (vel/acc explicit features per frame)
  * GNN over the skeleton (single-frame)
  * Per-rat fine-tuning head (frozen base + small head, trains per rat)

  Plus auxiliary training options:
    --noise_std        : Gaussian noise added to SLEAP input each batch
    --pc_loss_weight   : weight on auxiliary PC-space MSE term
                         (uses each rat's stored xyz PC weights)

Each checkpoint embeds enough metadata for evaluate_all.py to load it cleanly.

Usage examples:
  # A.1 per-rat head on top of frozen R1R2R3 temporal:
  python -m corrector.train_world_v2 --model perrat_head \\
       --base_ckpt corrector/checkpoints/R1R2R3_world_temporal_mlp.pt \\
       --rats R1 --tag R1_head_on_R1R2R3temporal

  # A.2 Gaussian noise on temporal MLP:
  python -m corrector.train_world_v2 --model temporal_mlp --rats R1 R2 R3 \\
       --tag R1R2R3_temporal_noise05 --noise_std 0.5

  # B.1 velocity/acceleration:
  python -m corrector.train_world_v2 --model velacc_mlp --rats R1 R2 R3 \\
       --tag R1R2R3_velacc

  # B.2 PC-space loss:
  python -m corrector.train_world_v2 --model temporal_mlp --rats R1 R2 R3 \\
       --tag R1R2R3_temporal_pcloss --pc_loss_weight 0.5

  # B.3 GNN:
  python -m corrector.train_world_v2 --model gnn --rats R1 R2 R3 \\
       --tag R1R2R3_gnn --hidden 64 --n_layers 3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from config import EDGES
from data_io import load_template

from corrector.data_world import (WindowedWorldDataset, WorldPairedDataset,
                                    session_split_multi)
from corrector.models import build_model

CKPT_DIR = _THIS.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Velocity/acceleration dataset wrapper — turns single-frame data into
# (B, 3, 23, 3) tensors with [pose, velocity, acceleration].
# ---------------------------------------------------------------------------

class VelAccDataset(Dataset):
    """Wraps a WorldPairedDataset to produce (vel/acc, target) pairs.

    Velocity is the difference from the previous frame in the SAME session;
    acceleration is the difference of velocities. We pre-compute these arrays
    per session for speed.
    """

    def __init__(self, base: WorldPairedDataset):
        self.base = base
        # Build per-session vel/acc arrays
        self._vel_per_session = []
        self._acc_per_session = []
        for arr in base._sl_aligned:
            vel = np.zeros_like(arr); vel[1:] = arr[1:] - arr[:-1]
            acc = np.zeros_like(arr); acc[2:] = vel[2:] - vel[1:-1]
            self._vel_per_session.append(vel)
            self._acc_per_session.append(acc)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        si, fi = self.base._index[idx]
        pose = self.base._sl_aligned[si][fi]
        vel = self._vel_per_session[si][fi]
        acc = self._acc_per_session[si][fi]
        x = np.stack([pose, vel, acc], axis=0).astype(np.float32)  # (3, 23, 3)
        y = self.base._dn[si][fi]
        return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Per-rat data helper for the fine-tune head
# ---------------------------------------------------------------------------

def per_rat_split(rats: list[str], train_frac=0.7, val_frac=0.15, seed=0):
    """Use the SAME per-rat splits the base model used. Same as session_split_multi."""
    return session_split_multi(rats, train_frac=train_frac, val_frac=val_frac, seed=seed)


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def bone_length_loss(pred, target, edges):
    e = torch.as_tensor(edges, dtype=torch.long, device=pred.device)
    pred_d = (pred[:, e[:, 0], :] - pred[:, e[:, 1], :]).norm(dim=-1)
    tgt_d = (target[:, e[:, 0], :] - target[:, e[:, 1], :]).norm(dim=-1)
    return ((pred_d - tgt_d) ** 2).mean()


class PCLossHelper:
    """Auxiliary PC-space loss using each rat's xyz template PC basis.

    Caveats: the corrector outputs are in DANNCE world space, but the
    template's PC basis is fit on z-FLIPPED, EGOCENTRICALLY-NORMALIZED SLEAP.
    To compute a sensible PC-space loss we'd need to perform the
    z-flip + egocentric normalization inside the loss, which depends on the
    Procrustes inverse for each session.

    A simpler proxy that's still useful: project the *raw* (corrector-output,
    target) pair through the rat's PC basis directly in DANNCE world space.
    The basis isn't perfectly aligned, but the mismatch is the same for both
    pred and target so it cancels — the loss still penalizes deviations along
    the directions of greatest pose variance.

    For correctness, we additionally normalize each rat's loss by its own
    feature_stds so different rats contribute commensurately.
    """

    def __init__(self, rats: list[str], device):
        self.basis = {}
        from corrector.evaluate_world import RAT_TEMPLATE
        for rat in rats:
            td = dict(load_template(rat, RAT_TEMPLATE[rat]))
            pcu = td["pcs_to_use"].ravel().astype(int)
            pw = torch.tensor(td["pc_weights"], dtype=torch.float32, device=device)
            fm = torch.tensor(td["feature_means"], dtype=torch.float32, device=device)
            stds = torch.tensor(td["feature_stds"][pcu], dtype=torch.float32, device=device)
            self.basis[rat] = (pw, fm, pcu, stds)

    def loss(self, pred, target, rat: str):
        pw, fm, pcu, stds = self.basis[rat]
        # Flatten and project. pred/target shape (B, 23, 3) → (B, 69)
        flat_p = pred.reshape(pred.shape[0], -1)
        flat_t = target.reshape(target.shape[0], -1)
        pc_p = (flat_p - fm) @ pw.t()
        pc_t = (flat_t - fm) @ pw.t()
        pc_p = pc_p[:, pcu] / (stds + 1e-6)
        pc_t = pc_t[:, pcu] / (stds + 1e-6)
        return ((pc_p - pc_t) ** 2).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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
    ap.add_argument("--tag", required=True,
                    help="checkpoint tag (e.g. R1R2R3_temporal_noise05)")
    ap.add_argument("--model", required=True,
                    choices=["mlp", "temporal_mlp", "velacc_mlp", "gnn",
                             "perrat_head"])
    # MLP / temporal options
    ap.add_argument("--ctx", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_hidden_layers", type=int, default=2)
    ap.add_argument("--n_layers", type=int, default=3, help="GNN message-passing layers")
    # Per-rat head
    ap.add_argument("--base_ckpt", default=None,
                    help="path to frozen base for perrat_head")
    # Training
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--bone_weight", type=float, default=0.1)
    ap.add_argument("--pc_loss_weight", type=float, default=0.0,
                    help="0 disables; typical values 0.1-1.0")
    ap.add_argument("--noise_std", type=float, default=0.0,
                    help="Gaussian noise std applied to SLEAP input each batch (mm)")
    ap.add_argument("--max_residual", type=float, default=60.0)
    ap.add_argument("--early_stop_patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = session_split_multi(args.rats, seed=args.seed)
    print("Split sizes:")
    for which in ("train", "val", "test"):
        sizes = {r: len(splits[which][r]) for r in args.rats}
        print(f"  {which}: {sizes}", flush=True)

    # Build datasets per the model needs
    print("\nBuilding datasets...", flush=True)
    if args.model == "temporal_mlp" or (args.model == "perrat_head" and args.ctx > 1):
        train_ds = WindowedWorldDataset(splits["train"], ctx=args.ctx,
                                         max_residual=args.max_residual)
        val_ds = WindowedWorldDataset(splits["val"], ctx=args.ctx,
                                       max_residual=args.max_residual)
    elif args.model == "velacc_mlp":
        base_train = WorldPairedDataset(splits["train"],
                                         max_residual=args.max_residual)
        base_val = WorldPairedDataset(splits["val"],
                                       max_residual=args.max_residual)
        train_ds = VelAccDataset(base_train)
        val_ds = VelAccDataset(base_val)
    elif args.model == "perrat_head":
        # For ctx==1 use the standard per-frame dataset, but we need the base's
        # input format. perrat_head's base might be temporal — handle below.
        train_ds = WorldPairedDataset(splits["train"],
                                       max_residual=args.max_residual)
        val_ds = WorldPairedDataset(splits["val"],
                                     max_residual=args.max_residual)
    else:
        train_ds = WorldPairedDataset(splits["train"],
                                       max_residual=args.max_residual)
        val_ds = WorldPairedDataset(splits["val"],
                                     max_residual=args.max_residual)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Build model
    if args.model == "temporal_mlp":
        model = build_model(args.model, ctx=args.ctx, hidden=args.hidden,
                            n_hidden_layers=args.n_hidden_layers).to(device)
    elif args.model == "velacc_mlp":
        model = build_model(args.model, hidden=args.hidden,
                            n_hidden_layers=args.n_hidden_layers).to(device)
    elif args.model == "gnn":
        model = build_model(args.model, hidden=args.hidden,
                            n_layers=args.n_layers).to(device)
    elif args.model == "perrat_head":
        if not args.base_ckpt:
            raise ValueError("perrat_head requires --base_ckpt")
        model = build_model(args.model, base_ckpt=args.base_ckpt,
                            hidden=args.hidden,
                            n_hidden_layers=args.n_hidden_layers).to(device)
        # If base wants a special input shape, rebuild datasets accordingly
        if model._base_kind == "temporal_mlp":
            ctx = model._base_ctx
            print(f"perrat_head: base is temporal_mlp ctx={ctx}; "
                  f"rebuilding datasets to windowed", flush=True)
            train_ds = WindowedWorldDataset(splits["train"], ctx=ctx,
                                             max_residual=args.max_residual)
            val_ds = WindowedWorldDataset(splits["val"], ctx=ctx,
                                           max_residual=args.max_residual)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                       shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0, pin_memory=True)
        elif model._base_kind == "velacc_mlp":
            print(f"perrat_head: base is velacc_mlp; "
                  f"rebuilding datasets with velocity/acceleration", flush=True)
            base_train = WorldPairedDataset(splits["train"],
                                             max_residual=args.max_residual)
            base_val = WorldPairedDataset(splits["val"],
                                           max_residual=args.max_residual)
            train_ds = VelAccDataset(base_train)
            val_ds = VelAccDataset(base_val)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                       shuffle=True, num_workers=0, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0, pin_memory=True)
    else:
        model = build_model(args.model, hidden=args.hidden,
                            n_hidden_layers=args.n_hidden_layers).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nmodel: {args.model}  total params: {n_params:,}  "
          f"trainable: {n_trainable:,}  device: {device}", flush=True)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)

    pc_loss_helper = None
    if args.pc_loss_weight > 0:
        pc_loss_helper = PCLossHelper(args.rats, device)
        # Build a per-frame rat lookup (for the average loss across batches that
        # may mix rats from different sessions). For simplicity we only support
        # single-rat batches via per-rat dataset enumeration here:
        if len(args.rats) > 1:
            print(f"WARNING: pc_loss with multiple rats uses uniform mixing "
                  f"weighted by training-set frequency. The specific rat per "
                  f"frame is not surfaced from the dataset, so we approximate "
                  f"by computing the loss in each rat's PC basis and averaging.",
                  flush=True)

    best_val = float("inf"); best_state = None; history = []
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_sse, train_n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Optional Gaussian noise on input (only on the SLEAP coords, not target)
            if args.noise_std > 0:
                x = x + torch.randn_like(x) * args.noise_std

            pred = model(x)
            loss_mse = ((pred - y) ** 2).mean()
            loss = loss_mse
            if args.bone_weight > 0:
                loss = loss + args.bone_weight * bone_length_loss(pred, y, EDGES)
            if pc_loss_helper is not None:
                loss_pc = sum(pc_loss_helper.loss(pred, y, rat)
                              for rat in args.rats) / len(args.rats)
                loss = loss + args.pc_loss_weight * loss_pc
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

    ckpt = CKPT_DIR / f"{args.tag}.pt"
    save = {
        "model_name": args.model,
        "hidden": args.hidden,
        "n_hidden_layers": args.n_hidden_layers,
        "n_layers": args.n_layers if args.model == "gnn" else None,
        "ctx": args.ctx if args.model == "temporal_mlp" else 1,
        "rats": args.rats, "tag": args.tag,
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "best_val_mse": best_val, "history": history,
        "splits": splits,
        "max_residual": args.max_residual,
        "noise_std": args.noise_std,
        "pc_loss_weight": args.pc_loss_weight,
        "bone_weight": args.bone_weight,
        "base_ckpt": args.base_ckpt,
        "session_residuals_train": getattr(
            getattr(train_ds, "base", train_ds), "session_residuals", []),
        "session_residuals_val": getattr(
            getattr(val_ds, "base", val_ds), "session_residuals", []),
    }
    torch.save(save, ckpt)
    print(f"\nsaved {ckpt}  best_val_mse={best_val:.3f}", flush=True)


if __name__ == "__main__":
    main()
