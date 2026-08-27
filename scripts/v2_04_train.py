"""Corrector v2, step 4 — train three corrector variants.

Self-contained (does not use corrector/data_world.py, whose loader takes the
already bone-regularised sleap_keys_3D). Keeping it standalone also serves the
"easy to deploy online" requirement: the inference path is
    2D -> triangulate(05_04) -> Procrustes -> model -> corrected pose.

Variants
  temporal : y_t = x_t + MLP(x_{t-4..t})                      (prior best base)
  velacc   : y_t = x_t + MLP([x_t, vel, acc])                 (prior best feature set)
  aspect   : y_t = x_t + MLP([x_{t-4..t}, cos/sin of heading-
                              relative-to-each-camera])       (NEW, motivated by
             the §8d finding that SLEAP error is viewing-aspect dependent)

Splits are by SESSION, never by frame.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from config import EDGES

OUT = 'results/corrector_v2'
CKPT = f'{OUT}/checkpoints'
os.makedirs(CKPT, exist_ok=True)
IN_DIM = 69
CTX = 5
NCTX = 12   # context features: aspect(6) + position(6)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG = np.random.default_rng(0)


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.0):
        super().__init__()
        L, d = [], in_dim
        for _ in range(layers):
            L += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout:
                L.append(nn.Dropout(dropout))
            d = hidden
        L.append(nn.Linear(d, IN_DIM))
        self.net = nn.Sequential(*L)

    def forward(self, z):
        return self.net(z)


class Temporal(nn.Module):
    kind = 'temporal'
    def __init__(self, hidden=128, layers=2):
        super().__init__(); self.head = MLPHead(CTX * IN_DIM, hidden, layers)
    def forward(self, win, asp):
        B = win.shape[0]
        z = win.reshape(B, -1)
        return win[:, -1] + self.head(z).reshape(B, 23, 3)


class VelAcc(nn.Module):
    kind = 'velacc'
    def __init__(self, hidden=128, layers=2):
        super().__init__(); self.head = MLPHead(3 * IN_DIM, hidden, layers)
    def forward(self, win, asp):
        B = win.shape[0]
        x = win[:, -1]
        vel = win[:, -1] - win[:, -2]
        acc = win[:, -1] - 2 * win[:, -2] + win[:, -3]
        z = torch.cat([x.reshape(B, -1), vel.reshape(B, -1), acc.reshape(B, -1)], 1)
        return x + self.head(z).reshape(B, 23, 3)


class Aspect(nn.Module):
    kind = 'aspect'
    def __init__(self, hidden=128, layers=2):
        super().__init__(); self.head = MLPHead(CTX * IN_DIM + NCTX, hidden, layers)
    def forward(self, win, asp):
        B = win.shape[0]
        z = torch.cat([win.reshape(B, -1), asp], 1)
        return win[:, -1] + self.head(z).reshape(B, 23, 3)


BUILD = {'temporal': Temporal, 'velacc': VelAcc, 'aspect': Aspect}


def bone_loss(pred, tgt, e):
    pd_ = (pred[:, e[:, 0]] - pred[:, e[:, 1]]).norm(dim=-1)
    td_ = (tgt[:, e[:, 0]] - tgt[:, e[:, 1]]).norm(dim=-1)
    return ((pd_ - td_) ** 2).mean()


def build_windows(M):
    """Indices t such that t-CTX+1..t all lie in the same segment, in order."""
    gid = M.gid.values
    idx = []
    start = 0
    for i in range(1, len(gid) + 1):
        if i == len(gid) or gid[i] != gid[start]:
            if i - start >= CTX:
                idx.extend(range(start + CTX - 1, i))
            start = i
    return np.array(idx, dtype=np.int64)


def main():
    z = np.load(f'{OUT}/dataset.npz')
    X, Y, A = z['X'], z['Y'], z['ASP']
    M = pd.read_csv(f'{OUT}/dataset_meta.csv')
    print(f'dataset X{X.shape}  sessions={M.session.nunique()}')

    tidx = build_windows(M)
    print(f'usable windows: {len(tidx):,} of {len(M):,} frames')

    # session-level split, stratified by rat
    sess = M[['rat', 'session']].drop_duplicates().reset_index(drop=True)
    tr, va, te = [], [], []
    for rat, g in sess.groupby('rat'):
        s = list(g.session); RNG.shuffle(s)
        n = len(s); ntr = max(1, int(round(0.6 * n))); nva = max(1, int(round(0.2 * n)))
        tr += [(rat, x) for x in s[:ntr]]
        va += [(rat, x) for x in s[ntr:ntr + nva]]
        te += [(rat, x) for x in s[ntr + nva:]]
    key = (M.rat + '/' + M.session).values
    def mask(pairs):
        want = {f'{r}/{s}' for r, s in pairs}
        return np.array([key[i] in want for i in tidx])
    mtr, mva, mte = mask(tr), mask(va), mask(te)
    print(f'  train {mtr.sum():,}  val {mva.sum():,}  test {mte.sum():,}')
    print(f'  test sessions: {[s for _, s in te]}')
    json.dump({'train': tr, 'val': va, 'test': te}, open(f'{OUT}/split.json', 'w'))

    # window tensors
    off = np.arange(-CTX + 1, 1)
    def make(midx):
        t = tidx[midx]
        win = X[t[:, None] + off[None, :]]                 # (N, CTX, 23, 3)
        return (torch.from_numpy(win), torch.from_numpy(Y[t]),
                torch.from_numpy(A[t]))
    Wtr, Ytr, Atr = make(mtr); Wva, Yva, Ava = make(mva); Wte, Yte, Ate = make(mte)

    # Centre on the last input frame's centroid (the model learns a residual
    # field), then divide by SCALE. Raw coordinates are +/-300 mm, which blows a
    # plain MLP up at lr 1e-3; working in units of SCALE mm keeps inputs ~O(1).
    # The RAW (unregularised) triangulation contains catastrophic per-keypoint
    # failures -- max |X-Y| is 2.4e6 mm and 16% of frames have a pose radius
    # >500 mm, versus a real rat radius of ~150 mm (DANNCE maxes out at 148 mm).
    # Those are exactly the failures the online bone-length bundle adjustment
    # exists to fix. We CLAMP rather than drop, so the corrector still sees
    # "this keypoint is wrong" and learns to pull it back, but the optimiser is
    # not dominated by 10^6-mm values.
    SCALE = 100.0
    CLAMP = 4.0                                            # +/-400 mm from centroid
    def norm(W, Yv):
        c = W[:, -1].mean(dim=1, keepdim=True)             # (N,1,3) centroid
        Wc = torch.clamp((W - c[:, None]) / SCALE, -CLAMP, CLAMP)
        Yc = torch.clamp((Yv - c) / SCALE, -CLAMP, CLAMP)
        return Wc, Yc, c
    e = torch.as_tensor(np.asarray(EDGES, int), dtype=torch.long, device=DEV)

    results = {}
    for name, cls in BUILD.items():
        t0 = time.time()
        torch.manual_seed(0)
        model = cls().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best, bad, best_state = np.inf, 0, None
        Wtr_c, Ytr_c, _ = norm(Wtr, Ytr); Wva_c, Yva_c, _ = norm(Wva, Yva)
        n = len(Wtr_c)
        for ep in range(200):
            model.train()
            perm = torch.randperm(n)
            for s in range(0, n, 4096):
                b = perm[s:s + 4096]
                w = Wtr_c[b].to(DEV); y = Ytr_c[b].to(DEV); a = Atr[b].to(DEV)
                p = model(w, a)
                loss = (torch.nn.functional.smooth_l1_loss(p, y, beta=0.1)
                        + 0.1 * bone_loss(p, y, e))
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                errs = []
                for s in range(0, len(Wva_c), 8192):
                    w = Wva_c[s:s + 8192].to(DEV); y = Yva_c[s:s + 8192].to(DEV)
                    a = Ava[s:s + 8192].to(DEV)
                    errs.append((model(w, a) - y).norm(dim=-1).flatten().cpu())
                v = float(torch.cat(errs).median()) * SCALE
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 20:
                    break
        model.load_state_dict(best_state)
        torch.save(best_state, f'{CKPT}/{name}.pt')
        # test MSE (per coordinate) and per-keypoint error
        Wte_c, Yte_c, cte = norm(Wte, Yte)
        with torch.no_grad():
            preds = []
            for s in range(0, len(Wte_c), 8192):
                preds.append(model(Wte_c[s:s+8192].to(DEV), Ate[s:s+8192].to(DEV)).cpu())
            P = torch.cat(preds)
        # back to millimetres for reporting
        err_corr = (P - Yte_c).norm(dim=-1).numpy() * SCALE
        err_base = (Wte_c[:, -1] - Yte_c).norm(dim=-1).numpy() * SCALE
        def pct(a, q): return float(np.percentile(a, q))
        results[name] = dict(
            epochs=ep + 1, val_median_mm=best, secs=time.time() - t0,
            test_med_corrected=float(np.median(err_corr)),
            test_med_baseline=float(np.median(err_base)),
            test_mean_corrected=float(err_corr.mean()),
            test_mean_baseline=float(err_base.mean()),
            test_p90_corrected=pct(err_corr, 90), test_p90_baseline=pct(err_base, 90),
            test_p99_corrected=pct(err_corr, 99), test_p99_baseline=pct(err_base, 99),
            frac_gt50_corrected=float((err_corr > 50).mean()),
            frac_gt50_baseline=float((err_base > 50).mean()),
            per_kp_corrected=np.median(err_corr, axis=0).tolist(),
            per_kp_baseline=np.median(err_base, axis=0).tolist())
        r = results[name]
        print(f'{name:9} ep={ep+1:3d} | median {r["test_med_baseline"]:6.2f}->{r["test_med_corrected"]:6.2f} '
              f'| mean {r["test_mean_baseline"]:7.2f}->{r["test_mean_corrected"]:7.2f} '
              f'| p90 {r["test_p90_baseline"]:7.2f}->{r["test_p90_corrected"]:7.2f} '
              f'| p99 {r["test_p99_baseline"]:8.1f}->{r["test_p99_corrected"]:7.1f} '
              f'| >50mm {r["frac_gt50_baseline"]*100:5.1f}%->{r["frac_gt50_corrected"]*100:4.1f}%', flush=True)
    json.dump(results, open(f'{OUT}/train_results.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/train_results.json and checkpoints')


if __name__ == '__main__':
    main()
