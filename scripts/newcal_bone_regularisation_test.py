"""How much of the port-sector DANNCE error would bone-length regularisation
actually remove?

The skeleton is a tree (23 nodes, 22 edges), so "regularise bone lengths" has
an exact one-pass form: root the tree at SpineM and walk outward, placing each
child along the DIRECTION the network predicted but at the REFERENCE length.
That keeps every predicted joint angle and changes only the segment lengths --
which is precisely the failure mode measured near the port.

The reference lengths come from DANNCE'S OWN far-from-port frames in the SAME
session (COM more than 300 mm from the water port), where DANNCE and SLEAP
agree.  No SLEAP information enters the correction, so measuring the corrected
pose against SLEAP afterwards is not circular -- and it mirrors what you could
actually deploy, since it needs nothing but DANNCE and a port location.

Evaluation, per session:
  near-port   per-keypoint RMSE between DANNCE and SLEAP, before and after
  far-port    the same, as the "healthy" reference the correction is trying to
              reach, and as a control that the operation is benign where
              nothing is wrong
  recovered   the fraction of the near-port EXCESS error that the correction
              removes:  (before_near - after_near) / (before_near - before_far)

DANNCE is Procrustes-aligned into SLEAP's frame first, with the fit taken on
far-from-port frames only so the alignment is not itself distorted by the
region under test.

Writes results/newcal/bone_regularisation_test.csv
"""
import os, sys
from collections import deque, defaultdict
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import dual as du
from config import EDGES, NODES, NODE_IDX as NI
from scripts.newcal_sleap_dannce_eval import robust_proc

OUT = 'results/newcal'
ROOT = NI['SpineM']
NEAR, FAR = 150.0, 300.0
SESSIONS = [('R7', '2026_10_26_1'), ('R8', '2026_10_26_1'), ('R9', '2026_10_26_1'),
            ('R7', '2026_10_23_1'), ('R8', '2026_10_23_1'), ('R9', '2026_10_23_1')]


def tree_order(edges, root=ROOT):
    """BFS parent list over the skeleton tree: [(parent, child), ...]."""
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    seen = {root}
    order, q = [], deque([root])
    while q:
        p = q.popleft()
        for c in adj[p]:
            if c not in seen:
                seen.add(c); order.append((p, c)); q.append(c)
    missing = set(range(len(NODES))) - seen
    if missing:
        raise RuntimeError(f'skeleton is not a connected tree; unreached: '
                           f'{[NODES[i] for i in missing]}')
    return order


ORDER = tree_order(EDGES)


def regularise(kp, target):
    """Impose reference bone lengths while keeping predicted bone directions.
    kp (T,23,3); target dict (parent,child)->length. Root position unchanged."""
    out = kp.copy()
    for p, c in ORDER:
        v = kp[:, c] - kp[:, p]
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        u = np.divide(v, np.maximum(n, 1e-9))
        out[:, c] = out[:, p] + u * target[(p, c)]
    return out


def main():
    rows = []
    for rat, session in SESSIONS:
        raw = sc.get_raw_session(rat, session, want_dual=True)
        d3 = du.dannce_aligned(raw)
        if d3 is None:
            continue
        n = min(len(raw['triang_3d']), len(d3))
        sl = np.asarray(raw['triang_3d'][:n], np.float32)
        dn = np.asarray(d3[:n], np.float32)
        port = np.asarray(raw['port_xy']).ravel()[:2]
        if not np.isfinite(port).all():
            print(f'{rat}/{session}: no port location', flush=True)
            continue
        com = sl.mean(axis=1)[:, :2]
        dist = np.hypot(com[:, 0] - port[0], com[:, 1] - port[1])
        near, far = dist < NEAR, dist > FAR
        if near.sum() < 500 or far.sum() < 500:
            print(f'{rat}/{session}: too few frames near/far ({near.sum()}/{far.sum()})',
                  flush=True)
            continue

        # alignment fitted on FAR frames only, so the region under test cannot
        # distort the transform used to judge it
        idx = np.where(far)[0]
        samp = np.random.default_rng(0).choice(idx, size=min(3000, len(idx)),
                                               replace=False)
        fit = robust_proc(dn[samp], sl[samp])
        dnw = fit['apply'](dn).astype(np.float32)

        # reference lengths from DANNCE's own far-from-port frames
        target = {(p, c): float(np.median(
            np.linalg.norm(dnw[far][:, c] - dnw[far][:, p], axis=-1)))
            for p, c in ORDER}
        corr = regularise(dnw, target)

        def rmse(a, m):
            return float(np.sqrt(((sl[m] - a[m]) ** 2).sum(axis=-1).mean()))

        b_near, a_near = rmse(dnw, near), rmse(corr, near)
        b_far, a_far = rmse(dnw, far), rmse(corr, far)
        excess = b_near - b_far
        rec = dict(rat=rat, session=session,
                   era='Oct26' if '10_26' in session else 'Oct23',
                   n_near=int(near.sum()), n_far=int(far.sum()),
                   frac_near=float(near.mean()),
                   rmse_near_before=b_near, rmse_near_after=a_near,
                   rmse_far_before=b_far, rmse_far_after=a_far,
                   excess_before=excess,
                   recovered_frac=float((b_near - a_near) / excess) if excess > 0 else np.nan)
        # per-keypoint, near-port
        for k, name in enumerate(NODES):
            d0 = np.linalg.norm(sl[near][:, k] - dnw[near][:, k], axis=-1)
            d1 = np.linalg.norm(sl[near][:, k] - corr[near][:, k], axis=-1)
            rec[f'kp_{name}_before'] = float(np.sqrt((d0 ** 2).mean()))
            rec[f'kp_{name}_after'] = float(np.sqrt((d1 ** 2).mean()))
        rows.append(rec)
        print(f'{rat}/{session}: near-port RMSE {b_near:.1f} -> {a_near:.1f} mm '
              f'(far-port reference {b_far:.1f}); recovered '
              f'{100*rec["recovered_frac"]:.0f}% of the excess', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/bone_regularisation_test.csv', index=False)
    pd.set_option('display.width', 220)
    print('\n=== summary ===')
    print(d[['rat', 'session', 'era', 'frac_near', 'rmse_far_before',
             'rmse_near_before', 'rmse_near_after', 'recovered_frac']]
          .round(3).to_string(index=False))
    print('\n=== per-keypoint, near-port, Oct 26 (median over rats) ===')
    o = d[d.era == 'Oct26']
    t = pd.DataFrame({
        'before': [o[f'kp_{n}_before'].median() for n in NODES],
        'after': [o[f'kp_{n}_after'].median() for n in NODES]}, index=NODES)
    t['delta'] = t.after - t.before
    print(t.sort_values('before', ascending=False).round(1).to_string())
    print(f'\nwrote {OUT}/bone_regularisation_test.csv')


if __name__ == '__main__':
    main()
