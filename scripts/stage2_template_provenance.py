"""Stage 2 — template provenance and the H2 mechanism test.

(a) Map every recorded build/refine match to its arena angle theta.
(b) Re-run the builder's OWN clustering exactly as
    ~/CLIRB/refine_templates_mult_session.py does it, and test whether the
    KMeans clusters segregate by arena sector.

Builder parameters replicated verbatim:
    WIDEN_FACTOR = 2.5, WIDEN_MATCH_METHOD = 'greedy_best', ACCEPTED_ERRORS = 3,
    K_CLUSTERS = 10, KMeans(n_init=10, random_state=0), z-scored flattened
    (N, W, n_pcs) windows, CLUSTER_SELECTION = 'largest',
    REFINED_BOUNDS_MODE = 'max_dev'
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial.geometry import ARENA, sector_index

WIDEN_FACTOR = 2.5
K_CLUSTERS = 10
ACCEPTED_ERRORS = 3
W = 30
SECTOR_W = 45          # match the builder-era coarse binning for readability
RNG = np.random.default_rng(0)
OUT = 'results/spatial_confound'


# ---------------------------------------------------------------- builder ops
def oob_and_dist_all(pc, tpl, band, W):
    n = pc.shape[0]
    if n < W:
        return np.array([], int), np.array([], float), np.array([], int)
    idx = np.arange(W)[None, :] + np.arange(n - W + 1)[:, None]
    win = pc[idx]
    oob = ((win >= tpl + band) | (win <= tpl - band)).sum(axis=(1, 2))
    nb = np.where(band == 0, 1.0, band)
    dist = (((win - tpl) / nb) ** 2).sum(axis=(1, 2))
    return oob, dist, np.arange(W - 1, n)


def matches_greedy(pc, tpl, band, W, acc=ACCEPTED_ERRORS):
    oob, dist, ends = oob_and_dist_all(pc, tpl, band, W)
    ok = oob <= acc
    cand, cd = ends[ok], dist[ok]
    keep = []
    for i in np.argsort(cd, kind='stable'):
        ef = int(cand[i])
        if all(abs(ef - a) >= W for a in keep):
            keep.append(ef)
    return np.array(sorted(keep), dtype=int)


def bounds_from_windows(windows, mode='max_dev'):
    mean = windows.mean(axis=0)
    lo, hi = windows.min(axis=0), windows.max(axis=0)
    if mode == 'max_dev':
        band = np.maximum(mean - lo, hi - mean)
    elif mode == 'half_range':
        band = (hi - lo) / 2.0
    else:
        band = (hi - lo)
    return mean, band


def session_pcs_and_theta(rat, session):
    """Return (pcs_used (T,2), theta_per_frame (T,), template) or None."""
    c = sc.get_raw_session(rat, session, want_dual=False)
    t = tp.load_template(rat, session)
    if c is None or t is None:
        return None
    pcs_full, _ = ev.project_session_to_pcs(c['triang_3d'], t, flip_z=True)
    pcu = np.asarray(t['pcs_to_use']).ravel().astype(int)
    com = c['triang_3d'].mean(axis=1)
    return pcs_full[:, pcu[:2]], ARENA.theta(com[:, :2]), t


# ------------------------------------------------------- (a) provenance -> th
def provenance_theta():
    rows = []
    seen = set()
    for rat in ['R4', 'R5', 'R6']:
        for p in sorted(glob.glob(f'/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data/{rat}/*/sleap/{rat}_template_*.npz')) + \
                 sorted(glob.glob(f'/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data/{rat}/templates/{rat}_template_*.npz')):
            name = os.path.basename(p)
            if (rat, name) in seen:
                continue
            seen.add((rat, name))
            z = np.load(p, allow_pickle=True)
            t = {k: z[k] for k in z.files}
            prov = tp.build_provenance(t)
            prov = prov[prov['exact']]
            if prov.empty:
                continue
            for sess, g in prov.groupby('session'):
                c = sc.get_raw_session(rat, sess, want_dual=False)
                if c is None:
                    continue
                com = c['triang_3d'].mean(axis=1)
                f = np.clip(g['frame'].to_numpy(int), 0, len(com) - 1)
                th = ARENA.theta(com[f, :2])
                for kind, tt in zip(g['kind'], th):
                    rows.append(dict(rat=rat, template=name, session=sess,
                                     kind=kind, theta=float(tt)))
    return pd.DataFrame(rows)


# ------------------------------------------- (b) reproduce builder clustering
def cluster_run(rat, sessions, label):
    """Run the builder's pool->KMeans exactly; return per-window records."""
    pool_win, pool_th, pool_sess = [], [], []
    for s in sessions:
        r = session_pcs_and_theta(rat, s)
        if r is None:
            continue
        pcs, theta, t = r
        tpl = t['template'][:, :2]
        band = t['bounds_to_use'][:, :2] * WIDEN_FACTOR
        ends = matches_greedy(pcs, tpl, band, W)
        for ef in ends:
            if ef - W + 1 < 0:
                continue
            pool_win.append(pcs[ef - W + 1:ef + 1])
            pool_th.append(theta[ef])
            pool_sess.append(s)
    if len(pool_win) < K_CLUSTERS * 2:
        return None
    X_windows = np.stack(pool_win)
    theta = np.asarray(pool_th)
    X = X_windows.reshape(len(X_windows), -1)
    Xz = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=0).fit(Xz)
    lab = km.labels_
    sec = sector_index(theta, SECTOR_W)

    ami = adjusted_mutual_info_score(lab, sec)
    mi = mutual_info_score(lab, sec)
    null = np.array([adjusted_mutual_info_score(lab, RNG.permutation(sec))
                     for _ in range(1000)])
    p = float((null >= ami).mean())

    sizes = np.bincount(lab, minlength=K_CLUSTERS)
    largest = int(np.argmax(sizes))
    sel = lab == largest
    sec_sel = sec[sel]
    top_share = float(np.bincount(sec_sel, minlength=8).max() / sel.sum())

    # how much narrower are max_dev bounds from the largest cluster vs a
    # sector-balanced sample of the same size?
    mean_l, band_l = bounds_from_windows(X_windows[sel])
    n_per = max(3, int(sel.sum() / max(1, len(np.unique(sec)))))
    bal_idx = []
    for s_ in np.unique(sec):
        ii = np.where(sec == s_)[0]
        bal_idx.extend(RNG.choice(ii, size=min(len(ii), n_per), replace=False))
    mean_b, band_b = bounds_from_windows(X_windows[np.array(bal_idx)])
    widen = float(band_b.mean() / band_l.mean())

    return dict(rat=rat, label=label, n_windows=int(len(lab)),
                n_sessions=len(set(pool_sess)),
                ami=float(ami), mi=float(mi), p_perm=p,
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                largest_cluster=largest, largest_n=int(sel.sum()),
                largest_top_sector_share=top_share,
                bounds_widen_if_balanced=widen,
                sector_hist_largest=np.bincount(sec_sel, minlength=8).tolist(),
                sector_hist_all=np.bincount(sec, minlength=8).tolist())


def main():
    os.makedirs(OUT, exist_ok=True)
    print('=' * 78); print('(a) PROVENANCE -> THETA'); print('=' * 78)
    prov = provenance_theta()
    prov.to_csv(f'{OUT}/template_provenance.csv', index=False)
    for (rat, tmpl, kind), g in prov.groupby(['rat', 'template', 'kind']):
        h = np.bincount(sector_index(g.theta.values, SECTOR_W), minlength=8)
        print(f'{rat} {tmpl:26s} [{kind:7s}] n={len(g):4d} '
              f'top=[{int(np.argmax(h))}] {h.max()/len(g)*100:3.0f}%  hist={h.tolist()}')

    print()
    print('=' * 78); print('(b) BUILDER CLUSTERING -> SECTOR SEGREGATION'); print('=' * 78)
    runs = []
    # historical refinements with recorded session lists
    hist = {
        ('R4', 'R4_template_refined'): ['2026_06_25_1', '2026_06_25_2', '2026_06_26_1'],
        ('R5', 'R5_template_4'): ['2026_06_24_1', '2026_06_25_1', '2026_06_26_2'],
        ('R5', 'R5_template_5'): ['2026_06_24_1', '2026_06_25_1', '2026_06_26_2'],
    }
    for (rat, name), ss in hist.items():
        r = cluster_run(rat, ss, f'historical:{name}')
        if r: runs.append(r); print(json.dumps(r))
    # general sweep: consecutive session triples per rat
    for rat in ['R4', 'R5', 'R6']:
        allses = sorted(os.path.basename(f)[:-4]
                        for f in glob.glob(f'data/cache/{rat}/*.npz'))
        for i in range(0, len(allses) - 3, 12):
            ss = allses[i:i + 3]
            r = cluster_run(rat, ss, f'sweep:{ss[0]}')
            if r: runs.append(r); print(json.dumps(r))
    pd.DataFrame(runs).to_csv(f'{OUT}/stage2_clustering.csv', index=False)
    print(f'\nwrote {OUT}/stage2_clustering.csv ({len(runs)} runs)')


if __name__ == '__main__':
    main()
