"""Is `ear_mean_z` a reliable rear detector on the current setup, and does its
reliability depend on where in the arena the rat is?

Three parts:

  1. THRESHOLD CALIBRATION.  Sweep the hysteresis detector's (rise, reset,
     min_frames) over a grid, running the SAME detector on each modality's own
     `ear_mean_z` trace, and score SLEAP's events against DANNCE's.  This is
     the honest question for the rig: if the online rule detects rears from
     SLEAP with these thresholds, how well do those rears correspond to the
     rears DANNCE says happened?

  2. SLEAP-ONLY SWEEP against a fixed DANNCE ground truth, i.e. hold DANNCE at
     the calibrated operating point and move only SLEAP's rise.  This separates
     "what threshold should the rig use" (part 1) from "how separable are true
     rears at all" (part 2), and gives the precision/recall trade-off curve.

  3. SECTOR BREAKDOWN at the chosen operating point.  Each DANNCE rear is
     assigned to an arena sector by the DANNCE centre-of-mass at its END frame
     (the project's convention -- gating is evaluated at the rear end, never the
     start), and per-sector recall / precision are computed.  SPATIAL_CONFOUND_
     RESULTS.md section 3 found online match rates vary ~6x by sector in the
     OLD arena; this asks whether the detector feeding that rule is itself
     spatially uneven.

  THE ARENA IS REFITTED PER SESSION.  `clirb.spatial.geometry.ARENA`
  (centre -66,-98, radius 332) was fitted on R4/R5/R6 in the old room and is
  not valid after the environment change -- a stale centre would smear the
  sector assignment.  `fit_arena` is re-run on each session's own DANNCE COM.

EVENT MATCHING IS BY INTERVAL OVERLAP, greedily largest-overlap-first and 1:1 --
the same primitive `ev.rear_matched` uses.  For a reward rule the operational
question is "when DANNCE says a rear happened, did the SLEAP detector fire
during it, and when it fired, was there a real rear", which is overlap, not
peak coincidence.  Peak-time matching was tried first and rejected: a rear's
`ear_mean_z` peak is a broad plateau, so the argmax can move by more than the
300 ms tolerance between two modalities that agree perfectly about the episode,
which measures peak localisation rather than detection.  Peak-time and
end-frame differences are instead reported for matched pairs as a separate
TIMING quality measure, which is what an online rule needs to trigger on time.
A stricter IoU >= 0.5 variant is reported alongside as a sensitivity check
against overlap matching getting easier at low thresholds.

Heights are up-positive: SLEAP's stored world z points down, the Procrustes
puts DANNCE in that same frame, so z is negated on both after alignment.

Writes:
  results/newcal/rear_sweep.csv          per session x threshold combo
  results/newcal/rear_sector.csv         per session x sector, at the operating point
  results/newcal/rear_sleap_sweep.csv    SLEAP-only rise sweep vs fixed DANNCE
  results/newcal/rear_arena_fits.csv     per-session arena fit diagnostics
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb import features as cf
from clirb.spatial import dual as du
from clirb.spatial.geometry import fit_arena, sector_index
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample

OUT = 'results/newcal'
MATCH_TOL_FR = int(round(0.300 * ev.FPS))     # +/-300 ms
RISES = list(range(80, 231, 10))
DELTAS = [10, 20, 30, 40]                      # reset = rise - delta
MIN_FRAMES = [5, 10, 20]                       # ev default is 10 (0.5 s)
SECTOR_WIDTHS = [30, 60]


# --------------------------------------------------------------------------- #
def detect_rears_fast(x, rise, reset, min_frames):
    """Vectorised, bit-identical replacement for ev.detect_rears (verified).
    Returns (n, 3) int array of (start, end, n_above)."""
    x = np.asarray(x)
    n = len(x)
    up, dn = x > rise, x < reset
    e = np.where(up, 1, np.where(dn, -1, 0))
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(e != 0, idx, -1))
    state = np.zeros(n, dtype=bool)
    ok = last >= 0
    state[ok] = e[last[ok]] == 1
    if not state.any():
        return np.empty((0, 3), dtype=int)
    d = np.diff(np.concatenate([[False], state, [False]]).astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.minimum(np.where(d == -1)[0], n - 1)
    cum = np.concatenate([[0], np.cumsum(up)])
    n_above = cum[ends + 1] - cum[starts]
    keep = n_above > min_frames
    return np.stack([starts[keep], ends[keep], n_above[keep]], axis=1)


def peaks_of(rears, x):
    """Frame of maximum height inside each detected rear."""
    return np.array([s + int(np.argmax(x[s:e + 1])) if e >= s else s
                     for s, e, _ in rears], dtype=int)


def match_overlap(a, b, min_iou=0.0):
    """Greedy largest-overlap-first 1:1 match of two (n,3) rear arrays.
    Returns (pairs, n_matched). min_iou>0 additionally requires that fraction
    of intersection-over-union."""
    if not len(a) or not len(b):
        return [], 0
    a0, a1 = a[:, 0][:, None], a[:, 1][:, None]
    b0, b1 = b[:, 0][None, :], b[:, 1][None, :]
    inter = np.minimum(a1, b1) - np.maximum(a0, b0) + 1
    union = np.maximum(a1, b1) - np.minimum(a0, b0) + 1
    ok = inter > 0
    if min_iou > 0:
        ok &= (inter / union) >= min_iou
    if not ok.any():
        return [], 0
    score = np.where(ok, inter, -1)
    pairs = []
    ua = np.zeros(len(a), bool); ub = np.zeros(len(b), bool)
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    for i, j in order:
        if score[i, j] <= 0:
            break
        if not ua[i] and not ub[j]:
            ua[i] = ub[j] = True
            pairs.append((int(i), int(j)))
    return pairs, len(pairs)


def prf(tp, n_pred, n_true):
    p = tp / n_pred if n_pred else np.nan
    r = tp / n_true if n_true else np.nan
    f = 2 * p * r / (p + r) if (n_pred and n_true and (p + r) > 0) else np.nan
    return p, r, f


# --------------------------------------------------------------------------- #
TRACEDIR = f'{OUT}/rear_traces'
os.makedirs(TRACEDIR, exist_ok=True)


def session_traces(rat, session):
    """(ear_mean_z SLEAP, ear_mean_z DANNCE, DANNCE COM xy, arena, fit cost,
    Procrustes residual) or None. Cached to disk -- the Procrustes fit and the
    arena fit are the expensive part and neither depends on the thresholds."""
    cp = f'{TRACEDIR}/{rat}_{session}.npz'
    if os.path.isfile(cp):
        z = np.load(cp)
        from clirb.spatial.geometry import ArenaFrame
        a = ArenaFrame(float(z['cx']), float(z['cy']), float(z['radius']))
        return (z['zs'], z['zd'], z['com'], a, float(z['cost']), float(z['resid']))
    raw = sc.get_raw_session(rat, session, want_dual=True)
    if raw is None or not raw.get('has_dual', False):
        return None
    d3 = du.dannce_aligned(raw)
    if d3 is None:
        return None
    n = min(len(raw['triang_3d']), len(d3))
    if n < 1000:
        return None
    sl_raw = np.asarray(raw['triang_3d'][:n], np.float32)
    dn_raw = np.asarray(d3[:n], np.float32)
    samp, _ = balanced_sample(sl_raw.mean(axis=1)[:, :2], rng=np.random.default_rng(0))
    if len(samp) < 500:
        return None
    fit = robust_proc(dn_raw[samp], sl_raw[samp])
    S = ev.causal_median_5_kp(sl_raw, warmup='online').copy()
    D = ev.causal_median_5_kp(fit['apply'](dn_raw), warmup='online').copy()
    S[:, :, 2] *= -1.0
    D[:, :, 2] *= -1.0
    zs = cf.compute_features(S)['ear_mean_z']
    zd = cf.compute_features(D)['ear_mean_z']
    com = D.mean(axis=1)[:, :2]
    arena, cost = fit_arena(com)
    np.savez_compressed(cp, zs=zs.astype(np.float32), zd=zd.astype(np.float32),
                        com=com.astype(np.float32), cx=arena.cx, cy=arena.cy,
                        radius=arena.radius, cost=cost, resid=fit['residual'])
    return zs, zd, com, arena, float(cost), float(fit['residual'])


def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]

    sweep, sector, sleap_sweep, fits = [], [], [], []
    cache = {}

    for i, r in enumerate(inv.itertuples(), 1):
        out = session_traces(r.rat, r.session)
        if out is None:
            print(f'[{i}/{len(inv)}] {r.rat}/{r.session}: unusable', flush=True)
            continue
        zs, zd, com, arena, cost, resid = out
        cache[(r.rat, r.session)] = (zs, zd, com, arena, r.era)
        fits.append(dict(rat=r.rat, session=r.session, era=r.era,
                         arena_cx=arena.cx, arena_cy=arena.cy,
                         arena_r=arena.radius, arena_fit_cost_mm=cost,
                         procrustes_residual_mm=resid,
                         com_r_med=float(np.median(arena.radius_of(com)))))

        # ---- part 1: matched-threshold sweep -------------------------------
        for mf in MIN_FRAMES:
            for rise in RISES:
                for dl in DELTAS:
                    reset = rise - dl
                    rs = detect_rears_fast(zs, rise, reset, mf)
                    rd = detect_rears_fast(zd, rise, reset, mf)
                    pairs, tp = match_overlap(rs, rd)
                    _, tp_iou = match_overlap(rs, rd, min_iou=0.5)
                    if pairs:
                        ps, pd_ = peaks_of(rs, zs), peaks_of(rd, zd)
                        dpk = np.array([abs(ps[i] - pd_[j]) for i, j in pairs])
                        dend = np.array([abs(rs[i, 1] - rd[j, 1]) for i, j in pairs])
                        pk_med, end_med = float(np.median(dpk)), float(np.median(dend))
                        pk_in_tol = float((dpk <= MATCH_TOL_FR).mean())
                    else:
                        pk_med = end_med = pk_in_tol = np.nan
                    sweep.append(dict(rat=r.rat, session=r.session, era=r.era,
                                      rise=rise, reset=reset, min_frames=mf,
                                      n_sleap=len(rs), n_dannce=len(rd), tp=tp,
                                      tp_iou50=tp_iou, peak_dt_med_fr=pk_med,
                                      end_dt_med_fr=end_med,
                                      peak_within_300ms=pk_in_tol))
        print(f'[{i}/{len(inv)}] {r.rat}/{r.session} ({r.era}) swept', flush=True)

    d = pd.DataFrame(sweep)
    d.to_csv(f'{OUT}/rear_sweep.csv', index=False)
    pd.DataFrame(fits).to_csv(f'{OUT}/rear_arena_fits.csv', index=False)

    # ---- choose the operating point on the most recent era ----------------
    def pooled(sub):
        g = sub.groupby(['rise', 'reset', 'min_frames'])[['tp', 'n_sleap', 'n_dannce']].sum()
        p, rr, f = zip(*[prf(t, a, b) for t, a, b in
                         zip(g.tp, g.n_sleap, g.n_dannce)])
        g = g.assign(precision=p, recall=rr, f1=f)
        return g.reset_index()

    best = {}
    for era in d.era.unique():
        g = pooled(d[d.era == era])
        g = g[g.n_dannce >= 20]
        row = g.sort_values('f1', ascending=False).iloc[0]
        best[era] = row
        print(f'\nbest matched threshold for {era}: rise={row.rise:.0f} '
              f'reset={row.reset:.0f} min_frames={row.min_frames:.0f} -> '
              f'P={row.precision:.3f} R={row.recall:.3f} F1={row.f1:.3f} '
              f'(n_dannce={row.n_dannce:.0f})', flush=True)

    from scripts.newcal_eras import ERAS as _E
    last = next((e for e in reversed(_E) if e in best), None)
    op = best[last] if last else list(best.values())[0]
    RISE, RESET, MF = int(op.rise), int(op.reset), int(op.min_frames)
    print(f'\noperating point taken from the most recent era: '
          f'rise={RISE} reset={RESET} min_frames={MF}', flush=True)

    # ---- part 2: SLEAP-only sweep against fixed DANNCE --------------------
    for (rat, session), (zs, zd, com, arena, era) in cache.items():
        rd = detect_rears_fast(zd, RISE, RESET, MF)
        for rise in RISES:
            rs = detect_rears_fast(zs, rise, rise - (RISE - RESET), MF)
            _, tp = match_overlap(rs, rd)
            sleap_sweep.append(dict(rat=rat, session=session, era=era,
                                    sleap_rise=rise, dannce_rise=RISE,
                                    n_sleap=len(rs), n_dannce=len(rd), tp=tp))
    pd.DataFrame(sleap_sweep).to_csv(f'{OUT}/rear_sleap_sweep.csv', index=False)

    # ---- part 3: sector breakdown at the operating point ------------------
    for (rat, session), (zs, zd, com, arena, era) in cache.items():
        rs = detect_rears_fast(zs, RISE, RESET, MF)
        rd = detect_rears_fast(zd, RISE, RESET, MF)
        pairs, _ = match_overlap(rs, rd)
        s_matched = np.zeros(len(rs), bool); d_matched = np.zeros(len(rd), bool)
        for a, b in pairs:
            s_matched[a] = d_matched[b] = True
        theta = arena.theta(com)
        rad = arena.radius_of(com)
        abs_err = np.abs(zs - zd)
        for W in SECTOR_WIDTHS:
            sec_all = sector_index(theta, W)
            # rears are assigned by the COM at the rear END frame
            sec_d = sector_index(theta[rd[:, 1]], W) if len(rd) else np.empty(0, int)
            sec_s = sector_index(theta[rs[:, 1]], W) if len(rs) else np.empty(0, int)
            for k in range(int(round(360 / W))):
                md = sec_d == k
                ms = sec_s == k
                occ = sec_all == k
                sector.append(dict(
                    rat=rat, session=session, era=era, sector_w=W, sector=k,
                    sector_centre_deg=(-180 + W * (k + 0.5)),
                    n_dannce=int(md.sum()), n_sleap=int(ms.sum()),
                    tp_recall=int(d_matched[md].sum()) if len(rd) else 0,
                    tp_precision=int(s_matched[ms].sum()) if len(rs) else 0,
                    occupancy_frames=int(occ.sum()),
                    med_abs_err_mm=float(np.median(abs_err[occ])) if occ.any() else np.nan,
                    med_radius_mm=float(np.median(rad[occ])) if occ.any() else np.nan,
                    med_radius_rears=float(np.median(rad[rd[md, 1]])) if md.any() else np.nan,
                ))
    pd.DataFrame(sector).to_csv(f'{OUT}/rear_sector.csv', index=False)
    print(f'\nwrote {OUT}/rear_sweep.csv, rear_sleap_sweep.csv, '
          f'rear_sector.csv, rear_arena_fits.csv')


if __name__ == '__main__':
    main()
