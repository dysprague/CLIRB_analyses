"""Stage 26 -- does the direction-split rear+turn detector actually separate
LEFT-turned from RIGHT-turned rears?

Background. §8x split the `rear_turn` seed pool by turn direction and reported
precision/recall per direction (F35). But those numbers are SLEAP-vs-DANNCE
*agreement* (tp=|both|, fp=|sleap_only|, fn=|dannce_only|), not behavioral
correctness -- they say the two modalities fire on the same frames, not that
those frames are genuinely left-turned rears. A height-only corridor carries no
directional information at all, so `ear_mean_z`'s ~94%/95% on BOTH directions
is consistent with "both modalities match every rear and agree while doing so".
That premise was never tested; §8x's tuning recommendation rests on it.

This script tests it directly. For each direction D in {left, right} it builds
the direction-specific SLEAP detector exactly as §8x/stage24 does, then scores
that detector against BOTH pools of seed events:

  hit(D, P) = fraction of pool P's seed rears that detector D fires on
              (match within PAIR_THRESH frames)

If the detectors are directionally selective:   hit(L,L) >> hit(L,R).
If they carry no directional information:       hit(L,L) ~= hit(L,R).

Two supporting diagnostics, because a high hit rate is trivial if a detector
simply fires everywhere:
  n_matches   -- how many events the detector fires on across the session
  self_overlap-- fraction of the LEFT detector's matches that the RIGHT
                 detector also fires on (within PAIR_THRESH). ~1.0 means the
                 two "direction-specific" detectors are the same detector.

Also reports, as a ceiling on what any direction split can achieve,
`sign_stability`: the fraction of rear+turn events where SLEAP and DANNCE
agree on the turn SIGN in the first place. If the two modalities disagree
about which way the rat turned, no detector built on one can be validated
against the other.

Session set, alignment, filtering and feature code are all imported from
stage24 -- this is the same pipeline, only the scoring differs.

Writes:
  results/spatial_confound/stage26_separation_per_session.csv
  results/spatial_confound/stage26_separation_summary.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'scripts')); os.chdir(REPO)

from clirb import events as ev
import stage24_multi_template_features as s24

OUT = 'results/spatial_confound'
W = s24.W
REFRACTORY = s24.REFRACTORY
PAIR_THRESH = s24.PAIR_THRESH

# The two combos F35 reported, plus the ruled-out torsion pairing as a contrast
# (§8v: fails on rearing GT -- does it at least separate directions?).
COMBOS = [
    ('ear_mean_z',),
    ('ear_mean_z', 'angle_skip_xy'),
    ('ear_mean_z', 'angle_torsion_xy'),
]


def split_by_direction(rearturn_ends, feat, w=W):
    """Sign of angle_torsion_xy at the frame of peak |torsion| in each window."""
    sig = feat['angle_torsion_xy']
    left, right = [], []
    for e in np.asarray(rearturn_ends, dtype=np.int64):
        s0 = max(0, e - w + 1)
        seg = sig[s0:e + 1]
        if len(seg) == 0:
            continue
        peak = seg[np.argmax(np.abs(seg))]
        (left if peak < 0 else right).append(e)
    return np.array(left, dtype=np.int64), np.array(right, dtype=np.int64)


def build_detector(combo, seed_ends, feat_sl, feat_dn):
    """stage24.match_combo's pipeline, stopped at the SLEAP match set.

    Returns (sleap_match_ends, n_gt) or (None, 0) if the pool is too small.
    """
    if len(seed_ends) < 5:
        return None, 0
    s_traj = np.stack([feat_sl[f] for f in combo], axis=1)
    d_traj = np.stack([feat_dn[f] for f in combo], axis=1)
    acc = max(1, round(1.5 * len(combo)))

    s_seg1, _ = s24.windows_at(s_traj, seed_ends)
    d_seg1, _ = s24.windows_at(d_traj, seed_ends)
    if len(s_seg1) < 5:
        return None, 0
    dannce_tmpl0, _, bounds1 = s24.rebuild_and_bounds(d_seg1, s_seg1)

    gt_f, _ = ev.get_template_match_indices(d_traj, dannce_tmpl0, bounds1,
                                            refractory_frames=REFRACTORY, accepted_errors=acc)
    if len(gt_f) == 0:
        return None, 0
    gt_f = ev.correct_matches(d_traj, dannce_tmpl0, gt_f)

    s_seg, _ = s24.windows_at(s_traj, gt_f)
    d_seg, _ = s24.windows_at(d_traj, gt_f)
    if len(s_seg) < 5:
        return None, len(gt_f)
    sleap_tmpl, _, bounds = s24.rebuild_and_bounds(s_seg, d_seg)

    sm, _ = ev.get_template_match_indices(s_traj, sleap_tmpl, bounds,
                                          refractory_frames=REFRACTORY, accepted_errors=acc)
    sm = ev.correct_matches(s_traj, sleap_tmpl, sm) if len(sm) else np.array([], dtype=int)
    return sm, len(gt_f)


def hit_rate(matches, pool_ends, thresh=PAIR_THRESH):
    """Fraction of pool_ends with a detector match within `thresh` frames."""
    if len(pool_ends) == 0:
        return np.nan, 0, 0
    if matches is None or len(matches) == 0:
        return 0.0, 0, len(pool_ends)
    m = np.asarray(matches)
    hits = int(sum(np.min(np.abs(m - e)) < thresh for e in pool_ends))
    return hits / len(pool_ends), hits, len(pool_ends)


def overlap_frac(a, b, thresh=PAIR_THRESH):
    """Fraction of a's events that b also fires on."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return np.nan
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean([np.min(np.abs(b - t)) < thresh for t in a]))


def sign_stability(left_ends, right_ends, feat_dn, w=W):
    """Fraction of SLEAP-labeled events where DANNCE agrees on the turn sign."""
    sig = feat_dn['angle_torsion_xy']
    agree = tot = 0
    for ends, want_neg in ((left_ends, True), (right_ends, False)):
        for e in np.asarray(ends, dtype=np.int64):
            s0 = max(0, e - w + 1)
            seg = sig[s0:e + 1]
            if len(seg) == 0:
                continue
            peak = seg[np.argmax(np.abs(seg))]
            agree += int((peak < 0) == want_neg)
            tot += 1
    return (agree / tot if tot else np.nan), tot


def main():
    sessions_by_rat = {rat: s24.off_new_sessions(rat) for rat in ['R4', 'R5', 'R6']}
    for rat, ss in sessions_by_rat.items():
        print(f'{rat}: {len(ss)} corrector-OFF + new-model sessions', flush=True)

    rows = []
    for rat, sessions in sessions_by_rat.items():
        for i, session in enumerate(sessions):
            sd = s24.session_data(rat, session)
            if sd is None:
                print(f'  [{rat}/{session}] unusable/excluded, skipping', flush=True)
                continue
            feat_sl, feat_dn = sd['feat_sl'], sd['feat_dn']
            left, right = split_by_direction(sd['seeds']['rear_turn'], feat_sl)
            sstab, n_sstab = sign_stability(left, right, feat_dn)

            for combo in COMBOS:
                det, pools = {}, {'left': left, 'right': right}
                for d in ('left', 'right'):
                    det[d], _ = build_detector(combo, pools[d], feat_sl, feat_dn)
                if det['left'] is None and det['right'] is None:
                    continue
                row = dict(rat=rat, session=session, combo='+'.join(combo),
                           n_left=len(left), n_right=len(right),
                           sign_stability=sstab, n_sign=n_sstab,
                           n_match_left=(0 if det['left'] is None else len(det['left'])),
                           n_match_right=(0 if det['right'] is None else len(det['right'])),
                           overlap_LR=overlap_frac(det['left'], det['right']),
                           overlap_RL=overlap_frac(det['right'], det['left']))
                for d in ('left', 'right'):
                    for p in ('left', 'right'):
                        hr, h, n = hit_rate(det[d], pools[p])
                        row[f'hit_{d}_on_{p}'] = hr
                        row[f'nhit_{d}_on_{p}'] = h
                        row[f'npool_{d}_on_{p}'] = n
                rows.append(row)
            print(f'  [{i+1}/{len(sessions)}] {rat}/{session} done '
                  f'(rear+turn: L={len(left)} R={len(right)})', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/stage26_separation_per_session.csv', index=False)
    print(f'\nwrote {OUT}/stage26_separation_per_session.csv ({len(d)} rows)', flush=True)

    # Pool over sessions by summing hit counts (not averaging per-session rates,
    # which would over-weight small sessions).
    summ = []
    for (combo,), g_all in [((c,), d[d.combo == '+'.join(c)]) for c in COMBOS]:
        name = '+'.join(combo)
        for rat in ['pooled', 'R4', 'R5', 'R6']:
            g = g_all if rat == 'pooled' else g_all[g_all.rat == rat]
            if not len(g):
                continue
            rec = dict(rat=rat, combo=name, n_sessions=len(g),
                       n_left=int(g.n_left.sum()), n_right=int(g.n_right.sum()),
                       sign_stability=float(np.average(g.sign_stability.fillna(0),
                                                       weights=g.n_sign.clip(lower=0) + 1e-9)),
                       median_matches_left=float(g.n_match_left.median()),
                       median_matches_right=float(g.n_match_right.median()),
                       overlap_LR=float(g.overlap_LR.mean(skipna=True)),
                       overlap_RL=float(g.overlap_RL.mean(skipna=True)))
            for dd in ('left', 'right'):
                for p in ('left', 'right'):
                    h = g[f'nhit_{dd}_on_{p}'].sum(); n = g[f'npool_{dd}_on_{p}'].sum()
                    rec[f'hit_{dd}_on_{p}'] = (h / n) if n else np.nan
            rec['sep_left'] = rec['hit_left_on_left'] - rec['hit_left_on_right']
            rec['sep_right'] = rec['hit_right_on_right'] - rec['hit_right_on_left']
            summ.append(rec)

    s = pd.DataFrame(summ)
    s.to_csv(f'{OUT}/stage26_separation_summary.csv', index=False)
    print(f'wrote {OUT}/stage26_separation_summary.csv', flush=True)

    print('\n=== separation test: hit rate of each direction detector on each pool ===')
    print('(sep = own-pool hit rate minus other-pool hit rate; ~0 means no directional info)\n')
    for combo in COMBOS:
        name = '+'.join(combo)
        p = s[(s.combo == name) & (s.rat == 'pooled')]
        if not len(p):
            continue
        r = p.iloc[0]
        print(f'{name}')
        print(f'   L detector: on L {r.hit_left_on_left:.3f} | on R {r.hit_left_on_right:.3f}'
              f'   -> sep {r.sep_left:+.3f}')
        print(f'   R detector: on R {r.hit_right_on_right:.3f} | on L {r.hit_right_on_left:.3f}'
              f'   -> sep {r.sep_right:+.3f}')
        print(f'   detector overlap L->R {r.overlap_LR:.3f}, R->L {r.overlap_RL:.3f}'
              f' | median matches/session L={r.median_matches_left:.0f} R={r.median_matches_right:.0f}')
        print(f'   SLEAP-DANNCE turn-sign agreement: {r.sign_stability:.3f}\n')


if __name__ == '__main__':
    main()
