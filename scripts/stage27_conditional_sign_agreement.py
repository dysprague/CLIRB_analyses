"""Stage 27 -- turn-sign agreement CONDITIONED ON A REAL TURN.

§8aa found SLEAP-vs-DANNCE turn-sign agreement plateaus at ~85% across 14
direction-bearing features and three structurally different formulations, and
concluded the ceiling looks intrinsic to the measurement. But that figure is
computed over ALL frames, and most frames have the rat barely turning -- where
the sign is meaningless by construction and the two modalities are effectively
coin-flipping on noise. A feature can look capped at 85% overall while agreeing
almost perfectly whenever the animal is actually turned.

That conditional number, not the unconditional one, is what decides whether
direction-specific reward is viable: an online rule only has to get the
direction right on frames where there IS a direction.

Method: same session set, alignment, filtering and features as stage21 (which
this imports). For each direction-bearing feature, sign agreement is recomputed
on the subset of frames where |SLEAP value| exceeds a threshold, swept as
quantiles of that feature's own |value| distribution (so the sweep is
comparable across mm / deg / deg-per-second units), plus absolute thresholds
for the interpretable turn features.

Frames are subsampled per session (SUBSAMPLE_PER_SESSION) so exact pooled
quantiles can be computed in one pass without holding all 7.2M frames.

Writes:
  results/spatial_confound/stage27_conditional_sign.csv        (quantile sweep)
  results/spatial_confound/stage27_conditional_sign_absolute.csv (abs thresholds)
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'scripts')); os.chdir(REPO)

from clirb import features as cf
import stage21_feature_reliability as s21

OUT = 'results/spatial_confound'
SUBSAMPLE_PER_SESSION = 6000
QUANTILES = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# Every feature whose SIGN encodes turn direction. heading_earhip is excluded:
# it is an absolute world heading, so its sign means "which half-plane is the
# rat facing", not "which way did it turn".
DIRECTION_FEATURES = [f for f in (list(cf.TURN_FEATURES) + list(cf.LATERAL_FEATURES)
                                  + ['yaw_rate', 'cum_heading_change'])]

# Absolute thresholds that mean something physically, for the features where a
# number can be named: degrees of accumulated turn, or mm of lateral offset.
ABS_THRESHOLDS = {
    'cum_heading_change': [15, 30, 45, 60, 90],
    'yaw_rate': [30, 60, 90, 120, 180],
    'hip_asym': [2, 5, 10, 15, 20],
    'shoulder_hip_asym': [2, 5, 10, 15, 20],
    'sagitta_spine': [2, 5, 10, 15, 20],
    'lateral_offset_head': [5, 10, 20, 30, 40],
    'angle_torsion_xy': [15, 30, 45, 60, 90],
    'angle_skip_xy': [15, 30, 45, 60, 90],
    'angle_body_head_xy': [15, 30, 45, 60, 90],
}


def main():
    rng = np.random.default_rng(0)
    # per rat, per feature: pooled subsample of (|sleap value|, signs agree?)
    store = {rat: {f: {'mag': [], 'agree': []} for f in DIRECTION_FEATURES}
             for rat in ['R4', 'R5', 'R6']}

    for rat in ['R4', 'R5', 'R6']:
        sessions = s21.dual_sessions(rat)
        print(f'{rat}: {len(sessions)} dual-modality sessions', flush=True)
        for i, session in enumerate(sessions):
            out, fit_info = s21.session_result(rat, session, rng)
            if fit_info is None or out is None:
                continue
            f_sl, f_dn = out
            n = f_sl['ear_mean_z'].shape[0]
            take = (np.arange(n) if n <= SUBSAMPLE_PER_SESSION
                    else np.sort(rng.choice(n, SUBSAMPLE_PER_SESSION, replace=False)))
            for feat in DIRECTION_FEATURES:
                a, b = np.asarray(f_sl[feat])[take], np.asarray(f_dn[feat])[take]
                store[rat][feat]['mag'].append(np.abs(a).astype(np.float32))
                store[rat][feat]['agree'].append((np.sign(a) == np.sign(b)))
            if i % 20 == 0:
                print(f'  [{i+1}/{len(sessions)}] {session}', flush=True)
        print(f'{rat}: done', flush=True)

    # ---- flatten ----
    flat = {rat: {f: (np.concatenate(v['mag']), np.concatenate(v['agree']))
                  for f, v in d.items()} for rat, d in store.items()}
    pooled = {f: (np.concatenate([flat[r][f][0] for r in flat]),
                  np.concatenate([flat[r][f][1] for r in flat]))
              for f in DIRECTION_FEATURES}

    def agree_above(mag, agree, thr):
        m = mag >= thr
        return (100.0 * agree[m].mean() if m.sum() else np.nan), int(m.sum())

    rows = []
    for feat in DIRECTION_FEATURES:
        for rat, src in [('pooled', pooled), ('R4', flat), ('R5', flat), ('R6', flat)]:
            mag, agree = (src[feat] if rat == 'pooled' else src[rat][feat])
            for q in QUANTILES:
                thr = float(np.quantile(mag, q)) if q > 0 else 0.0
                pct, n = agree_above(mag, agree, thr)
                rows.append(dict(rat=rat, feature=feat, unit=cf.UNITS[feat], quantile=q,
                                 threshold=thr, sign_agree_pct=pct, n_frames=n))
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/stage27_conditional_sign.csv', index=False)
    print(f'\nwrote {OUT}/stage27_conditional_sign.csv ({len(d)} rows)')

    rows = []
    for feat, thrs in ABS_THRESHOLDS.items():
        for rat, src in [('pooled', pooled), ('R4', flat), ('R5', flat), ('R6', flat)]:
            mag, agree = (src[feat] if rat == 'pooled' else src[rat][feat])
            for thr in thrs:
                pct, n = agree_above(mag, agree, float(thr))
                rows.append(dict(rat=rat, feature=feat, unit=cf.UNITS[feat], threshold=thr,
                                 sign_agree_pct=pct, n_frames=n,
                                 frac_of_frames=n / max(len(mag), 1)))
    da = pd.DataFrame(rows)
    da.to_csv(f'{OUT}/stage27_conditional_sign_absolute.csv', index=False)
    print(f'wrote {OUT}/stage27_conditional_sign_absolute.csv ({len(da)} rows)')

    print('\n=== sign agreement vs. |value| quantile (pooled) ===')
    p = d[d.rat == 'pooled']
    hdr = '  '.join(f'q{int(q*100):02d}' for q in QUANTILES)
    print(f'{"feature":24s} {hdr}')
    order = (p[p['quantile'] == 0.90].sort_values('sign_agree_pct', ascending=False).feature)
    for feat in order:
        g = p[p.feature == feat].set_index('quantile')
        vals = '  '.join(f'{g.loc[q].sign_agree_pct:4.1f}' for q in QUANTILES)
        print(f'{feat:24s} {vals}')


if __name__ == '__main__':
    main()
