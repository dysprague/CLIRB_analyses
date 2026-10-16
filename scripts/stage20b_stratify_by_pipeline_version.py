"""Stage 20b -- is the stage20 world-space keypoint ranking robust across
pipeline versions, or driven by one bad configuration?

Reuses stage20's per-session, per-keypoint sum-of-squared-error CSV
(no need to redo the Procrustes fits) and joins it against each session's
OWN `sleap/campy_config.yaml` to recover three independent version axes:

  - SLEAP model      (process_params.model_path)      -- 2 distinct models used
  - corrector_ckpt   (behavior_params.corrector_ckpt)  -- ON (a path) / OFF (null)
  - calibration      (behavior_params.calibration_path) -- 3 distinct calibrations

Writes:
  results/spatial_confound/stage20b_session_metadata.csv
  results/spatial_confound/stage20b_stratified_ranking.csv
"""
import os, sys, glob
import numpy as np
import pandas as pd
import yaml

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from config import NODE_IDX

DATA_ROOT = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
OUT = 'results/spatial_confound'
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]


def session_meta(rat, session):
    cfg = os.path.join(DATA_ROOT, rat, session, 'sleap', 'campy_config.yaml')
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg) as fh:
            c = yaml.safe_load(fh)
        model = c.get('process_params', {}).get('model_path', 'MISSING')
        calib = c.get('behavior_params', {}).get('calibration_path', 'MISSING')
        corr_raw = c.get('behavior_params', {}).get('corrector_ckpt', None)
        corrector = 'ON' if corr_raw else 'OFF'
    except Exception:
        return None
    model_short = 'old_n2298' if '2298' in model else ('new_n9975' if '9975' in model else model)
    calib_short = os.path.basename(calib.rstrip('/')) if calib != 'MISSING' else calib
    return dict(rat=rat, session=session, model=model_short, calibration=calib_short,
               corrector=corrector, model_path=model, calib_path=calib)


def pooled_rmse(df, group_mask):
    g = df[group_mask]
    n = g['n_frames'].sum()
    if n == 0:
        return None, 0
    mse = {kp: g[f'sqsum_{kp}'].sum() / n for kp in KP_NAMES}
    return mse, int(n)


def print_ranking(label, mse, n):
    print(f'\n{label}  ({n} frames)')
    order = sorted(mse, key=lambda k: mse[k])
    for rank, kp in enumerate(order, 1):
        print(f'  {rank:2d}. {kp:10s}  RMSE={np.sqrt(mse[kp]):6.2f} mm')
    return order


def main():
    sess = pd.read_csv(f'{OUT}/stage20_session_keypoint_sums.csv')
    meta_rows = [session_meta(r.rat, r.session) for r in sess.itertuples()]
    meta = pd.DataFrame([m for m in meta_rows if m is not None])
    meta.to_csv(f'{OUT}/stage20b_session_metadata.csv', index=False)
    print(f'wrote {OUT}/stage20b_session_metadata.csv ({len(meta)} sessions with readable config)')

    d = sess.merge(meta, on=['rat', 'session'], how='inner')
    print(f'\nmerged: {len(d)}/{len(sess)} analysis sessions matched to a config '
          f'({sess.n_frames.sum()} vs {d.n_frames.sum()} total frames)')

    print('\n=== Cross-tab: model x corrector (n sessions) ===')
    print(pd.crosstab(d.model, d.corrector))
    print('\n=== Cross-tab: calibration x corrector (n sessions) ===')
    print(pd.crosstab(d.calibration, d.corrector))
    print('\n=== Cross-tab: model x calibration (n sessions) ===')
    print(pd.crosstab(d.model, d.calibration))

    strata = []
    for corr in ['ON', 'OFF']:
        strata.append((f'corrector={corr}', d.corrector == corr))
    for model in d.model.unique():
        strata.append((f'model={model}', d.model == model))
    for calib in d.calibration.unique():
        strata.append((f'calibration={calib}', d.calibration == calib))
    # The rare bridge group that breaks the model/corrector confound (old
    # model, corrector OFF -- the very first session(s) per rat, before the
    # corrector was ever turned on).
    strata.append(('bridge: old_model + corrector_OFF', (d.model == 'old_n2298') & (d.corrector == 'OFF')))
    strata.append(('main: old_model + corrector_ON', (d.model == 'old_n2298') & (d.corrector == 'ON')))
    strata.append(('post-switch: new_model + corrector_OFF', (d.model == 'new_n9975') & (d.corrector == 'OFF')))

    rows = []
    rankings = {}
    for label, mask in strata:
        mse, n = pooled_rmse(d, mask)
        if mse is None:
            print(f'\n{label}: NO SESSIONS, skipping')
            continue
        order = print_ranking(label, mse, n)
        rankings[label] = order
        for rank, kp in enumerate(order, 1):
            rows.append(dict(stratum=label, keypoint=kp, rank=rank,
                             rmse_mm=np.sqrt(mse[kp]), n_frames=n))

    out = pd.DataFrame(rows)
    out.to_csv(f'{OUT}/stage20b_stratified_ranking.csv', index=False)
    print(f'\nwrote {OUT}/stage20b_stratified_ranking.csv')

    # Rank stability: Spearman correlation of the full ranking between key
    # stratum pairs, plus where does Snout / TailBase / the axial set land.
    from scipy.stats import spearmanr
    print('\n=== Rank stability (Spearman rho of keypoint ranking between strata) ===')
    pairs = [('corrector=ON', 'corrector=OFF'),
            ('model=old_n2298', 'model=new_n9975'),
            ('main: old_model + corrector_ON', 'bridge: old_model + corrector_OFF'),
            ('bridge: old_model + corrector_OFF', 'post-switch: new_model + corrector_OFF')]
    for a, b in pairs:
        if a in rankings and b in rankings:
            ra = {kp: i for i, kp in enumerate(rankings[a])}
            rb = {kp: i for i, kp in enumerate(rankings[b])}
            common = [kp for kp in ra if kp in rb]
            rho, p = spearmanr([ra[kp] for kp in common], [rb[kp] for kp in common])
            print(f'  {a:38s} vs {b:38s}  rho={rho:.3f}  p={p:.2e}  (n={len(common)} keypoints)')

    print('\n=== Where does Snout land in each stratum? ===')
    for label in rankings:
        r = rankings[label].index('Snout') + 1
        print(f'  {label:42s}  Snout rank {r}/{len(rankings[label])}')

    print('\n=== Where does TailBase land in each stratum? ===')
    for label in rankings:
        r = rankings[label].index('TailBase') + 1
        print(f'  {label:42s}  TailBase rank {r}/{len(rankings[label])}')


if __name__ == '__main__':
    main()
