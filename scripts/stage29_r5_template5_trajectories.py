"""Stage 29 -- what do the candidate feature trajectories look like, next to the
PC trajectories the online rule actually matches, for R5's template 5?

The matching rule scores a 30-frame TRAJECTORY against a template corridor, so
a feature's usefulness depends on the shape of that trajectory within the
behaviour -- not just on its frame-wise reliability (§8ab's finding: that is
why cum_heading_change works in the corridor and hip_asym, despite better
frame-wise numbers, does not). This renders them side by side.

For every rear+turn seed event in the R5 sessions running R5_template_5.npz:
  - PC0 / PC1 (the two components the production rule matches on), with the
    template's own target trajectory and bounds corridor drawn in.
  - cum_heading_change, hip_asym, and three of the better angle candidates.
Each panel shows the median trajectory with an IQR band, computed separately
for SLEAP and DANNCE, and split by turn direction (sign of the event's net
heading change) so direction-separability is visible by eye.

Writes:
  results/figures/spatial_confound/F38_r5_template5_trajectories.png
  results/spatial_confound/stage29_r5_template5_trajectories.csv
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'scripts')); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp, features as cf
from clirb.spatial import dual as du
import stage24_multi_template_features as s24

OUT = 'results/spatial_confound'
FIG = 'results/figures/spatial_confound'
os.makedirs(FIG, exist_ok=True)
W = ev.TEMPLATE_LEN
RAT, TEMPLATE_FILE = 'R5', 'R5_template_5.npz'

PANELS = ['PC0', 'PC1', 'cum_heading_change', 'hip_asym',
          'angle_skip_xy', 'angle_torsion_xy', 'angle_mid_xy', 'angle_body_head_xy']

SURFACE = '#fcfcfb'; INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e4e3df'
LEFT_C, RIGHT_C = '#2a78d6', '#c4423f'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 0.8, 'grid.color': GRID,
    'legend.frameon': False, 'figure.dpi': 160,
})


def sessions_on_template(rat, template_file):
    out = []
    for f in sorted(glob.glob(f'data/cache/{rat}/*.npz')):
        s = os.path.basename(f)[:-4]
        p = tp.find_template_file(rat, s)
        if p and os.path.basename(p) == template_file:
            out.append(s)
    return out


def collect():
    rows = []
    sessions = sessions_on_template(RAT, TEMPLATE_FILE)
    print(f'{RAT}: {len(sessions)} sessions on {TEMPLATE_FILE}', flush=True)
    tmpl_ref = None
    for i, session in enumerate(sessions):
        sd = s24.session_data(RAT, session)
        if sd is None:
            print(f'  [{session}] no dual data / failed fit guard, skipping', flush=True)
            continue
        raw = sc.get_raw_session(RAT, session, want_dual=True)
        tm = tp.load_template(RAT, session)
        d3 = du.dannce_aligned(raw)
        n = min(len(raw['triang_3d']), len(d3))
        s_pcs, _ = ev.project_session_to_pcs(raw['triang_3d'][:n], tm, flip_z=True)
        d_pcs, _ = ev.project_session_to_pcs(d3[:n], tm, flip_z=False)
        pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)[:2]
        if tmpl_ref is None:
            tmpl_ref = dict(template=np.asarray(tm['template'])[:, :2],
                            bounds=np.asarray(ev.template_bounds(tm)))

        sig = {'sleap': dict(sd['feat_sl']), 'dannce': dict(sd['feat_dn'])}
        sig['sleap']['PC0'] = s_pcs[:, pcu[0]]; sig['sleap']['PC1'] = s_pcs[:, pcu[1]]
        sig['dannce']['PC0'] = d_pcs[:, pcu[0]]; sig['dannce']['PC1'] = d_pcs[:, pcu[1]]

        h = sd['feat_sl']['heading_earhip']
        inc = np.concatenate([[0.0], cf.wrap_deg(np.diff(h))])
        for e in sd['seeds']['rear_turn']:
            if e < W - 1 or e >= n:
                continue
            s0 = e - W + 1
            net = float(inc[s0:e + 1].sum())
            direction = 'left' if net > 0 else 'right'
            for mod in ('sleap', 'dannce'):
                for name in PANELS:
                    v = np.asarray(sig[mod][name])[s0:e + 1]
                    if len(v) != W:
                        continue
                    rows.append(dict(session=session, end=int(e), modality=mod,
                                     signal=name, direction=direction,
                                     **{f't{j:02d}': float(v[j]) for j in range(W)}))
        print(f'  [{i+1}/{len(sessions)}] {session}: '
              f'{len(sd["seeds"]["rear_turn"])} rear+turn events', flush=True)
    return pd.DataFrame(rows), tmpl_ref


def main():
    d, tmpl_ref = collect()
    d.to_csv(f'{OUT}/stage29_r5_template5_trajectories.csv', index=False)
    print(f'\nwrote {OUT}/stage29_r5_template5_trajectories.csv ({len(d)} rows)')
    tcols = [f't{j:02d}' for j in range(W)]
    x = np.arange(W)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), constrained_layout=True)
    for ax, name in zip(axes.ravel(), PANELS):
        for direction, col in (('left', LEFT_C), ('right', RIGHT_C)):
            for mod, ls, alpha in (('sleap', '-', 0.18), ('dannce', '--', 0.0)):
                g = d[(d.signal == name) & (d.direction == direction) & (d.modality == mod)]
                if not len(g):
                    continue
                arr = g[tcols].to_numpy()
                med = np.median(arr, axis=0)
                ax.plot(x, med, ls, color=col, lw=1.6 if mod == 'sleap' else 1.2,
                        label=f'{direction}, {mod} (n={len(arr)})')
                if alpha:
                    lo, hi = np.percentile(arr, [25, 75], axis=0)
                    ax.fill_between(x, lo, hi, color=col, alpha=alpha, lw=0)
        if name in ('PC0', 'PC1') and tmpl_ref is not None:
            k = 0 if name == 'PC0' else 1
            t, b = tmpl_ref['template'][:, k], tmpl_ref['bounds'][:, k]
            ax.plot(x, t, color=INK, lw=1.8, zorder=5, label='template target')
            ax.fill_between(x, t - b, t + b, color=INK, alpha=0.10, lw=0, zorder=4)
        ax.axhline(0, color=GRID, lw=0.9, zorder=0)
        ax.set_title(name, fontsize=10, color=INK)
        ax.set_xlabel('frame in 30-frame window')
        ax.grid(axis='y', alpha=0.9, zorder=0); ax.set_axisbelow(True)
        ax.legend(fontsize=6.2, loc='best')
    fig.suptitle(f'{RAT} / {TEMPLATE_FILE} — rear+turn event trajectories: what the rule matches '
                 '(PC0/PC1, with template corridor) vs. candidate features\n'
                 'median with IQR band (SLEAP solid, DANNCE dashed), split by turn direction',
                 fontsize=11, color=INK)
    fig.savefig(f'{FIG}/F38_r5_template5_trajectories.png', bbox_inches='tight')
    print(f'wrote {FIG}/F38_r5_template5_trajectories.png')

    print('\n=== direction separability: |median(left) - median(right)| / pooled IQR ===')
    for name in PANELS:
        g = d[(d.signal == name) & (d.modality == 'sleap')]
        if not len(g):
            continue
        L = g[g.direction == 'left'][tcols].to_numpy()
        R = g[g.direction == 'right'][tcols].to_numpy()
        if not len(L) or not len(R):
            continue
        sep = np.abs(np.median(L, 0) - np.median(R, 0))
        spread = np.percentile(np.vstack([L, R]), 75, 0) - np.percentile(np.vstack([L, R]), 25, 0)
        print(f'  {name:22s} mean {np.mean(sep / np.maximum(spread, 1e-9)):.3f}   '
              f'max {np.max(sep / np.maximum(spread, 1e-9)):.3f}')


if __name__ == '__main__':
    main()
