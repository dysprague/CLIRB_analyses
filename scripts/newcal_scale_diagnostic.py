"""Is DANNCE's shrunken skeleton on 2026-10-26 a calibration SCALE difference,
or something deeper?

The two hypotheses make opposite predictions, and they are separable without
any cross-modality fitting:

  H1  UNIFORM SCALE.  The 2026_10_26 DANNCE calibration converts pixels to
      millimetres with a slightly different constant.  Then EVERY distance in
      DANNCE's world shrinks by the same factor -- the animal's body AND the
      arena it walks around.  The ratio (DANNCE arena radius / SLEAP arena
      radius) should fall by the same factor as (DANNCE body size / SLEAP body
      size), and a Procrustes fit with a free isotropic scale should recover
      that factor.

  H2  RECONSTRUCTION SCALE.  The animal's POSITION is still right -- it comes
      from the COM network plus triangulation, and the arena geometry with it --
      but the keypoints are placed at the wrong radius around that COM, because
      sDANNCE's fixed-size volume is projected into the images through a
      calibration whose effective magnification differs from the one the
      network was trained under.  Then the arena ratio stays ~1 while the body
      ratio falls, and an isotropic Procrustes over frames spanning the whole
      arena reports scale ~1 because the between-frame spread (~600 mm of arena)
      dwarfs the within-frame spread (~55 mm of body).

Measured per session, in each modality's OWN raw world coordinates (no
Procrustes anywhere -- that is the point):

  arena_r     radius of the circle fitted to that modality's own COM track
  body        median over frames of the mean distance from each keypoint to the
              body centroid
  body_x/y/z  the same per axis, to see whether any shrinkage is isotropic
  body vs radial position, to test whether the effect is depth-dependent
              (which would implicate intrinsics / focal length rather than a
              single global constant)

Writes results/newcal/scale_diagnostic.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import dual as du
from clirb.spatial.geometry import fit_arena
from config import NODE_IDX as NI

OUT = 'results/newcal'
BONES = [('SpineF', 'TailBase'), ('EarL', 'HipL'), ('ShoulderL', 'ShoulderR'),
         ('HipL', 'HipR'), ('SpineF', 'SpineL')]


def arena_of(com_xy):
    P = com_xy[np.isfinite(com_xy).all(axis=1)][::10]
    c0 = np.median(P, axis=0)
    r = np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1])
    P = P[r <= np.percentile(r, 99.5)]
    a, cost = fit_arena(P)
    return a, float(cost)


def body_stats(kp):
    c = kp.mean(axis=1, keepdims=True)
    d = kp - c
    out = dict(body=float(np.median(np.linalg.norm(d, axis=-1).mean(axis=1))))
    for ax, name in enumerate('xyz'):
        out[f'body_{name}'] = float(np.median(np.abs(d[:, :, ax]).mean(axis=1)))
    return out


def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual & inv.era.isin(['oct_b', 'oct_26', 'sept_baseline'])]
    # September is large; a few sessions are enough as the pre-change reference
    sept = inv[inv.era == 'sept_baseline'].sort_values('session').tail(6)
    use = pd.concat([sept, inv[inv.era != 'sept_baseline']])

    rows = []
    for r in use.itertuples():
        raw = sc.get_raw_session(r.rat, r.session, want_dual=True)
        d3 = du.dannce_aligned(raw)
        if d3 is None:
            continue
        n = min(len(raw['triang_3d']), len(d3))
        sl = np.asarray(raw['triang_3d'][:n], np.float32)
        dn = np.asarray(d3[:n], np.float32)
        a_sl, c_sl = arena_of(sl.mean(axis=1)[:, :2])
        a_dn, c_dn = arena_of(dn.mean(axis=1)[:, :2])
        bs, bd = body_stats(sl), body_stats(dn)
        rec = dict(rat=r.rat, session=r.session, era=r.era,
                   arena_r_sleap=a_sl.radius, arena_r_dannce=a_dn.radius,
                   arena_fit_cost_sleap=c_sl, arena_fit_cost_dannce=c_dn,
                   arena_ratio=a_dn.radius / a_sl.radius,
                   body_sleap=bs['body'], body_dannce=bd['body'],
                   body_ratio=bd['body'] / bs['body'])
        for name in 'xyz':
            rec[f'body_{name}_ratio'] = bd[f'body_{name}'] / bs[f'body_{name}']
        for a, b in BONES:
            la = np.median(np.linalg.norm(sl[:, NI[a]] - sl[:, NI[b]], axis=-1))
            lb = np.median(np.linalg.norm(dn[:, NI[a]] - dn[:, NI[b]], axis=-1))
            rec[f'bone_{a}_{b}_ratio'] = float(lb / la)
        # depth dependence: body size vs how far the animal is from the arena
        # centre (a proxy for distance to the cameras)
        com = dn.mean(axis=1)[:, :2]
        rad = a_dn.radius_of(com)
        size = np.linalg.norm(dn - dn.mean(axis=1, keepdims=True), axis=-1).mean(axis=1)
        q = np.nanpercentile(rad, [20, 80])
        inner, outer = rad <= q[0], rad >= q[1]
        rec['body_dannce_inner'] = float(np.median(size[inner])) if inner.any() else np.nan
        rec['body_dannce_outer'] = float(np.median(size[outer])) if outer.any() else np.nan
        rec['body_inner_over_outer'] = rec['body_dannce_inner'] / rec['body_dannce_outer']
        rows.append(rec)
        print(f'{r.rat}/{r.session} ({r.era}): arena ratio={rec["arena_ratio"]:.4f} '
              f'body ratio={rec["body_ratio"]:.4f}', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/scale_diagnostic.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['arena_r_sleap', 'arena_r_dannce', 'arena_ratio', 'body_sleap',
            'body_dannce', 'body_ratio', 'body_x_ratio', 'body_y_ratio',
            'body_z_ratio', 'body_inner_over_outer']
    print('\n=== era medians (all in each modality\'s OWN raw world coords) ===')
    print(d.groupby('era')[cols].median().round(4).to_string())
    print('\n=== per session, oct_b and oct_26 ===')
    print(d[d.era != 'sept_baseline'][['rat', 'session', 'era'] + cols]
          .round(4).to_string(index=False))
    print('\n=== bone-length ratios DANNCE/SLEAP ===')
    bc = [c for c in d.columns if c.startswith('bone_')]
    print(d.groupby('era')[bc].median().round(4).to_string())
    print(f'\nwrote {OUT}/scale_diagnostic.csv')


if __name__ == '__main__':
    main()
