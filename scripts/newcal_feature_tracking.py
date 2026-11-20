"""Do the two validated interpretable features still carry signal after the
environment change?

FEATURE_SELECTION_STATUS.md names two survivors:
  ear_mean_z          -- height / rearing.  5.41 mm SLEAP-vs-DANNCE RMSE,
                         93-94% precision / 96% recall on rearing (sec 8s-8x).
  cum_heading_change  -- turn.  99.5% turn-sign agreement conditioned on a real
                         turn (|value| >= 45 deg, 22% of frames) (sec 8ab).
`ear_snout_mean_z` (the ear+snout variant) and `heading_earhip`
(cum_heading_change's parent) are carried along for context.

Protocol is stage21's, feature definitions are `clirb.features` (the single
source of truth), so the September numbers here are directly comparable to the
published ones:
  one session-level robust Procrustes DANNCE -> SLEAP, then
  `ev.causal_median_5_kp(warmup='online')` on each modality independently, then
  features from each.

TWO DEVIATIONS FROM stage21, both forced by the new conditions:
  1. The spatially balanced Procrustes sample is drawn around each session's
     OWN COM centroid instead of `clirb.spatial.geometry.ARENA`, whose centre
     and radius were fitted in the OLD arena.
  2. stage21 requires a per-session template and drops sessions whose fit
     residual or overall RMSE trips the 60 mm guard.  R7/R8/R9 have no template
     yet, and the guard would drop exactly the degraded sessions under test, so
     nothing is dropped here -- the guard is recorded as a flag instead.

Three kinds of measurement per feature:
  continuous  RMSE / median / p90 absolute SLEAP-vs-DANNCE error, Pearson r
  functional  ear_mean_z: run the hysteresis rear detector independently on each
              modality's trace with the SAME fixed thresholds, match events at
              +/-300 ms, report precision/recall/F1 against DANNCE
  directional cum_heading_change: sign agreement, unconditional and conditioned
              on |SLEAP value| >= 45 deg, plus how often that condition holds

Heights are measured in an UP-POSITIVE frame: SLEAP's stored world z points
down (which is why `project_session_to_pcs` z-flips it), and the Procrustes
puts DANNCE in that same frame, so z is negated on both AFTER alignment.  This
changes no angle feature (those use x,y only) and no RMSE (a global sign flip
cancels in a difference); it only makes the rear thresholds readable as mm of
ear height.

Writes results/newcal/feature_tracking.csv (one row per session x feature).
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
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample

OUT = 'results/newcal'
FEATURES = ['ear_mean_z', 'ear_snout_mean_z', 'cum_heading_change', 'heading_earhip']

#: Rear thresholds on ear height (mm, up-positive). Chosen on the September
#: DANNCE distribution; held fixed across both eras so the detector itself is
#: not a free parameter. ~120 events/session in September for both modalities.
REAR_RISE, REAR_RESET = 120.0, 100.0
MATCH_TOL_FR = int(round(0.300 * ev.FPS))        # +/-300 ms, the project's tolerance
TURN_DEG = 45.0                                  # sec 8ab's "a real turn"
FIT_RESIDUAL_GUARD_MM = 60.0


def match_events(a_end, b_end, tol=MATCH_TOL_FR):
    """Greedy nearest-neighbour 1:1 match of two event-end lists."""
    a = np.sort(np.asarray(a_end, dtype=int))
    b = np.sort(np.asarray(b_end, dtype=int))
    used = np.zeros(len(b), dtype=bool)
    hits = 0
    for t in a:
        if not len(b):
            break
        d = np.abs(b - t)
        d[used] = np.iinfo(np.int64).max
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            hits += 1
    return hits


def session_rows(rat, session, era, rng):
    raw = sc.get_raw_session(rat, session, want_dual=True)
    if raw is None or not raw.get('has_dual', False):
        return []
    d3 = du.dannce_aligned(raw)
    if d3 is None:
        return []
    n = min(len(raw['triang_3d']), len(d3))
    if n < 1000:
        return []
    sl_raw = np.asarray(raw['triang_3d'][:n], dtype=np.float32)
    dn_raw = np.asarray(d3[:n], dtype=np.float32)
    samp, _ = balanced_sample(sl_raw.mean(axis=1)[:, :2], rng=rng)
    if len(samp) < 500:
        return []
    fit = robust_proc(dn_raw[samp], sl_raw[samp])
    dn_al = fit['apply'](dn_raw)
    overall = float(np.sqrt(((sl_raw - dn_al) ** 2).sum(axis=-1).mean()))
    guard = (fit['residual'] > FIT_RESIDUAL_GUARD_MM or
             overall > 2 * FIT_RESIDUAL_GUARD_MM)

    sl = ev.causal_median_5_kp(sl_raw, warmup='online').copy()
    dn = ev.causal_median_5_kp(dn_al, warmup='online').copy()
    sl[:, :, 2] *= -1.0          # up-positive; see module docstring
    dn[:, :, 2] *= -1.0
    f_sl = cf.compute_features(sl)
    f_dn = cf.compute_features(dn)

    rows = []
    for name in FEATURES:
        a, b = np.asarray(f_sl[name], float), np.asarray(f_dn[name], float)
        if name in cf.ANGLE_FEATURES:
            err = cf.wrap_deg(a - b)
        else:
            err = a - b
        r = dict(rat=rat, session=session, era=era, feature=name,
                 unit=cf.UNITS[name], n_frames=int(n),
                 guard_tripped=bool(guard),
                 rmse=float(np.sqrt(np.mean(err ** 2))),
                 med_abs=float(np.median(np.abs(err))),
                 p90_abs=float(np.percentile(np.abs(err), 90)),
                 bias=float(np.mean(err)),
                 sd_dannce=float(np.std(b)),
                 pearson_r=float(np.corrcoef(a, b)[0, 1]))

        if name in ('ear_mean_z', 'ear_snout_mean_z'):
            rs = ev.detect_rears(a, rise=REAR_RISE, reset=REAR_RESET)
            rd = ev.detect_rears(b, rise=REAR_RISE, reset=REAR_RESET)
            se = [x[1] for x in rs]; de = [x[1] for x in rd]
            hits = match_events(se, de)
            prec = hits / len(se) if se else np.nan
            rec = hits / len(de) if de else np.nan
            f1 = (2 * prec * rec / (prec + rec)
                  if (se and de and (prec + rec) > 0) else np.nan)
            r.update(n_events_sleap=len(se), n_events_dannce=len(de),
                     event_precision=prec, event_recall=rec, event_f1=f1)

        if name in cf.SIGNED_FEATURES:
            agree = np.sign(a) == np.sign(b)
            r['sign_agree_pct'] = float(100.0 * agree.mean())
            m = np.abs(a) >= TURN_DEG
            r['frac_real_turn'] = float(m.mean())
            r['sign_agree_turn_pct'] = (float(100.0 * agree[m].mean())
                                        if m.sum() > 100 else np.nan)
            r['rmse_turn'] = (float(np.sqrt(np.mean(err[m] ** 2)))
                              if m.sum() > 100 else np.nan)
        rows.append(r)
    return rows


def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]
    rng = np.random.default_rng(0)
    rows = []
    for i, r in enumerate(inv.itertuples(), 1):
        try:
            rows += session_rows(r.rat, r.session, r.era, rng)
        except Exception as e:
            print(f'[{i}/{len(inv)}] {r.rat}/{r.session}: ERROR {e!r}', flush=True)
            continue
        print(f'[{i}/{len(inv)}] {r.rat}/{r.session}', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/feature_tracking.csv', index=False)
    print(f'\nwrote {OUT}/feature_tracking.csv '
          f'({d.groupby(["rat","session"]).ngroups} sessions)')


if __name__ == '__main__':
    main()
