"""Stage 28 -- does `ear_mean_z` + a §8aa turn feature actually detect rear+turn
events, and do SLEAP and DANNCE agree on which events those are?

§8aa validated the new turn features for cross-modality RELIABILITY only (RMSE
and sign agreement, frame by frame). None has been through the matching
pipeline. This runs the exact §8v/stage24 feature-specific ground-truth
procedure on the `rear_turn` template for ear_mean_z + each candidate, and
reports three distinct things that §8x conflated:

  1. gt_size        -- how many events the rebuilt corridor matches per session.
                       §8v's diagnostic: a feature whose own "ground truth"
                       swallows the session is not behaviourally selective,
                       whatever its precision/recall says. Rear+turn seeds are
                       ~22/session, so a few-hundred-event corridor is fine and
                       ~1,200 is the degenerate ceiling.
  2. precision/recall -- SLEAP-vs-DANNCE ALIGNMENT: tp = both modalities fire
                       within 10 frames, fp = SLEAP only, fn = DANNCE only.
                       This is agreement between modalities, NOT behavioural
                       correctness (the distinction §8z had to make).
  3. seed_recovery  -- fraction of the actual rear+turn SEED events that the
                       SLEAP detector fires on. This is DETECTION, which
                       precision/recall does not measure: a detector can align
                       perfectly with DANNCE while both miss the behaviour.

`heading_earhip` is deliberately NOT tested as a matching feature despite being
the most precisely measured quantity in §8aa (8.17 deg). It is an ABSOLUTE
world heading, so a corridor built on it encodes "facing this compass
direction" -- which would bake an arena-orientation preference straight into
the online rule, the very class of spatial bias this whole investigation
exists to characterise. Only its direction-invariant derivative
(cum_heading_change) is appropriate here.

Same corrector-OFF/new-SLEAP-model session set as stage22-26.

Writes:
  results/spatial_confound/stage28_rear_turn_results.csv   (per session x combo x template)
  results/spatial_confound/stage28_rear_turn_summary.csv   (pooled + per-rat)
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'scripts')); os.chdir(REPO)

from clirb import events as ev
import stage24_multi_template_features as s24
import stage26_direction_separation as s26

OUT = 'results/spatial_confound'
W, REFRACTORY, PAIR_THRESH = s24.W, s24.REFRACTORY, s24.PAIR_THRESH

TEMPLATES = ['rear_turn', 'rearing']

COMBOS = [
    ('ear_mean_z',),                                 # baseline: height only
    ('ear_mean_z', 'angle_skip_xy'),                 # previous best turn pairing (§8x)
    ('ear_mean_z', 'cum_heading_change'),            # §8aa best turn feature
    ('ear_mean_z', 'hip_asym'),                      # §8aa best Family A
    ('ear_mean_z', 'shoulder_hip_asym'),
    ('ear_mean_z', 'lateral_offset_head'),
    ('ear_mean_z', 'sagitta_spine'),                 # contrast: best RMSE, poor sign
    ('ear_mean_z', 'yaw_rate'),                      # contrast: worst sign agreement
    ('ear_mean_z', 'angle_torsion_xy'),              # contrast: ruled out in §8v
    ('ear_mean_z', 'cum_heading_change', 'hip_asym'),
]


def evaluate(combo, seed_ends, feat_sl, feat_dn):
    """stage24's match_combo, instrumented to also return the SLEAP match set
    (for seed recovery) alongside TP/FP/FN and the ground-truth size."""
    if len(seed_ends) < 5:
        return None
    s_traj = np.stack([feat_sl[f] for f in combo], axis=1)
    d_traj = np.stack([feat_dn[f] for f in combo], axis=1)
    acc = max(1, round(1.5 * len(combo)))

    s_seg1, _ = s24.windows_at(s_traj, seed_ends)
    d_seg1, _ = s24.windows_at(d_traj, seed_ends)
    if len(s_seg1) < 5:
        return None
    dannce_tmpl0, _, bounds1 = s24.rebuild_and_bounds(d_seg1, s_seg1)

    gt_f, _ = ev.get_template_match_indices(d_traj, dannce_tmpl0, bounds1,
                                            refractory_frames=REFRACTORY, accepted_errors=acc)
    if len(gt_f) == 0:
        return None
    gt_f = ev.correct_matches(d_traj, dannce_tmpl0, gt_f)

    s_seg, _ = s24.windows_at(s_traj, gt_f)
    d_seg, _ = s24.windows_at(d_traj, gt_f)
    if len(s_seg) < 5:
        return None
    sleap_tmpl, dannce_tmpl, bounds = s24.rebuild_and_bounds(s_seg, d_seg)

    sm, _ = ev.get_template_match_indices(s_traj, sleap_tmpl, bounds,
                                          refractory_frames=REFRACTORY, accepted_errors=acc)
    dm, _ = ev.get_template_match_indices(d_traj, dannce_tmpl, bounds,
                                          refractory_frames=REFRACTORY, accepted_errors=acc)
    sm = ev.correct_matches(s_traj, sleap_tmpl, sm) if len(sm) else np.array([], dtype=int)
    dm = ev.correct_matches(d_traj, dannce_tmpl, dm) if len(dm) else np.array([], dtype=int)
    sleap_only, dannce_only, both = s24.get_match_overlap(dm, sm)

    rec, n_hit, n_pool = s26.hit_rate(sm, seed_ends)
    return dict(tp=len(both), fp=len(sleap_only), fn=len(dannce_only), n_gt=len(gt_f),
                n_sleap_match=len(sm), n_dannce_match=len(dm),
                n_seed=len(seed_ends), n_seed_hit=n_hit, seed_recovery=rec)


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
            for template in TEMPLATES:
                seed_ends = sd['seeds'][template]
                for combo in COMBOS:
                    r = evaluate(combo, seed_ends, sd['feat_sl'], sd['feat_dn'])
                    if r is None:
                        continue
                    rows.append(dict(rat=rat, session=session, template=template,
                                     combo='+'.join(combo), n_features=len(combo), **r))
            print(f'  [{i+1}/{len(sessions)}] {rat}/{session} done', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/stage28_rear_turn_results.csv', index=False)
    print(f'\nwrote {OUT}/stage28_rear_turn_results.csv ({len(d)} rows)', flush=True)

    summ = []
    for (template, combo), g_all in d.groupby(['template', 'combo']):
        for rat in ['pooled', 'R4', 'R5', 'R6']:
            g = g_all if rat == 'pooled' else g_all[g_all.rat == rat]
            if not len(g):
                continue
            tp_, fp_, fn_ = g.tp.sum(), g.fp.sum(), g.fn.sum()
            prec = tp_ / max(tp_ + fp_, 1)
            rec = tp_ / max(tp_ + fn_, 1)
            srec = g.n_seed_hit.sum() / max(g.n_seed.sum(), 1)
            summ.append(dict(rat=rat, template=template, combo=combo,
                             n_features=g.n_features.iloc[0], n_sessions=len(g),
                             tp=int(tp_), fp=int(fp_), fn=int(fn_),
                             precision=prec, recall=rec,
                             f1=2 * prec * rec / max(prec + rec, 1e-9),
                             median_gt=float(g.n_gt.median()),
                             seed_recovery=srec,
                             meets_target=(prec >= 0.8 and rec >= 0.8)))
    s = pd.DataFrame(summ)
    s.to_csv(f'{OUT}/stage28_rear_turn_summary.csv', index=False)
    print(f'wrote {OUT}/stage28_rear_turn_summary.csv', flush=True)

    for template in TEMPLATES:
        print(f'\n=== {template} (pooled) ===')
        print(f'{"combo":44s} {"P":>6} {"R":>6} {"seedRec":>8} {"medGT":>7}')
        p = s[(s.rat == 'pooled') & (s.template == template)].sort_values('f1', ascending=False)
        for r in p.itertuples():
            flag = '  <- 80/80' if r.meets_target else ''
            print(f'{r.combo:44s} {r.precision:6.3f} {r.recall:6.3f} '
                  f'{r.seed_recovery:8.3f} {r.median_gt:7.0f}{flag}')


if __name__ == '__main__':
    main()
