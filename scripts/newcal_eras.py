"""Shared era definitions for the post-recalibration comparison.

Three conditions, all with the SAME SLEAP model, sDANNCE + COM checkpoints and
corrector-off setting -- only the physical setup and the camera calibration
differ:

  sept_baseline  R4/R5/R6, 2026-09, calibration 2026_08_28. The reference.
  oct_a          calibration 2026_10_14. R5 2026_10_14_1 and R7/R8/R9 on
                 2026-10-19/20/21 -- the environment change + recalibration.
  oct_b          R7/R8/R9 on 2026-10-22, after a further physical change. The
                 campy_config files are identical to oct_a's, so the change is
                 physical only.
  oct_26         R7/R8/R9 on 2026-10-26. Three further changes at once, so this
                 era is NOT a clean continuation of oct_b: a new calibration
                 (2026_10_26), the task switched on (task_id 2, train_state 2,
                 context_switch_time [600, 1800]), and the online reward rule
                 changed from TemplateMatch to HeadHold -- a hysteresis detector
                 on ear_mean_z itself. Sessions are also 2400 s rather than
                 1800 s.

R5 2026_10_14_1 belongs to the oct_a calibration block but is a different
animal from R7/R8/R9, so it gets its own x-axis group rather than being pooled
with them.
"""
ERAS = ['sept_baseline', 'oct_a', 'oct_b', 'oct_26']

#: Categorical slots 1-3 of the validated reference palette (all-pairs safe).
ERA_COLOR = {'sept_baseline': '#2a78d6',   # blue
             'oct_a': '#eb6834',           # orange
             'oct_b': '#1baf7a',           # aqua
             'oct_26': '#eda100'}          # yellow

ERA_LABEL = {'sept_baseline': 'September baseline (cal 2026_08_28)',
             'oct_a': 'Oct 14–21 — environment change + recalibration',
             'oct_b': 'Oct 22 — setup changed again',
             'oct_26': 'Oct 26 — new calibration, task on, HeadHold rule'}

ERA_SHORT = {'sept_baseline': 'Sep', 'oct_a': 'Oct14-21', 'oct_b': 'Oct22',
             'oct_26': 'Oct26'}


def group_key(rat, session, era, sep='\n'):
    """x-axis group label for one session."""
    if era == 'sept_baseline':
        return f'{rat}{sep}Sep'
    if era == 'oct_b':
        return f'{rat}{sep}Oct22'
    if era == 'oct_26':
        return f'{rat}{sep}Oct26'
    if rat == 'R5':
        return f'R5{sep}Oct14'
    return f'{rat}{sep}Oct19-21'


def add_group(d, sep='\n'):
    """Add a `grp` column to a DataFrame with rat/session/era columns."""
    d = d.copy()
    d['grp'] = [group_key(r.rat, r.session, r.era, sep) for r in d.itertuples()]
    return d


def ordered_groups(d):
    """Groups in era order, rats alphabetical within era."""
    out = []
    for era in ERAS:
        sub = d[d.era == era]
        # R5 Oct14 before the R7/R8/R9 block within oct_a
        keys = sorted(sub.grp.unique(), key=lambda g: ('Oct14' not in g, g))
        out += [k for k in keys if k not in out]
    return out


def era_of_group(d, grp):
    s = d[d.grp == grp]
    return s.era.iloc[0] if len(s) else 'sept_baseline'
