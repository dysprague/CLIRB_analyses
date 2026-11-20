"""Warm session_cache for every session in the post-recalibration comparison.

Read-only w.r.t. olveczky_lab; writes only into CLIRB_analyses/data/cache and
the era inventory at results/newcal/session_inventory.csv.

Eras (the `era` column every downstream script groups on):
  sept_baseline : fully-processed 2026-09 sessions of R4, R5, R6
                  (calibration 2026_08_28, the pre-change reference)
  oct_a         : the first post-recalibration block, calibration 2026_10_14 --
                  R5 2026_10_14_1 plus R7/R8/R9 on 2026-10-19/20/21
  oct_b         : R7/R8/R9 on 2026-10-22, after a further physical change to
                  the setup.  Same SLEAP model, same sDANNCE/COM checkpoints,
                  same calibration path and corrector-off as oct_a -- the
                  campy_config files are byte-identical, so whatever changed is
                  physical, not a pipeline change.
"""
import os, sys, time
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
import session_cache as sc

DATA_ROOT = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
FLAGS = ['sleap_saved', 'dannce_saved', 'dannce_coms', 'dannce_inference',
         'aligned_data', 'sleap_dannce_keys']
#: The Oct-22 physical change applies to every session from that date onward
#: that still ran on the 2026_10_14 calibration (so 10-22 and 10-23), not just
#: to 10-22 itself.
OCT_B_FROM = '2026_10_22'
#: Everything from 2026-10-26 onward runs on the 2026_10_26 calibration with
#: the task on and the HeadHold rule -- 10-27 and 10-28 belong with 10-26, not
#: with the 10-22/23 block.
OCT_26_FROM = '2026_10_26'


def complete(df):
    m = pd.Series(True, index=df.index)
    for f in FLAGS:
        m &= df[f].astype(str).str.lower().eq('true')
    return df[m]


def era_of(session):
    if session.startswith('2026_09'):
        return 'sept_baseline'
    if session[:10] >= OCT_26_FROM:
        return 'oct_26'
    return 'oct_b' if session[:10] >= OCT_B_FROM else 'oct_a'


def main():
    old = pd.read_csv(f'{DATA_ROOT}/R4R5R6_runs.csv')
    new = pd.read_csv(f'{DATA_ROOT}/R7R8R9_runs.csv')

    sept = complete(old[old.session.str.startswith('2026_09')])
    r5oct = complete(old[old.session.str.startswith('2026_10')])
    new_c = complete(new)

    todo = [(r.rat, r.session) for r in sept.itertuples()] + \
           [(r.rat, r.session) for r in r5oct.itertuples()] + \
           [(r.rat, r.session) for r in new_c.itertuples()]
    todo = [(rat, s, era_of(s)) for rat, s in todo]
    n_era = pd.Series([e for _, _, e in todo]).value_counts().to_dict()
    print(f'{len(todo)} sessions to warm: {n_era}', flush=True)

    rows = []
    for i, (rat, session, era) in enumerate(todo, 1):
        t0 = time.time()
        try:
            raw = sc.get_raw_session(rat, session, want_dual=True)
        except Exception as e:
            print(f'[{i}/{len(todo)}] {rat}/{session}: ERROR {e!r}', flush=True)
            rows.append(dict(rat=rat, session=session, era=era, ok=False,
                             has_dual=False, n_frames=0, err=repr(e)))
            continue
        ok = raw is not None
        hd = bool(raw.get('has_dual', False)) if ok else False
        nf = int(raw['n_frames']) if ok else 0
        rows.append(dict(rat=rat, session=session, era=era, ok=ok,
                         has_dual=hd, n_frames=nf, err=''))
        print(f'[{i}/{len(todo)}] {rat}/{session} ({era}) ok={ok} dual={hd} '
              f'n={nf} {time.time()-t0:.1f}s', flush=True)

    out = 'results/newcal/session_inventory.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
