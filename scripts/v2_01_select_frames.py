"""Corrector v2, step 1 — balanced selection of contiguous frame segments.

Selection unit is a contiguous SEGMENT of `SEG_LEN` processed frames (temporal
models need causal context, so isolated frames are useless). Segments are
stratified by:

    rat  x  time-third of the rat's session sequence  x  arena sector (12x30 deg)
         x  posture (non-rear / rear / high-rear)

and drawn to an equal quota per cell, with rears enriched relative to their
natural frequency. Segments overlapping the SLEAP fine-tune labeled frames are
excluded (SLEAP has memorised those, so its error there is unrepresentative).

Everything is indexed on the PROCESSED-frame basis k, because that is what
sleap_keys_2D / the new inference / dannce_idx_for_sleap_cams_keys all use.
Arena/posture metadata lives on the ACQUISITION basis, bridged by cam0_frame[k].

Writes results/corrector_v2/segments.csv
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial.geometry import sector_index

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
OUT = 'results/corrector_v2'
os.makedirs(OUT, exist_ok=True)

# Segment length must support BOTH uses:
#   - corrector training: a 5-frame causal window
#   - Tier-2 evaluation: the 30-frame TEMPLATE window plus the +/-15-frame
#     alignment search buffer used by best_align_end
# 48 frames (2.4 s) covers both. Total frame budget is unchanged (the SMB decode
# span sets the cost), so this trades segment COUNT for segment LENGTH.
SEG_LEN = 48
CTX = 4               # first CTX frames are context only
SESSIONS_PER_RAT = 12
SEGS_PER_SESSION = 90

# I/O BUDGET. The SMB share sustains only ~5-8 MB/s, and reading a session's
# three 2 GB videos dominates everything. Decoding must run from frame 0 to the
# LAST needed frame, so the cost is set by the frame SPAN, not the frame count.
# We therefore draw segments from a bounded window: after the 5-min warmup
# (which is atypical behaviour anyway) and up to K_SPAN of the session.
# Extra segments inside that window are nearly free once decoded.
# ASSUMPTION: within-session time position is not a balancing dimension we need
# (we balance across the rat's SESSION SEQUENCE, which is the training-time axis
# that matters); the window still spans ~11 min of behaviour.
WARMUP_K = 6000       # skip the first 5 min (at 20 Hz)
K_SPAN = 0.60         # use frames up to this fraction of the session
SECTOR_W, NSEC = 30, 12
POSTURE_MIX = {'nonrear': 0.40, 'rear': 0.35, 'highrear': 0.25}
FT_MARGIN = 3         # exclude segments within this many frames of a fine-tune frame
RNG = np.random.default_rng(0)


def session_ok(rat, s):
    """Needs DANNCE pairing, 2D, video, and frame mapping."""
    p = f'{DR}/{rat}/{s}'
    return (os.path.isfile(f'{p}/processed/aligned_data.npz') and
            os.path.isfile(f'{p}/sleap/frame_mapping.csv') and
            all(os.path.isfile(f'{p}/sleap/Camera{c}/0.mp4') for c in range(3)))


def pick_sessions(rat, cache_sessions):
    """12 sessions spread evenly across the rat's session sequence."""
    ok = [s for s in cache_sessions if session_ok(rat, s)]
    if len(ok) <= SESSIONS_PER_RAT:
        return ok
    idx = np.linspace(0, len(ok) - 1, SESSIONS_PER_RAT).round().astype(int)
    return [ok[i] for i in sorted(set(idx))]


def main():
    ft = json.load(open('results/spatial_confound/finetune_labeled_frames.json'))
    rears = pd.read_csv('results/spatial_confound/rears.csv')
    rows = []

    for rat in ['R4', 'R5', 'R6']:
        cache = sorted(os.path.basename(f)[:-4]
                       for f in glob.glob(f'data/cache/{rat}/*.npz'))
        sessions = pick_sessions(rat, cache)
        n_third = max(1, len(sessions) // 3)
        print(f'{rat}: {len(sessions)} sessions selected', flush=True)

        for si, s in enumerate(sessions):
            third = min(2, si // n_third)
            fm = pd.read_csv(f'{DR}/{rat}/{s}/sleap/frame_mapping.csv',
                             skipinitialspace=True)
            cam = {c: fm[f'cam{c}_frame'].to_numpy(int) for c in range(3)}
            K = len(fm)
            qp = f'data/cache_quality/{rat}/{s}.npz'
            if not os.path.isfile(qp):
                print(f'  {rat}/{s}: no quality cache, skip'); continue
            with np.load(qp) as z:
                theta = z['theta']; radius = z['radius']
            T = len(theta)

            # posture on the acquisition basis, from the existing rear table
            g = rears[(rears.rat == rat) & (rears.session == s)]
            is_rear = np.zeros(T, bool); is_high = np.zeros(T, bool)
            if len(g):
                hi = g.apex_z.quantile(2 / 3)
                for _, r in g.iterrows():
                    a, b = int(r.start_frame), min(int(r.end_frame) + 1, T)
                    is_rear[a:b] = True
                    if r.apex_z >= hi:
                        is_high[a:b] = True

            # fine-tune frames to avoid (video-frame basis, any camera)
            ftf = np.array(ft.get(f'{rat}/{s}', []), dtype=int)
            bad_k = np.zeros(K, bool)
            if len(ftf):
                for c in range(3):
                    hit = np.isin(cam[c], ftf)
                    for d in range(-FT_MARGIN, FT_MARGIN + 1):
                        bad_k |= np.roll(hit, d)

            # candidate segment starts, inside the bounded I/O window
            k_lo = min(WARMUP_K, max(0, K - SEG_LEN - 1000))
            k_hi = max(k_lo + SEG_LEN + 1, int(K * K_SPAN))
            starts = np.arange(k_lo, min(k_hi, K - SEG_LEN))
            # segment must be contiguous in acquisition frames (no big gaps)
            gap_ok = (cam[0][starts + SEG_LEN - 1] - cam[0][starts]) <= SEG_LEN + 12
            ft_ok = ~np.array([bad_k[a:a + SEG_LEN].any() for a in starts])
            mid = cam[0][starts + SEG_LEN // 2]
            mid = np.clip(mid, 0, T - 1)
            wall_ok = radius[mid] > 200
            cand = starts[gap_ok & ft_ok & wall_ok]
            if len(cand) < 50:
                print(f'  {rat}/{s}: only {len(cand)} candidates, skip'); continue
            cmid = np.clip(cam[0][cand + SEG_LEN // 2], 0, T - 1)
            csec = sector_index(theta[cmid], SECTOR_W)
            cpos = np.where(is_high[cmid], 'highrear',
                            np.where(is_rear[cmid], 'rear', 'nonrear'))

            # stratified draw: posture mix x equal quota per sector
            for post, frac in POSTURE_MIX.items():
                want = int(round(SEGS_PER_SESSION * frac))
                per_sec = max(1, want // NSEC)
                pool = np.where(cpos == post)[0]
                if not len(pool):
                    continue
                taken = []
                for sec in range(NSEC):
                    ii = pool[csec[pool] == sec]
                    if len(ii):
                        taken.extend(RNG.choice(ii, min(len(ii), per_sec), replace=False))
                # top up from the whole posture pool if sectors were sparse
                if len(taken) < want:
                    rest = np.setdiff1d(pool, np.array(taken, dtype=int))
                    if len(rest):
                        taken.extend(RNG.choice(rest, min(len(rest), want - len(taken)),
                                                replace=False))
                for t in taken:
                    a = int(cand[t])
                    rows.append(dict(rat=rat, session=s, third=third,
                                     k_start=a, k_end=a + SEG_LEN,
                                     sector=int(csec[t]), posture=post,
                                     cam0=int(cam[0][a]), cam1=int(cam[1][a]),
                                     cam2=int(cam[2][a])))
            print(f'  {rat}/{s} (third {third}): {sum(1 for r in rows if r["session"]==s)} segments',
                  flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/segments.csv', index=False)
    print(f'\nwrote {OUT}/segments.csv: {len(d)} segments = {len(d)*SEG_LEN:,} frames')
    print(f'  sessions: {d.groupby("rat").session.nunique().to_dict()}')
    print('\n  posture mix:'); print(d.posture.value_counts(normalize=True).round(3).to_string())
    print('\n  sector balance (fraction per sector):')
    print(d.sector.value_counts(normalize=True).sort_index().round(3).to_string())
    print('\n  time-third balance:'); print(d.third.value_counts(normalize=True).round(3).to_string())


if __name__ == '__main__':
    main()
