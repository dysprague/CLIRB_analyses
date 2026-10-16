"""Hand-designed interpretable pose features -- the single source of truth.

These are the candidate building blocks for a redesigned online matching rule,
developed and evaluated in SPATIAL_CONFOUND_RESULTS.md §8s-§8z. Until now the
definitions lived as hand-synced copies inside stage21/22/23/24 and
render_sleap_dannce_feature_video.py; this module replaces all five. The
definitions here are byte-equivalent to those copies (verified numerically --
see `scripts/_verify_features_equivalence.py`), so results computed before the
consolidation remain reproducible.

Design notes, in case these get extended (§8s):
  - Snout is the worst-tracked keypoint in the skeleton and degrades every
    feature it touches. `ear_mean_z` beats `ear_snout_mean_z` precisely
    because it drops Snout, not despite it.
  - Angles are structurally noisier than distances/heights (~17-24 deg RMSE,
    19-27% sign-flip rate) because short spine segments amplify a few mm of
    positional noise. Prefer long lever arms.
  - Features are meant to be computed on causally-median-filtered keypoints
    (`ev.causal_median_5_kp(..., warmup='online')`), i.e. exactly what the
    online rule itself sees -- not raw positions.

Units: heights/distances mm, angles degrees, speed mm/s.
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from clirb import events as ev  # noqa: E402
from config import NODE_IDX  # noqa: E402

FPS = ev.FPS

SNOUT, EARL, EARR = NODE_IDX['Snout'], NODE_IDX['EarL'], NODE_IDX['EarR']
SPINEF, SPINEM, SPINEL, TAILBASE = (NODE_IDX['SpineF'], NODE_IDX['SpineM'],
                                    NODE_IDX['SpineL'], NODE_IDX['TailBase'])
HIPL, HIPR, SHOULDERL, SHOULDERR = (NODE_IDX['HipL'], NODE_IDX['HipR'],
                                    NODE_IDX['ShoulderL'], NODE_IDX['ShoulderR'])

HEIGHT_FEATURES = ['snout_z', 'ear_snout_mean_z', 'ear_mean_z']
TURN_FEATURES = ['angle_lower_xy', 'angle_upper_xy', 'angle_mid_xy', 'angle_skip_xy',
                 'angle_torsion_xy', 'angle_body_head_xy', 'angle_mid_head_xy',
                 'angle_torsion_ear_xy']
STRETCH_FEATURES = ['dist_snout_tb', 'dist_ears_tb', 'dist_spineF_tb', 'dist_earmean_hipmean']
#: The 6 angles that existed before the 2026-09-10 additions (§8w). stage22's
#: combo search was run against this shorter list and must keep using it to
#: stay reproducible.
TURN_FEATURES_PRE_8W = TURN_FEATURES[:6]
ANGLE_FEATURES = set(TURN_FEATURES)
VECTOR_FEATURES = {'com_velocity'}

FEATURE_NAMES = HEIGHT_FEATURES + TURN_FEATURES + STRETCH_FEATURES + ['speed']

UNITS = {**{f: 'mm' for f in HEIGHT_FEATURES + STRETCH_FEATURES},
         **{f: 'deg' for f in TURN_FEATURES},
         'speed': 'mm/s', 'com_velocity': 'mm/s'}


def signed_angle_xy(v1, v2):
    """Signed angle (degrees, -180..180) from v1 to v2 in the XY plane."""
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
    return np.degrees(np.arctan2(cross, dot))


def wrap_deg(diff):
    """Wrap an angular difference into -180..180."""
    return (diff + 180.0) % 360.0 - 180.0


def compute_features(kp, velocity='speed'):
    """Compute all interpretable features from one modality's keypoints.

    kp: (T, 23, 3) keypoints, normally causal-median-filtered.
    velocity: 'speed'  -> scalar `speed` (T,), mm/s, index 0 back-filled.
              'vector' -> `com_velocity` (T-1, 3), mm/s, as stage21's
                          reliability analysis consumes it.

    Returns dict name -> array.
    """
    if velocity not in ('speed', 'vector'):
        raise ValueError(f"velocity must be 'speed' or 'vector', got {velocity!r}")

    snout, earl, earr = kp[:, SNOUT], kp[:, EARL], kp[:, EARR]
    spinef, spinem, spinel, tb = kp[:, SPINEF], kp[:, SPINEM], kp[:, SPINEL], kp[:, TAILBASE]
    hipl, hipr, shoulderl, shoulderr = kp[:, HIPL], kp[:, HIPR], kp[:, SHOULDERL], kp[:, SHOULDERR]
    ear_mean = (earl + earr) / 2.0
    hip_mean = (hipl + hipr) / 2.0

    com = kp.mean(axis=1)
    vel = np.diff(com, axis=0) * FPS

    out = {
        'snout_z': snout[:, 2],
        'ear_snout_mean_z': np.stack([snout[:, 2], earl[:, 2], earr[:, 2]], axis=1).mean(axis=1),
        'ear_mean_z': np.stack([earl[:, 2], earr[:, 2]], axis=1).mean(axis=1),
        'dist_snout_tb': np.linalg.norm(snout - tb, axis=1),
        'dist_ears_tb': np.linalg.norm(ear_mean - tb, axis=1),
        'dist_spineF_tb': np.linalg.norm(spinef - tb, axis=1),
        'dist_earmean_hipmean': np.linalg.norm(ear_mean - hip_mean, axis=1),
        'angle_lower_xy': signed_angle_xy((spinem - tb)[:, :2], (spinef - spinem)[:, :2]),
        'angle_upper_xy': signed_angle_xy((spinef - spinem)[:, :2], (snout - spinef)[:, :2]),
        'angle_mid_xy': signed_angle_xy((spinem - spinel)[:, :2], (spinef - spinem)[:, :2]),
        'angle_skip_xy': signed_angle_xy((spinef - spinel)[:, :2], (snout - spinef)[:, :2]),
        'angle_torsion_xy': signed_angle_xy((hipr - hipl)[:, :2], (shoulderr - shoulderl)[:, :2]),
        'angle_body_head_xy': signed_angle_xy((spinef - tb)[:, :2], (ear_mean - spinef)[:, :2]),
        'angle_mid_head_xy': signed_angle_xy((spinem - spinel)[:, :2], (ear_mean - spinem)[:, :2]),
        'angle_torsion_ear_xy': signed_angle_xy((hipr - hipl)[:, :2], (earr - earl)[:, :2]),
    }

    if velocity == 'vector':
        out['com_velocity'] = vel
    else:
        speed = np.empty(kp.shape[0], dtype=np.float32)
        speed[1:] = np.linalg.norm(vel, axis=1)
        speed[0] = speed[1] if len(speed) > 1 else 0.0
        out['speed'] = speed
    return out


#: Backwards-compatible alias -- stage22/23/24 called this `compute_all_features`.
compute_all_features = compute_features
