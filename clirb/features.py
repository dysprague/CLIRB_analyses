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

#: Family A (2026-10-16) -- turn direction as a SIGNED DISTANCE rather than an
#: angle. An angle divides positional noise by the lever arm; a perpendicular
#: offset does not, so these should inherit the few-mm reliability of the
#: distance family instead of the 17-24 deg noise of the angle family (§8s).
#: All avoid Snout and TailBase, the two worst-tracked keypoints (§8p-§8r).
LATERAL_FEATURES = ['lateral_offset_head', 'hip_asym', 'shoulder_hip_asym', 'sagitta_spine']

#: Family B (2026-10-16) -- turning as a TEMPORAL quantity. Nothing in §8s-§8z
#: measured rotation over time; every turn candidate was instantaneous posture.
#: `cum_heading_change` is the motivated one: integrating yaw increments over
#: the match window suppresses noise instead of amplifying it (unlike
#: `yaw_rate`), and its dynamic range is a real turn's 90-180 deg rather than
#: trunk torsion's +/-30 -- the variance-to-range ratio the corridor needs.
#: It also gives LEFT and RIGHT genuinely different corridor SHAPES (a down-
#: ramp vs an up-ramp), which is what the height-only direction split lacked
#: (§8z found its two direction detectors overlapped 81-83%).
HEADING_FEATURES = ['heading_earhip', 'yaw_rate', 'cum_heading_change']

#: SIGN CONVENTION: every Family A and Family B feature is positive for a
#: LEFTWARD turn/bend (standard CCW-positive convention). `sagitta_spine` and
#: `hip_asym` are negated relative to their raw geometric form to hold that
#: convention -- see compute_features.
#: The 6 angles that existed before the 2026-09-10 additions (§8w). stage22's
#: combo search was run against this shorter list and must keep using it to
#: stay reproducible.
TURN_FEATURES_PRE_8W = TURN_FEATURES[:6]

#: Wrapped angles: a SLEAP-vs-DANNCE difference must be wrapped to -180..180.
#: `heading_earhip` is an absolute world-frame heading, so it wraps too -- but
#: its sign is arena-orientation-dependent and therefore NOT interpretable as
#: turn direction, unlike every other signed feature here.
ANGLE_FEATURES = set(TURN_FEATURES) | {'heading_earhip'}
VECTOR_FEATURES = {'com_velocity'}

#: Signed scalars whose SIGN encodes turn direction but whose magnitude must
#: NOT be wrapped (cumulative/rate quantities legitimately exceed 180).
SIGNED_FEATURES = set(LATERAL_FEATURES) | {'yaw_rate', 'cum_heading_change'}

FEATURE_NAMES = (HEIGHT_FEATURES + TURN_FEATURES + STRETCH_FEATURES
                 + LATERAL_FEATURES + HEADING_FEATURES + ['speed'])

#: The 16 features §8s-§8z were computed on, before Families A and B. Scripts
#: reproducing those results must keep using this list.
FEATURE_NAMES_PRE_FAMILY_AB = HEIGHT_FEATURES + TURN_FEATURES + STRETCH_FEATURES + ['speed']

UNITS = {**{f: 'mm' for f in HEIGHT_FEATURES + STRETCH_FEATURES + LATERAL_FEATURES},
         **{f: 'deg' for f in TURN_FEATURES},
         'heading_earhip': 'deg', 'cum_heading_change': 'deg', 'yaw_rate': 'deg/s',
         'speed': 'mm/s', 'com_velocity': 'mm/s'}


def signed_angle_xy(v1, v2):
    """Signed angle (degrees, -180..180) from v1 to v2 in the XY plane."""
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
    return np.degrees(np.arctan2(cross, dot))


def wrap_deg(diff):
    """Wrap an angular difference into -180..180."""
    return (diff + 180.0) % 360.0 - 180.0


def _signed_perp_xy(origin, axis_end, point, eps=1e-8):
    """Signed perpendicular XY distance (mm) of `point` from the line
    origin->axis_end. Positive = point lies to the LEFT of origin->axis_end
    (standard CCW-positive convention)."""
    f = (axis_end - origin)[:, :2]
    v = (point - origin)[:, :2]
    n = np.linalg.norm(f, axis=1)
    f_hat = f / np.maximum(n, eps)[:, None]
    out = f_hat[:, 0] * v[:, 1] - f_hat[:, 1] * v[:, 0]
    return np.where(n > eps, out, 0.0)


def _rolling_sum_causal(x, w):
    """Causal rolling sum over the trailing `w` samples; partial at the start."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x)) + 1
    return c[idx] - c[np.maximum(idx - w, 0)]


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

        # --- Family A: signed lateral distances (mm), positive = LEFT ---
        'lateral_offset_head': _signed_perp_xy(hip_mean, spinem, ear_mean),
        # raw (L - R) is positive when the head swings RIGHT, so take (R - L)
        'hip_asym': (np.linalg.norm(ear_mean - hipr, axis=1)
                     - np.linalg.norm(ear_mean - hipl, axis=1)),
        'shoulder_hip_asym': (np.linalg.norm(shoulderr - hipr, axis=1)
                              - np.linalg.norm(shoulderl - hipl, axis=1)),
        # SpineM bulges AWAY from the turn, so negate to keep positive = LEFT
        'sagitta_spine': -_signed_perp_xy(hip_mean, ear_mean, spinem),
    }

    # --- Family B: heading and its temporal derivatives/integrals ---
    # Body axis = hip_mean -> ear_mean: the longest well-tracked segment in the
    # skeleton (the same pair that won the stretch category at 8.15mm, §8s),
    # and it touches neither Snout nor TailBase.
    axis = (ear_mean - hip_mean)[:, :2]
    heading = np.degrees(np.arctan2(axis[:, 1], axis[:, 0]))
    out['heading_earhip'] = heading

    inc = np.zeros(kp.shape[0], dtype=np.float64)
    if kp.shape[0] > 1:
        inc[1:] = wrap_deg(np.diff(heading))
    out['yaw_rate'] = inc * FPS
    out['cum_heading_change'] = _rolling_sum_causal(inc, ev.TEMPLATE_LEN)

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
