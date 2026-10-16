import sys; sys.path.insert(0,'/home/yutaka-sprague/CLIRB_analyses')
import numpy as np
from config import NODE_IDX
from clirb import features as F

def pose(head_y=0.0, shoulder_rot=0.0):
    """Synthetic rat along +X: tail at -X, head at +X. head_y>0 = head swung LEFT."""
    kp = np.zeros((23, 3))
    kp[NODE_IDX['HipL']]      = (-30,  15, 0)
    kp[NODE_IDX['HipR']]      = (-30, -15, 0)
    kp[NODE_IDX['TailBase']]  = (-50,   0, 0)
    kp[NODE_IDX['SpineL']]    = (-15,   0, 0)
    kp[NODE_IDX['SpineM']]    = (  0,   0, 0)
    kp[NODE_IDX['SpineF']]    = ( 20,   0, 0)
    # shoulder girdle, optionally rotated CCW (positive = left) about its centre
    c, s = np.cos(np.radians(shoulder_rot)), np.sin(np.radians(shoulder_rot))
    for node, off in (('ShoulderL', (0, 12)), ('ShoulderR', (0, -12))):
        x, y = off
        kp[NODE_IDX[node]] = (20 + x*c - y*s, x*s + y*c, 0)
    kp[NODE_IDX['EarL']]  = (45, head_y + 8, 0)
    kp[NODE_IDX['EarR']]  = (45, head_y - 8, 0)
    kp[NODE_IDX['Snout']] = (58, head_y, 0)
    return kp

def feats(poses):
    return F.compute_features(np.stack(poses))

print('--- head swung LEFT (+y). Every Family A feature should be POSITIVE ---')
f = feats([pose(head_y=+20)] * 3)
for k in F.LATERAL_FEATURES:
    v = f[k][-1]
    print(f'  {k:22s} {v:+8.2f}  {"OK" if v > 0 else ("zero" if abs(v) < 1e-9 else "WRONG SIGN")}')

print('\n--- head swung RIGHT (-y). Every Family A feature should be NEGATIVE ---')
f = feats([pose(head_y=-20)] * 3)
for k in F.LATERAL_FEATURES:
    v = f[k][-1]
    print(f'  {k:22s} {v:+8.2f}  {"OK" if v < 0 else ("zero" if abs(v) < 1e-9 else "WRONG SIGN")}')

print('\n--- shoulder girdle rotated LEFT (+20 deg): shoulder_hip_asym should be POSITIVE ---')
f = feats([pose(head_y=0, shoulder_rot=+20)] * 3)
print(f'  shoulder_hip_asym      {f["shoulder_hip_asym"][-1]:+8.2f}')
f = feats([pose(head_y=0, shoulder_rot=-20)] * 3)
print(f'  (rotated RIGHT)        {f["shoulder_hip_asym"][-1]:+8.2f}  (should be negative)')

print('\n--- body rotating LEFT (CCW) over 40 frames: yaw_rate & cum_heading_change POSITIVE ---')
seq = []
for t in range(40):
    kp = pose(head_y=0)
    th = np.radians(2.0 * t)          # +2 deg/frame = leftward rotation
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    seq.append(kp @ R.T)
f = feats(seq)
print(f'  heading_earhip   start {f["heading_earhip"][0]:+7.2f} -> end {f["heading_earhip"][-1]:+7.2f} deg')
print(f'  yaw_rate         median {np.median(f["yaw_rate"][1:]):+8.2f} deg/s  (expect +2*{F.FPS:.0f})')
print(f'  cum_heading_change end  {f["cum_heading_change"][-1]:+8.2f} deg  (expect ~ +2*30 = +60)')

print('\n--- same, rotating RIGHT: both should be NEGATIVE ---')
seq = []
for t in range(40):
    kp = pose(head_y=0)
    th = np.radians(-2.0 * t)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    seq.append(kp @ R.T)
f = feats(seq)
print(f'  yaw_rate         median {np.median(f["yaw_rate"][1:]):+8.2f} deg/s')
print(f'  cum_heading_change end  {f["cum_heading_change"][-1]:+8.2f} deg')

print('\n--- wrap safety: rotating through +/-180 must not produce a spike ---')
seq = []
for t in range(60):
    kp = pose(head_y=0)
    th = np.radians(170 + 1.0 * t)    # sweeps through 180
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    seq.append(kp @ R.T)
f = feats(seq)
print(f'  max |yaw_rate| {np.max(np.abs(f["yaw_rate"][1:])):.2f} deg/s (expect ~{1.0*F.FPS:.0f}, NOT a 360 spike)')
print(f'  cum_heading_change end {f["cum_heading_change"][-1]:+.2f} deg (expect ~+30)')
