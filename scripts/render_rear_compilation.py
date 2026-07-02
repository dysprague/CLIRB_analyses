"""Compile a stitched video of every rear in a session, each padded by 5 s
before and after the rear interval. A small corner marker indicates when the
current frame is inside a template-match window — different colors for
matches that fired online vs matches that were excluded by the port-distance
gate.

Output: <out_dir>/<rat>_<session>_rear_compilation.mp4

Usage:
    python scripts/render_rear_compilation.py --rat R4 --session 2026_06_18_1
    python scripts/render_rear_compilation.py --rat R4 --session 2026_06_18_1 \\
        --camera 0 --pad_s 5 --out_dir rear_compilations
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from config import DATA_ROOT                                                 # noqa: E402
from skeleton import normalize_skeleton_batch                                # noqa: E402


SLEAP_HZ = 20
TEMPLATE_LEN_FRAMES = 30      # rule's pc_buffer_duration
REAR_PC0_THRESHOLD = 150.0
REAR_MIN_FRAMES = int(0.5 * SLEAP_HZ)   # 10
ONLINE_MATCH_TOL = 5          # ± frames tolerance for online ↔ offline pairing


# ---------------------------------------------------------------------------
# Loading + offline rule replay (lightweight copy of what the notebook does)
# ---------------------------------------------------------------------------
def _causal_median_5_kp(triang):
    pad = 4
    padded = np.concatenate([np.repeat(triang[:1], pad, axis=0), triang], axis=0)
    T = triang.shape[0]
    widx = np.arange(T)[:, None] + np.arange(5)[None, :]
    return np.median(padded[widx], axis=1).astype(np.float32)


def _detect_rears(pc0):
    above = pc0 > REAR_PC0_THRESHOLD
    if not above.any():
        return []
    edges = np.diff(above.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]
    return [(int(s), int(e), int(e - s)) for s, e in zip(starts, ends)
            if (e - s) >= REAR_MIN_FRAMES]


def _replay_rule(pcs_full, template, pcs_to_use, bounds_arr, accepted_errors=3):
    pcs = pcs_full[:, pcs_to_use]
    T, n_pc = pcs.shape
    win = template.shape[0]
    if T < win:
        return np.array([], dtype=int)
    base = np.arange(T - win + 1, dtype=np.int64)[:, None]
    windows = pcs[base + np.arange(win, dtype=np.int64)]
    above = windows >= (template + bounds_arr)
    below = windows <= (template - bounds_arr)
    outside_count = (above | below).reshape(windows.shape[0], -1).sum(axis=1)
    matches = []
    last_match = -win
    for t in range(win - 1, T):
        if outside_count[t - (win - 1)] <= accepted_errors and (t - last_match) >= win:
            matches.append(t)
            last_match = t
    return np.array(matches, dtype=int)


def load_and_analyze(rat, session):
    sleap_dir = os.path.join(DATA_ROOT, rat, session, 'sleap')
    triang = np.load(os.path.join(sleap_dir, 'triang_keys_3D.npy')).astype(np.float32)
    import pandas as pd
    log = pd.read_csv(os.path.join(sleap_dir, 'behavior_log.csv'),
                       skipinitialspace=True)
    fm = pd.read_csv(os.path.join(sleap_dir, 'frame_mapping.csv'),
                       skipinitialspace=True)
    pf_to_cam0 = fm['cam0_frame'].to_numpy(dtype=int)

    # Load template (per-session copy preferred).
    import glob as _g
    cands = sorted(_g.glob(os.path.join(sleap_dir, f'{rat}_template_*.npz')))
    tmpl_path = cands[0] if cands else os.path.join(
        DATA_ROOT, rat, 'templates', f'{rat}_template_1.npz')
    tmpl_npz = np.load(tmpl_path, allow_pickle=True)
    template = {k: tmpl_npz[k] for k in tmpl_npz.files}

    # Port location + exclusion radius.
    port_paths = sorted(_g.glob(os.path.join(
        sleap_dir, 'calibration', '*', 'water_port_location.npy')))
    port_xy = np.asarray(np.load(port_paths[0])).ravel()[:2] if port_paths else None

    min_port = 200.0
    cfg = os.path.join(sleap_dir, 'campy_config.yaml')
    if os.path.isfile(cfg):
        for line in open(cfg):
            if 'min_port_distance' in line and ':' in line:
                try:
                    min_port = float(line.split(':', 1)[1].split('#')[0].strip())
                except Exception:
                    pass
                break

    # PC projection — match online order (z-flip + normalize BEFORE median).
    # See diagnose_offline_only.py / notebook comment for why this matters.
    flipped = triang.copy(); flipped[:, :, 2] = -flipped[:, :, 2]
    rotated, _, _ = normalize_skeleton_batch(flipped)
    rotated_med = _causal_median_5_kp(rotated)
    flat = rotated_med.reshape(rotated_med.shape[0], -1)
    pcs_full = (flat - template['feature_means']) @ template['pc_weights'].T

    pcs_to_use = np.asarray(template['pcs_to_use']).ravel().astype(int)
    template_pc = template['template'][:, :len(pcs_to_use)]
    if 'bounds_to_use' in template:
        bounds_arr = template['bounds_to_use']
    else:
        scalar = float(template['bounds'])
        bounds_arr = np.tile(template['feature_stds'][pcs_to_use] * scalar,
                              (template_pc.shape[0], 1))

    rears = _detect_rears(pcs_full[:, pcs_to_use[0]])
    matches_cam0 = _replay_rule(pcs_full, template_pc, pcs_to_use, bounds_arr,
                                  accepted_errors=3)

    # Online triggers -> cam0 frames.
    online_pf = log.loc[log['triggered'] == 1, 'frame'].to_numpy(dtype=int)
    online_pf = online_pf[online_pf < len(pf_to_cam0)]
    online_cam0 = pf_to_cam0[online_pf]

    # COM port distance per cam0 frame.
    com_xy = triang.mean(axis=1)[:, :2]
    if port_xy is not None:
        port_dist = np.linalg.norm(com_xy - port_xy[None, :], axis=1)
    else:
        port_dist = np.full(len(triang), np.nan)

    # Classify each offline match: online-fired, port-excluded, or "other"
    # (offline match that wasn't online AND wasn't port-excluded — usually a
    # refractory clash with the online rule). Color the third class so it's
    # visible rather than silently dropped.
    online_set = set(int(f) for f in online_cam0)
    # Allow online tolerance — a match is considered "online-fired" if any
    # online_cam0 is within ONLINE_MATCH_TOL frames.
    online_arr = np.asarray(online_cam0)
    classes = []   # per match: 'online' / 'port_excluded' / 'other'
    for f in matches_cam0:
        if len(online_arr) and np.min(np.abs(online_arr - f)) <= ONLINE_MATCH_TOL:
            classes.append('online')
        else:
            d = port_dist[f] if f < len(port_dist) else np.nan
            if np.isfinite(d) and d < min_port:
                classes.append('port_excluded')
            else:
                classes.append('other')
    classes = np.array(classes)

    return {
        'rat': rat,
        'session': session,
        'sleap_dir': sleap_dir,
        'triang_len': len(triang),
        'rears': rears,
        'matches_cam0': matches_cam0,
        'match_classes': classes,
        'port_dist': port_dist,
        'min_port': min_port,
    }


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------
def _frame_in_template_window(frame, match_end):
    """True if `frame` lies in [match_end - 29, match_end]."""
    return match_end - (TEMPLATE_LEN_FRAMES - 1) <= frame <= match_end


def _classify_frame(frame, matches_cam0, match_classes):
    """Return ('online'/'port_excluded'/'other'/None) for the current frame
    based on whether it falls inside any template-match window. If multiple
    classes overlap, priority is online > port_excluded > other (so the
    marker reflects what the rig actually did, then why a match was suppressed).
    """
    # Only check matches whose end-frame is within template_len of `frame`.
    # Faster than scanning all matches.
    priority = {'online': 3, 'port_excluded': 2, 'other': 1}
    best = None
    best_pri = 0
    # Window of matches that could be relevant: end >= frame and end <= frame + 29.
    # Since matches_cam0 is sorted, use searchsorted.
    lo = np.searchsorted(matches_cam0, frame, side='left')
    hi = np.searchsorted(matches_cam0, frame + TEMPLATE_LEN_FRAMES, side='right')
    for k in range(lo, hi):
        end = matches_cam0[k]
        if _frame_in_template_window(frame, end):
            cls = match_classes[k]
            if priority.get(cls, 0) > best_pri:
                best = cls
                best_pri = priority.get(cls, 0)
    return best


CLASS_COLORS_BGR = {
    'online':         (0, 255, 0),     # green
    'port_excluded':  (0, 0, 255),     # red
    'other':          (0, 165, 255),   # orange — refractory clash / other
}


def render_compilation(rat, session, camera, pad_s, out_dir, fps_out=20):
    print(f'=== {rat}/{session} ===', flush=True)
    info = load_and_analyze(rat, session)
    rears = info['rears']
    if not rears:
        print('  no rears detected; skipping.', flush=True)
        return None
    matches = info['matches_cam0']
    classes = info['match_classes']
    n_online = int((classes == 'online').sum())
    n_excl = int((classes == 'port_excluded').sum())
    n_other = int((classes == 'other').sum())
    print(f'  rears: {len(rears)}, matches: {len(matches)} '
          f'(online={n_online}, port_excluded={n_excl}, other={n_other})',
          flush=True)

    pad = int(round(pad_s * SLEAP_HZ))
    triang_len = info['triang_len']

    video_file = os.path.join(info['sleap_dir'], f'Camera{camera}', '0.mp4')
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f'cannot open {video_file}')
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build clips, merging any whose padded windows overlap so that no source
    # frame appears in more than one clip. Each merged clip's frame range is
    # [first_rear_start - pad, last_rear_end + pad]; the 5 s pad only sits on
    # the outside of the merged group. Within a merged clip, all the source
    # frames play back contiguously (no cuts between the constituent rears).
    clips = []   # list of dicts: {'cs', 'ce', 'rears': [(rear_idx_1based, rs, re), ...]}
    for rear_idx, (rs, re, _) in enumerate(rears):
        clip_start = max(0, rs - pad)
        clip_end = min(triang_len, re + pad)
        if clips and clip_start < clips[-1]['ce']:
            # Padded window overlaps the previous clip — extend it.
            clips[-1]['ce'] = max(clips[-1]['ce'], clip_end)
            clips[-1]['rears'].append((rear_idx + 1, rs, re))
        else:
            clips.append({
                'cs': clip_start,
                'ce': clip_end,
                'rears': [(rear_idx + 1, rs, re)],
            })
    n_merged_clips = len(clips)
    rears_per_clip_max = max(len(c['rears']) for c in clips) if clips else 0
    print(f'  merged into {n_merged_clips} clips '
          f'(max {rears_per_clip_max} rears in a single clip)', flush=True)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{rat}_{session}_rear_compilation.mp4'
    writer = cv2.VideoWriter(str(out_path),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps_out, (vid_w, vid_h))
    if not writer.isOpened():
        raise RuntimeError(f'cannot open writer for {out_path}')

    WHITE = (255, 255, 255)
    YELLOW = (0, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (140, 140, 140)
    n_total_rears = len(rears)
    n_total_clip_frames = sum(c['ce'] - c['cs'] for c in clips)
    print(f'  output: {out_path}  ({n_total_clip_frames} frames ~= '
          f'{n_total_clip_frames / fps_out:.1f}s at {fps_out} fps)', flush=True)
    t0 = time.time()
    frames_written = 0

    # Header legend strip (drawn on every frame in top-left): small swatches.
    def _draw_overlay(img, frame_idx, clip_rears, clip_frame_local, clip_len,
                       active_class, active_rear_idx_1based):
        # Session timestamp MM:SS based on cam0 frame index.
        sec = frame_idx / SLEAP_HZ
        mm = int(sec // 60); ss = sec - mm * 60
        ts = f'session {mm:02d}:{ss:05.2f}'
        # Clip-local time.
        cl_sec = clip_frame_local / SLEAP_HZ
        cl_mm = int(cl_sec // 60); cl_ss = cl_sec - cl_mm * 60
        clip_ts = f'clip {cl_mm:01d}:{cl_ss:05.2f} / {clip_len / SLEAP_HZ:.1f}s'
        # Drop-shadow for legibility.
        def t(text, org, color, scale=0.75, thick=2):
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK,
                        thick + 2, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        thick, cv2.LINE_AA)

        t(ts, (15, 32), WHITE)
        t(clip_ts, (15, 62), WHITE)
        # Rear label: show the clip's rear range, then highlight the currently
        # active one if any. Single-rear clips look the same as before.
        rear_idxs = [r[0] for r in clip_rears]
        if len(rear_idxs) == 1:
            range_label = f'rear {rear_idxs[0]}/{n_total_rears}'
        else:
            range_label = (f'rears {rear_idxs[0]}-{rear_idxs[-1]}/'
                            f'{n_total_rears}')
        if active_rear_idx_1based is not None:
            rear_label = f'{range_label}  (active: rear {active_rear_idx_1based})'
            color = YELLOW
        else:
            rear_label = range_label
            color = WHITE
        t(rear_label, (15, 92), color)

        # Legend in the top-right region (always visible so the marker meaning
        # is obvious without context).
        legend_x = vid_w - 280
        cv2.rectangle(img, (legend_x - 10, 10), (vid_w - 10, 130), BLACK, -1)
        cv2.rectangle(img, (legend_x - 10, 10), (vid_w - 10, 130), GRAY, 1)
        t('template-match marker:', (legend_x, 32), WHITE, scale=0.45, thick=1)
        for k, (cls, label) in enumerate([
                ('online',         'online-fired match'),
                ('port_excluded',  'offline match, port-excluded'),
                ('other',          'offline-only (other)'),
        ]):
            cy = 50 + 24 * k
            cv2.circle(img, (legend_x + 12, cy + 6),
                        7, CLASS_COLORS_BGR[cls], -1)
            t(label, (legend_x + 30, cy + 12), WHITE, scale=0.45, thick=1)

        # Big corner marker (right side, just under the legend).
        marker_center = (vid_w - 50, 165)
        if active_class is not None:
            cv2.circle(img, marker_center, 22, CLASS_COLORS_BGR[active_class], -1)
            cv2.circle(img, marker_center, 22, BLACK, 2)
        else:
            cv2.circle(img, marker_center, 22, (60, 60, 60), 1)

    # Stream through clips in order; each clip plays back as a single
    # contiguous span [cs, ce). For each frame check whether it's inside any
    # of that clip's rears (clip may contain >1 rear after merging).
    for clip in clips:
        cs, ce = clip['cs'], clip['ce']
        clip_rears = clip['rears']
        cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
        clip_len = ce - cs
        for local_idx in range(clip_len):
            ok, frame = cap.read()
            if not ok:
                # Pad with black frame on read failure so timing stays sane.
                frame = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
            global_frame = cs + local_idx
            active_class = _classify_frame(global_frame, matches, classes)
            # Find which rear (if any) contains this frame.
            active_rear_idx_1based = None
            for (rear_idx_1, rs, re) in clip_rears:
                if rs <= global_frame < re:
                    active_rear_idx_1based = rear_idx_1
                    break
            _draw_overlay(frame, global_frame, clip_rears, local_idx, clip_len,
                          active_class, active_rear_idx_1based)
            writer.write(frame)
            frames_written += 1
            if frames_written % 500 == 0:
                el = time.time() - t0
                print(f'    {frames_written}/{n_total_clip_frames} '
                      f'({el:.1f}s elapsed, {frames_written/el:.0f} fps)',
                      flush=True)

    writer.release()
    cap.release()
    el = time.time() - t0
    print(f'  saved {out_path}  ({frames_written} frames in {el:.1f}s)',
          flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rat', required=True, action='append',
                    help='Repeatable. e.g. --rat R4 --rat R5')
    ap.add_argument('--session', required=True)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--pad_s', type=float, default=5.0,
                    help='Pad before/after each rear, in seconds (default 5).')
    ap.add_argument('--out_dir', type=str, default='rear_compilations')
    ap.add_argument('--fps_out', type=int, default=20)
    args = ap.parse_args()
    for rat in args.rat:
        render_compilation(rat, args.session, args.camera, args.pad_s,
                            args.out_dir, fps_out=args.fps_out)


if __name__ == '__main__':
    main()
