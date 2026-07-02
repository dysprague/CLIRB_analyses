"""Rear-compilation video with a per-rear static PC snapshot panel.

For each rear, finds the 30-frame window inside that rear with the lowest
MSE against the template (per-frame, per-PC squared error summed). That
snapshot is drawn statically in the bottom panel for the entire duration
the rear (and its surrounding pad) is shown. As the video plays through
the snapshot's session frames, a moving red dot on the snapshot trace shows
which timepoint is currently being displayed in the camera view above.

Output: <out_dir>/<rat>_<session>_rear_compilation_pc.mp4
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from scripts.render_rear_compilation import (load_and_analyze, SLEAP_HZ,    # noqa: E402
                                              TEMPLATE_LEN_FRAMES,
                                              CLASS_COLORS_BGR,
                                              _classify_frame,
                                              _causal_median_5_kp)


# Bottom panel geometry.
PC_PANEL_H = 400          # total height of the bottom panel
PC_SUBPANEL_H = 200       # each of PC1 and PC2 gets this height


# ---------------------------------------------------------------------------
# Clip building (merge overlapping padded windows; identical to parent script)
# ---------------------------------------------------------------------------
def _build_clips(rears, triang_len, pad):
    clips = []
    for rear_idx, (rs, re, _) in enumerate(rears):
        clip_start = max(0, rs - pad)
        clip_end = min(triang_len, re + pad)
        if clips and clip_start < clips[-1]['ce']:
            clips[-1]['ce'] = max(clips[-1]['ce'], clip_end)
            clips[-1]['rears'].append((rear_idx + 1, rs, re))
        else:
            clips.append({'cs': clip_start, 'ce': clip_end,
                          'rears': [(rear_idx + 1, rs, re)]})
    return clips


# ---------------------------------------------------------------------------
# Find the maximal-alignment 30-frame window for one rear
# ---------------------------------------------------------------------------
def find_max_alignment_window(pcs, template, bounds, rear_start, rear_end):
    """For one rear, return (best_end_frame, best_mse, snapshot_pcs).

    Searches all 30-frame windows whose END frame lies in [rear_start, rear_end).
    MSE is computed over (template_len * n_pcs) values: sum of per-frame
    per-PC squared error between the PC trace and the template trajectory.
    snapshot_pcs has shape (30, n_pcs).
    """
    n_pcs = template.shape[1]
    T = len(pcs)
    # Window end-frame candidates inside the rear, with enough lead-in
    # frames to form a full template-length window.
    end_min = max(rear_start, TEMPLATE_LEN_FRAMES - 1)
    end_max = min(rear_end, T) - 1
    if end_max < end_min:
        # Rear is at session start with insufficient lead-in. Skip.
        return None, None, None
    # Vectorized MSE over all candidate windows.
    starts = np.arange(end_min - (TEMPLATE_LEN_FRAMES - 1),
                        end_max - (TEMPLATE_LEN_FRAMES - 1) + 1)
    if len(starts) == 0:
        return None, None, None
    # Build (n_windows, template_len, n_pcs) tensor.
    idx = starts[:, None] + np.arange(TEMPLATE_LEN_FRAMES)[None, :]
    windows = pcs[idx, :n_pcs]                       # (n_windows, 30, n_pcs)
    diff = windows - template[None, :, :]            # (n_windows, 30, n_pcs)
    mse = (diff ** 2).mean(axis=(1, 2))              # (n_windows,)
    best = int(np.argmin(mse))
    best_start = int(starts[best])
    best_end = best_start + TEMPLATE_LEN_FRAMES - 1
    return best_end, float(mse[best]), windows[best].copy()


# ---------------------------------------------------------------------------
# Pre-render one static panel per rear
# ---------------------------------------------------------------------------
class _PCPanelRenderer:
    """Pre-renders one static PC snapshot panel per rear, then per-frame
    just slaps that panel down + draws a moving red dot when the current
    playback frame is inside the snapshot's session-frame range.
    """

    def __init__(self, pcs_full, template, template_bounds, panel_w, rears):
        # PCs we care about (rule's pcs_to_use is conventionally [0, 1]).
        self.pc = pcs_full[:, :template.shape[1]].astype(np.float32)
        self.template = template.astype(np.float32)            # (30, n_pcs)
        self.bounds = template_bounds.astype(np.float32)       # (30, n_pcs)
        self.panel_w = panel_w
        self.n_pcs = template.shape[1]

        # Y-axis range per PC: span the worst case of (template ± bounds,
        # observed PC values within the session) so plots always fit.
        def _yrange(i):
            tmpl_lo = (self.template[:, i] - self.bounds[:, i]).min()
            tmpl_hi = (self.template[:, i] + self.bounds[:, i]).max()
            obs_lo, obs_hi = np.percentile(self.pc[:, i], [0.5, 99.5])
            lo = min(tmpl_lo, obs_lo)
            hi = max(tmpl_hi, obs_hi)
            margin = (hi - lo) * 0.08
            return float(lo - margin), float(hi + margin)
        self.yr = [_yrange(i) for i in range(self.n_pcs)]

        # Layout constants.
        self.subpanel_y = [(0, PC_SUBPANEL_H),
                            (PC_SUBPANEL_H, PC_PANEL_H)]
        self.margin_left = 80
        self.margin_right = 20
        self.margin_v = 30

        # Per-rear: compute the snapshot + pre-render the static panel.
        self.rear_panels = {}    # rear_idx_1based -> dict
        for rear_idx_1, rs, re in [(i + 1, rs_, re_)
                                     for i, (rs_, re_, _) in enumerate(rears)]:
            best_end, mse, snapshot = find_max_alignment_window(
                self.pc, self.template, self.bounds, rs, re)
            if best_end is None:
                # No valid window — render a panel without a snapshot trace.
                self.rear_panels[rear_idx_1] = self._render_static_panel(
                    rear_idx_1, rs, re, snapshot=None, snapshot_start=None,
                    mse=None)
            else:
                snapshot_start = best_end - (TEMPLATE_LEN_FRAMES - 1)
                self.rear_panels[rear_idx_1] = self._render_static_panel(
                    rear_idx_1, rs, re, snapshot=snapshot,
                    snapshot_start=snapshot_start, mse=mse)

    def _y_to_px(self, val, pc_idx):
        y0, y1 = self.subpanel_y[pc_idx]
        plot_top = y0 + self.margin_v
        plot_bot = y1 - self.margin_v
        lo, hi = self.yr[pc_idx]
        if hi == lo:
            return (plot_top + plot_bot) // 2
        frac = (val - lo) / (hi - lo)
        return int(plot_bot - frac * (plot_bot - plot_top))

    def _template_x_positions(self):
        """Pixel x positions for the 30 template frames inside the panel."""
        plot_left = self.margin_left
        plot_right = self.panel_w - self.margin_right
        return np.array(
            [int(plot_left + (k / (TEMPLATE_LEN_FRAMES - 1))
                 * (plot_right - plot_left))
             for k in range(TEMPLATE_LEN_FRAMES)], dtype=np.int32)

    def _render_static_panel(self, rear_idx, rs, re, snapshot, snapshot_start,
                              mse):
        """Render the unchanging part of the bottom panel for one rear.
        Returns dict with the canvas + metadata needed at playback time.
        """
        canvas = np.full((PC_PANEL_H, self.panel_w, 3), 30, dtype=np.uint8)
        # Divider between PC subpanels.
        cv2.line(canvas, (0, PC_SUBPANEL_H), (self.panel_w, PC_SUBPANEL_H),
                 (60, 60, 60), 1)
        tmpl_x = self._template_x_positions()

        # Save per-PC pixel positions for the snapshot so the playback marker
        # can be drawn fast.
        snapshot_xy = []   # list of (np.ndarray x, np.ndarray y) per PC
        for pc_idx in range(self.n_pcs):
            y0, y1 = self.subpanel_y[pc_idx]
            plot_left = self.margin_left
            plot_right = self.panel_w - self.margin_right
            text_color = (200, 200, 200)

            cv2.putText(canvas, f'PC{pc_idx + 1}',
                        (10, y0 + self.margin_v + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1,
                        cv2.LINE_AA)
            lo, hi = self.yr[pc_idx]
            cv2.putText(canvas, f'{hi:.0f}',
                        (10, y0 + self.margin_v + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1,
                        cv2.LINE_AA)
            cv2.putText(canvas, f'{lo:.0f}',
                        (10, y1 - self.margin_v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1,
                        cv2.LINE_AA)
            # Zero line.
            if lo < 0 < hi:
                y0px = self._y_to_px(0.0, pc_idx)
                cv2.line(canvas, (plot_left, y0px),
                         (plot_right, y0px), (80, 80, 80), 1)

            # Bounds ribbon.
            tmpl_lo_y = np.array(
                [self._y_to_px(float(self.template[k, pc_idx]
                                      - self.bounds[k, pc_idx]),
                                pc_idx) for k in range(TEMPLATE_LEN_FRAMES)],
                dtype=np.int32)
            tmpl_hi_y = np.array(
                [self._y_to_px(float(self.template[k, pc_idx]
                                      + self.bounds[k, pc_idx]),
                                pc_idx) for k in range(TEMPLATE_LEN_FRAMES)],
                dtype=np.int32)
            poly = np.concatenate(
                [np.stack([tmpl_x, tmpl_hi_y], axis=1),
                 np.stack([tmpl_x[::-1], tmpl_lo_y[::-1]], axis=1)], axis=0)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [poly], (120, 90, 30))
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, dst=canvas)

            # Template line (yellow, thick).
            tmpl_y = np.array(
                [self._y_to_px(float(self.template[k, pc_idx]), pc_idx)
                 for k in range(TEMPLATE_LEN_FRAMES)], dtype=np.int32)
            tmpl_pts = np.stack([tmpl_x, tmpl_y], axis=1).reshape(-1, 1, 2)
            cv2.polylines(canvas, [tmpl_pts], False, (0, 220, 220), 2,
                          cv2.LINE_AA)

            # Snapshot trace (green) — the maximal-alignment 30-frame window.
            if snapshot is not None:
                snap_y = np.array(
                    [self._y_to_px(float(snapshot[k, pc_idx]), pc_idx)
                     for k in range(TEMPLATE_LEN_FRAMES)], dtype=np.int32)
                snap_pts = np.stack([tmpl_x, snap_y],
                                     axis=1).reshape(-1, 1, 2)
                cv2.polylines(canvas, [snap_pts], False, (50, 255, 50), 2,
                              cv2.LINE_AA)
                snapshot_xy.append((tmpl_x.copy(), snap_y))
            else:
                snapshot_xy.append((tmpl_x.copy(), None))

        # Top-of-panel label: which rear this snapshot is for, the session
        # frames it spans, the MSE.
        if snapshot is not None:
            t_start_sec = snapshot_start / SLEAP_HZ
            t_end_sec = (snapshot_start + TEMPLATE_LEN_FRAMES - 1) / SLEAP_HZ
            label = (f'Rear {rear_idx}  max-alignment snapshot: '
                      f'session frames {snapshot_start}-'
                      f'{snapshot_start + TEMPLATE_LEN_FRAMES - 1}  '
                      f'({t_start_sec:.1f}s - {t_end_sec:.1f}s)  '
                      f'MSE={mse:.2f}')
        else:
            label = (f'Rear {rear_idx}  (no valid template window — '
                      f'rear too close to session start)')
        cv2.putText(canvas, label, (self.margin_left + 5, PC_PANEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                    cv2.LINE_AA)
        # Legend.
        leg_y = 16
        cv2.putText(canvas, 'snapshot', (self.panel_w - 240, leg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 1,
                    cv2.LINE_AA)
        cv2.line(canvas, (self.panel_w - 300, leg_y - 4),
                 (self.panel_w - 250, leg_y - 4), (50, 255, 50), 2)
        cv2.putText(canvas, 'template', (self.panel_w - 130, leg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1,
                    cv2.LINE_AA)
        cv2.line(canvas, (self.panel_w - 200, leg_y - 4),
                 (self.panel_w - 150, leg_y - 4), (0, 220, 220), 2)

        return {
            'canvas': canvas,
            'snapshot_start': snapshot_start,   # session frame where snapshot's frame 0 lives
            'snapshot_xy': snapshot_xy,         # list per-PC of (x_array, y_array)
            'has_snapshot': snapshot is not None,
        }

    def render(self, current_frame, active_rear_idx_1based,
               displayed_rear_idx_1based):
        """Return a panel for the current playback frame. Uses the static
        pre-rendered panel for `displayed_rear_idx_1based` and overlays a
        red dot on the snapshot trace when current_frame falls inside the
        snapshot's frame range.
        """
        panel_info = self.rear_panels.get(displayed_rear_idx_1based)
        if panel_info is None:
            return np.full((PC_PANEL_H, self.panel_w, 3), 30, dtype=np.uint8)
        canvas = panel_info['canvas'].copy()
        snap_start = panel_info['snapshot_start']
        if panel_info['has_snapshot'] and snap_start is not None:
            k = current_frame - snap_start
            if 0 <= k < TEMPLATE_LEN_FRAMES:
                for pc_idx in range(self.n_pcs):
                    xs, ys = panel_info['snapshot_xy'][pc_idx]
                    if ys is not None:
                        cv2.circle(canvas, (int(xs[k]), int(ys[k])), 7,
                                    (0, 0, 255), -1)
                        cv2.circle(canvas, (int(xs[k]), int(ys[k])), 7,
                                    (255, 255, 255), 1, cv2.LINE_AA)
        return canvas


# ---------------------------------------------------------------------------
# Decide which rear's snapshot to display at each clip frame
# ---------------------------------------------------------------------------
def _displayed_rear_idx(global_frame, clip_rears):
    """Return the 1-based rear index whose snapshot panel should be shown
    for `global_frame`. Rule:
      - If the frame is inside any rear, show that rear's snapshot.
      - Otherwise (pad / gap), show the next upcoming rear's snapshot.
      - After the last rear has ended, show the last rear's snapshot.
    clip_rears is a list of (rear_idx_1based, rs, re), sorted by rs.
    """
    for rear_idx_1, rs, re in clip_rears:
        if rs <= global_frame < re:
            return rear_idx_1
    # No active rear — pick next upcoming.
    for rear_idx_1, rs, re in clip_rears:
        if global_frame < rs:
            return rear_idx_1
    # Past the last rear in this clip — show the last one.
    return clip_rears[-1][0]


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------
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

    # ---- Template + bounds ----
    import glob
    sleap_dir = info['sleap_dir']
    cands = sorted(glob.glob(os.path.join(sleap_dir, f'{rat}_template_*.npz')))
    if cands:
        tmpl_path = cands[0]
    else:
        from config import DATA_ROOT
        tmpl_path = os.path.join(DATA_ROOT, rat, 'templates',
                                  f'{rat}_template_1.npz')
    tmpl_npz = np.load(tmpl_path, allow_pickle=True)
    template = tmpl_npz['template'][:, :2]
    pcs_to_use = np.asarray(tmpl_npz['pcs_to_use']).ravel().astype(int)
    if 'bounds_to_use' in tmpl_npz.files:
        bounds_arr = tmpl_npz['bounds_to_use']
    else:
        scalar = float(tmpl_npz['bounds'])
        bounds_arr = np.tile(tmpl_npz['feature_stds'][pcs_to_use] * scalar,
                              (template.shape[0], 1))

    # ---- PC projection — matches the rule's order ----
    from skeleton import normalize_skeleton_batch
    triang = np.load(os.path.join(sleap_dir,
                                    'triang_keys_3D.npy')).astype(np.float32)
    flipped = triang.copy(); flipped[:, :, 2] = -flipped[:, :, 2]
    rotated, _, _ = normalize_skeleton_batch(flipped)
    rotated_med = _causal_median_5_kp(rotated)
    flat = rotated_med.reshape(rotated_med.shape[0], -1)
    pcs_full = (flat - tmpl_npz['feature_means']) @ tmpl_npz['pc_weights'].T

    # ---- Pre-render one static panel per rear ----
    print('  pre-rendering per-rear panels ...', flush=True)
    pc_panel = _PCPanelRenderer(pcs_full, template, bounds_arr, vid_w, rears)
    print(f'    rendered {len(pc_panel.rear_panels)} panels', flush=True)

    # ---- Build merged clips ----
    clips = _build_clips(rears, triang_len, pad)
    n_merged_clips = len(clips)
    rears_per_clip_max = max(len(c['rears']) for c in clips) if clips else 0
    print(f'  merged into {n_merged_clips} clips '
          f'(max {rears_per_clip_max} rears in a single clip)', flush=True)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{rat}_{session}_rear_compilation_pc.mp4'
    out_w = vid_w
    out_h = vid_h + PC_PANEL_H
    writer = cv2.VideoWriter(str(out_path),
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps_out, (out_w, out_h))
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

    def _draw_top_overlay(img, frame_idx, clip_rears, clip_frame_local,
                            clip_len, active_class, active_rear_idx_1based,
                            in_snapshot):
        sec = frame_idx / SLEAP_HZ
        mm = int(sec // 60); ss = sec - mm * 60
        ts = f'session {mm:02d}:{ss:05.2f}'
        cl_sec = clip_frame_local / SLEAP_HZ
        cl_mm = int(cl_sec // 60); cl_ss = cl_sec - cl_mm * 60
        clip_ts = f'clip {cl_mm:01d}:{cl_ss:05.2f} / {clip_len / SLEAP_HZ:.1f}s'

        def t(text, org, color, scale=0.75, thick=2):
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK,
                        thick + 2, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        thick, cv2.LINE_AA)

        t(ts, (15, 32), WHITE)
        t(clip_ts, (15, 62), WHITE)
        rear_idxs = [r[0] for r in clip_rears]
        if len(rear_idxs) == 1:
            range_label = f'rear {rear_idxs[0]}/{n_total_rears}'
        else:
            range_label = f'rears {rear_idxs[0]}-{rear_idxs[-1]}/{n_total_rears}'
        if active_rear_idx_1based is not None:
            rear_label = f'{range_label}  (active: rear {active_rear_idx_1based})'
            color = YELLOW
        else:
            rear_label = range_label
            color = WHITE
        t(rear_label, (15, 92), color)
        if in_snapshot:
            t('IN MAX-ALIGNMENT SNAPSHOT', (15, 124),
              (50, 255, 50), scale=0.65, thick=2)

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
        marker_center = (vid_w - 50, 165)
        if active_class is not None:
            cv2.circle(img, marker_center, 22, CLASS_COLORS_BGR[active_class], -1)
            cv2.circle(img, marker_center, 22, BLACK, 2)
        else:
            cv2.circle(img, marker_center, 22, (60, 60, 60), 1)

    # ---- Stream through clips ----
    for clip in clips:
        cs, ce = clip['cs'], clip['ce']
        clip_rears = clip['rears']
        cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
        clip_len = ce - cs
        for local_idx in range(clip_len):
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
            global_frame = cs + local_idx
            active_class = _classify_frame(global_frame, matches, classes)
            active_rear_idx_1based = None
            for (rear_idx_1, rs, re) in clip_rears:
                if rs <= global_frame < re:
                    active_rear_idx_1based = rear_idx_1
                    break
            displayed_rear_idx = _displayed_rear_idx(global_frame, clip_rears)
            # Is the current frame inside the displayed rear's snapshot window?
            panel_info = pc_panel.rear_panels.get(displayed_rear_idx)
            in_snapshot = False
            if panel_info and panel_info['has_snapshot']:
                k = global_frame - panel_info['snapshot_start']
                in_snapshot = 0 <= k < TEMPLATE_LEN_FRAMES
            _draw_top_overlay(frame, global_frame, clip_rears, local_idx,
                              clip_len, active_class, active_rear_idx_1based,
                              in_snapshot)
            pc_img = pc_panel.render(global_frame, active_rear_idx_1based,
                                       displayed_rear_idx)
            combined = np.vstack([frame, pc_img])
            writer.write(combined)
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
    ap.add_argument('--rat', required=True, action='append')
    ap.add_argument('--session', required=True)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--pad_s', type=float, default=5.0)
    ap.add_argument('--out_dir', type=str, default='rear_compilations')
    ap.add_argument('--fps_out', type=int, default=20)
    args = ap.parse_args()
    for rat in args.rat:
        render_compilation(rat, args.session, args.camera, args.pad_s,
                            args.out_dir, fps_out=args.fps_out)


if __name__ == '__main__':
    main()
