"""Compile a video of ONLY the offline matches that the current end-frame gate
counts as INSIDE the port-exclusion zone (d_end < min_port), for one session.

Each match contributes a clip of +-PAD_S seconds around its match-end (trigger)
frame; overlapping clips are merged into one continuous segment. SLEAP keypoints
(from triang_keys_3D.npy, causal-5 median filtered like the online convention)
are reprojected onto Camera0 and drawn on every frame. The water-port location
is marked, and a HUD shows which match is active, its COM->port distance, and a
marker at the exact match-end frame.

This is the "inside-zone" set -- the matches that the mean-gate analysis flags as
possibly mislabeled (animal lunging to the port right as the template completes).

Frame spaces (verified): triang_keys_3D.npy is indexed in cam0/video-frame space
(1:1 with Camera*/0.mp4), and replay_rule_offline slides over triang, so the
match indices ARE video-frame indices -- we seek Camera0/0.mp4 directly.

Usage:
    python scripts/render_inside_port_matches.py --rat R6 --session 2026_07_06_2
    python scripts/render_inside_port_matches.py --rat R6 --session 2026_07_06_2 \
        --pad_s 3 --camera 0
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from config import DATA_ROOT, EDGES                                         # noqa: E402
from corrector.data_world_2d import (load_session_calibration,              # noqa: E402
                                      project_3d_to_2d_batch)
from scripts.render_triang_overlay import _smooth_keypoints, _draw_skel     # noqa: E402

# analysis pipeline (project -> PCs -> replay template rule) lives here
sys.path.insert(0, str(_THIS.parent.parent / "scratch"))
import port_zone_timing_analysis as pz                                       # noqa: E402

OUT_ROOT = _THIS.parent.parent / "corrector" / "videos" / "inside_port_review"

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE  = (255, 255, 255)
ORANGE = (0, 165, 255)


def compute_inside_matches(sd, template_len=pz.TEMPLATE_LEN):
    """Return (match_end_frames, d_end) for matches counted INSIDE the zone."""
    triang = sd["triang_3d"]
    n = len(triang)
    match_end, _ = pz.replay_rule_offline(sd)
    match_end = np.asarray(match_end, dtype=int)
    port_xy = np.asarray(sd["port_xy"]).ravel()[:2]
    com_xy = triang.mean(axis=1)[:, :2]
    port_dist = np.linalg.norm(com_xy - port_xy[None, :], axis=1)
    d_end = port_dist[np.clip(match_end, 0, n - 1)]
    min_port = sd["min_port_distance"]
    inside = d_end < min_port
    return match_end[inside], d_end[inside], min_port, port_dist


def merge_windows(centers, pad, n_frames):
    """Build [start,end) windows of +-pad around each center; merge overlaps.
    Returns list of (start, end, [ (center, orig_index) ... ]) so the HUD can
    label which match(es) fall in each merged segment."""
    order = np.argsort(centers)
    segs = []  # each: [start, end, list of (center, idx)]
    for oi in order:
        c = int(centers[oi])
        s = max(0, c - pad)
        e = min(n_frames, c + pad + 1)
        if segs and s <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], e)
            segs[-1][2].append((c, int(oi)))
        else:
            segs.append([s, e, [(c, int(oi))]])
    return segs


def render(rat, session, camera, pad_s, fps, smooth_size, smooth_causal):
    sd = pz.load_session(rat, session)
    if sd is None:
        raise RuntimeError(f"could not load {rat}/{session}")
    if sd["port_xy"] is None:
        raise RuntimeError("no water_port_location for this session")

    triang = sd["triang_3d"].astype(np.float32)
    n_frames = len(triang)
    m_end, d_end, min_port, port_dist = compute_inside_matches(sd)
    # match number in the FULL (all matches) ordering, for reference in the HUD
    all_end, _ = pz.replay_rule_offline(sd)
    all_end = np.asarray(all_end, dtype=int)
    match_no = {int(f): int(np.where(all_end == f)[0][0]) + 1 for f in m_end}

    print(f"{rat}/{session}: {len(all_end)} total matches, "
          f"{len(m_end)} inside the port zone (d_end < {min_port:.0f} mm)", flush=True)
    if len(m_end) == 0:
        print("nothing to render.", flush=True)
        return None

    pad = int(round(pad_s * fps))
    segs = merge_windows(m_end, pad, n_frames)
    total_clip_frames = sum(e - s for s, e, _ in segs)
    print(f"  {len(segs)} merged segments, {total_clip_frames} frames "
          f"(~{total_clip_frames/fps:.0f}s) with +-{pad_s}s padding", flush=True)

    # smoothing (matches online causal-median convention)
    if smooth_size > 1:
        triang = _smooth_keypoints(triang, smooth_size, smooth_causal)

    calib, cal_date = load_session_calibration(rat, session)
    cam = calib[camera]
    port3d = np.asarray(sd["port_xy"]).ravel()[:3].astype(np.float32)
    port2d = project_3d_to_2d_batch(port3d.reshape(1, 1, 3), cam)[0, 0]

    video_file = Path(DATA_ROOT) / rat / session / "sleap" / f"Camera{camera}" / "0.mp4"
    capw = cv2.VideoCapture(str(video_file))
    if not capw.isOpened():
        raise FileNotFoundError(f"cannot open {video_file}")
    vid_w = int(capw.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(capw.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{rat}_{session}_cam{camera}_inside_port_pad{int(pad_s)}s.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (vid_w, vid_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_path}")

    # project only the frames we need, per segment (cheap vs video IO)
    print(f"rendering {out_path.name} ({vid_w}x{vid_h}) ...", flush=True)
    t0 = time.time()
    written = 0
    n_inside = len(m_end)
    for si, (s, e, members) in enumerate(segs):
        kp2d = project_3d_to_2d_batch(triang[s:e], cam)
        capw.set(cv2.CAP_PROP_POS_FRAMES, s)
        centers = sorted(members)  # (center_frame, orig_idx)
        for local, f in enumerate(range(s, e)):
            ok, frame = capw.read()
            if not ok:
                break
            _draw_skel(frame, kp2d[local], GREEN)
            # port marker (exclusion-zone anchor)
            cv2.circle(frame, tuple(port2d.astype(int)), 14, RED, 2, cv2.LINE_AA)
            cv2.putText(frame, "port", (int(port2d[0]) + 16, int(port2d[1]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)

            # which match is this frame closest to (for the HUD)?
            cf, ci = min(centers, key=lambda cc: abs(cc[0] - f))
            mno = match_no.get(cf, -1)
            dist = float(d_end[np.where(m_end == cf)[0][0]])
            dframe = f - cf  # negative before trigger, 0 at trigger
            at_trigger = (dframe == 0)

            # HUD banner
            cv2.rectangle(frame, (0, 0), (vid_w, 70), (0, 0, 0), -1)
            hud = (f"INSIDE-ZONE MATCH {mno}   COM->port {dist:.0f} mm "
                   f"(< {min_port:.0f})   frame {f}")
            cv2.putText(frame, hud, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        WHITE, 2, cv2.LINE_AA)
            sub = (f"{rat}/{session}  cam{camera}  |  match {_INSIDE_ORDER.get(cf, '?')} "
                   f"of {n_inside} inside-zone  |  +-{int(pad_s)}s window")
            cv2.putText(frame, sub, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        YELLOW, 1, cv2.LINE_AA)

            # trigger countdown / flash at the match-end frame
            if at_trigger:
                cv2.rectangle(frame, (2, 2), (vid_w - 2, vid_h - 2), ORANGE, 6)
                cv2.putText(frame, ">>> MATCH TRIGGER <<<",
                            (vid_w // 2 - 220, vid_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, ORANGE, 3, cv2.LINE_AA)
            else:
                tcol = ORANGE if abs(dframe) <= 5 else WHITE
                rel = "to trigger" if dframe < 0 else "since trigger"
                cv2.putText(frame, f"t{dframe:+d}f ({abs(dframe)/fps:.1f}s {rel})",
                            (vid_w - 470, vid_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, tcol, 2, cv2.LINE_AA)

            writer.write(frame)
            written += 1
        if (si + 1) % 5 == 0 or si == len(segs) - 1:
            print(f"  segment {si+1}/{len(segs)}  ({written} frames, "
                  f"{time.time()-t0:.1f}s)", flush=True)

    writer.release()
    capw.release()
    print(f"saved {out_path}  ({written} frames, {written/fps:.0f}s)", flush=True)
    return out_path


_INSIDE_ORDER = {}   # frame -> 1-based rank among this session's inside matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pad_s", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--smooth_size", type=int, default=5)
    ap.add_argument("--no_smooth_causal", dest="smooth_causal",
                    action="store_false", default=True)
    args = ap.parse_args()

    # pre-compute the inside-order map for the sub-HUD (frame-sorted 1..N)
    sd = pz.load_session(args.rat, args.session)
    m_end, _, _, _ = compute_inside_matches(sd)
    for k, f in enumerate(sorted(int(x) for x in m_end), start=1):
        _INSIDE_ORDER[f] = k

    render(args.rat, args.session, args.camera, args.pad_s, args.fps,
           args.smooth_size, args.smooth_causal)


if __name__ == "__main__":
    main()
