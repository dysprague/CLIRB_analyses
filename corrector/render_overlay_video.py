"""
Render side-by-side overlay videos for visual inspection of the corrector.

Left panel:  raw Camera<N> frame with raw SLEAP (cyan) and DANNCE (magenta) overlaid.
Right panel: same frame with CORRECTED SLEAP (cyan) and DANNCE (magenta) overlaid.

Both panels project keypoints into the SLEAP camera coordinate system, mirroring
qc_utils.generate_qc_video:
  - SLEAP coords are projected directly with the SLEAP calibration.
  - DANNCE coords (and corrected SLEAP, which is in DANNCE space) are first
    inverse-Procrustes'd back into SLEAP world space, then projected.

Usage:
    python -m corrector.render_overlay_video \
        --rat R3 --session 2026_02_06_1 --model mlp \
        --start_frame 0 --n_frames 1000 --camera 0
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import median_filter

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent.parent / "experiments"))

from config import EDGES, sleap_path, calibration_path
from data_io import load_aligned_data, load_sleap_dannce_keys
from projection import project_3d_to_2d_for_camera
from qc_utils import find_sleap_dannce_alignment
from skeleton import normalize_skeleton_batch

from corrector.models import build_model

OUT_DIR = _THIS.parent / "videos"
OUT_DIR.mkdir(exist_ok=True)

SLEAP_MEDFILT = 11
DANNCE_MEDFILT = 25


def correct_egocentric(model, sl_eg: np.ndarray, device, batch=8192) -> np.ndarray:
    out = np.empty_like(sl_eg, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(sl_eg), batch):
            x = torch.from_numpy(sl_eg[i:i + batch].astype(np.float32)).to(device)
            out[i:i + batch] = model(x).cpu().numpy()
    return out


def egocentric_to_world(eg_pts: np.ndarray, rotmat_xy: np.ndarray,
                        spine_m: np.ndarray) -> np.ndarray:
    """Inverse of skeleton.normalize_skeleton_batch.

    eg_pts    (T, 23, 3)  egocentric coords
    rotmat_xy (T, 2, 2)   xy rotation used during normalization (world->ego)
    spine_m   (T, 3)      original SpineM world coordinate
    """
    out = eg_pts.copy()
    R_inv = np.transpose(rotmat_xy, axes=(0, 2, 1))
    out[:, :, :2] = np.einsum("fij,fkj->fki", R_inv, eg_pts[:, :, :2])
    out += spine_m[:, None, :]
    return out


def draw_skel(img, pts2d, color, edges=EDGES, radius=4, thickness=1):
    pts = pts2d.astype(np.int32)
    h, w = img.shape[:2]
    valid = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    for e in edges:
        if valid[e[0]] and valid[e[1]]:
            cv2.line(img, tuple(pts[e[0]]), tuple(pts[e[1]]), color, thickness, cv2.LINE_AA)
    for i in range(len(pts)):
        if valid[i]:
            cv2.circle(img, tuple(pts[i]), radius, color, -1, cv2.LINE_AA)


def render(rat: str, session: str, model_name: str,
           start_frame: int, n_frames: int, camera: int,
           fps: int = 20, alignment_n_frames: int = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Load checkpoint -----
    ckpt_path = _THIS.parent / "checkpoints" / f"{rat}_{model_name}.pt"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"],
                        hidden=ck.get("hidden", 128),
                        n_hidden_layers=ck.get("n_hidden_layers", 2))
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()

    # ----- Load raw SLEAP and DANNCE in their native coordinate systems -----
    print(f"Loading {rat}/{session} ...", flush=True)
    keys = load_sleap_dannce_keys(rat, session)
    aligned = load_aligned_data(rat, session)

    sleap_3d_raw = keys["sleap_keys_3D"].astype(np.float64)        # (n_sleap, 23, 3)
    dannce_3d = keys["dannce_keys_3D"].astype(np.float64)
    if dannce_3d.ndim == 4:
        dannce_3d = dannce_3d.squeeze(axis=1).transpose(0, 2, 1)
    else:
        dannce_3d = dannce_3d.transpose(0, 2, 1)

    sleap_3d_raw = median_filter(sleap_3d_raw, size=(SLEAP_MEDFILT, 1, 1))
    dannce_3d = median_filter(dannce_3d, size=(DANNCE_MEDFILT, 1, 1))

    aligned_indices = aligned["dannce_idx_for_sleap_cams"].astype(int).ravel()

    # ----- Procrustes alignment (SLEAP world -> DANNCE world) -----
    # qc_utils default: try z_flip; pick lower-residual solution.
    print(f"Fitting Procrustes (n_frames={alignment_n_frames})...", flush=True)
    align = find_sleap_dannce_alignment(
        sleap_3d_raw, dannce_3d, aligned_indices,
        n_sample_frames=alignment_n_frames, seed=42, try_z_flip=True,
    )
    print(f"  residual={align['residual']:.2f}  scale={align['s']:.4f}  "
          f"z_flipped={align['z_flipped']}", flush=True)

    # ----- Apply corrector in egocentric DANNCE-space -----
    # The corrector was trained on SLEAP & DANNCE both passed through
    # exp_utils.load_session_data, which z-flips SLEAP and resamples DANNCE to
    # the SLEAP frame rate, then both go through normalize_skeleton_batch.
    sl_zflip = sleap_3d_raw.copy()
    sl_zflip[:, :, 2] = -sl_zflip[:, :, 2]                          # z-flip per load_session_data
    sl_eg, sl_rot_xy, sl_spineM_zflip = normalize_skeleton_batch(sl_zflip)
    sl_eg_corr = correct_egocentric(model, sl_eg.astype(np.float32), device)
    sl_world_corr_zflip = egocentric_to_world(sl_eg_corr, sl_rot_xy, sl_spineM_zflip)
    # Undo z-flip so corrected SLEAP lives in the same space as raw SLEAP
    sl_world_corr_sleapspace = sl_world_corr_zflip.copy()
    sl_world_corr_sleapspace[:, :, 2] = -sl_world_corr_sleapspace[:, :, 2]

    # ----- Per-frame DANNCE in SLEAP space (for left panel) -----
    end_frame = min(start_frame + n_frames, len(sleap_3d_raw), len(aligned_indices))
    rng = slice(start_frame, end_frame)
    dn_per_sleap_frame = np.zeros((end_frame - start_frame, 23, 3))
    for j, i in enumerate(range(start_frame, end_frame)):
        di = aligned_indices[i] if i < len(aligned_indices) else 0
        if di < len(dannce_3d):
            dn_per_sleap_frame[j] = dannce_3d[di]
    dn_in_sleap = align["apply_inverse"](dn_per_sleap_frame)        # DANNCE -> SLEAP space

    # ----- Project everything to Camera<camera> using SLEAP calibration -----
    cal_folder = calibration_path(rat, session)
    print(f"Calibration folder: {cal_folder}", flush=True)
    print(f"Projecting {end_frame - start_frame} frames to Camera{camera}...",
          flush=True)
    sleap_2d_raw = project_3d_to_2d_for_camera(
        sleap_3d_raw[rng], cal_folder, camera_idx=camera)
    sleap_2d_corr = project_3d_to_2d_for_camera(
        sl_world_corr_sleapspace[rng], cal_folder, camera_idx=camera)
    dannce_2d = project_3d_to_2d_for_camera(
        dn_in_sleap, cal_folder, camera_idx=camera)

    # ----- Open Camera<camera> video -----
    sp = sleap_path(rat, session)
    cam_name = f"Camera{camera}"
    video_file = Path(sp) / cam_name / "0.mp4"
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_file}")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_w = vid_w * 2
    out_h = vid_h
    out_path = OUT_DIR / f"{rat}_{session}_cam{camera}_f{start_frame}-{end_frame}_{model_name}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_path}")

    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    YELLOW = (0, 255, 255)
    print(f"Rendering to {out_path}...", flush=True)
    t0 = time.time()
    n = end_frame - start_frame
    for fi in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        left = frame.copy()
        right = frame.copy()
        draw_skel(left, sleap_2d_raw[fi], CYAN)
        draw_skel(left, dannce_2d[fi], MAGENTA)
        draw_skel(right, sleap_2d_corr[fi], CYAN)
        draw_skel(right, dannce_2d[fi], MAGENTA)
        cv2.putText(left, "SLEAP raw", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN, 2, cv2.LINE_AA)
        cv2.putText(left, "DANNCE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(right, "SLEAP corrected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, CYAN, 2, cv2.LINE_AA)
        cv2.putText(right, "DANNCE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(left, f"frame {start_frame + fi}", (10, vid_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
        cv2.putText(right, f"{rat}/{session}  Camera{camera}", (10, vid_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
        writer.write(np.hstack([left, right]))
        if (fi + 1) % 200 == 0 or fi == n - 1:
            print(f"  {fi + 1}/{n} ({time.time()-t0:.1f}s)", flush=True)
    writer.release()
    cap.release()
    print(f"saved {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--n_frames", type=int, default=1000)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--alignment_n_frames", type=int, default=1000)
    args = ap.parse_args()
    render(args.rat, args.session, args.model, args.start_frame, args.n_frames,
           args.camera, args.fps, args.alignment_n_frames)


if __name__ == "__main__":
    main()
