"""End-to-end offline pipeline on a small subset of *processed* frame indices
from one session.

Mirrors the live campy-CLIRB pipeline:
  - Resolve processed_frame -> per-camera video frame via frame_mapping.csv
    (live pipeline can have cam0/1/2 at slightly different frame_ids for the
    same processed sample — handle that).
  - Read the actual video frames (BGR via cv2 -> RGB).
  - tf.image.resize(bilinear, antialias=False) 1200x1920 -> 600x960.
  - SLEAP forward pass (HWC float32/255).
  - Confmap argmax -> px coords *8 + 0.5 -> full-res px.
  - Behavior-side cleanup (per-cam COM filter + NaN -> prev).
  - Triangulate (undistort + SVD DLT).
  - Compare against saved sleap_keys_2D and triang_keys_3D at the matching
    processed_frame indices.

Reports per-frame 2D residuals (offline vs saved), per-frame 3D residuals,
and wall-clock breakdown for storage / throughput planning.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")
sys.path.insert(0, str(REPO))

from data_io import load_sleap_keys_2d, load_sleap_keys_3d  # noqa: E402

from corrector.online_pipeline import (  # noqa: E402
    N_KEYPOINTS, SleapModel, build_calibration_for_session,
    find_peaks_from_confmaps, load_frame_mapping, postprocess_peaks,
    preprocess_2d_behavior, preprocess_for_sleap, read_session_frames_per_cam,
    triangulate_session)


MODEL_DIR = ("/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/models/"
             "250731_105225.single_instance.n=10383.og")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rat", default="R2")
    ap.add_argument("--session", default="2025_11_01_1")
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--start_proc", type=int, default=5000,
                     help="starting processed_frame index")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    args = ap.parse_args()

    fm = load_frame_mapping(args.rat, args.session)
    processed_indices = list(range(args.start_proc,
                                     args.start_proc + args.n_frames))
    per_cam = np.stack([
        fm.loc[p, [f"cam{ci}_frame" for ci in range(3)]].values
        for p in processed_indices
    ]).astype(int)

    print("=== Offline pipeline subset run ===")
    print(f"  rat={args.rat}  session={args.session}")
    print(f"  processed_frames=[{args.start_proc}, "
          f"{args.start_proc + args.n_frames})  (n={args.n_frames})")
    print(f"  per-cam video frame example: proc {processed_indices[0]} -> "
          f"cams {per_cam[0].tolist()}")

    # 1. Read frames
    t0 = time.perf_counter()
    frames = read_session_frames_per_cam(args.rat, args.session, per_cam)
    t_read = time.perf_counter() - t0
    n_cams = frames.shape[1]
    print(f"\n[1/6] read_session_frames_per_cam: {t_read:.2f}s "
          f"({1000 * t_read / args.n_frames:.1f}ms/sample)  "
          f"shape={frames.shape}")

    # 2. Preprocess (RGB, resize bilinear, /255)
    t0 = time.perf_counter()
    pre = preprocess_for_sleap(frames)
    t_pre = time.perf_counter() - t0
    print(f"[2/6] preprocess_for_sleap: {t_pre:.2f}s  out={pre.shape}")

    # 3. SLEAP forward
    t0 = time.perf_counter()
    model = SleapModel(args.model_dir)
    t_load = time.perf_counter() - t0
    print(f"[3a/6] SleapModel load: {t_load:.2f}s")
    t0 = time.perf_counter()
    outs = model.predict(pre)
    t_pred = time.perf_counter() - t0
    print(f"[3b/6] SLEAP forward: {t_pred:.2f}s "
          f"({1000 * t_pred / (args.n_frames * n_cams):.1f}ms/image)")

    cm = outs["SingleInstanceConfmapsHead"]
    if cm.shape[-1] == N_KEYPOINTS:
        cm = np.transpose(cm, (0, 3, 1, 2))

    # 4. Peaks + postprocess
    t0 = time.perf_counter()
    pk, vals = find_peaks_from_confmaps(cm, threshold=0.01)
    peaks_full = postprocess_peaks(pk, vals)
    peaks_full = peaks_full.reshape(args.n_frames, n_cams, N_KEYPOINTS, 3)
    t_peaks = time.perf_counter() - t0
    print(f"[4/6] peaks + postprocess: {t_peaks:.2f}s")

    # 5. Behavior cleanup
    t0 = time.perf_counter()
    clean = preprocess_2d_behavior(peaks_full)
    t_clean = time.perf_counter() - t0
    print(f"[5/6] preprocess_2d_behavior: {t_clean:.2f}s")

    # 6. Triangulate
    t0 = time.perf_counter()
    calib = build_calibration_for_session(args.rat, args.session)
    print(f"        calibration date: {calib['cal_date']}")
    pts3d = triangulate_session(clean[..., :2], calib)
    t_tri = time.perf_counter() - t0
    print(f"[6/6] triangulate_session: {t_tri:.2f}s "
          f"({1000 * t_tri / args.n_frames:.1f}ms/sample)")

    # --- Compare ---
    # IMPORTANT index convention:
    #   - sleap_keys_2D.npy is indexed by processed_frame (one row per network
    #     forward pass). Use processed_indices to index it.
    #   - triang_keys_3D.npy is indexed by cam0_frame (one row per video frame;
    #     gaps from dropped frames are linearly interpolated by behavior.py).
    #     Use cam0_frame to index it.
    saved_2d = load_sleap_keys_2d(args.rat, args.session)
    saved_3d = load_sleap_keys_3d(args.rat, args.session)
    saved_2d_sub = saved_2d[processed_indices]
    cam0_frames = per_cam[:, 0]
    saved_3d_sub = saved_3d[cam0_frames]

    print("\n=== 2D residual (offline-clean vs saved) — per-cam aggregates ===")
    for ci in range(n_cams):
        d = np.linalg.norm(
            clean[:, ci, :, :2] - saved_2d_sub[:, ci, :, :2], axis=-1)
        valid = (saved_2d_sub[:, ci, :, 2] > 0) & np.isfinite(d)
        dv = d[valid]
        cdiff = np.abs(clean[:, ci, :, 2] - saved_2d_sub[:, ci, :, 2])
        print(f"  cam{ci}: 2D px  median={np.median(dv):6.2f}  "
              f"p90={np.percentile(dv, 90):7.2f}  "
              f"max={dv.max():7.2f}  n={dv.size}  |  "
              f"conf |Δ| med={np.median(cdiff):.4f}  "
              f"frac_>0.05={(cdiff > 0.05).mean():.3f}")

    print("\n=== 3D residual (offline vs saved triang_keys_3D) ===")
    d3 = np.linalg.norm(pts3d - saved_3d_sub, axis=-1)
    print(f"  per-frame median across kp:")
    for i, p in enumerate(processed_indices):
        print(f"    proc {p:5d}: median={np.median(d3[i]):7.2f}mm  "
              f"p90={np.percentile(d3[i], 90):7.2f}mm  "
              f"max={d3[i].max():7.2f}mm")
    print(f"  overall: median={np.median(d3):7.2f}mm  "
          f"p90={np.percentile(d3, 90):7.2f}mm  "
          f"max={d3.max():7.2f}mm")

    # --- Throughput / storage ---
    per = (t_read + t_pre + t_pred + t_peaks + t_clean + t_tri) / args.n_frames
    full_min = per * 36000 / 60.0
    sleap_only_per_frame = (t_pred + t_peaks + t_clean + t_tri + t_pre) \
        / args.n_frames
    print("\n=== Throughput / storage estimate ===")
    print(f"  per-sample wall-clock (cpu): {1000 * per:.1f}ms "
          f"-> 36000-frame session ~ {full_min:.1f} min")
    print(f"  breakdown:")
    print(f"    video read         : {1000 * t_read/args.n_frames:6.1f}ms")
    print(f"    preprocess (resize): {1000 * t_pre/args.n_frames:6.1f}ms")
    print(f"    SLEAP forward      : {1000 * t_pred/args.n_frames:6.1f}ms "
          f"(CPU; GPU will be ~10-20x faster)")
    print(f"    peaks+clean+triang : "
          f"{1000 * (t_peaks + t_clean + t_tri)/args.n_frames:6.1f}ms")
    print(f"  per-sample disk usage: 2D (3 cams x 23 kp x 3) = 207 bytes; "
          f"3D 552 bytes -> 36000 frames = ~27 MB/session")


if __name__ == "__main__":
    main()
