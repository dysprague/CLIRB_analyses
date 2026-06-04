"""Compare a newly fine-tuned SLEAP model against an old SLEAP model and DANNCE.

Given a session that has SLEAP videos + saved DANNCE keypoints, this:

  1. Resolves a window of processed-frame indices to per-camera video frames
     via frame_mapping.csv (mirrors the live pipeline, where cam0/1/2 can sit
     at slightly different frame_ids for the same processed sample).
  2. Reads those video frames once and runs the *full offline SLEAP pipeline*
     (the same one in corrector.online_pipeline that mirrors campy-CLIRB) with
     BOTH the new model and the old model -- only the network weights differ,
     so this is a clean apples-to-apples comparison through identical
     preprocessing / peak-finding / triangulation.
  3. Loads DANNCE for the same window, resamples it to SLEAP frame rate, fits a
     per-session Procrustes (SLEAP_old <-> DANNCE), and inverse-transforms
     DANNCE into SLEAP world space so it overlays on the SLEAP camera.
  4. Projects all three 3D streams onto the chosen SLEAP camera and renders a
     three-panel video:  [ New SLEAP | Old SLEAP | DANNCE ].
  5. Prints per-keypoint median 3D residuals (new-SLEAP-vs-DANNCE and
     old-SLEAP-vs-DANNCE, both in DANNCE space) so "did fine-tuning help" has a
     number, not just a visual impression.

Usage:
    python -m corrector.compare_models_video \
        --new_model /path/to/finetuned.og \
        --rat R2 --session 2025_11_01_1 \
        --camera 0 --start_frame 5000 --n_frames 1000

`--old_model` defaults to the current production single-instance model.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))

from config import EDGES, N_KEYPOINTS, sleap_path, calibration_path  # noqa: E402
from data_io import load_sleap_dannce_keys, load_aligned_data  # noqa: E402
from projection import project_3d_to_2d_for_camera  # noqa: E402
from qc_utils import find_sleap_dannce_alignment  # noqa: E402

from corrector.online_pipeline import (  # noqa: E402
    SleapModel, build_calibration_for_session, find_peaks_from_confmaps,
    load_frame_mapping, postprocess_peaks, preprocess_2d_behavior,
    preprocess_for_sleap, read_session_frames_per_cam, triangulate_session)
from corrector.render_world_overlay import draw_skel  # noqa: E402

OUT_DIR = _THIS.parent / "videos"

# Visual-parity smoothing (matches qc_utils.generate_qc_video / render scripts).
SLEAP_MEDFILT = 11
DANNCE_MEDFILT = 25

DEFAULT_OLD_MODEL = ("/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/models/"
                     "250731_105225.single_instance.n=10383.og")

# Confmap head name emitted by the single-instance SavedModel.
CONFMAP_HEAD = "SingleInstanceConfmapsHead"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_sleap_inference(model_dir: str, pre: np.ndarray, n_frames: int,
                        n_cams: int, calib: dict,
                        peak_threshold: float = 0.01,
                        batch: int = 30) -> np.ndarray:
    """Run one SLEAP model over already-preprocessed frames and triangulate.

    pre : (n_frames * n_cams, 600, 960, 3) float32 in [0, 1] -- output of
          preprocess_for_sleap, shared across both models so frames are read
          and resized only once.
    batch : images per forward pass. 600x960 confmaps are memory-heavy, so the
          full set is chunked to stay within GPU memory (the desktop GPU shares
          VRAM with Xorg). Peak-finding/triangulation run on the full array.

    Returns (n_frames, 23, 3) world-space 3D keypoints (SLEAP space).
    The TF model is loaded and released inside this call to keep only one
    model resident on the GPU at a time.
    """
    t0 = time.perf_counter()
    model = SleapModel(model_dir)
    print(f"    loaded {Path(model_dir).name} ({time.perf_counter() - t0:.1f}s)",
          flush=True)

    t0 = time.perf_counter()
    cm_chunks = []
    for i in range(0, len(pre), batch):
        out = model.predict(pre[i:i + batch])[CONFMAP_HEAD]
        if out.shape[-1] == N_KEYPOINTS:  # HWC -> (B, n_kp, H, W)
            out = np.transpose(out, (0, 3, 1, 2))
        cm_chunks.append(out)
    cm = np.concatenate(cm_chunks, axis=0)
    print(f"    forward pass: {time.perf_counter() - t0:.1f}s "
          f"({pre.shape[0]} images, batch={batch})", flush=True)

    pk, vals = find_peaks_from_confmaps(cm, threshold=peak_threshold)
    peaks_full = postprocess_peaks(pk, vals)
    peaks_full = peaks_full.reshape(n_frames, n_cams, N_KEYPOINTS, 3)
    clean = preprocess_2d_behavior(peaks_full)
    pts3d = triangulate_session(clean[..., :2], calib).astype(np.float32)

    # Release the model + TF graph before the next one loads.
    del model, cm_chunks, cm
    gc.collect()
    return pts3d


# ---------------------------------------------------------------------------
# DANNCE -> SLEAP space
# ---------------------------------------------------------------------------
def load_dannce_in_sleap_space(rat: str, session: str,
                               sleap_ref_3d: np.ndarray,
                               cam0_frames: np.ndarray,
                               alignment_n_frames: int = 1000):
    """Load DANNCE for the window and bring it into SLEAP world space.

    Returns (dannce_in_sleap, dannce_in_dannce, alignment) where:
      dannce_in_sleap   : (T, 23, 3) DANNCE projected back to SLEAP space
                          (for overlay on the SLEAP camera)
      dannce_in_dannce  : (T, 23, 3) DANNCE in its native space
                          (for residual stats against aligned SLEAP)
      alignment         : the Procrustes dict (apply / apply_inverse)

    sleap_ref_3d is the OLD-model full-session SLEAP 3D used to fit the
    Procrustes (matches qc_utils.generate_qc_video, which fits on saved SLEAP).
    cam0_frames are the cam0 video-frame indices for the rendered window;
    DANNCE is keyed by cam0_frame in the resampled stream.
    """
    keys = load_sleap_dannce_keys(rat, session)
    aligned = load_aligned_data(rat, session)

    dannce_3d = keys["dannce_keys_3D"]
    if dannce_3d.ndim == 4:
        dannce_3d = dannce_3d.squeeze(axis=1).transpose(0, 2, 1)
    else:
        dannce_3d = dannce_3d.transpose(0, 2, 1)
    dannce_3d = median_filter(dannce_3d.astype(np.float32),
                              size=(DANNCE_MEDFILT, 1, 1))

    sl_full = median_filter(keys["sleap_keys_3D"].astype(np.float32),
                            size=(SLEAP_MEDFILT, 1, 1))
    aligned_idx = aligned["dannce_idx_for_sleap_cams"].astype(int).ravel()

    # DANNCE resampled onto the SLEAP timeline (one row per SLEAP frame).
    di = np.clip(aligned_idx[: len(sl_full)], 0, len(dannce_3d) - 1)
    dannce_at_sleap = dannce_3d[di]  # (n_sleap, 23, 3)

    # Fit Procrustes on full-session paired (SLEAP_old, DANNCE).
    n = min(len(sl_full), len(dannce_at_sleap))
    alignment = find_sleap_dannce_alignment(
        sl_full[:n], dannce_at_sleap[:n],
        n_sample_frames=min(alignment_n_frames, n), try_z_flip=True)
    print(f"  Procrustes residual={alignment['residual']:.2f} mm  "
          f"z_flipped={alignment['z_flipped']}", flush=True)

    # Window selection: dannce_at_sleap is keyed by SLEAP frame == cam0_frame.
    win = np.clip(cam0_frames, 0, len(dannce_at_sleap) - 1)
    dannce_in_dannce = dannce_at_sleap[win]
    dannce_in_sleap = alignment["apply_inverse"](dannce_in_dannce).astype(
        np.float32)
    return dannce_in_sleap, dannce_in_dannce, alignment


# ---------------------------------------------------------------------------
# Residual reporting
# ---------------------------------------------------------------------------
def report_residuals(new_sleap_3d, old_sleap_3d, dannce_in_dannce, alignment):
    """Per-keypoint median 3D residual (mm) of each SLEAP model vs DANNCE,
    computed in DANNCE space (SLEAP brought in via alignment['apply'])."""
    from config import NODES
    new_in_d = alignment["apply"](new_sleap_3d)
    old_in_d = alignment["apply"](old_sleap_3d)
    d_new = np.linalg.norm(new_in_d - dannce_in_dannce, axis=-1)  # (T, 23)
    d_old = np.linalg.norm(old_in_d - dannce_in_dannce, axis=-1)

    print("\n=== 3D residual vs DANNCE (mm), median over window ===")
    print(f"  {'keypoint':<12} {'old':>8} {'new':>8} {'Δ(new-old)':>12}")
    new_med = np.median(d_new, axis=0)
    old_med = np.median(d_old, axis=0)
    for j, name in enumerate(NODES):
        delta = new_med[j] - old_med[j]
        flag = "  better" if delta < -0.5 else ("  worse" if delta > 0.5 else "")
        print(f"  {name:<12} {old_med[j]:8.2f} {new_med[j]:8.2f} "
              f"{delta:12.2f}{flag}")
    print(f"  {'-' * 44}")
    print(f"  {'OVERALL':<12} {np.median(d_old):8.2f} {np.median(d_new):8.2f} "
          f"{np.median(d_new) - np.median(d_old):12.2f}")
    improved = (new_med < old_med).sum()
    print(f"  keypoints improved by new model: {improved}/{len(NODES)}")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render(new_model: str, old_model: str, rat: str, session: str,
           start_frame: int, n_frames: int, camera: int, fps: int = 20,
           peak_threshold: float = 0.01, out_subdir_name: str | None = None,
           batch: int = 30):
    # --- Resolve processed-frame window -> per-camera video frames ---
    fm = load_frame_mapping(rat, session)
    proc = [p for p in range(start_frame, start_frame + n_frames)
            if p in fm.index]
    if len(proc) < n_frames:
        print(f"  note: {n_frames - len(proc)} processed frames missing from "
              f"frame_mapping; rendering {len(proc)}", flush=True)
    per_cam = np.stack([
        fm.loc[p, [f"cam{ci}_frame" for ci in range(3)]].values for p in proc
    ]).astype(int)
    n_frames = len(proc)
    cam0_frames = per_cam[:, 0]
    n_cams = per_cam.shape[1]

    # --- Read + preprocess frames ONCE (shared across both models) ---
    print(f"Reading {n_frames} frames x {n_cams} cams ...", flush=True)
    frames = read_session_frames_per_cam(rat, session, per_cam)
    pre = preprocess_for_sleap(frames)

    calib = build_calibration_for_session(rat, session)
    print(f"  calibration date: {calib['cal_date']}", flush=True)

    # --- Inference: new then old (one model resident at a time) ---
    print("Running NEW model ...", flush=True)
    new_3d = run_sleap_inference(new_model, pre, n_frames, n_cams, calib,
                                 peak_threshold, batch=batch)
    print("Running OLD model ...", flush=True)
    old_3d = run_sleap_inference(old_model, pre, n_frames, n_cams, calib,
                                 peak_threshold, batch=batch)

    # Visual-parity median filter (matches QC/render conventions).
    new_3d_s = median_filter(new_3d, size=(SLEAP_MEDFILT, 1, 1))
    old_3d_s = median_filter(old_3d, size=(SLEAP_MEDFILT, 1, 1))

    # --- DANNCE into SLEAP space ---
    print("Aligning DANNCE ...", flush=True)
    dannce_in_sleap, dannce_in_dannce, alignment = load_dannce_in_sleap_space(
        rat, session, sleap_ref_3d=old_3d_s, cam0_frames=cam0_frames)

    # --- Residual report (unsmoothed for honest numbers) ---
    report_residuals(new_3d, old_3d, dannce_in_dannce, alignment)

    # --- Project all three onto the chosen SLEAP camera ---
    cal_folder = calibration_path(rat, session)
    print(f"\nProjecting to Camera{camera} ...", flush=True)
    new_2d = project_3d_to_2d_for_camera(new_3d_s, cal_folder, camera_idx=camera)
    old_2d = project_3d_to_2d_for_camera(old_3d_s, cal_folder, camera_idx=camera)
    dannce_2d = project_3d_to_2d_for_camera(dannce_in_sleap, cal_folder,
                                            camera_idx=camera)

    # --- Set up writer ---
    cam_name = f"Camera{camera}"
    video_file = Path(sleap_path(rat, session)) / cam_name / "0.mp4"
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_file}")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_tag = Path(new_model).stem
    old_tag = Path(old_model).stem
    sub_tag = out_subdir_name or f"{new_tag}_vs_{old_tag}"
    out_subdir = OUT_DIR / sub_tag
    out_subdir.mkdir(parents=True, exist_ok=True)
    last = cam0_frames[-1]
    out_path = (out_subdir /
                f"{rat}_{session}_cam{camera}_f{cam0_frames[0]}-{last}.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (vid_w * 3, vid_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {out_path}")

    CYAN = (255, 255, 0); MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255); YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)

    print(f"Rendering {out_path.name} ...", flush=True)
    t0 = time.time()
    for fi in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cam0_frames[fi]))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)

        p_new = frame.copy(); p_old = frame.copy(); p_dan = frame.copy()
        draw_skel(p_new, new_2d[fi], GREEN)
        draw_skel(p_old, old_2d[fi], CYAN)
        draw_skel(p_dan, dannce_2d[fi], MAGENTA)

        cv2.putText(p_new, "NEW SLEAP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, GREEN, 2, cv2.LINE_AA)
        cv2.putText(p_old, "OLD SLEAP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, CYAN, 2, cv2.LINE_AA)
        cv2.putText(p_dan, "DANNCE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(p_new, f"{rat}/{session}  Camera{camera}",
                    (10, vid_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2,
                    cv2.LINE_AA)
        cv2.putText(p_dan, f"frame {int(cam0_frames[fi])}", (10, vid_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

        writer.write(np.hstack([p_new, p_old, p_dan]))
        if (fi + 1) % 200 == 0 or fi == n_frames - 1:
            print(f"  {fi + 1}/{n_frames} ({time.time() - t0:.1f}s)", flush=True)

    writer.release(); cap.release()
    print(f"saved {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_model", required=True,
                    help="path to the fine-tuned SLEAP SavedModel (.og dir)")
    ap.add_argument("--old_model", default=DEFAULT_OLD_MODEL,
                    help="baseline SLEAP SavedModel (.og dir)")
    ap.add_argument("--rat", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--start_frame", type=int, default=0,
                    help="starting processed_frame index")
    ap.add_argument("--n_frames", type=int, default=1000)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--peak_threshold", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=30,
                    help="images per SLEAP forward pass (lower if GPU OOMs)")
    ap.add_argument("--out_subdir", default=None,
                    help="override video subdir name "
                         "(default: <new>_vs_<old>)")
    args = ap.parse_args()
    render(args.new_model, args.old_model, args.rat, args.session,
           args.start_frame, args.n_frames, args.camera, args.fps,
           args.peak_threshold, args.out_subdir, args.batch)


if __name__ == "__main__":
    main()
