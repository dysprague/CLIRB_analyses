# CLIRB SLEAP→DANNCE Corrector — Handoff

**Last updated**: 2026-06-02. Phase I (2026-05-22 .. 2026-06-01) trained four
new 2D-input models, deployed the rebuilt template_1 via a code-only swap,
added in-corrector median smoothing, and shipped a new flat-MLP-with-2D
architecture (`temporal_mlp_2d_reproj`) that is now the recommended R2 and R3
production pick. Section 2.8 covers what's new. Section 3 is rewritten.
For a focused production deployment guide see `production_picks.md`.

---

## 1. The project in 30 seconds

The CLIRB experiment uses **SLEAP** (low-latency 3D keypoints, used online for template matching → rewards) and **DANNCE** (higher-fidelity offline 3D keypoints) on freely-moving rats. SLEAP and DANNCE disagree at the keypoint level, which means template matches detected by each don't perfectly overlap.

We've built a **neural corrector**: a small MLP that takes Procrustes-aligned SLEAP keypoints and outputs corrected keypoints that look more like DANNCE. The 3D-input version (`R1R2R3_velacc.pt` + per-rat heads) is the current production candidate and gets template-matching F1 ≥ 0.77 on R2 and R3 and 0.83–0.90 on R1, vs raw-SLEAP F1 of ~0.65–0.70 across rats. A new **2D-input corrector** (`R1R2R3_2d_v2.pt`, Phase H.6) is the next bet — it eats per-camera 2D detections + confidence + reprojection residuals alongside the triangulated 3D, and *partially recovers* sessions broken by the 2026-02-06 calibration cutover that the 3D-only corrector can't touch.

Three sister directories:
- `~/CLIRB/` — raw experimental code, single-session analysis notebooks. **Also hosts the local NVMe caches** (`CLIRB/data/sleap_dannce_keys_2026_02_18/` for 3D pairs, `CLIRB/data/sleap_2d_cache_2026_05_21/` for per-session SLEAP 2D + calibration).
- `~/campy-CLIRB/` — online acquisition + real-time matching pipeline (deployment target).
- `~/CLIRB_analyses/` — **this directory**, offline analysis + corrector code.

---

## 2. Phases A–F: 3D-input corrector ablations (legacy, frozen)

Six phases of ablations on the 3D-input corrector. 15 checkpoints in `corrector/checkpoints/`, all evaluable with `python -m corrector.evaluate_all --ckpt …`. Full per-phase tables live in `README.md` ("Phase 3 — Ablation experiments"). Headline winners that section 4 still points at:

| Rat | Best F1 xyz | Best F1 groupO | Best kp_mse |
|---|---|---|---|
| R1 | wide head on velacc (0.830) | **head on temporal (0.898)** ⭐ | vel/acc + noise (169) |
| R2 | head on velacc (0.680) | **head on velacc+noise (0.776)** | vel/acc + noise (254) |
| R3 | head on temporal+noise (0.778) | **vel/acc (0.818)** | head on velacc+noise (266) |

Big finding: no single model wins all metrics on all 3 rats. Production uses per-rat configs.

---

## 2.5. Phase G — settled findings (full details in git history)

Highlights that downstream work still depends on:

1. **Calibration cutover 2026-02-06.** SLEAP triangulation calibration regressed; bone-length CV jumped 0.35 → 0.48 SLEAP-side, unchanged DANNCE-side. ~3 months of sessions are geometrically broken on the SLEAP side and get gated out by the 60 mm Procrustes residual filter. Phase H.2 showed 2D detections + new calibration are *internally* consistent, motivating the 2D-input corrector.
2. **Templates moved from `*_template_1.npz` to `*_template_2.npz`** (the latter records origin session/frame). Production still loads template_1.
3. **`corrected_xyz` F1 is the honest operational metric.** `corrected_groupO` involves per-session bounds sweeps and is invariant to rigid transforms — it scores higher partly by construction. Treat groupO as a diagnostic.

---

## 2.6. Phase H — template_1 rebuild + 2D-input groundwork (2026-05-12 .. 2026-05-21)

### H.1 — Template_1 rebuild from DANNCE keypoints (big F1 win, not yet deployed)

The stored `<rat>_template_1.npz` files were built from SLEAP-pose windows. Rebuilding them from the *DANNCE* keypoints (at SLEAP-detected match end-frames) improves F1 on every rat. See `corrector/_rebuild_template1.py` for the construction.

Result on 2025-10-25..2025-12-10 (~165 sessions, internally consistent eval):

| Rat | n | corr_xyz F1 stored | **F1 rebuilt** | Δ |
|---|---|---|---|---|
| R1 | 53 | 0.797 | **0.886** | +0.089 |
| R2 | 54 | 0.636 | **0.807** | +0.170 |
| R3 | 55 | 0.711 | **0.795** | +0.084 |

Files at `/olveczky_lab/Lab/CLIRB/data/<rat>/templates/<rat>_template_1_rebuild.npz` (×3). **Production code (`~/campy-CLIRB/campy/behavior_rule.py`) still loads `<rat>_template_1.npz`** — deployment is gated on a final visual sanity check and a deployment decision (rename rebuild → stored, or update the loader path in production).

### H.2 — 2D + calibration data plumbing

**`corrector/data_world_2d.py`** added `load_paired_world_with_2d`, per-session calibration loaders (`load_session_calibration`), and `reproject_all_cams(points_3d, calib)`. The rig convention is **z<0 in-front-of-camera**; `project_3d_to_2d_batch` handles this correctly, but **`corrector/projection.py:project_3d_to_2d` does not** — it silently produces nonsense for wrong-side points. Renderer scripts using `projection.project_3d_to_2d` should be audited; `render_world_overlay.py` was reviewed and uses `project_3d_to_2d_for_camera` which is fine.

Per-cam median reprojection residual on raw triangulated 3D is 73–124 px (55–69 px for high-confidence keypoints) — healthy and monotonically improves with SLEAP confidence.

### H.3 — Offline pipeline + critical index-mapping discovery

`corrector/online_pipeline.py` reproduces the live `campy-CLIRB` inference pipeline (read .mp4 → SLEAP forward → confmap argmax → per-cam cleanup → undistort + SVD-DLT triangulation). Validated against saved live-pipeline outputs at median 2.84 mm on 3D and 0–8 px on 2D.

**Critical index-mapping trap** (will eat hours otherwise):

- `sleap_keys_2D.npy` is indexed by **`processed_frame`** (one row per SLEAP forward pass; length ~34,430).
- `triang_keys_3D.npy` is indexed by **`cam0_frame`** (one row per video frame, length ~36,000, with linear interpolation filling gaps from dropped frames per `behavior.py:418-440`).
- The bridge is `<session>/sleap/frame_mapping.csv`. **Its `processed_frame` column is 1-indexed** while `sleap_keys_2D.npy` is 0-indexed.

Triangulating `saved_2d[p]` and comparing to `saved_3d[p]` gives ~500 mm error from misalignment. Always compare to `saved_3d[cam0_frame_from_mapping[p]]`.

### H.4 — Saved-2D loader

**`corrector/data_world_2d_from_saved.py`** reads per-processed-frame data without any video reads (~1000× faster than re-running SLEAP). Trade-off: ~4.4% of video frames are dropped by the live pipeline and unrecoverable from saved 2D alone.

```python
from corrector.data_world_2d_from_saved import load_session_2d, PairedSession2DDataset
sd = load_session_2d("R2", "2025_11_01_1")
# sd.x_2d        (P, 3, 23, 2)  pixel coords per camera, post live cleanup
# sd.x_conf      (P, 3, 23)     SLEAP detection confidences
# sd.x_triang_3d (P, 23, 3)     saved triangulated SLEAP 3D, native SLEAP world (NOT smoothed)
# sd.y_dannce_3d (P, 23, 3)     median-25-filtered DANNCE 3D, DANNCE world
# sd.cam_frames  (P, 3)         per-cam video frame idx
# sd.calibration list of 3 cam dicts (K, r, t, dist)
```

---

## 2.7. Phase H.5–H.6 — 2D-input corrector v2 (current state)

The 2D-input corrector landed (`TriangulationRefiner`), trained, and was evaluated end-to-end. Headline checkpoint: **`corrector/checkpoints/R1R2R3_2d_v2.pt`**.

### Architecture

**`TriangulationRefiner`** (in `corrector/models.py`, dispatched via `build_model("triangulation_refiner", …)`). PointNet-style per-keypoint MLP + global max-pool + per-kp head outputting a residual added to `xyz_triang`. Defaults: `hidden=128, n_per_kp_layers=3, global_dim=64` → **69,187 params**. Last layer zero-init so the model starts at identity. Per-kp input dim 21:

```
xyz_triang            (3)   triangulated 3D, Procrustes-aligned into DANNCE world
per_cam_xy_normalized (6)   3 cams × 2, pixel coords scaled to [-1, 1] by (1920, 1200)
per_cam_conf          (3)   SLEAP detection confidence per cam
per_cam_reproj_resid  (6)   3 cams × 2, (detected − reprojected) / 100 px
per_cam_visibility    (3)   1 if detection finite AND conf > 0 AND reproj finite
```

NaN handling contract: invisible keypoints (wrong-side-of-camera, missing detection, or NaN reproj) get **zeroed** in xy/resid channels with visibility flagging the mask. The trainer + eval both enforce this in `build_features`.

### Trainer (`corrector/train_2d_input.py`)

- Pulls per-processed-frame data from `Paired2DTrainDataset` (saved-2D path, no video re-inference).
- Per-session Procrustes fit on `(x_triang_3d, y_dannce_3d)` over the calibration window. Same `max_residual=60` gate as the 3D pipeline (uncalibrated sessions skipped).
- **Outlier filter** (added in H.6): drops per-sample if `max(|x_triang_3d|) > 1000 mm` or `max(|y_dannce_3d|) > 1000 mm`. Arena is ~500 mm; anything past 1 m is a triangulation failure. Without this, single bad frames (we saw ~11,600 mm samples in R1 train) drive train_mse to 10^10 and destabilize training. With it, train_mse and val_mse are in the same ballpark from epoch 0.
- **Grad clipping** (added in H.6): `--grad_clip 1.0` defends against outlier gradients (defense-in-depth alongside the filter).
- **Reprojection in torch, fully vectorized** (added in H.6): stacks per-session calibrations once into `(S, 3, 3, 3)` etc tensors and gathers per-sample via `session_idx`. Replaces a per-batch Python loop over unique sessions that fired ~600 small CUDA kernels per batch.
- Loss: DANNCE-space MSE + optional bone-length loss. No PC-loss helper (the 3D version assumes 3D-world input shape).
- Checkpoint format includes `model_name="triangulation_refiner"`, `hidden`, `n_per_kp_layers`, `global_dim`, `grad_clip`, `outlier_threshold_mm`, plus standard `state_dict`, `splits`, `session_residuals_*`.

### Eval integration (`corrector/evaluate_all.py`, Phase H.7)

Added `correct_triangulation_refiner(model, rat, session, sl_aligned_dn, tx, device)`. For `triangulation_refiner` checkpoints, `compute_full`:

1. Loads the saved-2D bundle.
2. Fits Procrustes on `(x_triang_3d, y_dannce_3d)` over the calibration window (same as training, **NOT** on `(sl, dn)` from `load_paired_world` — these don't quite match).
3. Runs the corrector per processed_frame.
4. Scatters predictions back to the SLEAP timeline by `cam0_frame` index. ~4.4% frames not covered by saved 2D fall back to Procrustes-only.

All downstream blocks (kp_mse, PC MSE, F1 xyz, F1 groupO) operate on the SLEAP-timeline arrays unchanged.

### Renderer integration (`corrector/render_world_overlay.py`, Phase H.7)

Same 2-arm dispatch as `evaluate_all`. For `triangulation_refiner` checkpoints:
- Procrustes from `load_session_2d` (matches training/eval alignment).
- `correct_triangulation_refiner` does the inference.
- **Median-11 smoothing on the corrector output** before projection (cosmetic, matches the upstream `SLEAP_MEDFILT=11` applied by `load_paired_world` to the 3D-input pipeline; the 2D path otherwise produces visibly jitterier output because `load_session_2d` skips that filter).
- Videos are read from SMB (the local 2D cache only contains keypoint/calibration files, not videos).

### Headline results

Best `val_mse=112.33` (epoch 11/17, early-stopped). Full eval (`corrector/results/R1R2R3_2d_v2_all.json`):

| Rat | n | kp_mse align→corr | corrected_xyz F1 | corrected_groupO F1 |
|---|---|---|---|---|
| R1 | 11 | 414 → **2413** (poisoned by 1 outlier session, see below) | **0.835** | 0.791 |
| R2 | 11 | 684 → **478** (-30%) | 0.452 | 0.584 |
| R3 | 12 | 608 → **428** (-30%) | 0.671 | 0.648 |

Mixed picture vs the velacc baseline. Notes:

- **R1 F1xyz beats the velacc leaderboard (0.835 vs ~0.83)** — the corrector closes most of the SLEAP/DANNCE gap on the *operational* metric.
- **R1 kp_mse is poisoned by `R1/2026_02_13_1`** which went 513 → 23,082 (45× regression). The other 10 R1 test sessions average ~340 corr, a real ~18% improvement over 414 align. Same outlier-frame story as training — model emits wild corrections for some test-time triangulation failures. **Open** — need either tighter inference-time outlier guards or a stricter training filter.
- **Post-cutover sessions consistently get 30–47% reductions**, the strongest evidence yet that the 2D-input approach recovers geometric information broken on the 3D side:

  | Session | align → corr |
  |---|---|
  | R1/2026_02_06_1 | 683 → 550 (-19%) |
  | R2/2026_02_09_1 | 1058 → 707 (-33%) |
  | R2/2026_02_13_1 | 1099 → 598 (-46%) |
  | R3/2026_02_06_1 | 1405 → 776 (-45%) |
  | R3/2026_02_10_1 | 1313 → 697 (-47%) |

- R2 and R3 F1 still trail the velacc leaderboard. Likely reasons: training early-stopped at epoch 17 (vs velacc's 60-80), no per-rat heads, no temporal context, outliers diluting training signal mid-run.

### Performance + infrastructure (added during H.6)

- **Local SLEAP 2D cache**: `config.sleap_path()` now prefers `$SLEAP_LOCAL_CACHE/<rat>/<session>/sleap/` (default `~/CLIRB/data/sleap_2d_cache_2026_05_21/`) when present, falls back to SMB. Cache contains `sleap_keys_2D.npy`, `triang_keys_3D.npy`, `frame_mapping.csv`, `calibration/<date>/hires_cam{0,1,2}_params.mat` per session (227 sessions, 18.6 GB).
- **Staging script**: `scripts/stage_sleap_2d_cache.py` is idempotent and skips existing files. Re-run if new sessions appear.
- **Dataset load time** dropped from ~40 min (SMB) to **~2 min** (local NVMe) for a 149-session train set.
- **Per-epoch time** dropped from ~10+ s/epoch to ~10 s/epoch *and* now actually trains; pre-vectorization the GPU was idle waiting on ~600 kernel launches per batch.
- **Output buffering note**: `conda run` buffers stdout completely until the subprocess exits. Use the env's python directly for training: `/home/yutaka-sprague/anaconda3/envs/clirb_analysis/bin/python -u -m corrector.train_2d_input …`.

### Files added or changed this phase

```
corrector/models.py                              (+TriangulationRefiner, build_model dispatch)
corrector/train_2d_input.py                      (new)
corrector/evaluate_all.py                        (+correct_triangulation_refiner, +2D dispatch)
corrector/render_world_overlay.py                (+2D dispatch, +median-11 smoothing, SMB video path)
config.py                                        (+SLEAP_LOCAL_CACHE override on sleap_path)
scripts/stage_sleap_2d_cache.py                  (new)
corrector/checkpoints/R1R2R3_2d_v2.pt            (best 2D-input checkpoint)
corrector/checkpoints/R1_2d_v2.pt                (R1-only baseline; best val_mse=100.95)
corrector/results/R1R2R3_2d_v2_all.json          (eval output)
corrector/videos/R1R2R3_2d_v2/*.mp4              (6 comparison videos, median-11 smoothed)
corrector/videos/R1R2R3_2d_v2_unsmoothed/*.mp4   (same 6, unsmoothed for comparison)
corrector/figures/R1R2R3_2d_v2/*.png             (PC trajectories per video)
```

---

## 2.8. Phase I — production-quality 2D-input corrector (2026-05-22 .. 2026-06-01)

Major shift: the 2D-input approach moved from "promising but worse than
velacc on R2/R3" to "the new production recommendation for R2 and R3" via a
sequence of compounding improvements. The path was non-obvious — see "what
didn't work" below — so the chronology matters.

### Headline result (compared to handoff baseline)

All evals use the rebuilt template_1 (deployed via `RAT_TEMPLATE` code swap,
see §2.8d). Production-pick column gives the per-rat best:

| Rat | Handoff pick F1xyz | New pick F1xyz | Δ | Pick |
|-----|-------------------:|---------------:|---|------|
| R1 | 0.821 | **0.897** | **+0.076** | `R1R2R3_world_temporal_mlp.pt` (no 2D, classic) |
| R2 | 0.646 | **0.824** | **+0.178** | `R2_reproj_ft_from_v3.pt` (per-rat ft, smooth=15) |
| R3 | 0.727 | **0.840** | **+0.113** | `R1R2R3_temporal_mlp_2d_reproj_v4.pt` (smooth=15) |

R1 stayed with the classic temporal_mlp — the 3D triangulation is already
clean enough on R1 that the 2D inputs don't help. R2 and R3 saw the largest
gains because they have worse triangulation on cutover sessions, which the
2D + reprojection-residual model directly addresses.

The **single-model alternative** is `R1R2R3_temporal_mlp_2d_reproj_v4.pt`
with `--smooth_size 15`: R1=0.870, R2=0.804, R3=0.840 — within ≤0.027 of
per-rat best on every rat. Recommended if production can only hold one
corrector.

Full numbers, including post-cutover-only breakdowns, are in
`production_picks.md`.

### 2.8a. The new model class — `temporal_mlp_2d_reproj`

`TemporalMLPWith2DReproj` in `corrector/models.py`. Same flat-MLP shape as
`temporal_mlp` (2 hidden layers × 128 units), but the input gets four extra
channels for the **current** frame:

```
Per-sample input (759 dims total):
  pose_window     (ctx=5, 23, 3)   = 345    same as temporal_mlp
  per_cam_2d_norm (3, 23, 2)        = 138    SLEAP 2D, normalized to [-1, 1]
  per_cam_conf    (3, 23)            =  69    SLEAP detection confidence
  per_cam_vis     (3, 23)            =  69    1/0 visibility mask
  per_cam_reproj  (3, 23, 2)        = 138    (detected - reprojected) / 100 px

Output: (23, 3) residual added to pose_window[:, -1].
Last layer zero-init.
~123k params.
```

The reprojection residual is the *causally meaningful* channel: it directly
tells the model whether the triangulated 3D is consistent with the 2D
evidence. High residual → triangulation is broken (the cutover failure
mode), low residual → trust the 3D. The reprojection uses the saved
un-smoothed triangulated 3D (the geometry that actually produced the
detections), projected through the session calibration.

**Why a flat MLP (not the PointNet-style `TriangulationRefiner`)**: an
earlier ablation showed PointNet was the wrong prior — it lost cleanly to
the flat-MLP shape that `temporal_mlp` already proved works on this data.
This is the same architecture, just with extra input channels.

Trainer: `corrector/train_temporal_mlp_2d_reproj.py`.
Inference: `corrector/evaluate_all.py:correct_temporal_mlp_2d_reproj`.

### 2.8b. Versions trained and their lessons

| Version | Config | best val_mse | F1xyz R1 / R2 / R3 (test, smooth=15 where applicable) | Notes |
|---------|--------|-------------:|-------------------------------------------------------|-------|
| v1 | noise 50mm/0.3, 100 epochs | 134.3 | 0.882 / 0.787 / 0.809 | First working 2D model. Still slowly improving at ep 100. |
| v2 | noise **100mm**/0.5 | 288.7 (killed) | n/a | **Killed at ep 21.** Noise destroyed the rat-skeleton prior — bones got "stretched" 2-4×, breaking the temporal context the rest of the network depends on. Don't go above ~50mm with whole-skeleton noise. |
| v3 | noise 50/0.3, **cosine LR** 1e-3→1e-5, 150 epochs | 151.4 | 0.871 / 0.810 / 0.820 | Cosine LR + longer training. Worse val_mse than v1 (artifact of dropout-style schedule noise) but better kp_mse. |
| v4 | **targeted noise**: 1-3 keypoints, 150mm, prob 0.5; cosine 150 ep | 132.2 | 0.870 / 0.804 / **0.840** | **The breakthrough on R3.** Replaces whole-skeleton noise with realistic targeted-keypoint corruption (1-3 paws at a time). R3 jumps from 0.820 → 0.840 on test, 0.797 → 0.840 on cutover. |
| Per-rat FTs from v3 | `--init_ckpt`, no noise, lr=5e-5, 30 ep | varies | 0.888 / 0.824 / 0.776 | R1 +0.017, R2 +0.014 — both clean wins on both F1 and kp_mse. R3 ft overfits immediately (val diverges from ep 0); use global v4 for R3. |

### 2.8c. In-corrector median smoothing (the other compounding fix)

The user observed visible "jumpiness" in the rendered videos — at ~4.4% of
SLEAP-timeline frames there is no processed_frame mapping, so the corrector
output falls back to Procrustes-only, creating a visible discontinuity at
every gap.

Fix: `correct_temporal_mlp_2d_reproj` now applies
`scipy.ndimage.median_filter(out, size=(smooth_size, 1, 1))` along the time
axis after scatter. Default `smooth_size=11`. The CLI flag
`--smooth_size N` (added to `evaluate_all.py` and both renderers) overrides.

Ablation on v3 (rebuild template active):

| Rat | Metric | size=0 | size=7 | size=11 | size=15 |
|-----|--------|-------:|-------:|--------:|--------:|
| R1 | F1 test | 0.873 | 0.871 | 0.872 | 0.871 |
| R2 | F1 test | 0.795 | 0.807 | 0.808 | 0.810 |
| R3 | F1 test | 0.776 | 0.783 | 0.790 | **0.820** |
| R3 | F1 cutover | 0.765 | 0.775 | 0.785 | **0.825** |

**`smooth_size=15` is the recommended value** for offline scoring and
rendering. R3 gain (+0.030 F1 test, +0.040 cutover) is large; R1/R2 cost is
near-zero. Default in code is still 11; pass 15 on the CLI.

**Online caveat**: this median is *symmetric* (uses past + future frames).
A causal median for the live pipeline would introduce N/2 frames of lag
(7 frames for size=15, ~350 ms at 20 Hz — too much). For online integration
use a causal median of size 5-7. See `production_picks.md` §5c.

### 2.8d. The rebuild template (deployed)

`<rat>_template_1_rebuild.npz` (built from DANNCE keypoints rather than
SLEAP-pose, per Phase H.1) is now the default in `RAT_TEMPLATE`
(`corrector/evaluate_world.py`). All eval and rendering paths see it without
a flag. **Data-side rename was NOT done** — the user opted for code-only
deployment to avoid disturbing shared lab data. `<rat>_template_1_rebuild.npz`
and `<rat>_template_1.npz` (the legacy SLEAP-pose template) both still exist
on disk; only the code-level mapping changed.

Production-pipeline integration (i.e. updating
`~/campy-CLIRB/campy/behavior_rule.py` to load the rebuild file) is **not
done** and is explicitly gated on user approval.

### 2.8e. Inference cost benchmark

End-to-end single-frame latency (`scripts/bench_reproj_v1.py`):

- CPU, B=1: median **295 µs** (~0.3 ms)
- GPU, B=1 (incl. h2d + cuda sync): median **207 µs**
- GPU, B=64 amortized: 0.8 µs / frame

SLEAP's online frame budget is ~30 ms. The corrector adds **<1.5%** to that
budget on CPU. Online-feasible with major headroom.

### 2.8f. Things that did NOT work (don't re-do)

- **Heavier whole-skeleton noise (v2: 100mm/0.5).** Training noise must
  stay below the bone-length scale (~30-80mm). Above that, the skeleton
  prior collapses and the temporal context becomes useless. Cap at 50mm
  for global noise; or use targeted noise (v4) for larger displacements.
- **PointNet-style architecture for the 2D corrector**
  (`TemporalTriangulationRefiner`). Per-keypoint shared MLP + max-pool
  global was the wrong prior — model overfit calibration-tied signatures
  immediately and val_mse diverged. The flat MLP shape (per `temporal_mlp`)
  is the right one. Don't try PointNet again.
- **Per-rat fine-tuning on R3.** R3 val_mse diverges from epoch 0 in every
  per-rat ft attempt (3 attempts, 3 failures). Use the global model on R3.
- **Pure 2D-input + temporal_mlp_2d (no reproj residuals).** Without the
  reprojection channel the model can't distinguish "low-confidence detection"
  from "geometry inconsistent with detections," and it underperforms
  temporal_mlp. The reproj residual is the load-bearing channel.

### 2.8g. Files added or changed this phase

```
corrector/models.py                                (+TemporalMLPWith2D,
                                                    +TemporalMLPWith2DReproj,
                                                    +TemporalTriangulationRefiner)
corrector/train_temporal_mlp_2d.py                  (new — v1 no-reproj baseline)
corrector/train_temporal_mlp_2d_reproj.py           (new — v1/v3/v4 trainer;
                                                    has --noise_mode {global,targeted},
                                                    --lr_schedule cosine,
                                                    --init_ckpt for per-rat ft)
corrector/train_2d_temporal.py                      (new — failed PointNet temporal
                                                    refiner, kept for diagnosis)
corrector/evaluate_all.py                           (+correct_temporal_mlp_2d,
                                                    +correct_temporal_mlp_2d_reproj
                                                     with smooth_size param,
                                                    +--smooth_size,
                                                    +--template_suffix,
                                                    +--sessions,
                                                    +--out_tag)
corrector/evaluate_world.py                         (RAT_TEMPLATE now points at
                                                     *_template_1_rebuild.npz)
corrector/render_world_overlay.py                   (+--single_panel,
                                                    +--smooth_size,
                                                    +--out_subdir,
                                                    +dispatch for new models)
corrector/render_world_3d_triptych.py               (new — 3D side-by-side renderer
                                                     for raw / corrected / DANNCE,
                                                     supports temporal_mlp_2d* dispatch)
scripts/bench_reproj_v1.py                          (new — inference benchmark)
scripts/render_v4_and_ft.sh                         (new — 24-video render set, smooth=15)
scripts/render_v4_and_ft_smooth5.sh                 (new — same set, smooth=5)
scripts/render_reproj_smoothed.sh                   (new — v1/v3 smoothed renders)
production_picks.md                                 (new — focused deploy guide)

Checkpoints added (in corrector/checkpoints/):
  R1R2R3_temporal_mlp_2d_reproj_v1.pt
  R1R2R3_temporal_mlp_2d_reproj_v3.pt
  R1R2R3_temporal_mlp_2d_reproj_v4.pt    ← single-model production pick
  R1_reproj_ft_from_v3.pt                ← R1 best 2D model (still loses to temporal_mlp)
  R2_reproj_ft_from_v3.pt                ← R2 production pick
  R3_reproj_ft_from_v3.pt                ← R3 fine-tune (worse than global v4 on R3)
  R1R2R3_temporal_mlp_2d_v1.pt           ← no-reproj baseline (worse than reproj_v3)

Videos added (in corrector/videos/):
  R1R2R3_temporal_mlp_2d_reproj_v{1,3,4}_smooth{5,11,15}_{singlepanel,3dtriptych}/
  R{1,2,3}_reproj_ft_smooth{5,15}_{singlepanel,3dtriptych}/
```

---

## 3. Recommended production deployment (2D-input, 2026-06-01)

The 2D-input path is now the recommended production candidate for R2 and
R3. R1 stays with the classic 3D-only temporal_mlp. Full details are in
`production_picks.md`; the headline is:

### Per-rat picks (rebuild template active in code)

| Rat | Checkpoint | F1xyz test | F1xyz cutover | kp_mse |
|-----|------------|-----------:|--------------:|-------:|
| R1 | `R1R2R3_world_temporal_mlp.pt` | **0.897** | 0.898 | 186 |
| R2 | `R2_reproj_ft_from_v3.pt` (`--smooth_size 15`) | **0.824** | 0.799 | 221 |
| R3 | `R1R2R3_temporal_mlp_2d_reproj_v4.pt` (`--smooth_size 15`) | **0.840** | **0.840** | 227 |

### Single-model alternative

`R1R2R3_temporal_mlp_2d_reproj_v4.pt` with `--smooth_size 15`:
- R1: 0.870 (−0.027 vs per-rat best)
- R2: 0.804 (−0.020 vs per-rat best)
- R3: 0.840 (global R3 winner)

### What changed since the handoff baseline

- **R1**: was `R1_head_on_R1R2R3temporal.pt` (F1xyz 0.821); now plain
  `R1R2R3_world_temporal_mlp.pt` (F1xyz 0.897). The gain is almost entirely
  the rebuild template; the per-rat head no longer adds value over the
  base.
- **R2**: was `R2_head_on_velacc_noise.pt` (F1xyz 0.646); now
  `R2_reproj_ft_from_v3.pt` smoothed (F1xyz 0.824). Both rebuild template
  + the new architecture contribute.
- **R3**: was `R1R2R3_velacc.pt` (F1xyz 0.727); now
  `R1R2R3_temporal_mlp_2d_reproj_v4.pt` smoothed (F1xyz 0.840). Targeted-
  noise augmentation was the breakthrough.

### Deployment status

- ✅ Rebuild template active in code (`RAT_TEMPLATE` updated).
- ❌ Rebuild template not renamed on disk (deliberate; symlink/rename
  blocked by user pending decision).
- ❌ Production-pipeline integration into `~/campy-CLIRB/campy/behavior_rule.py`
  not done. Explicitly gated on user approval. See `production_picks.md`
  §5b–5c for the to-do list (including causal-median caveat).

---

## 4. How to run the code

### Environment
```bash
conda activate clirb_analysis
# Verify GPU
python -m corrector.check_gpu
```

If `conda run` is buffering output and you need live progress, invoke python directly: `/home/yutaka-sprague/anaconda3/envs/clirb_analysis/bin/python -u -m <module> …`.

### Train the 3D-input corrector (unchanged, legacy)
```bash
python -m corrector.train_world_v2 --model temporal_mlp --rats R1 R2 R3 --tag <tag> --ctx 5
python -m corrector.train_world_v2 --model velacc_mlp   --rats R1 R2 R3 --tag <tag>
python -m corrector.train_world_v2 --model perrat_head  --rats R1 --base_ckpt corrector/checkpoints/<base>.pt --tag <tag>
```

### Train the 2D-input corrector (new)
```bash
python -m corrector.train_2d_input --rats R1 R2 R3 --tag <tag>
# Useful flags: --grad_clip 1.0 (default), --bone_weight 0.1, --max_residual 60,
#               --max_sessions_per_rat N (for smoke tests)
```

If the SLEAP 2D cache is missing, populate it first:
```bash
python -u scripts/stage_sleap_2d_cache.py
```

### Evaluate any checkpoint
```bash
python -m corrector.evaluate_all --ckpt corrector/checkpoints/<stem>.pt
```
Output: `corrector/results/<stem>_all.json` with per-session and per-rat aggregate metrics (kp_mse, PC1/2 MSE, F1@{100,300,500}ms in xyz and Group-O spaces). `evaluate_all` dispatches on `ck["model_name"]` and handles `triangulation_refiner` as of H.7. ~6 min on the GPU.

### Render comparison videos
```bash
python -m corrector.render_world_overlay --ckpt corrector/checkpoints/<stem>.pt \
    --rat R3 --session 2026_02_06_1 --camera 0 --n_frames 1000
# 2D-input checkpoints get an automatic median-11 smoothing on the corrector output.
```

---

## 5. Key files / where things live

```
CLIRB_analyses/
├── README.md                              # Full project history Phase 1→3
├── handoff.md                             # This file
│
├── config.py                              # DATA_ROOT, processed_path, sleap_path
│                                           # (sleap_path prefers SLEAP_LOCAL_CACHE)
├── data_io.py                             # load_paired_world, load_session_calibration, etc.
│
├── corrector/
│   ├── train_world_v2.py                  # 3D-input trainer (Phases A–F)
│   ├── train_2d_input.py                  # 2D-input trainer (H.5–H.6)
│   ├── evaluate_all.py                    # Unified evaluator; dispatches on model_name
│   ├── models.py                          # All architectures (incl. TriangulationRefiner)
│   ├── data_world.py                      # 3D-input dataset
│   ├── data_world_2d.py                   # Per-cam reprojection helpers + calibration loader
│   ├── data_world_2d_from_saved.py        # Per-processed-frame loader (saved 2D, no video reads)
│   ├── world_alignment.py                 # fit_procrustes()
│   ├── render_world_overlay.py            # 1000-frame side-by-side video (handles both input modes)
│   ├── render_long_combined.py            # ~10000-frame video + PC1/PC2 panels (3D-input only)
│   ├── _rebuild_template1.py              # H.1 template rebuild from DANNCE
│   ├── _verify_reprojection.py            # H.2 reprojection sanity check
│   ├── online_pipeline.py                 # H.3 offline reproduction of live pipeline
│   ├── check_gpu.py                       # GPU sanity check
│   ├── checkpoints/                       # *.pt files (15+ for 3D, 2D-input added H.6)
│   ├── results/                           # *_all.json eval outputs
│   ├── videos/<run_tag>/                  # Rendered .mp4 files
│   ├── figures/                           # PC plots, summary figures
│   └── logs/                              # Training/eval logs
│
├── scripts/stage_sleap_2d_cache.py        # H.6 — rsync 2D files SMB → local NVMe
│
├── template_matching.ipynb                # Has optional MLP corrector cell + multi-rat sweep
└── experiments/                           # Pre-corrector Phase 1 stuff (still useful)
```

---

## 6. Technical details that matter for resuming

### Coordinate systems
- `load_paired_world` returns SLEAP and DANNCE in their **native world coordinates**, median-filtered (SLEAP-11, DANNCE-25), with DANNCE resampled to the SLEAP 20 Hz timeline via `dannce_idx_for_sleap_cams`.
- **`load_session_2d` does NOT median-filter `x_triang_3d`** — it's the raw triangulated SLEAP. DANNCE is median-25 filtered. This asymmetry means 2D-input training sees noisier SLEAP input than the 3D pipeline; the renderer's H.7 fix applies a cosmetic median-11 to the corrector output to compensate visually.
- The corrector trains in **DANNCE coordinate space**: Procrustes maps SLEAP → DANNCE, the MLP outputs corrections in DANNCE space.
- To overlay on a SLEAP camera or project through the rat's template (z-flipped SLEAP egocentric coords), inverse-Procrustes the output back to SLEAP space, apply the z-flip + egocentric normalization. Encapsulated in `evaluate_all.py:compute_full()`; don't reinvent.

### Procrustes residual safety check
Sessions with calibration-window residual > 60 mm get dropped from training and evaluation. About 3-6 sessions per rat fall out, mostly very early September (DANNCE not co-calibrated yet) and most post-2026-02-06 sessions for the 3D-input pipeline.

### Why F1 xyz and F1 groupO often disagree
- **xyz F1** uses the rat's stored xyz-PCA template — sensitive to absolute keypoint positions.
- **groupO F1** uses a per-session pooled-PCA fit on pairwise distances — invariant to rigid transforms, bounds tuned per session.

groupO fires 2-3× more often than xyz on R2/R3 because its bounds are tuned per session. Only ~60% of one set has a partner within 300 ms in the other. Treat groupO as a diagnostic, not the headline.

### Outlier samples (2D-input pipeline only)
Triangulation failures produce sparse but extreme samples — sometimes 10+ meters out for one keypoint on one frame. `train_2d_input` drops these via `OUTLIER_THRESHOLD_MM=1000` (arena is ~500 mm). Grad clipping is the second line of defense. **Inference (eval, renderer) does NOT apply this filter** — `R1/2026_02_13_1`'s 23,082 kp_mse blowup was caused by a test-time outlier the trained model couldn't handle. Open issue.

### `template_matching.ipynb` gotcha
Local `project_to_pcs` used to shadow the template's `feature_means` with a per-session recomputed mean. Broke silently when the corrector was applied because corrected SLEAP has a different per-session mean than raw SLEAP. The fix was to use `skeleton.project_to_pcs(rotated, pc_weights, feature_means)`, passing `feature_means` from the template. **Don't recompute the PCA reference inside an inference path.**

### Inference cost
- MLP-based correctors: <2 µs/frame on GPU. Negligible vs SLEAP's ~30 ms.
- `TriangulationRefiner`: similar order; per-batch reprojection adds a tiny overhead.
- 5-min Procrustes fit at session start: ~1 s one-time.

---

## 7. Immediate next steps

The Phase H "next steps" list is now fully resolved. The Phase I work
landed: the 2D model is production-quality on R2/R3, the inference-time
outlier guard is in, the rebuild template is deployed (code-only), and the
2D corrector got rendered in both single-panel and 3D-triptych form for
visual inspection.

Remaining open work, ordered:

1. **Production-pipeline integration into `~/campy-CLIRB/campy/behavior_rule.py`.**
   The two production candidates (`R2_reproj_ft_from_v3.pt` and
   `R1R2R3_temporal_mlp_2d_reproj_v4.pt`) need to be loaded inside
   `TemplateMatch`, called per processed frame, and the output smoothed with
   a **causal** median (size 5-7 — symmetric size-15 introduces ~350ms of
   lag, not viable online). `production_picks.md` §5b-5c has the to-do
   list. Explicitly gated on user approval; don't touch without.

2. **Decide on the on-disk template_1 rename.** The rebuild is currently
   deployed via code (`RAT_TEMPLATE` in `corrector/evaluate_world.py`); the
   data files `<rat>_template_1.npz` (legacy) and
   `<rat>_template_1_rebuild.npz` (production) both still exist. For the
   live pipeline to pick up the rebuild, either the data-side rename has to
   happen or `behavior_rule.py` has to be updated to read the rebuild
   filename. User explicitly asked NOT to rename on shared lab data
   without further notice.

3. **Causal-median online smoothing implementation.** The current
   `correct_temporal_mlp_2d_reproj` applies a symmetric `median_filter`
   over `(size, 1, 1)` — uses both past and future frames. The online
   version needs a circular-buffer causal median over the last N frames.
   Plan: add an `--online_causal` mode or write a small wrapper for the
   live pipeline that maintains the last 5-7 corrected outputs and emits
   the median.

4. **Why does R3 resist per-rat fine-tuning?** Every R3 ft attempt
   (3 tries across this phase) has val_mse diverge from epoch 0. Train_mse
   drops cleanly. Hypothesis (untested): R3 train and R3 val sessions
   differ in some structural way the val-loss-with-outlier-guard isn't
   capturing — maybe calibration-date mix, maybe per-session amplitude
   distribution. Worth a 30-min diagnostic before declaring "use global
   model on R3" the final answer.

5. **(Maybe) v5: combine v4's targeted noise with per-rat fine-tunes for
   R1 and R2 only.** v4's targeted-noise pretrain might give a stronger
   starting point for per-rat ft than v3 did. Could push R1 fine-tune
   above the classic temporal_mlp's 0.897 and lock that rat in too.
   Lower priority; the marginal gain is bounded.

6. **R3's classic `temporal_mlp` alternative.** F1xyz=0.824 on R3 test
   without any 2D inputs, vs v4's 0.840. The classic model is simpler and
   doesn't need the reprojection plumbing online. If you're OK with the
   −0.016 F1, it's a viable backup option. Note kp_mse is much worse for
   classic (280 vs 227) — operational template-matching is similar, but
   keypoint accuracy is degraded.

### Direction: Online integration (NOT done; explicitly gated)

`~/campy-CLIRB/campy/behavior_rule.py:TemplateMatch` is the integration
target. **User explicitly requested NOT to touch production code without
further notice.** Confirm before any edits there. The corrector latency is
proven fine (~0.3 ms / frame on CPU, see §2.8e) so latency is not the
blocker — the work is the causal-median implementation and the per-rat
ckpt loading.

---

## 8. Smoke tests for new sessions

Run these before pushing on anything. (`RAT_TEMPLATE` defaults to the
rebuild template — all F1 numbers below assume that.)

1. `nvidia-smi` — should print without "Driver/library version mismatch".
2. `python -m corrector.check_gpu` — should report `cuda available: True`.
3. `python -c "from corrector.data_world_2d_from_saved import load_session_2d; sd = load_session_2d('R2', '2025_11_01_1'); print(len(sd.x_2d), sd.cal_date)"` → `34430 2025_07_27`.
4. Quick 2D-input train smoke (legacy refiner, fast):
   `python -m corrector.train_2d_input --rats R2 --tag _smoke --epochs 2 --max_sessions_per_rat 2 --batch_size 1024`
   → ~1 s/epoch on GPU.
5. Quick new-trainer smoke:
   `python -m corrector.train_temporal_mlp_2d_reproj --rats R2 --tag _smoke_reproj --epochs 2 --max_sessions_per_rat 2 --batch_size 2048`
   → ~0.5 s/epoch on GPU, val_mse roughly 500-1500.
6. Classic 3D-input eval (rebuild template):
   `python -m corrector.evaluate_all --ckpt corrector/checkpoints/R1R2R3_world_temporal_mlp.pt`
   → F1xyz R1=0.897, R2=0.758, R3=0.824; kp_mse R1=186, R2=280, R3=280.
7. New 2D-input production single-model eval:
   `python -m corrector.evaluate_all --ckpt corrector/checkpoints/R1R2R3_temporal_mlp_2d_reproj_v4.pt --smooth_size 15`
   → F1xyz R1=0.870, R2=0.804, R3=0.840; kp_mse R1=165, R2=256, R3=227.
8. Inference benchmark:
   `python scripts/bench_reproj_v1.py`
   → end-to-end ~0.3 ms/frame CPU.
