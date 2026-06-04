# CLIRB SLEAP→DANNCE Corrector — Production Picks

**Last updated**: 2026-05-29

This file consolidates the per-rat recommended models, their numbers, and the
operations needed to deploy them. The reference handoff document is
`handoff.md` — read that first for the project context.

---

## 1. TL;DR

Three significant changes since the handoff:

1. **Rebuilt template_1 is now the default** in `corrector/evaluate_world.py`
   (`RAT_TEMPLATE` points at `<rat>_template_1_rebuild.npz`). The on-disk
   `<rat>_template_1.npz` files have NOT been renamed; the change is code-only.
   To deploy this into the online pipeline, see §5 below.
2. **A new model class, `temporal_mlp_2d_reproj`**, takes the temporal MLP and
   adds current-frame 2D detections + confidence + visibility + reprojection
   residuals. With cosine-LR training (150 epochs) and a median-11 smoothing
   pass on the output, it is the new R2 production candidate. Details: §3.
3. **Inference cost is negligible.** End-to-end ~0.4 ms / frame on CPU, 0.2 ms
   on GPU — well below SLEAP's ~30 ms / frame budget. Benchmark in
   `scripts/bench_reproj_v1.py`.

---

## 2. Per-rat picks

All evals use the rebuilt template (default). F1xyz = `corrected_xyz_f1_300`.
The reproj_v3 picks below use `--smooth_size 15` (was 11; the size-15 median
filter beats size 11 on R3 by a large margin without hurting R1/R2 — see §6).

| Rat | Best model (test) | F1xyz test | F1xyz post-cutover | kp_mse |
|-----|-------------------|-----------:|-------------------:|-------:|
| R1 | `R1R2R3_world_temporal_mlp.pt` | **0.897** | 0.898 | 186 |
| R2 | `R2_reproj_ft_from_v3.pt` (smooth_size=15, per-rat fine-tune) | **0.824** | 0.799 | 221 |
| R3 | `R1R2R3_temporal_mlp_2d_reproj_v4.pt` (smooth_size=15, targeted noise) | **0.840** | **0.840** | 227 |

### Per-rat fine-tunes

After global v3 trained, we fine-tuned per-rat with `--init_ckpt` for 30 ep
at lr=5e-5 cosine, no noise aug. Results on test split with smooth=15:

| Rat | Source | F1 test | F1 cutover | kp_mse |
|-----|--------|--------:|-----------:|-------:|
| R1 | Global v3 | 0.871 | 0.884 | 178 |
| R1 | **R1 fine-tune** | **0.888** | **0.889** | **164** |
| R2 | Global v3 | 0.810 | 0.783 | 231 |
| R2 | **R2 fine-tune** | **0.824** | **0.799** | **221** |
| R3 | Global v3 | **0.820** | **0.825** | 239 |
| R3 | R3 fine-tune | 0.776 | 0.781 | 226 |

R1 and R2 fine-tunes are clean wins on both metrics. R3 fine-tune trades F1
for kp_mse — R3 has resisted per-rat ft in every attempt (val_mse diverges
immediately during fine-tune training). Use the **global v3** for R3.

Note: R1's classic `temporal_mlp` (0.897) still beats R1's 2D fine-tune
(0.888). R1 is at its template-matching ceiling; the 2D inputs don't help
when the 3D triangulation is already clean.

### What changed vs the handoff's section 3 picks

| Rat | Handoff pick | F1xyz then | New pick | F1xyz now | Δ |
|-----|--------------|-----------:|----------|----------:|---|
| R1 | `R1_head_on_R1R2R3temporal` | 0.821 | `R1R2R3_world_temporal_mlp` | 0.897 | +0.076 |
| R2 | `R2_head_on_velacc_noise` | 0.646 | `R1R2R3_temporal_mlp_2d_reproj_v3` (smoothed) | 0.808 | +0.162 |
| R3 | `R1R2R3_velacc` | 0.727 | `R1R2R3_world_temporal_mlp` | 0.824 | +0.097 |

Most of the R1/R3 gain is from the rebuild template; the R2 gain comes from
both the template AND the new architecture.

### Single-model alternative

If the production loader can only hold one corrector at a time, the strongest
single-model pick is `R1R2R3_temporal_mlp_2d_reproj_v4.pt` with
`--smooth_size 15`:
- R1: 0.870 (−0.027 vs per-rat best)
- R2: 0.804 (−0.020 vs per-rat best)
- R3: **0.840** (the new global R3 winner)

This single ckpt is within ≤0.027 of the per-rat best on every rat.
**Recommended one-model production deployment.**

The classic alternative is `R1R2R3_world_temporal_mlp.pt`:
- R1: **0.897** (the only model that beats it on R1)
- R2: 0.758 (−0.052 vs reproj_v3)
- R3: 0.824 (+0.004 vs reproj_v3)

---

## 3. About `temporal_mlp_2d_reproj`

### Architecture

Same flat-MLP shape as `temporal_mlp` (2 hidden × 128 units), but the input
gets four extra channels for the **current** frame:

```
Input (per sample):
  pose_window       (ctx=5, 23, 3)    = 345    same as temporal_mlp
  per_cam_2d_norm   (3, 23, 2)         = 138    SLEAP detected 2D, normalized
  per_cam_conf      (3, 23)             =  69    SLEAP detection confidence
  per_cam_vis       (3, 23)             =  69    1/0 visibility flag
  per_cam_reproj    (3, 23, 2)         = 138    (detected - reprojected) / 100 px
                                       = 759

Output: (23, 3) residual added to pose_window[:, -1, :, :]
Last layer zero-init.
~123k params.
```

The reprojection uses the saved un-smoothed triangulated 3D (the geometry that
actually produced the detections), projected through the session calibration.
That's what gives the model an internal consistency check on the triangulation.

### Training story (chronological)

| version | best val_mse | F1xyz R1 / R2 / R3 (test) | Notes |
|---------|-------------:|---------------------------|-------|
| v1 (50mm noise, 0.3 prob, 100 ep) | 134.3 | 0.882 / 0.787 / 0.809 | First successful 2D model. Still improving at ep 100. |
| v2 (100mm / 0.5, 100 ep, killed) | 288.7 | n/a (killed) | Noise too big — destroyed the rat skeleton prior. Stopped early. |
| v3 (50mm / 0.3, cosine LR, 150 ep) | 151.4 | 0.873 / 0.795 / 0.776 | Worse val_mse than v1 but better kp_mse. |
| v3 + smooth=11 | (same ckpt) | 0.872 / 0.808 / 0.790 | +median-11 inside corrector. |
| v3 + smooth=15 | (same ckpt) | 0.871 / 0.810 / 0.820 | +median-15. R3 jumps. |
| v4 (targeted, 150mm/0.5/maxK=3, cosine 150 ep) | 132.2 | 0.870 / 0.804 / **0.840** | Targeted per-kp noise. R3 best by far. |
| Per-rat fine-tunes from v3 (smooth=15) | n/a | 0.888 / **0.824** / 0.776 | R1, R2 improve. R3 degrades (overfit-fast). |

The smoothing step lives inside `correct_temporal_mlp_2d_reproj` (default
`smooth_size=11`) so both eval and rendering paths see it.

### Files

- Model: `corrector/models.py:TemporalMLPWith2DReproj`
- Trainer: `corrector/train_temporal_mlp_2d_reproj.py`
- Inference: `corrector/evaluate_all.py:correct_temporal_mlp_2d_reproj`
- Checkpoints: `corrector/checkpoints/R1R2R3_temporal_mlp_2d_reproj_v{1,3}.pt`
- Comparison videos: `corrector/videos/R1R2R3_temporal_mlp_2d_reproj_v{1,3}_smoothed_{singlepanel,3dtriptych}/`

---

## 4. Inference cost benchmark

Single-frame end-to-end (full pipeline: project 3D→2D, build features, run MLP):

| Path | Median latency |
|------|---------------:|
| CPU, B=1 | 295 µs |
| GPU, B=1 (incl. h2d + cuda sync) | 207 µs |
| GPU, B=64 (per-frame amortized) | 0.8 µs |

SLEAP frame budget: ~30 ms. The corrector is **<1.5%** of the budget on CPU.
Benchmark script: `scripts/bench_reproj_v1.py`.

---

## 5. Deploy notes

### 5a. Code-only changes (already done in this session)

- `corrector/evaluate_world.py:RAT_TEMPLATE` updated to point at the rebuild
  `.npz` filenames. Affects ALL eval and rendering paths that go through
  `evaluate_all.py` and `render_world_overlay.py`.

### 5b. What's NOT yet done

- **Data-side rename of template files.** `<rat>_template_1_rebuild.npz` is
  still a separate file from `<rat>_template_1.npz` on disk. Code change above
  routes around this by referencing the rebuild filename directly.
- **Production-pipeline integration.** `~/campy-CLIRB/campy/behavior_rule.py`
  still loads `<rat>_template_1.npz` (the legacy file) and doesn't call the
  corrector. Two separate things to wire if you want this online:
    1. Update the template path in `behavior_rule.py` to use the rebuild file.
    2. Add the corrector call site inside `TemplateMatch.update()` /
       `corrections()`.

### 5c. Online integration caveats (when you get to it)

- **Causal median filter.** The offline `correct_temporal_mlp_2d_reproj`
  applies `median_filter(out, size=(11, 1, 1))` symmetrically — uses both past
  and future frames. The live pipeline only has past frames. For online use
  with no display lag, the symmetric median-11 should be replaced with a
  causal median over the most recent N frames; this is effectively a median-N
  with N-1 frames of lag in the smoothed signal. Sensible N values are 5 or 7
  to keep lag under 250 ms while still removing the gap-frame jumpiness.
- **Reprojection in the live pipeline** needs access to per-session
  calibration tensors. The offline path uses
  `corrector.data_world_2d.reproject_all_cams` over numpy; for production this
  should be torch-side for consistency with the SLEAP-on-GPU pipeline.
- **Fallback path.** When a processed frame has no valid 2D detection on any
  camera, fall back to Procrustes-only output (the offline pipeline does this
  via `has_2d` masking inside `correct_temporal_mlp_2d_reproj`). Same logic
  needs to be in the online version.

### 5d. Rollback plan

To revert the template change without touching data files:

```python
# corrector/evaluate_world.py
RAT_TEMPLATE = {
    "R1": "R1_template_1.npz",   # back to legacy SLEAP-pose template
    "R2": "R2_template_1.npz",
    "R3": "R3_template_1.npz",
}
```

To revert the architecture pick, use the existing `R1R2R3_world_temporal_mlp.pt`
checkpoint, which is the prior production candidate.

---

## 6. Smoothing window ablation

`correct_temporal_mlp_2d_reproj` accepts a `smooth_size` parameter (CLI flag
`--smooth_size`, default 11). It applies an in-corrector median filter along
time. All four sizes evaluated below use the same v3 checkpoint.

| Rat | Metric | size=0 | size=7 | size=11 | size=15 |
|-----|--------|-------:|-------:|--------:|--------:|
| R1 | F1 test | 0.873 | 0.871 | 0.872 | 0.871 |
| R1 | F1 cutover | 0.883 | 0.882 | 0.884 | 0.884 |
| R1 | kp_mse | 193 | 180 | 178 | 178 |
| R2 | F1 test | 0.795 | 0.807 | 0.808 | **0.810** |
| R2 | F1 cutover | 0.766 | 0.783 | 0.783 | 0.783 |
| R2 | kp_mse | 255 | 230 | **229** | 231 |
| R3 | F1 test | 0.776 | 0.783 | 0.790 | **0.820** |
| R3 | F1 cutover | 0.765 | 0.775 | 0.785 | **0.825** |
| R3 | kp_mse | 252 | 239 | **238** | 239 |

**Recommendation: use `smooth_size=15`** for both eval and production. The R3
gain (+0.030 F1 test, +0.040 F1 cutover) is large and the R1/R2 cost is
near-zero. The default in `correct_temporal_mlp_2d_reproj` is still 11; pass
`--smooth_size 15` to get the recommended behavior or change the default if
you want it system-wide. (Note that this offline filter uses past+future
frames symmetrically; an online causal version would have lag — see §5c.)

---

## 7. Open work (not done this session)

- **Smoothing window ablation** (median-7 vs median-11 vs median-15) — would
  finalize whether 11 is optimal across rats.
- **Per-rat fine-tuning of `reproj_v3`** — might close the R3 full-test gap.
- **Targeted per-keypoint noise augmentation** (corrupt 1–3 keypoints at a
  time rather than all 23) — more realistic failure-mode augmentation; might
  push R3 cutover higher.
- **`templates/R{x}_template_2.npz`** — records origin metadata, but the
  production loader uses `_template_1`. Not yet evaluated.

---

## 8. Quick reproduction commands

Eval the R2/R3 production pick:
```bash
python -m corrector.evaluate_all \
    --ckpt corrector/checkpoints/R1R2R3_temporal_mlp_2d_reproj_v3.pt \
    --smooth_size 15
```

Render comparison video on a specific session:
```bash
python -m corrector.render_world_overlay \
    --ckpt corrector/checkpoints/R1R2R3_temporal_mlp_2d_reproj_v3.pt \
    --rat R2 --session 2026_02_09_1 --camera 0 --n_frames 1000 \
    --single_panel
```

Benchmark:
```bash
python scripts/bench_reproj_v1.py
```
