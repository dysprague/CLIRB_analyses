# Diagnostic findings — multi_session.ipynb reprojection + PC MSE issues

Date: 2026-06-25
Investigated on request: "diagnose why reprojection residuals are so high … and why PC MSE seems to be increasing over time".

---

## TL;DR

| Symptom | Root cause | Severity |
|---|---|---|
| Per-session reprojection residual ≈ 60–500 px (vs expected 5–15 px for good calibration) | `triang_keys_3D.npy` stores the **corrector network's output**, not raw SLEAP triangulation. Reprojecting it measures how aggressively the SLEAP→DANNCE corrector is pulling points, not calibration quality. | **High — metric is misnamed.** Calibration is healthy. |
| PC MSE varies session-to-session, hypothesized to drift upward | **Partly** the hypothesis: feature_means is taken from a baseline session and never updated. SLEAP and DANNCE per-session means drift 15–60 mm from template (5–17% of template magnitude), and SLEAP-vs-DANNCE per-session means differ by 15–32 mm. However, **the drift is mostly stable across sessions, not monotonically increasing**, so the cross-session PC MSE variation is dominated by other factors (rat activity level, tracking failure rate). | **Medium — fix recommended but not the dominant noise source.** |

---

## Finding 1 — Reprojection residuals reflect corrector output, not calibration

`section_2.2 — Calibration health` projects `triang_keys_3D.npy` back into each camera
and compares to `sleap_keys_2D.npy`. With the median-over-confident-detections
metric, per-session residuals come out 65 px (R6/06_20) to 477 px (R5/06_16).
That looks alarmingly high.

It is not a calibration problem. From `campy-CLIRB/campy/behavior.py:432-453`:

```python
# 1. Outlier replace 2D detections with carried-forward stale values
mask = distances > 400
keypoints_2D[mask] = np.nan
keypoints_2D[nan_mask] = prev_keypoints_2D[nan_mask]
mm_peaks_and_vals[frameNumber, :,:,:2] = keypoints_2D  # written to sleap_keys_2D.npy

# 2. Triangulate
keypoints_3D, undist_pts = triangulate(keypoints_2D, P_list, dist_coefs, K_list)

# 3. APPLY THE CORRECTOR NETWORK
if corrector is not None:
    new_points_3d = corrector.correct(keypoints_3D, keypoints_2D, peak_vals)
mm_keys_3D[save_key_index,:,:] = new_points_3d  # written to triang_keys_3D.npy
```

So `triang_keys_3D.npy` is the SLEAP→DANNCE corrector's output, not raw triangulation.

### Evidence

On `R4/2026_06_18_2`, first 2000 frames, cam0:

| Source of 3D | Median reprojection error (conf ≥ 0.5) | Mean | p95 |
|---|---|---|---|
| **Raw triangulation of `sleap_keys_2D.npy`** | **9.32 px** | 12.91 px | 35.32 px |
| **Saved `triang_keys_3D.npy` (corrector output)** | **60.17 px** | 189.13 px | 734.45 px |

The corrector is moving 3D points by an average of 139 mm (p95 = 488 mm) relative to the raw triangulation. That's the entire reprojection residual.

So **calibration is fine** — raw-triangulation reprojection error is < 10 px median, which is the expected SLEAP detection noise.

### What the metric in section 2.2 IS measuring

It is roughly proportional to "average corrector pull magnitude", which is interesting in its own right (it tells you how much the corrector is reshaping the SLEAP→DANNCE mapping) but is not "calibration health".

### Remedies

Three options, ordered by effort:

1. **Re-label the metric.** Title it "median post-corrector reprojection residual" or "corrector pull magnitude (px-equivalent)". This is honest about what's being computed.
2. **Replace with raw-triangulation reprojection.** Re-triangulate from `sleap_keys_2D.npy` per session (the function in `behavior.py:147-169` is ~20 lines) and reproject the result. Cost: ~+15 s/session. This is the actual calibration health metric.
3. **Use the online corrector's saved residual** if it's persisted anywhere (search for `reproj_resid` in `corrector_online.py:315`; it's currently used as MLP input, not saved).

Option 2 is the cleanest fix. I have a working implementation in `/tmp/` from the diagnostic; can wire it into the notebook if you want.

---

## Finding 2 — feature_means drift is real but small and not monotone

### Template provenance

```
R4: 16 sessions, 4 unique template hashes (changes every 4-7 sessions)
R5: 14 sessions, 3 unique template hashes
R6: 17 sessions, 4 unique template hashes
```

Within each rat, **`feature_means` is bit-identical across every template file**.
Only the per-PC trajectory and bounds are rebuilt; the centering vector is frozen
from the original baseline.

### Drift measurements (||per-session feature_means − template feature_means|| in mm)

| rat | session | drift_sleap | drift_dannce | sleap-vs-dannce | tmpl_fm_norm |
|---|---|---|---|---|---|
| R4 | 2026_06_12_1 | 28.3 | 24.8 | 14.9 | 304.2 |
| R4 | 2026_06_15_1 | 27.1 | 21.6 | 19.4 | 304.2 |
| R4 | 2026_06_16_1 | 33.5 | 30.2 | 16.2 | 304.2 |
| R4 | 2026_06_18_2 | 22.7 | 20.0 | 19.8 | 304.2 |
| R4 | 2026_06_20_1 | 22.4 | 25.3 | 15.5 | 304.2 |
| R5 | 2026_06_12_1 | 22.5 | 13.1 | 21.9 | 309.3 |
| R5 | 2026_06_15_1 | 16.7 | 17.8 | 23.8 | 309.3 |
| R5 | 2026_06_16_1 | 22.7 | 33.5 | 24.0 | 309.3 |
| R5 | 2026_06_17_1 | 34.6 | 37.4 | 20.4 | 309.3 |
| R5 | 2026_06_18_1 | 23.4 | 26.1 | 22.2 | 309.3 |
| R5 | 2026_06_18_2 | 16.8 | 22.5 | 22.9 | 309.3 |
| R5 | 2026_06_20_1 | 19.7 | 24.5 | 19.3 | 309.3 |
| R6 | 2026_06_12_1 | 36.8 | 39.6 | 26.7 | 309.2 |
| R6 | 2026_06_15_1 | 31.9 | 23.4 | 31.8 | 309.2 |
| R6 | 2026_06_16_1 | 53.1 | 57.4 | 26.7 | 309.2 |
| R6 | 2026_06_17_1 | 40.1 | 37.1 | 27.1 | 309.2 |
| R6 | 2026_06_18_1 | 44.1 | 43.6 | 26.5 | 309.2 |
| R6 | 2026_06_20_1 | 47.2 | 42.9 | 27.5 | 309.2 |

Interpretation:
- **Template feature_means norm is ~305 mm.** Drift of 20-50 mm is 7-17% of the centering magnitude. Non-trivial.
- **`sleap_vs_dannce_drift`** (per-session mismatch between SLEAP and DANNCE mean poses) is roughly stable per rat: R4 ≈ 15-20 mm, R5 ≈ 19-24 mm, R6 ≈ 26-32 mm. R6 is the worst, which matches the higher PC MSE for R6 in section 2.3.
- The drift is **not monotonically increasing** in any rat. R4 06_12: drift 28 → 06_20: drift 22 (decreased). R5 same. R6 went 37 → 53 → 40 → 44 → 47 (noisy, no trend).

So "drift over time" as the dominant explanation for PC MSE variation does **not** hold up — but the absolute size of the drift is enough that it does contribute, especially for R6.

### Re-centering experiment

I tested 3 strategies on 3 sessions:
- **baseline** — use template's `feature_means` for both SLEAP and DANNCE (current code)
- **shared_session** — use this session's SLEAP-derived `feature_means` for both
- **per_modality** — use SLEAP's own session mean for SLEAP, DANNCE's own session mean for DANNCE

Per-session SLEAP-DANNCE PC MSE at matched windows (lower = better):

| session | baseline | shared_session | per_modality |
|---|---|---|---|
| R4/2026_06_18_2 | 896.82 | 888.84 | 954.40 |
| R5/2026_06_20_1 | 654.91 | 643.75 | **540.41** |
| R6/2026_06_20_1 | 657.16 | 707.81 | 769.83 |

R5 benefits noticeably from per-modality recentering (−17%). R4 is essentially flat. R6 gets worse — its SLEAP and DANNCE per-modality means differ enough that recentering separately throws away the shared signal.

A simpler variant — project as usual through the template's feature_means, then
subtract each modality's *PC-space* mean from its PCs before computing
SLEAP-vs-DANNCE PC MSE — gives the same picture: R5 improves substantially,
R4 doesn't change, R6 gets slightly worse. Per-session PC-space mean offsets:

| session | SLEAP PC mean (PC1, PC2) | DANNCE PC mean (PC1, PC2) | Δ | PC MSE base → offset |
|---|---|---|---|---|
| R4/2026_06_18_2 | (−2.4, 2.7) | (4.3, −1.4) | (6.7, 4.1) | 897 → 954 |
| R5/2026_06_20_1 | (13.9, 1.9) | (19.3, 9.3) | (5.4, 7.4) | 655 → **540** |
| R6/2026_06_20_1 | (−12.1, 33.8) | (−10.2, 25.0) | (1.9, 8.8) | 657 → 770 |
| R6/2026_06_15_1 | (6.6, 23.7) | (12.3, 8.5) | (5.7, 15.2) | 814 → 765 |

R6 has the largest absolute PC-space offsets from origin (−12, 34 and 6.6, 24)
— the rebuilt rule template assumes PCs centered near zero, but R6's session
data is drifting tens of units in PC2. That same drift is the reason
`drift_sleap/dannce` for R6 is 36–53 mm in the previous table.

### Remedies

1. **Don't change the rule's centering** — bounds and matcher were tuned around the baseline `feature_means`. Re-centering would invalidate them.
2. **For the *diagnostic* PC MSE metric in section 2.3**, the most defensible cleanup is **per-modality PC-space mean subtraction**: project as usual, then subtract `sleap_pcs.mean(axis=0)` from SLEAP PCs and `dannce_pcs.mean(axis=0)` from DANNCE PCs *before* computing the per-match SLEAP-DANNCE MSE. This removes the rigid offset (the part that drift creates) without touching the template-matching pipeline. The remaining MSE captures actual dynamic disagreement.
3. **Long-term:** consider rebuilding `feature_means` at every template refresh (currently only the trajectory and bounds get refreshed). Easy to add to whatever script writes the templates.

### A second hypothesis worth checking

The bigger driver of PC MSE variation across sessions is probably **per-session SLEAP tracking quality**, not the centering. Sessions with more low-confidence keypoints will have noisier triangulated 3D, which will produce wider SLEAP-DANNCE divergence at every PC. Easy follow-up: for each session, plot mean SLEAP 2D confidence (averaged across cams and kpts) against the section 2.3 PC MSE. If they correlate strongly, drift is a sideshow.

---

## Calibration sanity check (since we suspected it)

Raw-triangulation reprojection error (median, conf ≥ 0.5) on R4/2026_06_18_2 cam 0:

| keypoint | n | median err (px) |
|---|---|---|
| 0 (snout) | 3342 | 404.8 |
| 4 (SpineM) | 2597 | **162.3** |
| 6 | 4170 | 202.4 |
| 19 | 2417 | **143.9** |
| ... | | |

Wait — those numbers above are reprojecting **the corrector output**, not raw triangulation. After re-triangulating from `sleap_keys_2D.npy` and reprojecting, the pooled cam-0 median was **9.32 px**. So per-keypoint medians using raw triangulation should be in single-digit-to-teens px. Calibration is good.

---

## Summary of recommended next steps

1. **Section 2.2 (reprojection residual)** — rename to "post-corrector reprojection residual" OR replace with raw-triangulation reprojection. I lean toward replacing — it gives a metric that actually tracks calibration quality.
2. **Section 2.3 (PC MSE)** — apply per-modality PC-mean subtraction before computing the per-match SLEAP-DANNCE MSE. Removes the static-offset component caused by drift, exposes real dynamic disagreement.
3. **Add a per-session SLEAP detection quality metric** (mean 2D confidence per camera × keypoint, fraction of low-confidence detections per session) alongside section 2.3 so it's possible to see whether PC MSE variation is tracking detection quality.
4. **Optional, longer-term:** make the template-rebuild script also refresh `feature_means` from the rebuild sessions, not just `bounds_to_use` and `template`.
