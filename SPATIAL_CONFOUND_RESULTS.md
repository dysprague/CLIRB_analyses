# Spatial confound in template matching — results

**Question.** Are the same behaviors rewarded differently in different parts of the arena?

**Answer. Yes, substantially — in two of the three animals.** Holding true behavior fixed
(nearest-neighbour matched on the full 30-frame DANNCE trajectory), match probability still varies
across the arena perimeter by **0.75 (R5) and 0.37 (R6)**, both p < 0.0001, when rears are defined
entirely from DANNCE so that event selection cannot be contaminated by SLEAP. **R4's effect (0.28)
does not survive that test (p = 0.075)** — see §8a(2). So there is a real, large, position-dependent
bias in what the online rule sees, and the animals' localization is not purely natural settling.

⚠️ **Read §8a before quoting anything below.** The follow-up round corrected the headline from
"all three rats" to "two of three", and rejected the leading mechanistic hypothesis (M1).

Run 2026-08-05 on all 151 locally cached sessions (R4/R5/R6), 16,315 rears, 136 dual-modality
sessions. Plan: `SPATIAL_CONFOUND_PLAN.md`. Code: `clirb/`, `scripts/stage*.py`.
Outputs: `results/spatial_confound/`. **Figures: `results/figures/spatial_confound/`**
(`scripts/make_figures.py`).

| figure | shows |
|---|---|
| **F1** `F1_arena_occupancy.png` | rear locations per rat on the fitted arena, port + gate arc |
| **F2** `F2_pmatch_by_sector.png` | P(match \| rear) by sector with session-clustered CIs (§2) |
| **F3** `F3_nn_matched_delta.png` | **the main result** — behaviour-matched Δ P(match) by sector (§3b) |
| **F4** `F4_quality_vs_theta.png` | the negative result — no quality metric tracks the gradient (§4) |
| **F5** `F5_residual_field.png` | the residual field μ(θ), same shape in all 3 rats (§5) |
| **F6** `F6_behaviour.png` | occupancy concentration over training + spatial win-stay (§7) |
| **F7** `F7_cluster_sector_ami.png` | builder KMeans clusters × arena sector vs null (§6) |

---

## 1. Setup

| | |
|---|---|
| arena | circle, centre **(−66, −98) mm**, R ≈ **332 mm** (across-angle wall-radius std 4.0 mm) |
| animals are at the wall | median COM radius 310 mm; **96%** of rears at r > 250 mm → θ is the coordinate |
| water port | θ ≈ **−12°**; the 200 mm gate covers the wall arc **−46°…+23°** |
| rears | 16,315 over 151 sessions (R4 4,431 · R5 7,979 · R6 3,905) |
| P(match \| rear) | R4 0.441 · R5 0.411 · R6 0.327 |
| analysis set | gate-on sessions, outside the gate zone: **10,397** rears |

`clirb/events.py` reproduces the documented invariant exactly (R6/2026_07_08_1 → **60 rears /
20 matched**), so the extraction is behaviour-identical to the notebooks.

---

## 2. The effect is large

P(match | rear), 30° sectors, gate zone excluded, 95% session-clustered bootstrap CI:

| rat | worst sector | best sector | spread |
|---|---|---|---|
| R4 | 0.297 at +165° | 0.600 at +15° | **0.30** |
| R5 | 0.210 at +15° | 0.596 at +45° [0.555–0.638] | **0.39** |
| R6 | 0.235 at −135° | 0.514 at −45° | **0.28** |

R5's peak sector holds 2,903 of its 5,213 analysed rears — it has localized onto its best sector.
Note R5's +15° and +45° sectors are **adjacent** yet differ by 0.39 in reward probability.

---

## 3. It survives conditioning on true behavior — the main result

Two independent designs, both using DANNCE (6 cameras) as the referee against SLEAP (3 cameras).

**Referee construction (the methodological crux).** `multi_session`'s `analyse_dual_session` rebuilds
its DANNCE template *from SLEAP-selected matches*, which imports the very bias under test. Instead:
per-session modality offset from a **sector-balanced** frame sample → seed = production template −
offset (this also fixes phase: R4/R6 templates are the rear *descent*, R5's is the *ascent*) → align
every rear's DANNCE window → average a **sector-stratified** subsample → reference. Nothing in the
chain uses `matched`. Validation: ρ(d_dannce, matched) = **−0.331 / −0.178 / −0.337** (all
p < 1e-50) — the referee tracks behavior quality — while ρ(median d_dannce, P(match)) *across
sectors* is +0.12 / −0.44 / +0.12 (all n.s.), so it is not merely echoing the outcome.

**(a) Within d_dannce quality bins** (10 bins, ≥20 rears/sector/bin), variance of P(match) across
sectors vs a within-bin sector-shuffle null:

| rat | observed var | null mean | p | implied sector SD in P at fixed behavior |
|---|---|---|---|---|
| R4 | 0.0125 | 0.0056 | <0.0001 | **0.083** |
| R5 | 0.0248 | 0.0028 | <0.0001 | **0.148** |
| R6 | 0.0067 | 0.0031 | 0.0040 | **0.060** |

**(b) Nearest-neighbour matching on the full 60-dim DANNCE trajectory** — for each rear, the mean
match rate of its closest behavioural twins located in *other* sectors. Signed difference per sector:

| rat | n | worst sector | best sector | spread | p (full-pipeline null, §3c) |
|---|---|---|---|---|---|
| R4 | 1,388 | −0.155 at +165° | +0.149 at +135° | **0.30** | 0.005 |
| R5 | 3,086 | −0.408 at +15° | **+0.309 at +45°** | **0.72** | <0.0001 |
| R6 | 1,182 | −0.173 at +45° | +0.166 at −45° | **0.34** | <0.0001 |

For R5: the *same* rear is ~31 points more likely to be rewarded at +45° and ~41 points less likely
at +15°. Those sectors are adjacent.

### 3c. How the NN design works, and what it does not license

**Procedure.** For each rear, take the DANNCE window at its DANNCE-aligned end — 30 frames × 2 PCs
— flatten to 60-dim and z-score each dimension across rears (so Euclidean distance weights every
frame×PC equally). Fit `NearestNeighbors(k=40)`, exclude self, and keep only neighbours that are
(a) in a **different** sector and (b) within a global distance threshold (25th percentile of all
neighbour distances). Then `Δᵢ = matchedᵢ − mean(matched over those neighbours)`, aggregated to a
per-sector mean. `matched` is always the offline replay of the online rule on SLEAP; the neighbour
search is entirely in DANNCE space, so outcome never enters the matching.

**Robustness — the knobs don't drive it.** Sweeping k ∈ {10,20,40,80} × threshold ∈ {10th,25th,50th
pct}, the sector spread stays 0.24–0.45 (R4), 0.63–0.90 (R5), 0.26–0.41 (R6) across all 12
settings — always far above the null. (The two tightest settings for R5/R6 leave too few rears and
return NaN.)

**Stronger null.** The §3b p-values came from permuting sector labels *after* the deltas were
computed, which leaves the pipeline's own use of sector labels intact. Re-running the **entire**
NN procedure on shuffled sector labels, 200×:

| rat | observed weighted var | null mean | null p95 | p |
|---|---|---|---|---|
| R4 | 0.0063 | 0.0023 | 0.0039 | **0.005** |
| R5 | 0.0502 | 0.0008 | 0.0016 | **<0.0001** |
| R6 | 0.0100 | 0.0013 | 0.0025 | **<0.0001** |

All survive. **Correction to §3b: R4 is p = 0.005, not p < 0.0001**, under this more honest null.

**The caveat that matters.** These deltas are **contrasts against a mixture**, not independent
per-sector estimates. Each rear is compared to whatever behavioural twins exist elsewhere, so the
comparison set is dominated by whichever sectors are densely occupied. For R5, one sector holds
**54%** of the compared rears — so its +0.31 is largely "+45° vs the pooled rest", and the −0.41 at
+15° is largely "+15° vs a set dominated by +45°". The two are not independent measurements, and the
n-weighted mean delta is +0.133 for R5 (vs −0.026 / −0.013 for R4/R6), reflecting exactly this
asymmetry rather than a bug.

**Read the 0.72 as a contrast range, not a difference of two independent quantities.** The
within-quality-bin analysis (§3a) has no such structure and gives the more conservative,
independently-estimated figure: sector SD of 0.148 for R5.

**Conditioning barely shrinks the effect.** Max−min sector log-odds contrast before → after
conditioning on d_dannce: R4 1.169 → 1.341, R5 1.814 → 1.744, R6 1.241 → 1.397. Behaviour
composition (H0) explains almost none of the spatial gradient.

---

## 4. It is NOT a gross tracking-quality failure (H1 as originally posed: rejected)

Per-frame truth-free metrics over all 152 sessions. **These are the corrected v2 numbers**, computed
after fixing the processed/acquisition frame-basis mismatch described in §8c — `reproj_px` is now the
reprojection error of the *stored* 3D pose (the one the analysis and the online rule use), not the
self-consistency of a re-triangulation. Spearman ρ with P(match) across sectors:

| metric | R4 | R5 | R6 |
|---|---|---|---|
| reprojection residual | −0.10 (p 0.77) | +0.10 (0.77) | −0.39 (0.26) |
| mean confidence | +0.22 (0.52) | −0.32 (0.34) | −0.02 (0.96) |
| n_cams conf≥0.5 | +0.10 (0.76) | +0.44 (0.18) | +0.35 (0.32) |
| bone-length CV | +0.38 (0.25) | +0.16 (0.63) | +0.38 (0.28) |
| jitter | +0.08 (0.81) | +0.13 (0.71) | +0.15 (0.68) |
| FOV margin at apex | +0.31 (0.36) | −0.16 (0.65) | +0.29 (0.43) |

**Nothing reaches significance**, so the conclusion is unchanged — but it now rests on correctly
aligned data. Reprojection residual runs 15–21 px across sectors. Bone-length CV varies spatially
(0.10 → 0.18) but in the *wrong* direction — R5's best sector has among its worst rigidity.

*Worth a follow-up:* R5's best sector (+45°, P = 0.596) does have the **lowest** reprojection
residual (15.2 px vs up to 19.4) and the **highest** `n_cams_conf50` (2, the only sector reaching 2).
Directionally that is a quality story, but with 11 sectors it does not survive as a correlation
(ρ = +0.10 / +0.44 for reproj / n_cams). Worth revisiting with per-rear rather than per-sector
statistics.

*Rig note (corrected):* the median number of cameras with keypoint confidence ≥ 0.5 is **2**, and
per-camera means are **0.549 / 0.547 / 0.348** — camera 2 is the weak one, but not as weak as the
misaligned v1 numbers suggested.

---

## 5. What it actually is: a position-dependent bias inside the tolerance band

The SLEAP−DANNCE residual in PC space has a systematic spatial mean μ(θ):

| rat | μ(PC0) range across sectors | as % of the rule's PC0 tolerance | residual magnitude / tolerance |
|---|---|---|---|
| R4 | −5.8 → +10.2 | **19%** | 0.33–0.37 |
| R5 | −27.3 → −9.7 | **24%** | 0.40–0.53 |
| R6 | −5.1 → +6.6 | **12%** | 0.28–0.37 |

The online rule is a tight corridor (`bounds_to_use`, mean PC0 tolerance 73–125) that admits at most
3 out-of-bounds samples in 60. A systematic offset worth 12–24% of that corridor — on top of noise
already worth 30–50% of it — is more than enough to move match probability by the observed amounts.
For R5 the residual is *smallest in magnitude* at +45°, its best and most-occupied sector.

This is a subtle bias, not a tracking failure, which is exactly why §4's gross-quality metrics miss it.

**The bias field belongs to the rig, not the animal (F5).** μ(θ) has the *same shape* in all three
rats — pairwise Pearson r = **+0.953 (R4·R5), +0.870 (R4·R6), +0.947 (R5·R6)**, all p ≤ 0.0002 over
12 shared sectors — differing only by a per-rat constant (R4 +1.0, R5 −18.7, R6 +0.7). After
removing that offset, the cross-rat SD at a given θ is **1.11** vs a within-rat spread across θ of
**4.19**: the spatial pattern is ~4× larger than the between-animal disagreement.

This is the strongest single piece of evidence that the confound is a real, reproducible property of
the 3-camera reconstruction geometry — and it means the correction is transferable across animals.

---

## 6. H2 (template-import loop): mechanism confirmed, prediction NOT supported

The builder's clustering **does** carry position information, as predicted. Re-running
`refine_templates_mult_session.py`'s own procedure verbatim (widen ×2.5, `greedy_best`,
`KMeans(k=10, n_init=10, random_state=0)` on z-scored flattened windows), cluster label vs arena
sector: **AMI significantly above the shuffle null in 15/16 runs** (14 at p = 0.000, one at 0.005,
one at 0.008; the 16th — R5's sparsest early triple — at p = 0.086; AMI 0.026–0.227 vs null mean
≈ 0). The chosen (largest) cluster is often single-sector dominated (up to 94%), and `max_dev`
bounds built from it are **1.5–3.0× tighter** than bounds from a sector-balanced sample.

Template provenance is likewise concentrated — R5's templates 3/4/5/6 were built 92%/92%/96%/**100%**
from sector +45°.

**But the prediction fails.** Across 6 templates with enough data, the build sector's P(match) rank
is **4.33 vs 4.08 expected by chance** (Wilcoxon p = 0.56); the build sector is the best sector in
only **1/6**; and mean P(match) in the build sector is slightly *below* the sector average
(−0.036). The one clean hit is R5_template_4 (build share 0.69, build = best, 0.589 vs 0.191 sector
mean).

**Conclusion:** the template-building procedure demonstrably can import location-specific structure,
and it is worth fixing on those grounds, but it is **not** the driver of the observed gradient.

---

## 7. Behaviour: only R5 shows adaptation to the gradient

| test | R4 | R5 | R6 |
|---|---|---|---|
| rear-location concentration, early → late (circular R) | 0.386 → 0.446 (p 0.016) | **0.388 → 0.707 (p < 1e-10)** | 0.502 → 0.475 (p 0.61) |
| spatial win-stay: P(move sector) after reward vs miss | 0.787 vs 0.823 (p 0.11) | **0.534 vs 0.748 (p < 1e-4)** | 0.721 vs 0.704 (p 0.42) |
| rearing within 200 mm of port, trend across sessions | ρ −0.02 (p 0.89) | **ρ −0.52 (p < 0.001)** | ρ −0.18 (p 0.23) |
| align-MSE vs cumulative rears in that sector (session index partialled out) | +0.077 | **−0.367** | +0.040 |

R5 concentrates onto its best sector, stays put after reward and moves after failure, learns to stop
rearing in the ungated-reward port zone, and genuinely improves its rear shape with practice in a
sector. R4 and R6 show none of this. Note the raw practice correlations (−0.16, −0.25, −0.00) are
mostly a **time trend** — after partialling out session index only R5's survives, and R4's flips sign.

R4 and R6 do 42% and 39% of their rearing within 200 mm of the port, where reward is impossible by
design; R5 is at 25% and falling.

**Gate-off natural experiment (2026_07_21–23, all three rats): inconclusive.** Port-zone rearing went
0.414→0.436 (R4), 0.271→0.135 (R5), 0.400→0.243 (R6) — no rebound, but only 3 sessions and
confounded with R5's ongoing downward trend.

---

## 8. What did not work

**The counterfactual relocation simulation failed its self-consistency check and is not reported as a
result.** Recombining a behavior window with a residual snippet drawn from the same sector should
reproduce that sector's observed match rate; the simulated-vs-observed correlation across sectors was
**+0.008 (R4), +0.645 (R5), −0.021 (R6)**, with simulated absolute rates 4–10× too low. The cause is
that behavior and tracking residual are **not independent** — SLEAP error depends on pose — so
swapping noise across rears breaks a real correlation and overstates the damage. The conditioning
designs in §3 answer the same question without that assumption. A valid version would need a
pose-conditioned noise model.

`results/spatial_confound/counterfactual_map.csv` is retained for reference but should not be quoted.

---

## 8a. Follow-up round — three of the four hypotheses tested and mostly rejected

*(Run after §8b was written. §8b below is kept as the record of what was hypothesised;
read this section for what survived.)*

### (1) Detector selection (M3) is NOT the explanation — and rearing really is 6× uneven

Rears re-defined entirely from **DANNCE** PC0, put on the SLEAP scale by the per-session
sector-balanced offset (a single spatially-flat constant), then the identical hysteresis detector:

| rat | sector rate range, SLEAP-defined | DANNCE-defined | corr of the two profiles | totals |
|---|---|---|---|---|
| R4 | 7.3× | 5.5× | +0.987 | 3,855 vs 3,951 |
| R5 | 4.8× | 6.9× | +0.926 | 7,125 vs 7,121 |
| R6 | 6.1× | 5.3× | +0.904 | 3,765 vs 3,823 |

The two detectors agree almost perfectly. **The ~6× sector variation in rearing rate is genuine
behaviour, not detector bias**, and M3 is not driving §3.

### (2) §3 re-run on DANNCE-defined rears — replicates for R5 and R6, NOT for R4

| rat | n | spread (SLEAP-defined) | spread (DANNCE-defined) | p (full-pipeline null) |
|---|---|---|---|---|
| R4 | 1,256 | 0.304 | 0.284 | **0.075 — not significant** |
| R5 | 2,305 | 0.716 | **0.751** | <0.0001 |
| R6 | 1,526 | 0.339 | **0.365** | <0.0001 |

**Correction to the headline.** With events defined independently of SLEAP, the effect is robust and
large in **R5 and R6** but **not established in R4** (already the weakest at p = 0.005 in §3c). The
claim should be *two of three rats*, not all three. R5's signature is unchanged: +0.318 at +45°,
−0.433 at +15°.

### (3) M1 (heading × yaw-only egocentrics) — REJECTED

First attempt was confounded: the raw SLEAP−DANNCE world-frame difference is ~500 mm and sweeps a
circle with θ. That is a **coordinate-frame mismatch** between the two systems (~100–110° relative
rotation), which is exactly why `corrector/` fits a per-session 7-DoF Procrustes — not a tracking
bias. Redone with a Procrustes fit on a **sector-balanced** frame sample:

| rat | across-sector SD, WORLD bias (xy) | across-sector SD, EGOCENTRIC bias (xy) | ratio ego/world | Procrustes residual |
|---|---|---|---|---|
| R4 | 4.58 mm | 2.27 mm | 0.50 | 20.3 mm |
| R5 | 3.28 mm | 1.92 mm | 0.59 | 16.1 mm |
| R6 | 6.47 mm | 2.01 mm | 0.31 | 23.5 mm |

M1 predicted world ≈ constant and egocentric varying. Observed is the **opposite**: the world-frame
disagreement varies 1.7–3.2× *more* across sectors than the egocentric one. The egocentric transform
does not manufacture the spatial bias — it partially cancels it.

### (4) Triangulation geometry is uniform; rotation-invariant features make it WORSE

- **Condition number** of the triangulation covariance at rear height is **1.48–1.53** and
  essentially constant across sectors (σ_major ≈ 0.45 mm). Geometry is well-conditioned everywhere,
  so degenerate triangulation (M2) is not a spatial explanation — though the principal-error
  **azimuth does rotate** systematically with position.
- **Pairwise distances** (rotation-invariant) show a *higher* across-sector CV of the SLEAP−DANNCE
  discrepancy than egocentric coordinates do: 0.134 vs 0.071 (R4), 0.133 vs 0.065 (R5), 0.127 vs
  0.072 (R6). Removing rotation **amplifies** the sector dependence rather than removing it.

Both independently corroborate the rejection of M1 — and they mean *"switch to rotation-invariant
features"* is **not** the fix.

### Where that leaves the mechanism

What survives is a **genuine, position-dependent disagreement between the two reconstructions**
(~3–6 mm SD across sectors in world coordinates, after global alignment) that is *not* explained by
rotation, by triangulation conditioning, or by event selection. Given SLEAP has 3 cameras to
DANNCE's 6, SLEAP is the likelier source. **This was subsequently resolved in favour of M6 (SLEAP
2D keypoint bias varying with viewing aspect) — see §8d.**

**Next test:** per-camera, per-keypoint 2D residual as a function of the animal's apparent aspect
angle and its image position — this separates a network-bias account (M6) from a calibration
account. **Attempted and BLOCKED — see §8c.**

---

## 8c. The M6-vs-calibration test is blocked: `triang_keys_3D` is not reproducible from `sleap_keys_2D`

The test needs an independent 3D pose (DANNCE) expressed in the **camera** frame. That requires a
bridge from the stored 3D to the calibration, and the bridge does not exist with the artifacts on
disk.

**`triang_keys_3D` is in a different frame from the camera calibration.** Reprojecting it directly
into the SLEAP cameras gives a **~490 px** median residual, versus ~9 px for a 3D pose
re-triangulated from `sleap_keys_2D`. So the stored 3D cannot be projected into the images as-is.

**And it cannot be re-derived.** Four triangulation conventions were tried against
`sleap_keys_2D` + `calibration/2026_05_04`:

| variant | bone-length CV | Procrustes → `triang_3d` |
|---|---|---|
| **stored `triang_keys_3D`** | **0.118** | — |
| DLT, normalized undistorted coords (`quality.py`) | 0.366 | scale 0.45, resid 209 mm |
| DLT, confidence-weighted (pow 1 / 2) | 0.271 / 0.276 | scale 0.31, resid 148 mm |
| DLT, `P = K[R\|t]`, distorted pixels | 0.365 | scale 0.44, resid 199 mm |
| DLT, `P = K[R\|t]`, undistorted pixels | 0.365 | scale 0.45, resid 200 mm |

An image y-flip (`y → 1200−y`, a commented-out line in the online code) was also tried and is worse
(bone CV 0.441). Every variant is ~2.5–3× worse in skeleton rigidity than the stored 3D and none
aligns to it.

**The likely missing step, found in `~/campy-CLIRB/campy/behavior.py`:** the online `triangulate()`
is *byte-for-byte the same DLT* used here (normalized undistorted coords, `P = I·[R|t]`, plain SVD)
— but it is followed by **`correct_triangulations()` → `optimize_with_bones()`**, a
`scipy.optimize.least_squares` bundle adjustment that adds a **bone-length penalty** using
`campy/data/bone_lengths.npy` (`w_bone=1.0`, `max_nfev=15`). That is almost certainly why the stored
3D has bone CV 0.118 while a raw DLT gives 0.366.

What it does *not* explain is the ~490 px reprojection error: a bundle adjustment that minimises
reprojection error should keep the same frame and reproject *better*, not worse. So there is a frame
or convention difference on top of the bone regularisation that I could not identify by trial.

**Do not guess further.** The way to close this is to either (a) port `correct_triangulations` and
run it on `sleap_keys_2D`, or (b) confirm with whoever maintains the rig exactly how
`triang_keys_3D` is generated and with which calibration. Until then DANNCE cannot be placed in the
camera frame, and M6 cannot be separated from a calibration account.

### ✅ RESOLVED — it was a frame-basis indexing error on my part, not a data bug

**Retraction.** An earlier version of this section claimed `sleap_keys_2D.npy` did not correspond to
`triang_keys_3D.npy` and called it a pipeline bug affecting every session. **That was wrong.** The
files are fine; I was indexing them against each other incorrectly.

`sleap_keys_2D.npy` is stored on the **processed-frame** basis — frames that were not processed fast
enough online are simply not saved (here 32,160 of 36,000 rows, zero-padded to the full length).
`triang_keys_3D.npy` is on the **cam0 acquisition-frame** basis. The bridge is `frame_mapping.csv`:

```
triang_keys_3D[cam0_frame[k]]   <-->   sleap_keys_2D[k]
```

| indexing | reprojection of stored 3D | corr(cam0 median u, 3D x) |
|---|---|---|
| `triang[i]` vs `keys2d[i]` (what I did) | 253.5 px | +0.233 |
| `triang[k]` vs `keys2d[cam0[k]]` (wrong direction) | 275.7 px | +0.061 |
| **`triang[cam0[k]]` vs `keys2d[k]`** (correct) | **17.4 px** | **+0.988** |

Implemented as `quality.reprojection_residual_aligned()` / `scatter_to_acquisition()`, and the
quality cache is rebuilt at `QUALITY_VERSION = 2`.

**What this changes.** §4's metrics are **not** void — they were simply computed on misaligned data
and are now recomputed correctly. Two specific earlier claims were artifacts and are withdrawn:
"only ~1 camera has median confidence ≥ 0.5" (the correct median is **2**) and camera-2 mean
confidence of 0.31 (correct: **0.348**, with cams 0/1 at 0.549/0.547). See §4 for the corrected
results.

Note also that the corrected `reproj_px` (~15 px) is a **more meaningful** metric than the original:
it is the reprojection error of the *actual stored 3D pose* the analysis uses, not the
self-consistency of a re-triangulation.

<details>
<summary>Superseded diagnosis (kept for the record)</summary>

### The earlier, incorrect diagnosis: "`sleap_keys_2D.npy` does not correspond to `triang_keys_3D.npy`"

Chasing the bridge further ruled out every geometric explanation and found a data-integrity problem
instead:

- **Not a frame convention.** A 7-DoF similarity fit optimised *directly against the 2D detections*
  only reduces the reprojection error from 138–497 px to 70–178 px, with inconsistent fitted scales
  (0.18–0.41) and rotations (14–20°). No rigid+scale transform relates them.
- **Not a temporal lag.** Cross-correlating per-frame shape descriptors over ±60 frames gives a
  maximum |r| of 0.12–0.15 with no peak.
- **Not the frame mapping.** Indexing through `frame_mapping_cam0` (both directions) changes nothing
  (253 → 276 px; 497 → 497 px).

The decisive measurement — does each 3D reconstruction track the animal's position *in the image*?

| | corr(cam0 median u, x) |
|---|---|
| 3D re-triangulated from `sleap_keys_2D` | **+0.987, +0.981, +0.992** |
| stored `triang_keys_3D` | **+0.233, +0.100, +0.117** |

The re-triangulation follows the 2D detections almost perfectly; the stored 3D is essentially
uncorrelated with them. **The two files in a session describe different data.**

**This is across the board, not session-specific.** Over all **152** cached sessions, taking the best
absolute correlation between any (camera, image axis) and any 3D centroid axis:

| rat | n | min | median | max |
|---|---|---|---|---|
| R4 | 54 | 0.038 | 0.160 | 0.617 |
| R5 | 49 | 0.091 | 0.233 | 0.684 |
| R6 | 49 | 0.070 | 0.192 | 0.570 |

**0 / 152 sessions exceed 0.7.** (`results/spatial_confound/keys2d_vs_triang_correspondence.csv`.)

**Which file is trustworthy?** `triang_keys_3D`. It is corroborated by a completely independent
6-camera system: DANNCE-defined and `triang_3d`-defined rear rates correlate **r = 0.90–0.99** with
near-identical totals (§8a(1)). So `triang_keys_3D` tracks the real animal, and **`sleap_keys_2D.npy`
is the mismatched file** — stale, from a different run, or written on an index basis that is not
recoverable from what is stored.

The evidence that led to it: a 7-DoF similarity fit against the 2D only got 138–497 px down to
70–178 px; cross-correlating shape descriptors over ±60 frames peaked at |r| ≤ 0.15; and over 152
sessions the best correlation between any (camera, image axis) and any 3D centroid axis had a median
of 0.16–0.23 with none above 0.7. All of that is explained by the frame-basis mismatch above — the
two arrays were being compared at different moments in time.

**§§1–3 and 5–7 were never affected** either way: every main result uses `triang_3d` and DANNCE only.

</details>

---

## 8b. Candidate mechanisms *(as hypothesised; see §8a for what survived)*

Ranked by how much evidence supported them **before** the §8a follow-ups.

### M1 — Heading × yaw-only egocentrics *(was leading; now REJECTED, see §8a(3))*

`normalize_skeleton_batch` centres on SpineM and rotates SpineF→SpineM to +X **in the XY plane
only** — a yaw-only transform that leaves pitch and roll in the coordinates. Measured during rears,
the animal's mean absolute heading tracks arena angle with **circular r = +0.96** (R5 +0.959,
R6 +0.969; per-sector concentration R = 0.18–0.70). So heading is effectively a function of position.

Triangulation error from 3 cameras is **anisotropic in world coordinates** — an error ellipsoid set
by camera geometry. Rotating world-frame anisotropic error by a heading that itself rotates with θ
converts a spatially *uniform* error magnitude into a **sector-dependent bias in egocentric PC
space**. This explains, in one mechanism:

- why μ(θ) has the same shape in all three rats (§5) — it is camera geometry, not animal;
- why μ(θ) is smooth and roughly one cycle in θ (F5);
- why reprojection residual, confidence and FOV margin are flat (§4) — the error *magnitude* is
  uniform; only its *direction in egocentric coordinates* rotates.

**Corroboration already in the repo:** `README.md` reports that "Group O" — pairwise distances with
pooled PCA, i.e. a **rotation-invariant** feature set — was the best non-network fix, worth 2–6.6 pp
of F1. That is what one would expect if orientation-coupled error were the problem.

**Test:** propagate the per-position triangulation covariance through the measured heading and the
template's `pc_weights`; it should predict μ(θ) quantitatively. Cheap and decisive.

### M2 — Reprojection residual is structurally blind to the error that matters

With **only ~1 camera** at median keypoint confidence ≥ 0.5 (§4), depth along the viewing ray is
weakly constrained. Error along that ray is *invisible to reprojection residual by construction* —
it reprojects back onto the same pixel. So §4's null is partly a statement about the metric being
blind, not about error being absent. Which ray is degenerate depends on position and heading, giving
the same signature as M1. **Test:** report the triangulation covariance's condition number and
principal axis per position, rather than the reprojection residual.

### M3 — Rear-detector selection *(the most serious threat to §3)*

Rears are defined by a threshold on **SLEAP** PC0 (>200), so the detector is exposed to the very
bias under test. Detection rate per wall-minute varies **5.6× (R4), 6.9× (R5), 5.7× (R6)** across
sectors (R5: 15.2/min at +45° vs 2.2/min at −15°). Where μ(θ) inflates PC0, marginal movements are
admitted as "rears" and then fail the stricter template test → low P(match); where it deflates PC0,
only vigorous rears are admitted and they match well.

**Conditioning on DANNCE behavior does not repair this** — the analysed set is already selected, so
it is a collider/selection effect, not a covariate. Some of the detection-rate variation is surely
genuine behaviour (R5 really does rear more at +45°), but the two cannot be separated with a
SLEAP-defined detector. **Test:** re-define rears from DANNCE PC0 or from apex height and re-run
§3 end to end. Until that is done, §3's magnitudes should carry this caveat.

### M4 — Residual behavioural differences below the 2-PC referee

Conditioning used the 2-PC DANNCE trajectory. Rearing against a flat wall, in the port hardware's
recess, or where the animal can brace differently is physically different, and `apex_z` does vary by
sector (2.2–6.9 in R4). **Test:** condition on the full 23-keypoint DANNCE pose instead of 2 PCs.

### M5 — The template encodes a heading *mixture* (a refinement of H2)

§6 tested only the **modal** build sector and found nothing. But the template is built from a mixture
of headings, and M1 says heading is what matters. A heading-weighted version — does P(match | θ)
track the similarity between a sector's heading distribution and the build set's? — is the sharper
test and has not been run.

### M6 — Per-camera SLEAP bias varying with viewing aspect

The 2D network may place keypoints systematically differently depending on the animal's apparent
aspect; as aspect changes with position, a 3D bias follows. Camera 2 (mean confidence 0.31 vs
0.49/0.49) would contribute this unevenly. **Test:** per-camera, per-keypoint residual vs aspect angle.

### A note on the per-rat offsets in §5

Each rat has its **own** `pc_weights` and `feature_means`, so the absolute level of μ is not
comparable across animals — R5's −18.7 vs R4/R6's ≈ +1 is a basis difference, not a biological one.
Only the **shape** of μ(θ) is comparable, and that is what agrees at r = 0.87–0.95.

---

## 8d. Mechanism resolved: SLEAP network bias (M6), not calibration

With the frame-basis fix, `triang_keys_3D` is confirmed to be **in the camera-calibration frame**
(~17 px reprojection), so DANNCE can be carried into the camera frame by a single Procrustes fit and
compared against the SLEAP 2D detections directly. `residual = detection − projection(DANNCE)`,
decomposed over ~0.6–1.0 M camera-keypoint observations per rat on rear-at-wall frames:

| component | what it means | R4 | R5 | R6 |
|---|---|---|---|---|
| median \|residual\| | — | 11.65 px | 10.90 px | 11.44 px |
| **[A]** common field vs image position | calibration / lens distortion / global misalignment | 5.21 | 5.24 | 4.72 |
| **[B]** keypoint-specific (after A) | network bias | 5.90 | 5.88 | 5.43 |
| **[C]** aspect-dependent, per keypoint | **M6 signature** | **7.95** | **9.36** | **8.19** |
| | **B/A** | 1.13 | 1.12 | 1.15 |
| | **C/A** | **1.52** | **1.79** | **1.73** |

**The aspect-dependent, keypoint-specific component is the largest — 1.5–1.8× the common
image-position field — in all three rats.** A calibration or distortion account predicts the
opposite (A dominant, B and C near zero). **M6 is supported; the calibration account is not.**

The most aspect-sensitive keypoints are the head and limb extremities — **EarR (47–53 px), EarL,
ShoulderL/R, SpineM, FootL, HipR, TailBase** — exactly the parts a 2D pose network mislocates when
self-occlusion changes with viewing angle.

**An unexpected corroboration of the referee.** The DANNCE projection explains the SLEAP 2D
detections **better than SLEAP's own 3D does**: median residual 10.9–11.7 px vs DANNCE, but
16.5–17.3 px vs `triang_3d`. A 6-camera reconstruction mapped in by a rigid transform beats the
3-camera reconstruction on its *own* detections. That independently supports using DANNCE as the
behavioural referee in §3.

---

## 8e. Per-rear quality — validated metrics, and quality does NOT mediate the sector effect

Per-sector statistics have only 11 points per rat. Attaching the corrected per-frame quality metrics
to each individual rear (`results/spatial_confound/rears_with_quality.csv`, n = 2,269–5,728 per rat
in the analysis set) gives far more power.

**(a) Quality weakly but reliably predicts whether the rule fired.** Spearman ρ with `matched`:

| metric | R4 | R5 | R6 |
|---|---|---|---|
| FOV margin at apex | **+0.182** (p 2e-19) | −0.024 (n.s.) | **+0.097** (4e-6) |
| n_cams conf≥0.5 | +0.084 (4e-5) | **+0.113** (1e-17) | +0.063 (3e-3) |
| mean confidence | +0.040 (0.05) | **+0.127** (5e-22) | +0.061 (4e-3) |
| reprojection residual | +0.042 (0.04) | **−0.081** (8e-10) | −0.040 (0.06) |

Real but small (|ρ| ≤ 0.18). For R4, matched rears have a median FOV margin of 143 px vs 107 px for
unmatched.

**(b) Quality strongly predicts the SLEAP–DANNCE discrepancy — the mechanistic link.** ρ with the
per-rear |SLEAP − DANNCE| PC residual:

| metric | R4 | R5 | R6 |
|---|---|---|---|
| **reprojection residual** | **+0.250** (8e-32) | **+0.379** (4e-178) | **+0.383** (4e-79) |
| mean confidence | −0.150 (3e-12) | −0.307 (2e-114) | −0.134 (2e-10) |
| n_cams conf≥0.5 | −0.140 (9e-11) | −0.271 (2e-88) | −0.009 (n.s.) |

This validates both sides: the corrected reprojection residual **is** measuring real tracking error,
and the SLEAP–DANNCE discrepancy tracks it. Neither could be shown on the misaligned v1 data.

**(c) But quality does not mediate the spatial effect — conditioning makes it *larger*.** Sector
log-odds contrast before → after adding all six quality metrics as covariates:

| rat | base | + quality | change |
|---|---|---|---|
| R4 | 1.269 | 1.739 | **+37%** |
| R5 | 1.711 | 2.810 | **+64%** |
| R6 | 1.236 | 1.354 | **+10%** |

Tracking quality is mildly *suppressing* the sector effect, not producing it. **§4's conclusion now
holds with per-rear power and validated metrics: the spatial reward gradient is not explained by
tracking quality**, even though tracking quality is real, measurable, and does drive the
SLEAP–DANNCE discrepancy.

**Where that leaves the mechanism.** The confound is a position-dependent *bias* arising from
aspect-dependent SLEAP keypoint error (§8d) — but its effect on reward is not captured by any scalar
quality summary, because what matters is the *direction* the bias pushes the PC trajectory relative
to the template corridor, not how large the error is.

---

## 8f. Mechanism of the aspect-dependence — and why it does NOT close the chain

### What causes the viewpoint-specific 2D error (stage 11)

Four accounts tested against the DANNCE reference reprojected into each SLEAP camera:

| account | prediction | result |
|---|---|---|
| **foreshortening** | error peaks end-on | **supported** — facing the camera 12.5–13.6 px (conf 0.605–0.625) vs oblique 10.1–11.4 px (conf 0.686–0.698) |
| **lateral self-occlusion** | far member of a bilateral pair worse | **supported** — median far/near ratio **1.26**, confidence 0.084 lower; ankle 1.44, foot 1.36, knee 1.36; ears 0.99 |
| **left/right swaps** | swapping reduces error | **rejected** — swap better in **1.9%** of frames; swapping triples-to-quintuples error |
| **aspect-dependent labels** | net is confidently wrong | **rejected** — ρ(conf, error) = **−0.32**, negative for all 23 keypoints; "virtual" vs "visible" landmark amplitude identical (6.31 vs 6.55 px) |

So the network is *appropriately uncertain* where it errs: this is **missing information**, not removable
bias — which is exactly why the corrector could not beat identity (§ CORRECTOR_V2_PLAN).

**Unexplained:** EarR has 2× the aspect amplitude of EarL (10.2 vs 4.9 px) and *no* near/far effect. A
symmetric mechanism cannot produce a left/right asymmetry, so something physically asymmetric
(headcap, tether, lighting) is likely. Worth inspecting frames.

### ❌ The propagation test fails: aspect error explains only ~20–25% of μ(θ)

The decisive test: measure the 2D error field in a **body-relative image frame** binned by (keypoint,
camera, viewing aspect) — all 828 cells populated, ≥100 obs each — then push it through the **actual**
triangulation Jacobian at each arena position and compare with the measured μ(θ).

A first attempt (`stage12b`) was invalid: it predicted for a *raw DLT* triangulation but compared
against μ(θ) measured on `triang_3d`, which has already been bone-regularised. Corrected to an
apples-to-apples, per-frame test (`stage12c` — same pipeline both sides, each frame's own DANNCE
pose, position and heading):

| rat | Spearman | Pearson | predicted range | measured range |
|---|---|---|---|---|
| R4 | +0.441 (p 0.15) | +0.378 | 3.6 | 13.6 |
| R5 | +0.133 (p 0.68) | +0.075 | 3.2 | 8.3 |
| R6 | +0.105 (p 0.75) | +0.269 | 3.0 | 15.4 |
| **pooled** | **+0.296 (p 0.080)** | | | |

**The sign is right and the pooled trend is marginal, but the amplitude is 3–4× short.** Re-measuring
the field with the confidence filter and outlier cut removed (admitting the failure tail) made it
*worse* (pooled ρ +0.185), so the shortfall is not an artifact of my own censoring.

**Conclusion: the aspect-dependent 2D error is real and well characterised, but it is not the main
driver of the spatial reward bias.** Roughly three-quarters of the amplitude comes from something else.

**The most likely gap.** The field was binned by aspect alone, but 2D error must also depend on
**posture** — a full rear and a crouch present completely different self-occlusion at the same viewing
aspect — and posture varies systematically by sector (apex_z spans 2.2–6.9 in R4). Conditioning the
field on pose as well as aspect is the next iteration. Until that is done, the causal chain from
camera geometry to reward asymmetry remains **open**.

---

## 9. Caveats

- **n = 3 rats.** Every cross-rat pattern is descriptive. The within-rat, within-behaviour designs in
  §3 carry the inferential weight.
- **DANNCE is the referee, not ground truth.** It has its own error. The §3 estimates are therefore
  lower bounds to the extent DANNCE error correlates with SLEAP error spatially. The referee
  validation (§3) is reassuring but not proof.
- **Nearest-neighbour "twins" are not identical behaviors.** Residual behavioural differences within a
  neighbourhood could contribute; the shuffle null controls sampling, not that.
- **d_dannce is a scalar** summary of a 30-frame 2-PC trajectory — the NN design (§3b) is the stronger
  of the two conditioning analyses for exactly this reason.
- **Sector −15° is sparse** in the analysis set because it sits inside the port-gate arc.
- **align_mse is template-dependent** and not comparable across rats or template epochs; only
  within-rat statements are made from it.
- Templates are **non-monotone** (R4 runs 1→2→refined→2→1→2→3); `template_file` is treated as a factor.

---

## 10. Recommendations

1. **Report and analyse per sector.** `multi_session` §1.8b/§1.8e split by port zone but not by
   sector; per-session match-rate trends partly track *where the rat stood*, and cross-session
   learning curves are contaminated by occupancy drift (severely so for R5, whose concentration
   nearly doubles).
2. **Fix the template builder anyway** (§6). Minimal change to
   `~/CLIRB/refine_templates_mult_session.py`: keep the largest cluster's mean as the template but
   compute `max_dev` bounds over a **sector-balanced** sample. Current bounds are 1.5–3× tighter than
   that, purely because the pool is spatially lopsided.
3. **Give the corrector a real success criterion.** The right target is *flattening the
   behaviour-conditioned sector effect in §3*, not the current F1 tables. That is now a single number
   per rat and directly measures what the corrector is for.
4. **Consider a per-sector bounds calibration** in the online rule — the residual field μ(θ) in §5 is
   measurable and stable enough to correct for directly.

---

## 11. Artifacts

```
clirb/events.py                   single source of truth (rear detector, matcher, gating, attempts)
clirb/templates.py                template loading + provenance
clirb/spatial/{geometry,quality,dual}.py
scripts/build_event_table.py      -> results/spatial_confound/rears.csv (16,315 rears)
scripts/build_quality_cache.py    -> data/cache_quality/ (152 sessions, per-frame metrics)
scripts/stage1_quality_maps.py    -> quality_by_sector_*.csv, stage1_results.json
scripts/stage2_template_provenance.py -> template_provenance.csv, stage2_clustering.csv
scripts/stage3_dannce_referee.py  -> rears_dannce.csv, residual_snippets_<rat>.npz
scripts/stage4_counterfactual.py  -> counterfactual_map.csv (FAILED validation), stage4_results.json
scripts/stage5_settling.py        -> occupancy/gate/winstay/template_swaps csv, stage5_results.json
scripts/stage6_synthesis.py       -> residual_field_<rat>.csv, h2_build_vs_best.csv, nn_matching.json
```

The notebooks have **not** yet been repointed at `clirb.events` — that is the remaining refactor step
(the module is verified equivalent, so it is mechanical).
