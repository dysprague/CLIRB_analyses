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

⚠️ **Read §8i too.** `triang_3d`/"SLEAP" for essentially every session before 2026-07-29 was passed
through an online, R1/R2/R3-trained correction network before being stored — not raw SLEAP
triangulation. §8c's bone-length-bundle-adjustment explanation is retracted (that code never runs).

⚠️ **And §8j.** §8f's "aspect explains only 20-25%, three-quarters unexplained" was measured on a
fine-tuned research model's 2D output, not the real production SLEAP, and on 1/15th the data. Redone
on the real thing, aspect explains roughly half to four-fifths of the amplitude and the pooled
correlation is significant. Don't quote the 20-25% figure.

⚠️ **And §8k.** §8g/§8h's posture/limb-config verdicts were also on the small fine-tuned dataset.
Retested on the real data: scalar posture gives a real (if rat-dependent) improvement over aspect
alone; limb-configuration clustering is actively harmful, not merely unhelpful. Don't quote §8g/§8h's
numbers as final.

⚠️ **And §8l — a mitigation candidate, not just more mechanism (but see §8m/§8n/§8o).** Dropping the
four limb keypoints cuts the sector-bias range by ~35-38% in all 3 rats — this part holds up.
§8l's specific claim that body rotation angle beats everything else does **not** survive a
full-scale (~172-session) replication using `multi_session.ipynb`'s own methodology — there, axial
keypoints wins and body rotation angle alone is the *worst* performer. §8n explains why (a
structural bias in how ground truth gets built, not a simple "which number is right"). Don't cite
body rotation angle as a recommendation without reading §8n.

⚠️ **And §8o.** §8m's "axial keypoints wins" comparison scored axial on 1 PCA component against
full23's 2 — an unfair dimensionality gap, not a fair test. Redone with a proper 2-component axial
variant (`axial2`), the result is a wash (axial2: precision 0.825 vs full23's 0.838; recall 0.835
vs 0.762), not a win. Describe axial keypoints as "comparable to full23, trading precision for
recall," not as an outright winner — see §8o before citing §8m's numbers.

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
| **F8** `F8_posture_by_sector.png` | posture (rear-extent) index by sector, all 3 rats (§8g) |
| **F9** `F9_propagation_before_after.png` | predicted vs measured μ(θ), aspect-only vs aspect+posture (§8g) |
| **F10** `F10_field_aspect_vs_posture.png` | why posture-conditioning didn't help: the 2D field is aspect-structured, not posture-structured (§8g) |
| **F11** `F11_method_comparison.png` | aspect-only vs +scalar-posture vs +limb-configuration propagation, side by side (§8h) |
| **F12** `F12_original_sleap_correction.png` | propagation test redone on real production SLEAP 2D, ~15× the data (§8j) — supersedes the "20-25%" figure |
| **F13** `F13_original_posture_comparison.png` | posture/limb-config retested on real SLEAP data (§8k) — scalar posture helps, limb config hurts |
| **F14** `F14_axial_sector_bias.png` | sector-bias range, full 23kp vs axial 9kp only (§8l) — ~35-38% reduction |
| **F15** `F15_p_rewarded_by_sector.png` | P(rewarded\|DANNCE rear) by sector, 4 feature spaces (§8l) — superseded, see §8m/§8n |
| **F16** `F16_p_rewarded_spread_summary.png` | sector-spread and pooled P(rewarded) summary bars (§8l) — superseded, see §8m/§8n |
| **F17** `F17_overlap_overall.png` | aggregate + per-session-over-time precision/recall, with sample size (§8m) |
| **F18** `F18_overlap_by_sector.png` | precision/recall by sector × rat, DANNCE-ground-truthed (§8m) |
| **F19** `F19_overlap_feature_comparison.png` | 6 feature spaces compared at full scale, incl. fair 2-PC axial2 — axial2 is a wash with full23, not a win; bodyrot alone loses (§8m/§8o) |

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

**❌ RETRACTED — see §8i.** `correct_triangulations()` is a dead stub that returns its input
unchanged; `optimize_with_bones()` is never called anywhere in the repository. Bone-length bundle
adjustment cannot be why `triang_3d` is rigid. The real explanation, and a previously-unstated
premise affecting the entire document, is in §8i.

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

**❌ RETRACTED — see §8j.** This entire propagation test (and stage11, and the field it's built on)
used 2D keypoints from a **fine-tuned research model**, re-inferred on a small curated frame subset
for the `corrector_v2` project — not the actual production SLEAP model, on adequate sample size, that
generated the online rule's real behavior. Redone on the real thing at ~15× the data, aspect alone
explains roughly half to four-fifths of the amplitude, not one-quarter, and the pooled correlation is
significant (ρ = +0.46, p = 0.004). "Roughly three-quarters of the amplitude comes from something
else" does not survive.

**The most likely gap.** The field was binned by aspect alone, but 2D error must also depend on
**posture** — a full rear and a crouch present completely different self-occlusion at the same viewing
aspect — and posture varies systematically by sector (apex_z spans 2.2–6.9 in R4). Conditioning the
field on pose as well as aspect is the next iteration. Until that is done, the causal chain from
camera geometry to reward asymmetry remains **open**.

---

## 8g. Posture-conditioning tested and REJECTED as the missing ~75%

§8f's leading suspect for why the aspect-only propagation test predicts only
20-25% of the measured μ(θ) amplitude was posture: "a full rear and a crouch
present completely different self-occlusion at the same viewing aspect, and
posture varies systematically by sector." That is now tested directly.

**Method.** The 2D error field (stage12a) was re-measured binned by
(keypoint, camera, aspect, **posture**) instead of aspect alone — posture =
max keypoint height above SpineM on the DANNCE pose, in the calibration frame
(the same body-relative quantity as `geometry.apex_height`, referenced to
SpineM so it does not depend on a session's absolute floor level). Tertile
bins (49-73 / 73-110 / 110-158 mm) fit on the pooled distribution: n=91,317
frames, 15 sessions, 2,469/2,484 (kp, cam, aspect, posture) cells with ≥50
observations (`stage12d_measure_2d_field_posture.py` →
`field2d_posture.npz`). The same apples-to-apples, per-frame propagation test
as §8f (stage12c) was re-run looking up the posture-specific field cell
(`stage12e_propagate_posture.py`).

**Result: no improvement — if anything, slightly worse.**

| | pooled Spearman ρ | R4 | R5 | R6 | predicted range |
|---|---|---|---|---|---|
| aspect only (stage12c) | +0.30 (p=0.08) | +0.44 (p=0.15) | +0.13 (p=0.68) | +0.10 (p=0.75) | 3.0-3.6 |
| aspect + posture (stage12e) | +0.19 (p=0.27) | +0.32 (p=0.31) | +0.27 (p=0.39) | **-0.15** (p=0.65) | 2.7-4.4 |

(measured range is unchanged, 8.3-15.4, since only the prediction side changed.)
R6 flips sign; the pooled correlation, already only marginal, gets weaker; the
predicted amplitude is still ~3-4 units against a measured 8-15 — the same
3-4× shortfall as before. **Posture, as operationalized here, does not close
the gap** (F9).

**Why not, mechanistically (F8, F10).** The premise holds up: posture really
does vary by sector — 25-53 mm across sectors per rat, with a similar shape
across all three rats (F8), the same rig-level signature seen in μ(θ) itself
(§5). But the 2D error field this pipeline measures barely depends on it:
pooled over all keypoints and cameras, median |2D bias| spans **3.0 px across
the 12 aspect bins** (4.4-7.4 px) but only **0.4 px across the 3 posture
bins** (5.65-6.09 px, F10). The aspect-dependent SLEAP error is, empirically,
almost aspect-*only* — conditioning on this posture index gives the
propagation step almost no extra structure to exploit. Correction to §8f:
posture was the leading candidate for the missing amplitude; it is now ruled
out, at least as operationalized here (a single scalar body-height-above-
SpineM proxy, tertile-binned).

**What is NOT ruled out.**
- A richer posture representation — limb configuration rather than body
  height (how many paws are on the wall vs. floor, forelimb extension) —
  could still carry occlusion-relevant information this scalar misses.
- The gap could sit outside the 2D-error-and-triangulation-geometry chain
  entirely: the bone-length bundle adjustment applied to `triang_3d` (§8c)
  redistributes error non-linearly in a way a linearized (Jacobian)
  propagation cannot capture, or there is a residual calibration/lens effect
  not visible in the DANNCE-vs-SLEAP reprojection comparison.
- Sample size: 15 sessions split three ways over posture bins is enough for
  the field to be well-estimated on average (2,469/2,484 cells ≥50 obs) but
  not obviously enough to rule out a subtler, non-tertile-shaped posture
  effect.

**Status: the causal chain from camera geometry to reward asymmetry remains
open.** The confound itself (§3) and its rig-level signature (§5) are solid;
the search for the last ~75% of the aspect-mechanism's proximate cause (§8f)
is not resolved by this test.

Scripts: `scripts/stage12d_measure_2d_field_posture.py` →
`results/spatial_confound/field2d_posture.npz`;
`scripts/stage12e_propagate_posture.py` →
`propagation_matched_posture.csv`, `stage12e_results.json`. Figures:
`scripts/make_figures_stage12_posture.py` → F8-F10.

---

## 8h. Limb-configuration posture tested — same result as §8g

§8g's own "not ruled out" list named the obvious next try: "a richer posture representation — limb
configuration rather than body height." Tested directly.

**Method.** Instead of a scalar height, posture = a KMeans cluster label (k=4) fit on the
**egocentric pose shape** — `skeleton.normalize_skeleton_batch` applied to the DANNCE pose (removes
translation and yaw, exactly the representation the template builder itself already uses for shape
clustering elsewhere in this codebase), flattened to 69-dim, z-scored, clustered on 30,439 pooled
frames (cluster sizes 6,047–9,529, reasonably balanced). The 2D error field was re-measured binned by
(keypoint, camera, aspect, **cluster**) — 3,208/3,312 cells with ≥50 obs
(`stage12f_measure_2d_field_limbconfig.py` → `field2d_limbconfig.npz`) — and the same apples-to-apples
propagation test re-run assigning each prediction frame to the nearest centroid of its own egocentric
shape (`stage12g_propagate_limbconfig.py`).

**Result: same as scalar posture — no improvement (F11).**

| conditioning | pooled ρ | R4 | R5 | R6 | predicted range |
|---|---|---|---|---|---|
| aspect only | +0.30 (p=0.08) | +0.44 | +0.13 | +0.10 | 3.0-3.6 |
| + scalar posture (§8g) | +0.19 (p=0.27) | +0.32 | +0.27 | -0.15 | 2.7-4.4 |
| + limb configuration | +0.25 (p=0.14) | +0.26 | +0.18 | -0.05 | 2.7-3.9 |

Limb configuration lands between the other two — marginally better pooled ρ than scalar posture, still
worse than aspect-only, still not significant, R6 still wrong-signed, predicted range still ~3-4 units
against a measured 8-15. **No conditioning variant tried so far explains the gap.**

**Same diagnosis as §8g, slightly softer.** Marginal range of the measured 2D bias: aspect 3.0 px,
scalar posture 0.4 px, limb-configuration clusters **0.8 px** (medians 5.55-6.34 px across the 4
clusters). Richer than a scalar, but still an order of magnitude below aspect's own structure — there
still isn't much for the propagation step to use. A 4-cluster split of ~30k frames is a coarse
description of "limb configuration"; a finer one risks the same data-sparsity problem already
flagged in §8g.

**Status unchanged: the causal chain remains open.** Two independent, reasonably-motivated posture
descriptors have now been tried and both fail to move the needle. Whatever explains the missing ~75%
is either (a) a posture signal neither of these representations captures, (b) a mechanism outside the
2D-error/triangulation-geometry chain (see §8i), or (c) beyond what a linearized, per-keypoint-Jacobian
propagation can express regardless of the conditioning variable.

Scripts: `scripts/stage12f_measure_2d_field_limbconfig.py` → `field2d_limbconfig.npz`;
`scripts/stage12g_propagate_limbconfig.py` → `propagation_matched_limbconfig.csv`,
`stage12g_results.json`. Figure: F11 (`make_figures_stage12_posture.py`).

---

## 8i. Correction to §8c: bone-length bundle adjustment never runs — it's an online corrector network

While testing whether §8c's bone-length bundle-adjustment guess explains the remaining
propagation gap, direct inspection of `~/campy-CLIRB/campy/behavior.py` shows §8c's "likely missing
step" was wrong, and finds something that changes a premise running through this entire document.

**`correct_triangulations()` is a dead stub:**

```python
def correct_triangulations(points_3d, P_list, undist_pts, edges, bone_length_avg, w_bone=1.0):
    return points_3d  # TODO: add triangulation corrections
```

`optimize_with_bones()` — the real `scipy.optimize.least_squares` bone-length bundle adjustment §8c
described — is fully implemented but **called from nowhere in the repository** (confirmed by
`grep -rn "correct_triangulations\|optimize_with_bones"` across the whole codebase: only their own
`def` lines match). Bone-length bundle adjustment cannot explain `triang_3d`'s low bone-length CV —
that code path never executes.

**What actually runs, read directly from the real-time loop** (`behavior.py`, the frame-processing
loop):

```python
keypoints_3D, undist_pts = triangulate(keypoints_2D, P_list, dist_coefs, K_list)
if corrector is not None:
    new_points_3d = corrector.correct(keypoints_3D, keypoints_2D, peak_vals)
else:
    new_points_3d = keypoints_3D
```

`corrector` is a **neural network** (`campy/corrector_online.py`) — a temporal MLP trained to map
SLEAP-space poses toward DANNCE-space — loaded from a checkpoint path set per-session in
`sleap/campy_config.yaml`'s `corrector_ckpt` field. That field, checked across every R4/R5/R6
session on disk:

| era | dates | `corrector_ckpt` | sessions |
|---|---|---|---|
| main analysis window | 2026-06-09 → 2026-07-27/28 | `R1R2R3_temporal_mlp_2d_reproj_v4_dn2sl.pt` (**active**) | 177 |
| everything since | 2026-07-29 → 2026-08-28 | `null` (disabled) | growing |

**`triang_3d`/`triang_keys_3D`, for essentially this entire project, was never raw SLEAP
triangulation.** It is SLEAP triangulation passed through a correction network trained on a
*different cohort (R1/R2/R3)*, applied online, out-of-distribution, to R4/R5/R6. Every place this
document says "SLEAP" for a pre-2026-07-29 session means "SLEAP, corrected by a cross-cohort MLP" —
a previously unstated premise.

**Natural experiment: does the corrector explain the rigidity?** Median bone-length CV
(`clirb/spatial/quality.py: bone_length_cv`, same 23-edge skeleton as everywhere else in this
project), 5 corrector-ON sessions vs. 5 corrector-OFF sessions per rat, all from the local cache:

| rat | corrector ON | corrector OFF |
|---|---|---|
| R4 | 0.095 | 0.195 |
| R5 | 0.099 | 0.214 |
| R6 | 0.089 | 0.188 |
| **pooled** | **0.095** | **0.196** |

Every single session, both groups, same direction, ~2× gap, zero overlap. **The corrector network —
not bone-length bundle adjustment — is why `triang_3d` is rigid.** (It's not quite as rigid as the
0.118 originally quoted in §8c for the full 151-session cohort — that number pools sessions this
5-vs-5 sample doesn't exactly match — but the on/off gap itself is unambiguous.)

**Does removing the corrector change the spatial reward gradient? Attempted — confounded and
inconclusive.** The corrector-OFF sessions (2026-08-05 onward) also run under **newer templates**
(`R4_template_4`, `R5_template_5/6`, `R6_template_4` — not seen anywhere in the June-July analysis
window) built from data collected *without* the corrector. Any ON-vs-OFF sector-pattern difference is
therefore at least as attributable to template-provenance concentration (already established in
§2/§6 as a real, if not primary, mechanism) as to the corrector itself. Descriptively: R5's
corrector-OFF match rate collapses toward zero in most sectors while staying comparable to its
ON-era value specifically in its historically-dominant sector (+45°) — exactly the H2 signature
already described in §2/§6, not independent evidence for a corrector-driven account. **A clean
version of this test needs a session where the corrector is toggled with the template held fixed,
which does not exist in the current dataset.**

**Consequence for §8d-8g.** Stage 11/12's aspect-dependent 2D error field is measured from raw 2D
detections vs. the DANNCE reprojection (`corrector_v2/keys2d`, upstream of any 3D correction step),
so those findings stand. But the **target** that propagation is trying to reach — the true,
operationally-relevant μ(θ) of §5 — is measured on the corrector-adjusted `triang_3d`. The ~75% of
its amplitude that aspect (§8f), scalar posture (§8g), and limb-configuration posture (§8h) all fail
to reach may include the correction network's *own* idiosyncratic, position-dependent
generalization error — trained on R1/R2/R3, applied out-of-distribution to R4/R5/R6 — layered on top
of the raw 2D detection bias. Nobody in this project had reason to consider this mechanism until the
`corrector_ckpt` field turned up. **Not yet tested.**

**Next, more direct test:** run the correction network offline, out-of-band, on a sector-balanced
sample of raw triangulations, and measure *its own* output bias by sector the same way §5 measures
μ(θ) — this isolates how much of the true μ(θ) the correction step itself contributes, independent of
whatever the raw SLEAP detections were already going to produce.

Scripts: `scripts/stage13_corrector_natural_experiment.py` → `corrector_onoff_bone_cv.csv`;
`scripts/stage13b_corrector_onoff_sectors.py` → `corrector_onoff_rears.csv`.

---

## 8j. Retry on the real production SLEAP 2D: the "20-25%" conclusion does not survive

Prompted by a direct question: stage11 and stage12a-c (the entire aspect-mechanism chain, §8d-8f)
measured "the 2D error field" from `results/corrector_v2/keys2d/*.npz` — 2D keypoints from a
**fine-tuned research model**, re-inferred by the `corrector_v2` project on a small, curated,
balanced subset of frames (v2_01/02_*.py) — never the actual production SLEAP model that ran online
and produced the real reward-rule behavior for R4/R5/R6. The real production 2D lives in
`sleap/sleap_keys_2D.npy` per session.

**The frame-basis gotcha, confirmed correct.** `sleap_keys_2D.npy` is dense over the **processed-frame
counter** (row k = the k-th successfully-processed frame; padded past that count) — a different basis
from `triang_keys_3D.npy`'s cam0-acquisition-frame counter (bridged by `frame_mapping.csv`'s
`cam0_frame` column, per §8c). The bridge actually needed here is a different one: `aligned_data.npz`'s
`dannce_idx_for_sleap_cams_keys`, which is verified — both by direct row comparison against
`corrector_v2`'s own 'k' field, and by the Procrustes-residual check below — to share that SAME
processed-frame-counter basis: `dannce_idx_for_sleap_cams_keys[k]` gives the DANNCE frame for
`sleap_keys_2D[k]`. `corrector_v2`'s 'k' field was already using this bridge correctly (confirmed by
comparing its fine-tuned keypoint values against `sleap_keys_2D[k]` at the same k — close, as expected
for two models on the same frame) — so there was no NEW frame-alignment bug in stage11/12; the issue
was purely the choice of 2D source and how much of it was used.

**A real bug found and fixed along the way.** Rebuilding the field on all processed frames (not the
curated subset) initially gave nonsense — Procrustes residuals of 600-900 mm on every single session,
despite the underlying triangulated poses being individually well-behaved. Root cause: a
`fit_procrustes(A, B)` call had `A`/`B` swapped relative to the convention used everywhere else in this
project (`robust_proc(X, tgt)` instead of `robust_proc(tgt, X)`), which flips which direction
`.apply()` maps and silently produces a garbage-but-plausible-looking transform when applied backwards.
Confirmed by direct comparison: the correct-direction fit gives ~9-11 mm median residual (matching
`corrector_v2`'s own baseline); the swapped direction gives 600-900 mm on the identical data. Fixed in
`stage14a_original_sleap_field_and_dataset.py`.

**Rebuilt at real scale.** Field measured on all processed, DANNCE-dual frames across the same 14
sessions used in §8d-8h (one dropped for a missing file) — **540,259 frames, ~15× the 36,674 used
before** — Procrustes residuals 9.0-11.0 mm per session (consistent with `corrector_v2`'s own
baseline), field coverage 824/828 (keypoint, camera, aspect) cells with ≥100 obs.

**Result: aspect alone now explains far more, and the pooled correlation is significant (F12).**

| rat | Spearman ρ (fine-tuned, n=36,674) | Spearman ρ (original SLEAP, n=540,259) | predicted range | measured range |
|---|---|---|---|---|
| R4 | +0.44 (p 0.15) | +0.35 (p 0.27) | 4.4 | 8.6 |
| R5 | +0.13 (p 0.68) | +0.48 (p 0.11) | 4.2 | 5.2 |
| R6 | +0.11 (p 0.75) | **+0.79 (p 0.002)** | 4.7 | 9.8 |
| **pooled** | +0.30 (p 0.080) | **+0.46 (p 0.004)** | | |

Predicted-to-measured range now runs roughly 50-80% per rat (was 20-25% pooled) and the pooled
correlation clears significance for the first time in this mechanism chain. R6 in particular goes from
the weakest (p=0.75) to the strongest (p=0.002) individual result.

**Why the earlier estimate was so far off — two compounding factors, not one.** (1) *Sample size*:
the previous test estimated each sector's median from as few as ~200-800 frames drawn from a small
curated pool; with 1,400-71,000 frames per sector now available, sector-median estimates are far less
noisy, which alone shrinks the apparent measured range (R4: 13.6 → 8.6) independent of any mechanism.
(2) *Model identity*: the fine-tuned model was fine-tuned specifically to agree with DANNCE, so its
residual against DANNCE is smaller, more zero-centered, and less structured — measured Δ PC0 with the
fine-tuned model spans both signs (-6 to +9, F12 left); with the original, un-fine-tuned production
model it is **uniformly positive** across every sector of every rat (0 to +14, F12 right) — a real,
substantial, consistently-signed bias, not noise around zero. The aspect mechanism was always going to
explain more of a *real, structured* bias than of a smaller, denoised one.

**What this changes.** §8d-8f's qualitative mechanism characterization (foreshortening, self-occlusion,
confidence tracks the error honestly) was not re-derived on the original 2D and likely still holds
qualitatively, but its specific numbers (bilateral near/far ratios, aspect-profile amplitudes) have not
been independently re-verified at this sample size. §8g/§8h's posture and limb-configuration tests were
run on the same smaller fine-tuned dataset as §8f and **have not been retested** on the original SLEAP
data — given how much the baseline moved, they should not be treated as settled until they are.
§8i's corrector finding is unaffected (it concerns 3D pose provenance, not 2D source).

**Revised status.** The aspect-dependent SLEAP 2D-detection mechanism, propagated through real
triangulation geometry, is now the best-supported explanation for the *majority* of μ(θ)'s amplitude,
not a fifth of it. The remaining ~20-50% gap is far more modest than "roughly three-quarters," and
posture, limb configuration, and the online corrector (§8i) all remain plausible contributors to what's
left — but none of them needs to explain nearly as much as previously thought.

Scripts: `scripts/stage14a_original_sleap_field_and_dataset.py` → `field2d_original.npz`,
`dataset_original.npz`; `scripts/stage14b_propagate_original.py` → `propagation_matched_original.csv`,
`stage14b_results.json`. Figure: F12 (`make_figures_stage12_posture.py`).

---

## 8k. §10's recommendations 1 and 3, done: mechanism confirmed at scale; posture retested with a real split verdict

One combined pass over the real production 2D (`stage15_original_sleap_deep_dive.py`) redid both:
stage11's mechanism characterization (recommendation 3) and the posture/limb-configuration
propagation tests (recommendation 1), reusing the same 14 sessions, DANNCE alignment, and corrected
Procrustes direction validated in §8j.

### Mechanism characterization — confirms §8d exactly, nothing changes

| test | fine-tuned model (old) | real SLEAP, 540k frames (new) |
|---|---|---|
| H-OCC median far/near ratio | 1.26 | **1.34** |
| H-SWAP: frac. frames swap is better | 1.9% | **1.2%** |
| SHAPE side/end ratio | foreshortening-like | **0.91 — foreshortening-like** |
| H-LAB median ρ(confidence, error) | −0.32 | **−0.25** (negative for all 23/23 keypoints) |
| virtual vs. visible aspect amplitude | 6.31 vs. 6.55 px (~equal) | 6.45 vs. 8.01 px (still no support for H-LAB) |

Every qualitative conclusion replicates: bilateral occlusion is real and strongest in Foot/Shoulder/
Elbow/Ankle/Hip (Ear ≈ 1.0, symmetric as expected); L/R swaps are decisively rejected; the aspect
profile is foreshortening-like (error peaks end-on, not side-on); and confidence tracks error
negatively for every single keypoint — the network is honestly uncertain, not confidently wrong. The
fine-tuned-model characterization in §8d was already sound; this was a confirmation, not a correction.

### Posture retest — a real, but split, verdict (F13)

| conditioning | pooled ρ | R4 | R5 | R6 |
|---|---|---|---|---|
| aspect only | +0.46 (p 0.004) | +0.35 (p 0.27) | +0.48 (p 0.11) | +0.79 (**p 0.002**) |
| + scalar posture | **+0.50 (p 0.002)** | **+0.61 (p 0.036)** | +0.34 (p 0.29) | +0.64 (p 0.026) |
| + limb configuration | +0.07 (p 0.69) | +0.44 (p 0.15) | −0.16 (p 0.62) | −0.02 (p 0.95) |

Unlike §8g/§8h's fine-tuned-model retest (where *both* posture descriptors failed uniformly), the two
diverge sharply on real data:

**Scalar posture (height above SpineM) gives a real, if rat-dependent, improvement.** Pooled ρ rises
to the best result in this entire chain (+0.50, p=0.002). R4 crosses into significance for the first
time (+0.35 → +0.61, p=0.036). R5 and R6 both decline somewhat (R6 stays significant at p=0.026), so
the net pooled gain is modest, not uniform — but it is a real, positive contribution this time, not
the null result §8g reported.

**Limb-configuration clustering (KMeans-on-egocentric-shape, k=4) is actively harmful at scale.**
Pooled ρ collapses to +0.07 (n.s.), driven by R5 and R6 flipping to small negative correlations —
destroying R6's previously-strongest result in this entire investigation. Field coverage was
comparable to the posture field (3,200/3,312 vs. 2,437/2,484 cells ≥50 obs), so this isn't a coverage
artifact: the clustering itself is introducing noise, or capturing structure in the ~60,000-frame
fitting sample that doesn't generalize to prediction-time frames.

**Revised status.** Body height is a small, genuine contributor beyond aspect alone — not the
dominant missing piece §8f originally went looking for, but not nothing. Limb-configuration
clustering, at least as operationalized here, should be dropped rather than iterated on; the simpler
descriptor beats the richer one. The remaining gap (aspect+posture pooled ρ=+0.50 still leaves
~50% of variance unexplained by rank correlation, and R5 in particular stays non-significant
throughout) keeps recommendations 2 (source-level correction) and 6/7 (isolate the corrector,
confirm with a clean natural experiment) live.

Scripts: `scripts/stage15_original_sleap_deep_dive.py` → `field2d_original_posture.npz`,
`field2d_original_limbconfig.npz`, `aspect_profile_by_keypoint_original.csv`,
`aspect_mechanism_by_keypoint_original.csv`, `stage16_results.json`;
`scripts/stage15b_propagate_original_posture.py` → `propagation_matched_original_posture.csv`;
`scripts/stage15c_propagate_original_limbconfig.py` → `propagation_matched_original_limbconfig.csv`.
Figure: F13 (`make_figures_stage12_posture.py`).

---

## 8l. A mitigation, not just a mechanism: keypoint/feature selection for the online rule

§8h's per-keypoint `aspect_amp` table suggested a practical question distinct from "what causes
the confound": could **restricting which keypoints/features feed the online rule** reduce it
directly, as a deployable mitigation? Two tests, both on the original production SLEAP 2D.

### (a) Sector-bias remeasurement with an axial-only keypoint subset

Dropping all four limbs (Shoulder/Elbow/Wrist/Hand, Hip/Knee/Ankle/Foot) and keeping only the 9
axial/core keypoints with the lowest `aspect_amp` (Snout, EarL, EarR, SpineF, SpineM, SpineL,
TailBase, HipL, HipR), a fresh 1-component PCA fit on a sector-balanced DANNCE sample correlates
0.93-0.95 with the real production PC0 (confirms it still captures rearing) — computed entirely
from the already-local `dataset_original.npz`, no new SMB reads. Comparing the raw SLEAP−DANNCE
sector-bias range (same PC0-like units, so directly comparable to the full-23kp baseline, F14):

| rat | full 23 kp | axial 9 kp only | reduction |
|---|---|---|---|
| R4 | 121.9 | 76.9 | 37% |
| R5 | 133.3 | 82.3 | 38% |
| R6 | 224.3 | 146.5 | 35% |

A consistent **~35-38% reduction in raw sector-bias magnitude**, in all three rats, from dropping
the limbs alone. Two hand-crafted single features were also tried on the same data — snout height
(egocentric z above SpineM) and a body-rotation angle (SpineF-SpineM vs. SpineM-TailBase) — but
these are in different units (mm, degrees) from PC0, so their raw ranges aren't directly comparable
to the baseline; see (b) for the fair, common-currency comparison.

### (b) P(rewarded): does this survive contact with the actual online-rule mechanics?

Defined precisely: for a rear **independently, unbiasedly detected by DANNCE** (DANNCE PC0
calibrated onto the raw-SLEAP PC0 scale via a sector-balanced **linear regression** — not just an
additive offset, since raw/uncorrected SLEAP has a substantially wider PC0 range than DANNCE;
sign/validity confirmed against an independent height proxy, r ≈ ±0.86-0.88, consistent direction
in all 3 rats), what fraction does the SLEAP-side trajectory *also* match, per the real
hysteresis-detect + 30-frame-corridor-with-`accepted_errors` mechanics? Three example sessions per
rat (already validated in §8j/8k), full per-session trajectories re-triangulated from
`sleap_keys_2D.npy` this time (not pooled sector samples), 170-328 DANNCE-detected rears per rat.

**The fair comparison fits a fresh template + bounds for every candidate — including the full
23-keypoint one — from the same procedure on the same raw-SLEAP data**, so the comparison isolates
feature selection alone rather than conflating it with "freshly-fit beats historically-fit" (an
earlier pass that compared against the *historical* production bounds directly is kept separately
below, because it turned up something else worth reporting).

| feature | R4 P / spread | R5 P / spread | R6 P / spread |
|---|---|---|---|
| full 23 kp (fresh-fit) | 0.96 / **0.194** | 0.92 / 0.143 | 0.87 / **0.500** |
| axial 9 kp | 0.99 / 0.083 | 0.93 / 0.169 | 0.95 / 0.121 |
| snout height | 0.98 / 0.118 | 0.96 / 0.143 | 0.97 / 0.111 |
| **body rotation angle** | **0.99 / 0.059** | **0.99 / 0.111** | **0.99 / 0.048** |

**Body rotation angle (the SpineF-SpineM vs. SpineM-TailBase angle) wins outright** (F15, F16):
lowest sector-spread in *all three* rats simultaneously, while also matching the highest or
tied-highest overall P(rewarded). It's the single spine-curl number, computed straight from 3D
geometry — no PCA, no keypoint list to argue about, and by construction rotation-invariant, so it
doesn't inherit the camera-aspect coupling that limb keypoints (and, to a lesser extent, the
axial-PCA composite) carry. Axial keypoints and snout height both help R4 and R6 (R6's baseline
spread of 0.500 is partly small-n noise — some sectors have only 2 events in this 3-session
sample — but even so neither axial nor snout height matches bodyrot's consistency), and are roughly
a wash for R5.

**An unplanned confirmation of §8i.** Scoring raw/uncorrected SLEAP against the REAL, historically-fit
production bounds (calibrated on corrector-adjusted data) rather than a freshly-fit one gives P(rewarded)
of just 0.06 / 0.36 / 0.33 (R4/R5/R6) — raw SLEAP alone barely passes the real tolerance band at all.
That the online corrector is doing serious work to make SLEAP poses tight enough to match is
independent evidence for §8i's finding, from a completely different angle.

**Caveats.** Only 3 sessions/rat (170-328 events) — some per-sector cells are small (n=2 in a few
R6 sectors); the reference/bounds for each candidate is fit from (mostly) the same event pool being
scored, so absolute P values are optimistic in-sample estimates — but this affects all four
candidates equally, so the *relative* ranking (which is what matters here) should be robust. A
proper held-out evaluation, and running this on the full 14-session pool, is the natural next step
before treating body rotation angle as a real production recommendation rather than a promising lead.

Scripts: `scripts/stage17a_sector_bias_axial_and_features.py` → `sector_bias_feature_comparison.csv`
(figure F14, `make_figure_stage17a.py`); `scripts/stage17b_p_rewarded_by_feature.py` →
`p_rewarded_by_feature.csv` (figures F15-F16, `make_figures_stage17.py`).

**⚠️ Superseded by §8m/§8n.** §8l's ranking (body rotation angle wins) does not survive a
larger-scale, more rigorous replication using `multi_session.ipynb`'s own overlap methodology —
see below. Read §8n before citing body rotation angle as a recommendation.

---

## 8m. Redone at full scale, the notebook's own way — the ranking flips

§8l's ranking was measured on 3 sessions/rat with templates fit and scored on the same small pool.
Redone properly: `multi_session.ipynb` §2.1's own overlap logic (SLEAP matched against the real
production template; DANNCE compared via *rebuilt* templates), extended with an explicit
DANNCE-ground-truthing step, run across **all ~172 dual-modality sessions** instead of 3.

**Per session:**
1. SLEAP matches against the real production template, bounds × **1.25** (candidates only).
2. Rebuild a DANNCE-only template from DANNCE's values at those Pass-1 SLEAP-match windows
   (bounds = std(SLEAP seg) + std(DANNCE seg), same formula §2.1 already uses). Rerun matching in
   DANNCE space against it → the ground-truth anchor set (`dannce_matches_gt`).
3. For each of 5 feature spaces (full 23kp/2-PC baseline, axial 9kp PCA, snout height, body
   rotation angle, snout+bodyrot combined): rebuild *both* a SLEAP and a DANNCE template from the
   two modalities' values at those ground-truth windows (same summed-std bounds formula). Score
   each modality independently against its own rebuilt template.
4. Pair the two match-index lists within 10 frames (§2.1's own `get_match_overlap`) → SLEAP-only
   (FP), DANNCE-only (FN), both (TP).

`accepted_errors` scales with feature count to preserve relative tolerance:
`round(1.5 × n_features)` — 2 for the four single/double-feature-but-small spaces, 3 for the
2-feature baseline and the snout+bodyrot combo (matches the real rule's 3-of-60 ≈ 5% convention).

**Baseline (full 23kp) accuracy is now believable, not inflated** (F17): precision 0.79–0.87,
recall 0.74–0.79 across the three rats, pooled over 64,776 events. F17 also plots, per session,
the total ground-truth rear count and total SLEAP-match count behind every precision/recall point
— several early sessions (e.g. R4's first few) run on samples as small as 5-15 events, which is
worth keeping in mind when reading the noisiest stretches of the time series.

**Real sector-dependence survives this methodology too** (F18): every rat shows genuine
sector-to-sector swings in both precision and recall (e.g. R6's precision drops to ~0.59 in one
sector), confirming the confound isn't an artifact of any one scoring method.

**The feature-space ranking flips** (F19):

| feature | precision | recall |
|---|---|---|
| full 23kp (baseline) | 0.838 | 0.762 |
| **axial 9kp** | **0.914** | **0.882** |
| snout height | 0.888 | 0.814 |
| **body rotation alone** | **0.690** | **0.632** |
| snout + bodyrot | 0.828 | 0.734 |

Axial keypoints is now the clear winner; body rotation angle alone is now the *worst* performer,
consistently across all three rats on both metrics — the opposite of §8l. Adding body rotation to
snout height also makes that combination worse than snout height alone.

Scripts: `scripts/stage18_dannce_ground_truth_overlap.py` → `stage18_events.csv`,
`stage18_session_summary.csv`; figures F17-F19 (`make_figures_stage18.py`).

---

## 8n. Why the ranking flips: a structural bias in how "ground truth" gets built, not a bug

Three hypotheses, tested directly (`scripts/stage18b_diagnose_disagreement.py`):

**A — sample size/independence.** Partially responsible for §8l's numbers being *unreliable*
(3 sessions, in-sample fit-then-score), but doesn't by itself explain a *ranking reversal*.

**B — bounds instability, ruled out.** Session-to-session coefficient of variation of the rebuilt
template's mean level and bounds width is **no worse for body rotation than for axial or snout
height** (bounds-width CV 0.115-0.201 for bodyrot vs. 0.115-0.214 for axial, 0.114-0.261 for
snout_z). Median ground-truth events per session is 37-86, not tiny. This is not the explanation.

**C — selection-criterion mismatch, confirmed as the driver.** Correlation of each feature with
the production PC0 (the quantity that selects the ground-truth set at *every* stage of this
pipeline):

| feature | R4 | R5 | R6 |
|---|---|---|---|
| snout height | +0.906 | +0.945 | +0.921 |
| body rotation | **−0.453** | **−0.572** | **−0.564** |

Snout height is nearly redundant with PC0 (r≈0.9+) — unsurprising, since PC0 largely *is* "how
reared-up is the animal." Axial keypoints correlates similarly well (0.93-0.95, §8h). **Body
rotation captures a genuinely different, much more weakly-related axis of posture** (r²≈0.2-0.33
with PC0). Since the ground-truth anchor set at every stage is selected by PC0-based matching, it
systematically favors whichever candidate feature is redundant with PC0 — the "good" windows PC0
selects are, almost by construction, also good windows for snout height and axial keypoints, but
not particularly good or bad windows for body rotation specifically. Distribution checks (skew
0.4-1.0, no wraparound: <2% of frames near 0°, 0% near 180°) rule out a pathological bodyrot
distribution as a contributing factor.

**The honest conclusion is not "stage18 is right, §8l was wrong" — it's that the two tests measure
different things.** §8l's ground truth (a pure DANNCE height-threshold, never touching SLEAP or
PC0) is structurally *fairer* to a deliberately orthogonal feature like body rotation. §8m/§2.1's
ground truth (built by bootstrapping through PC0-based matching at every stage) structurally
rewards features redundant with PC0. Neither is "the" answer for "is body rotation a good
online-rule feature" in isolation — a clean test would need ground truth that doesn't route through
PC0 at all, scored with body rotation's own independently-built template and bounds (i.e. §8l's
design, but at §8m's scale and with a proper held-out split). That test has not yet been run.

**Practical takeaway:** axial keypoints looks robust either way (wins or ties in both tests) and is
the safer near-term recommendation. Body rotation angle is not yet disqualified — it failed a test
that structurally disadvantages orthogonal features — but it needs the PC0-independent, at-scale
test described above before being recommended for anything.

**`multi_session.ipynb` updated to match.** §2.1's `analyse_dual_session` (and its markdown header)
now implements the three-stage rebuild described in §8m — SLEAP Pass 1 (widened bounds) → rebuild
DANNCE-only, get the ground-truth match set → rebuild *both* SLEAP and DANNCE from that set → score
independently. Same return-dict keys as before (plus one new one, `dannce_matches_gt`) so
downstream §2.2/§2.3 cells are unaffected. Only cells 47-48 changed; verified against a backup with
`nbformat.validate`.

Scripts: `scripts/stage18b_diagnose_disagreement.py` → `diag_feature_pc0_correlation.csv`,
`diag_template_stability.csv`, `diag_bodyrot_distribution.csv`.

*(Note: this diagnostic script calls `fit_axial_basis`/`feature_traj` with the pre-§8o
single-component calling convention and has not been rerun since §8o's signature change — treat
its numbers above as frozen at the time they were computed, not as reproducible by re-running the
script as-is.)*

---

## 8o. A real confound caught by inspection, not by a test: axial was 1-PC, full23 was 2-PC

Before trusting §8m/§8n's "axial keypoints wins" result, a direct question exposed an asymmetry
none of the prior analysis had flagged: full23 scores each session on **2** principal components
(`pcs_to_use = (0, 1)`, confirmed by direct code inspection to be exactly `(0, 1)` for every one of
the 17 curated §8k/§8l/§8m sessions and all 172 dual-modality §8m/§8n sessions — no exceptions
found), while axial's PCA (`fit_axial_basis`) fit only **1** component. More scoring dimensions
generally means a *harder* corridor-matching test (more independent constraints to satisfy
simultaneously) but §8n's `accepted_errors` scaling (`round(1.5 × n_features)`) doesn't fully
compensate for that — so axial's apparent dominance in §8m was at least partly an artifact of it
having an easier (lower-dimensional) test, not evidence the axial keypoints themselves are more
informative.

**Fix:** added `axial2`, a proper 2-component axial PCA (`AXIAL_VARIANTS = {'axial': 1, 'axial2':
2}` in `stage18_dannce_ground_truth_overlap.py`), fit the same way as `axial` (sector-balanced
pooled fit per rat, each component independently sign-oriented to correlate positively with the
matching real PC — component *k* against `pcs_to_use[k]`), with `accepted_errors` scaled the same
way (3, matching full23's). Reran all ~172 dual-modality sessions with 6 feature spaces.

| feature | precision | recall |
|---|---|---|
| full 23kp (2-PC baseline) | 0.838 | 0.762 |
| axial 9kp (1-PC, unfair) | 0.914 | 0.881 |
| **axial 9kp (2-PC, fair)** | **0.825** | **0.835** |
| snout height (1-PC) | 0.888 | 0.814 |
| body rotation alone (1-PC) | 0.690 | 0.632 |
| snout + bodyrot (2-PC) | 0.828 | 0.734 |

**The fair comparison is a wash, not a win.** Matched PC-for-PC against full23, axial2 has
*slightly worse* precision (0.825 vs 0.838) and *better* recall (0.835 vs 0.762). Most of axial
(1-PC)'s apparent dominance in §8m (0.914/0.881 vs full23's 0.838/0.762) was the 1-PC-vs-2-PC
dimensionality gap, not the axial keypoint subset being more informative than the full 23. This
doesn't overturn §8n's separate finding about body rotation (that one is about *which* feature the
PC0-based ground truth structurally favors, orthogonal to this dimensionality point) — but it does
mean **axial keypoints should be described as "roughly comparable to full23, with a
precision/recall trade toward recall," not as an outright winner.**

Single-feature spaces (snout height, body rotation) are unaffected by this issue — they were always
correctly compared 1-PC-worth-of-information vs full23's 2, which is a fair apples-to-apples-per-PC
comparison, not a fair total-information comparison; that asymmetry is inherent to comparing a
1-dimensional trajectory against a 2-dimensional one and doesn't have a clean fix short of also
giving snout height/bodyrot a paired second dimension (e.g. snout+bodyrot, which already exists as
the 2-feature combination and is comparable to full23 and axial2 as-is).

Figure F19 updated to show all 6 feature spaces with axial and axial2 side by side and a divider
marking the "fair, 2-PC-matched" group.

Scripts: `scripts/stage18_dannce_ground_truth_overlap.py` (now `FEATURES = ['full23', 'axial',
'axial2', 'snout_z', 'bodyrot', 'snout_bodyrot']`) → `stage18_events.csv` (73,727 events),
`stage18_session_summary.csv`; figure `make_figures_stage18.py` → F19.

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

*(Revised 2026-09-02, after §8l. Former items are done — see §8k/§8l — and are kept below,
struck through, for the record; everything else is reordered by priority given what §8l found.)*

1. ~~Retest posture and limb configuration on the real production SLEAP 2D.~~ **Done, §8k.** Split
   verdict: scalar posture gives a real, rat-dependent improvement (pooled ρ +0.46→+0.50, R4 becomes
   significant); limb-configuration clustering is actively harmful (pooled ρ collapses to +0.07,
   destroying R5/R6's signal) and should not be pursued further in its current form.
2. ~~Redo stage11's mechanism characterization on the real dataset.~~ **Done, §8k.** Every qualitative
   conclusion replicates at 540k frames: occlusion ratio 1.34 (was 1.26), swap-improves-fit in only
   1.2% of frames (was 1.9%), foreshortening-like aspect profile, confidence negatively tracks error
   for all 23/23 keypoints. No changes needed to §8d's story.
3. ~~Validate body rotation angle at full scale.~~ **Done, §8m — reversed.** At full scale
   (~172 sessions), axial keypoints wins and body rotation angle alone is the *worst* performer.
   §8n traces this to a structural bias (ground truth is built via PC0-based matching at every
   stage, which favors PC0-redundant features) rather than settling which number is "right" —
   axial keypoints is the safer recommendation; body rotation angle needs a PC0-independent,
   at-scale, held-out test before being recommended for anything. **Caveat, §8o:** the "axial
   wins" framing itself scored axial on 1 PC vs full23's 2 — an unfair dimensionality gap.
   PC-matched (axial2), it's a wash, not a win (see item 4).
4. **Prototype axial keypoints (9kp, dropping all 4 limbs) as the online feature — but budget for
   "roughly comparable to full23," not a guaranteed win.** §8o's PC-matched comparison (axial2,
   2 components, same as full23's PC0+PC1) shows precision 0.825 vs full23's 0.838 (slightly worse)
   and recall 0.835 vs 0.762 (better) — a real precision/recall trade, not the clear dominance
   §8m's 1-PC-vs-2-PC comparison suggested. Still worth prototyping (it wins or ties on every test
   run so far, §8h/§8l/§8m/§8o, and the recall gain may matter more than the precision cost for an
   online reward rule), but design the pilot to detect a wash, not to confirm a win. Pair with a
   broader source-level correction using the (keypoint, camera, aspect, posture) field as a
   complementary, more general fallback for whatever axial keypoints alone doesn't catch (aspect
   explains 50-80% of μ(θ)'s amplitude, posture adds a further real increment — §8j, §8k).
4b. **Run a PC0-independent, full-scale, held-out test of body rotation angle** before ruling it
   out — §8l's design (DANNCE height-threshold ground truth, never touching SLEAP/PC0) was
   structurally fairer to it than §8m's; it just needs §8m's sample size and a proper train/test
   split.
5. **Report and analyse per sector.** `multi_session` §1.8b/§1.8e split by port zone but not by
   sector; per-session match-rate trends partly track *where the rat stood*, and cross-session
   learning curves are contaminated by occupancy drift (severely so for R5, whose concentration
   nearly doubles).
6. **Ship a per-sector bounds calibration** in the online rule now — the residual field μ(θ) in §5 is
   measurable and stable enough to correct for directly, and doesn't need items 3-4 finished first.
7. **Fix the template builder anyway** (§6). Minimal change to
   `~/CLIRB/refine_templates_mult_session.py`: keep the largest cluster's mean as the template but
   compute `max_dev` bounds over a **sector-balanced** sample. Current bounds are 1.5–3× tighter than
   that, purely because the pool is spatially lopsided.
8. **Isolate the online corrector's own contribution to μ(θ)** (§8i). Run the R1/R2/R3-trained
   `corrector_ckpt` MLP offline on a sector-balanced sample of raw triangulations and measure its
   output bias by sector directly. R5 stays non-significant through every conditioning tried so far
   (aspect, +posture, +limb config) — the corrector is now the leading candidate specifically for
   whatever is different about R5. §8l's finding that raw SLEAP barely passes the *real* production
   bounds (P=0.06-0.36) independently confirms the corrector is doing real work.
9. **Give `corrector_v2` a real success criterion.** The right target is *flattening the
   behaviour-conditioned sector effect in §3*, not the old F1 tables. (This is `corrector_v2`, the
   offline post-hoc project — see [[clirb-corrector-v2]] — not the online `corrector_ckpt` of §8i,
   a separate, older, R1/R2/R3-trained model that was already active for essentially the whole cohort.)
10. **Promote the §8j/§8k/§8l data pipeline into `clirb/` as a documented, reusable module.** Getting this
   right required two non-obvious fixes — sourcing 2D from `sleap_keys_2D.npy` (not
   `corrector_v2/keys2d`) via `dannce_idx_for_sleap_cams_keys`, and a `fit_procrustes(A, B)` argument
   order that silently produces a plausible-looking but wrong transform if swapped. Both are exactly
   the kind of mistake that's easy to reintroduce in a future one-off script; worth a shared,
   tested `clirb.spatial.original_sleap` module rather than leaving `stage14a`/`stage15` as the only
   record.
11. **Get a clean corrector on/off natural experiment.** The current one (§8i) is confounded by a
    concurrent template change. Worth doing for completeness alongside item 8.

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
scripts/stage11_aspect_mechanism.py     -> aspect_mechanism_by_keypoint.csv (§8d)
scripts/stage12a-c_*.py                 -> field2d.npz, propagation_matched.csv (§8f, aspect only)
scripts/stage12d/e_*_posture.py         -> field2d_posture.npz, propagation_matched_posture.csv (§8g)
scripts/stage12f/g_*_limbconfig.py      -> field2d_limbconfig.npz, propagation_matched_limbconfig.csv (§8h)
scripts/stage13_corrector_natural_experiment.py -> corrector_onoff_bone_cv.csv (§8i)
scripts/stage13b_corrector_onoff_sectors.py     -> corrector_onoff_rears.csv (§8i, confounded)
scripts/stage14a_original_sleap_field_and_dataset.py -> field2d_original.npz, dataset_original.npz (§8j, 540k frames)
scripts/stage14b_propagate_original.py          -> propagation_matched_original.csv (§8j — supersedes stage12c)
scripts/stage15_original_sleap_deep_dive.py     -> field2d_original_posture/limbconfig.npz, aspect_mechanism_by_keypoint_original.csv (§8k)
scripts/stage15b/c_propagate_original_*.py      -> propagation_matched_original_posture/limbconfig.csv (§8k)
scripts/stage17a_sector_bias_axial_and_features.py -> sector_bias_feature_comparison.csv (§8l)
scripts/stage17b_p_rewarded_by_feature.py       -> p_rewarded_by_feature.csv (§8l)
scripts/make_figures_stage12_posture.py -> F8-F13
scripts/make_figure_stage17a.py, make_figures_stage17.py -> F14-F16
scripts/stage18_dannce_ground_truth_overlap.py  -> stage18_events.csv, stage18_session_summary.csv (§8m)
scripts/stage18b_diagnose_disagreement.py       -> diag_feature_pc0_correlation/template_stability/bodyrot_distribution.csv (§8n)
scripts/make_figures_stage18.py -> F17-F19
multi_session.ipynb §2.1 (cells 47-48)          -> UPDATED 2026-09-03 to the §8m three-stage rebuild
```

The notebooks have **not** yet been repointed at `clirb.events` — that is the remaining refactor step
(the module is verified equivalent, so it is mechanical).
