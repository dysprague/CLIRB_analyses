# Corrector v2 — plan (raw-triangulation input, balanced sampling)

Goal: train a SLEAP→DANNCE corrector on **uncorrected** input (re-triangulated from
`sleap_keys_2D`), with sampling balanced across rats/sessions/arena-sectors and enriched for
rears, then re-run the spatial-confound analysis on the corrected keypoints to see whether the
position-dependent reward bias shrinks.

Written 2026-08-06, before implementation. **Two premises did not survive checking — see §1.**

---

## 1. Pre-flight findings (these change the plan)

### 1.1 The 07_20 calibration is NOT better for post-07_20 sessions

All post-07_20 sessions on disk are saved with `2026_05_04`, as you said. But swapping in
`2026_07_20` does not improve anything (15 sessions, R4/R5/R6, pre and post):

| | median self-consistency reprojection | median bone-length CV |
|---|---|---|
| **pre**-07_20 sessions, 05_04 cal | **9.24 px** | 0.263 |
| **pre**-07_20 sessions, 07_20 cal | 9.57 px | 0.262 |
| **post**-07_20 sessions, 05_04 cal | **9.60 px** | 0.262 |
| **post**-07_20 sessions, 07_20 cal | 9.85 px | 0.263 |

05_04 is *slightly better* in **both** eras, by the same ~0.3 px margin. Agreement with the
independent 6-camera DANNCE is a wash — median |SLEAP−DANNCE| after Procrustes is 9.6–11.6 mm and
the winner alternates by session (05_04 better on R5/07_21; 07_20 better on R4/07_21, R6/07_22;
split on pre-era sessions).

**Interpretation.** If a camera had physically moved on 07_20, the stale 05_04 calibration would be
markedly *worse* on post-07_20 sessions. It isn't — the pre→post degradation is 0.36 px with 05_04,
i.e. nothing. There is no evidence the cameras moved.

### 1.2 The two calibrations are the same solution in a different world frame

- **Intrinsics are byte-identical**: `|K_0504 − K_0720|_F = 0.00` for all three cameras. Only the
  extrinsics differ, so 07_20 was an extrinsics-only re-solve, not a full recalibration.
- Extrinsics differ by a **~179° rotation** and a **~2.1–2.3 m camera-centre shift** — i.e. a
  world-origin/orientation redefinition, roughly a flip.

### 1.3 They do NOT share a world frame — but are related by one fixed similarity transform

Triangulating the same 2D with both calibrations:

| session | raw \|X_0504 − X_0720\| | after 7-DoF fit: scale | residual |
|---|---|---|---|
| R5/2026_07_21_1 | 606.4 mm | 0.9924 | **0.92 mm** |
| R4/2026_07_21_1 | 624.1 mm | 0.9959 | **0.40 mm** |
| R6/2026_07_22_1 | 623.8 mm | 0.9955 | **0.44 mm** |
| R5/2026_06_23_1 | 608.9 mm | 0.9955 | **0.45 mm** |

So the reconstructions are **identical in shape to sub-millimetre**, differing by a rigid transform
plus a **0.5% scale**. Mixing sessions across calibrations is therefore trivial: estimate the
transform once (it is session-independent) and map everything into one frame. The 0.5% scale is the
only substantive difference and is worth resolving against DANNCE's metric scale.

### 1.4 Why the frames differ: the world origin was flipped, the cameras did not move

Relative camera geometry is frame-independent, so it isolates real movement from bookkeeping:

| pair | baseline 05_04 | baseline 07_20 | relative-rotation difference |
|---|---|---|---|
| cam0–cam1 | 1904.1 mm | 1899.5 mm | 0.58° |
| cam0–cam2 | 1893.1 mm | 1888.2 mm | 0.46° |
| cam1–cam2 | 1815.8 mm | 1804.6 mm | 0.40° |

Baselines agree to 4.6–11.2 mm out of ~1.9 m; inter-camera angles to < 0.6°. **The cameras did not
move.**

And the world-frame change `W = R_c2ᵀ·R_c1` is the *same* for all three cameras — 178.799° /
178.930° / 178.804° about axis ≈ **[0, 0, −1]**, agreeing with each other to 0.40–0.58°. The camera
centres confirm a pure yaw flip, (x, y, z) → (−x, −y, z), with Z unchanged (≈ −400 mm in both).

**Cause:** the world frame is defined by the calibration target, not the cameras. A 180° yaw flip is
the signature of the **checkerboard origin being detected at the opposite corner** — a symmetric
board is ambiguous under 180° rotation about its normal, and a board lying flat on the arena floor
has a vertical normal. Equivalent: the board was physically placed rotated 180°.

### 1.5 Scale: 05_04 is metrically correct, 07_20 is ~0.4% small

Robust (trimmed) Procrustes from each SLEAP triangulation to the independent 6-camera DANNCE,
15 sessions:

| | Procrustes scale to DANNCE | \|1 − s\| | bone-length ratio to DANNCE |
|---|---|---|---|
| **05_04** | **1.0006** | **0.0006** | **0.9945** |
| 07_20 | 1.0040 | 0.0040 | 0.9925 |

**05_04 is already metric** (scale 1.0006 against DANNCE); 07_20 needs a 0.4% stretch. Both measures
agree, and the numbers are internally consistent: `s_0504 / s_0720 = 0.99675` vs the 0.99626
predicted independently from the camera baselines — agreement to 0.05%.

*(An earlier non-robust run appeared to favour 07_20; that was one degenerate Procrustes fit —
scale 0.081 — dragging the median. Trimmed fitting removes it.)*

**This also explains why 07_20 reprojects worse (§1.1).** Intrinsics were byte-identical, so only
extrinsics were re-solved — and that solve came out ~0.4% small. With focal lengths unchanged but
baselines 0.4% short, the three rays no longer intersect as cleanly, which is exactly the ~0.25 px
reprojection penalty seen in every one of 15 sessions. In a checkerboard calibration a pure scale
error of this kind almost always traces to the **assumed square size** (e.g. 30.0 mm entered for a
board that is 29.9 mm, or a different board used). Combined with the 180° origin flip, both symptoms
point to the 07_20 calibration having been run with a differently-placed and differently-specified
target — not to anything having changed about the rig.

**Consequence for the plan: use 05_04 throughout.** It is metrically correct, fits marginally
better, and needs no cross-frame mapping. See Q1.

---

## 1.6 DECISIONS LOCKED (2026-08-06) — all questions resolved

| | decision |
|---|---|
| **Q1 calibration** | `2026_05_04` for every session. Metrically correct (scale 1.0006 to DANNCE); cameras never moved. |
| **Q2 balance** | Balanced set across time. **Superseded by the fine-tune requirement below.** |
| **Q3 templates** | Do **both** — (a) reuse existing templates = corrector-only gain; (b) rebuilt sector-balanced templates = theoretical ceiling. |
| **Q4 architectures** | **All three** — `temporal_mlp`, `velacc_mlp`, and a new **aspect-aware** variant; compare. |
| **Q5 intent** | Scoped experiment, but the inference path must be clean enough to deploy online later. |

### The fine-tuned SLEAP model changes the input pipeline

The SLEAP network was fine-tuned on 2026-07-21 but only deployed online on 2026-07-29. So the stored
`sleap_keys_2D` comes from the **old** model for all but 16 of 163 sessions (R4 6, R5 8, R6 2). The
corrector must operate on **fine-tuned** outputs, so 2D must be **re-inferred from video**.

Model: `/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/models/260721_165454.single_instance.n=9975`
(single-instance, 23 parts, `input_scaling` 0.5, colour input).

**Feasibility established:**

- SLEAP 1.6.3 / **sleap-nn 0.2.0 on PyTorch 2.12 + CUDA** (no TensorFlow). The legacy Keras `.h5`
  weights load into the PyTorch backbone — *"Successfully mapped 22/22 … all weight assignments
  verified"*. Inference runs at **165 FPS** on the RTX 4070 SUPER.
- ⚠️ **`--frames` is broken** in this build (`'<' not supported between instances of 'str' and 'int'`
  → 0 frames predicted). Selected frames must be written to a clip instead.
- ⚠️ **Re-encoding must be lossless.** Lossy mp4v vs lossless: median 0.51 px, **p95 10.1 px, max
  88 px** — not acceptable. Lossless **FFV1** and a PNG directory agree **bit-exactly**
  (max diff 0.0000 px), and FFV1 runs 3.6× faster than PNG (165 vs 46 FPS). → **use FFV1 `.avi`
  clips**, chunked and deleted after inference (~1.2 MB/frame; 1.4 TB free).
- Video I/O: random seek costs **5.5 s**, sequential decode runs at **500 FPS**. → decode
  sequentially, never seek. Cost ≈ full decode of 3 videos per session ≈ 3.6 min + ~6 GB SMB read.
- Per-camera frame indices differ by up to 2 frames, so use `cam{c}_frame` for camera *c*, **not**
  `cam0_frame` for all three.

### ⚠️ Fine-tune leakage — must be excluded

`labels_gt.train.slp` holds **8,977** labeled frames and `labels_gt.val.slp` **998**, drawn from
**276 videos = 92 sessions × 3 cameras spanning the whole date range** (not just 07-21). SLEAP has
effectively memorised those frames, so its error there is unrepresentative. **Exclude every
fine-tune labeled (video, frame) from the corrector's training and evaluation sets.**

### Temporal models need contiguous frames

`temporal_mlp` uses a 5-frame causal window, so isolated frames are unusable. **Selection unit is a
contiguous 16-frame segment** (~0.8 s), stratified by its midpoint. The first 4 frames of each
segment serve as context only; frames 5–16 are training targets. This also makes decoding cheaper.

**Scale chosen:** 12 sessions/rat = 36 sessions × ~200 segments × 16 frames ≈ **115k frames**
(~86k usable targets), ~3 h of data build. Assumption: this is enough for a 2-layer MLP corrector —
the previous corrector used more frames but far less diverse sampling.

---

## 2. Data construction (the genuinely new part)

Per session, on the **processed-frame basis** (which is what `sleap_keys_2D` uses):

1. `k2 = sleap_keys_2D` — drop the zero-padded tail (`K = len(frame_mapping)`, e.g. 32,160/36,000).
2. **Re-triangulate** `k2` with the session's calibration → raw, uncorrected 3D.
   *(This is the `quality.undistort_all` + `triangulate_batch` path; it reprojects at ~9 px and has
   bone CV ≈ 0.26 vs 0.118 for the stored corrected 3D — i.e. genuinely uncorrected, as intended.)*
3. **DANNCE target**: `dannce_keys_3D[dannce_idx_for_sleap_cams_keys[k]]` — verified this is the
   processed-basis index, and equals `dannce_idx_for_sleap_cams[cam0_frame[k] + 1]` (the documented
   off-by-one).
4. Map into a common world frame (§1.3 transform if mixing calibrations).
5. Per-session 7-DoF Procrustes SLEAP→DANNCE on a **sector-balanced** frame sample (not the first
   5 min as the current `data_world.py` does — a time-based epoch is spatially biased).
6. Drop sessions whose Procrustes residual exceeds a threshold (existing pipeline uses 60 mm).

This input is also **more deployable than the current corrector**: it consumes exactly what the rig
has at inference time (2D detections → raw triangulation), whereas the existing corrector takes
`sleap_keys_3D`, which is already bone-regularised.

---

## 3. Balanced sampling

The point of balancing here is not just class balance — a **sector-balanced** training set is
precisely the fix implied by the spatial-confound result, since an unbalanced set would teach the
corrector the occupancy prior (R5 spends 41% of its rears in one sector).

Proposed strata, sampled to equal quota within each level:

| dimension | levels | rationale |
|---|---|---|
| rat | R4, R5, R6 | equal, so no rat dominates |
| calibration era | pre / post 07_20 | post is only ~7–9 of ~50 sessions per rat → floor, not equality (see Q2) |
| session | cap per session | stops long/numerous sessions dominating |
| **arena sector** | 12 × 30° | **the key one** — kills the occupancy prior |
| posture | non-rear / rear / high-rear | enrichment |

Enrichment target (rears are ~5–15% of frames naturally): **40% non-rear, 35% rear, 25% high-rear**,
where "high rear" = top apex-height tertile within rears. Sector-balanced *within* each posture
stratum.

Gating: drop frames with non-finite triangulation, missing DANNCE, or |SLEAP−DANNCE| beyond a
robust outlier cut.

Target size ~0.5–1 M frames (≈4.4 M available), which trains comfortably on the RTX 4070 SUPER.

---

## 4. Training

Reuse `corrector/models.py` and the existing recipe (MSE + 0.1 × bone-length loss, Adam, wd 1e-5,
batch 4096, early stop at 8 epochs without val improvement).

- **Baseline:** `temporal_mlp` (5-frame causal, world) — the prior best base.
- **Second:** `velacc_mlp` — vel/acc features gave the largest Phase-B gains (R2 +0.06, R3 +0.04 F1
  groupO).
- **Split by SESSION**, never by frame, to avoid leakage.

Given §8d (the error is aspect-dependent and concentrated in ears/shoulders/feet), a variant that
gets the animal's **heading relative to each camera** as an input feature is the theoretically
motivated addition — see Q4.

---

## 5. Evaluation — including the re-run you asked for

**Tier 1, keypoint level:** kp MSE, per-keypoint MSE (does it fix EarR/EarL/shoulders/feet?),
bone-length CV, on held-out sessions.

**Tier 2, the spatial-confound re-run** — this is the real success criterion:

| metric | current value | success = |
|---|---|---|
| NN-matched sector spread in P(match) (§3b) | R5 0.75, R6 0.37 | shrinks |
| within-quality-bin sector variance (§3a) | R4/R5/R6 0.0125/0.0248/0.0067 | shrinks |
| μ(θ) range as % of template tolerance (§5) | 12–24% | shrinks |
| M6 aspect-dependent component C/A (§8d) | 1.52 / 1.79 / 1.73 | → toward ≤1 |
| overall P(match \| rear) | 0.44 / 0.41 / 0.33 | should not collapse |

**⚠️ A real design problem here — see Q3.** Tier 2 needs corrected poses expressed in the *template's*
PC space. The templates were fit on `triang_keys_3D`-space egocentric poses; the corrector outputs
DANNCE-space poses from a differently-framed raw triangulation. The egocentric transform cancels
translation and yaw but **not** scale, pitch or roll, so this mapping needs to be done explicitly
and validated (e.g. confirm that uncorrected raw-triangulation poses pushed through the same path
reproduce the existing rear/match counts before trusting the corrected ones).

---

## 6. Sequencing

| | step | output |
|---|---|---|
| 1 | fix the inter-calibration transform; build `corrector/data_raw2d.py` | paired (raw-3D, DANNCE) loader |
| 2 | sanity: uncorrected poses through the full path reproduce known rear/match counts | validates Tier 2 |
| 3 | balanced sampler + dataset build | frame index + stats |
| 4 | train baseline + velacc | checkpoints |
| 5 | Tier 1 eval | per-keypoint table |
| 6 | Tier 2 re-run | the answer |

---

## 7. Questions before I start

**Q1 — Calibration. ✅ RESOLVED 2026-08-06: use `2026_05_04` for every session.** It is metrically
correct (§1.5), fits marginally better (§1.1), and the cameras never moved (§1.4). The
cross-calibration mapping and the "calibration era" stratum both drop out of the pipeline.

**Q2 — Training-time balance** *(reframed now that Q1 is settled: this is no longer about
calibration, but the stratum is still worth keeping for a different reason)*. Behaviour drifts
strongly over training — R5's rear-location concentration nearly doubles (R 0.39 → 0.71) — so a
sample dominated by late sessions would teach the corrector late-training postures and positions.
Proposal: balance across **early / mid / late thirds of each rat's session sequence**, proportional
with a floor rather than strictly equal. OK?

**Q3 — Templates for the Tier-2 re-run.** To ask "did reward bias improve", corrected poses must go
through a template. Options: **(a)** map corrected poses back into the `triang_3d` frame and reuse
the existing per-session templates — measures the effect on the *deployed* rule, but the frame
mapping is a validation risk; **(b)** rebuild templates from corrected data (sector-balanced, per
the §12 recommendation) — cleaner, but then "improvement" conflates the corrector with a better
template. I lean (a) as primary and (b) as a follow-up. Preference?

**Q4 — Architecture scope.** Just the two (temporal_mlp, velacc_mlp), or should I also try an
aspect-aware variant (heading-relative-to-camera as input), which §8d motivates directly? The latter
is a new model class, so more work but the most likely to actually fix the confound.

**Q5 — Downstream intent.** Is corrector v2 meant to eventually *replace* `triang_keys_3D` for all
analysis and/or run online, or is this a scoped experiment to test whether correction removes the
spatial bias? That changes how much I invest in the inference path.
