# Spatial confound in template matching — full analysis plan

**Question.** *Are the same behaviors rewarded differently in different parts of the arena?*

The animals localize their rearing to one region. Is that natural settling on a solution, or is it
shaped by a systematic, position-dependent bias in SLEAP tracking that makes the online template
rule fire more readily in some places than others?

Written 2026-08-05. Supersedes `sleap_dannce_error_analysis.py` (the R1/R2/R3 session-scalar
ancestor of this analysis). All pilot numbers below were computed from the local cache during
planning; scripts under `scratchpad/` are throwaway — the real versions are specified in §5.

---

## 1. Executive summary

A pilot over the full locally cached cohort (15,642 rears, 142 sessions, R4/R5/R6) establishes:

1. The arena is a **circle, R ≈ 332 mm**, and the animals are **always at the wall** (median COM
   radius 310 mm; 96% of rears at r > 250 mm). The spatial coordinate that matters is **perimeter
   angle θ**, not a 2-D grid.
2. **P(match | rear) varies about 2× around the perimeter** — a 0.20–0.23 spread in P per rat.
3. **The entire spatial effect flows through the observed PC-space alignment.** Conditioning on
   align-MSE quartile nearly flattens it. So sector acts on the matcher by moving where the rear
   lands in *observed* PC space — which is `true pose + tracking error`, and SLEAP alone cannot say
   which.
4. **The templates were built from matches concentrated in one sector.** R5's templates 3/4/5 were
   built 95%/95%/100% from matches in sector (0°,45°] — the exact sector R5 later occupies and where
   it has its highest match rate. R6's template 3 was built in sectors it does *not* occupy, and R6
   has the lowest match rate of the three.

5. **The template-building procedure makes that mechanical.** `~/CLIRB/refine_templates_mult_session.py`
   pools widened-band matches, runs `KMeans(k=10)` **on trajectory shape alone**, takes the
   **largest cluster**, and sets bounds to the max deviation *within* it. Nothing balances across
   the arena. If position modulates shape, the clusters are partly sectors, "largest cluster" means
   "most-occupied sector," and the bounds tighten around that sector's biased trajectory.

Points 4–5 were not anticipated and reframe the project: there is a plausible feedback loop in which
the template acquisition procedure *imports* a location-specific tracking bias and then rewards it.
Both are testable from data already on disk — the template `.npz` provenance fields plus a re-run of
the builder's own clustering — with no DANNCE and no new infrastructure.

The plan therefore tests three hypotheses (§2), builds the infrastructure to do it properly (§5–6),
and runs six analysis stages (§7–12) ending in a counterfactual map: **P(match | identical behavior,
sector)**.

---

## 2. Hypotheses

**H0 — Natural settling.** The rat converges on a location for reasons unrelated to tracking (wall
support, proximity to the port, thigmotaxis, idiosyncratic preference). Reward follows the animal;
the spatial reward gradient is a *consequence* of where it practices, because practice makes its
rears there more stereotyped.

**H1 — Static tracking field.** SLEAP's 3-camera reconstruction has a position-dependent bias μ(θ)
and noise Σ(θ) — from occlusion, camera geometry, or FOV coverage at rear apex. This creates a
reward gradient independent of behavior, which then reinforces occupancy.

**H2 — Template-import loop.** *(new; suggested by the pilot, then confirmed as mechanically
possible by reading the builder)* The template is built and refined from matches that occurred
wherever the rat happened to be standing, so it encodes **that sector's tracking bias** as if it
were part of the target behavior. Rears in the build sector match better; rears elsewhere match
worse. Occupancy → template built there → gradient favouring there → occupancy reinforced.

This is **structural, not accidental**. `~/CLIRB/refine_templates_mult_session.py` refines a template
by: pooling widened-band matches (`WIDEN_FACTOR = 2.5`) across a hand-listed set of sessions →
`KMeans(n_clusters=10, random_state=0)` on the z-scored, flattened `(N, 30, 2)` PC windows → taking
the **largest cluster** → per-frame mean as the new template, with bounds
`band = max(mean − min, max − mean)` over that cluster (`REFINED_BOUNDS_MODE = 'max_dev'`).

The clustering is on **trajectory shape alone**. If position modulates shape — which is precisely
the confound under test — then the KMeans clusters *are partly sectors*, "largest cluster" means
"most-occupied sector," and `max_dev` bounds over a spatially homogeneous cluster are tight around
that one sector's biased trajectory. Nothing in the procedure balances across the arena. So H2 does
not require anything to have gone wrong; it is what this recipe does whenever B ≠ 0.

### Discriminating predictions

| observation | H0 | H1 | H2 |
|---|---|---|---|
| spatial gradient survives conditioning on DANNCE-measured behavior | no | **yes** | **yes** |
| gradient peak coincides with best truth-free tracking quality (conf, reprojection, FOV) | no | **yes** | not necessarily |
| gradient peak coincides with the **template build sector** | no | not necessarily | **yes** |
| gradient moves when the template is rebuilt elsewhere | no | no | **yes** |
| occupancy concentrates *before* the template is built there | **yes** | no | no (after) |
| same gradient shape in all three rats | no | **yes** | no (rat-specific) |
| align-MSE in a sector improves with practice in that sector | **yes** | no | no |

H1 and H2 are not exclusive — H2 is a mechanism by which a modest H1 field gets amplified into a
large reward gradient. The realistic outcome is a mixture, and the deliverable is the mixing
proportions, not a verdict.

---

## 3. Estimand and causal model

For rear event *i* at perimeter angle *θ*:

- `b_i` — true 30-frame pose trajectory (unobserved)
- `z_i = f(b_i) + ε(θ_i)` — the SLEAP PC trajectory the online rule sees; `ε` has position-dependent
  bias `μ(θ)` and covariance `Σ(θ)`
- `T` — the template + `bounds_to_use`, themselves estimated from a spatially biased sample of past `z`
- `R_i = 1[rule(z_i; T) fires] ∧ 1[port gate passes]`

Observed spatial variation in P(R | θ) decomposes into:

| | source | status |
|---|---|---|
| **A** | behavior composition — P(b \| θ) differs | H0; a real behavioral effect, not nuisance |
| **B** | tracking bias — μ(θ), Σ(θ) | H1; **the confound** |
| **B′** | template–location coupling — T fit where ε(θ_build) applied | H2; **the confound, amplified** |
| **C** | task-rule geometry — the `min_port_distance` gate | known by design; annotate + exclude |
| **D** | data loss — dropped frames / NaN triangulation varying with θ | must be measured, not assumed 0 |

**Target quantity:** `P(R | b = b₀, θ)` for a fixed behavior `b₀` — estimated by counterfactual
simulation in Stage 4 (§10). Everything before that stage exists to estimate `μ(θ)`, `Σ(θ)` and a
position-independent measure of `b`.

**A note on A.** A is not purely "the rat chose to rear better there." Rearing against a wall at one
angle is *physically* a different act than at another — local geometry, lighting, and the port's
presence shape posture. That is still a behavioral cause, not a tracking artifact, so the A/(B,B′)
split is the primary decomposition. A gets split further in Stage 5 (§11) into
posture-shaped-by-geometry vs skill-acquired-by-practice.

---

## 4. Data inventory

Everything needed is already in the local cache — **no SMB reads for the main analysis**, except
template `.npz` files (tiny) and calibration `.mat` files.

| | value |
|---|---|
| rats | R4, R5, R6 |
| sessions cached | 151 (R4 54, R5 49, R6 48) |
| with aligned DANNCE (`has_dual`) | **136** |
| with `sleap_keys_2D` (`has_keys2d`) | **151** |
| cache size / location | 11 GB, `data/cache/<rat>/<session>.npz` |
| session length | 24k / 36k / 42k frames @ 20 Hz (20–35 min) |
| SLEAP cameras | **3** (mean conf 0.49 / 0.49 / **0.31**) |
| DANNCE cameras | **6** |
| calibration | single folder `2026_05_04` for the whole cohort |
| arena | circle, centre **(−66, −98)**, R ≈ **332 mm**, wall-radius std 4.0 mm |
| water port | (273, −168) → **θ ≈ −12°**, r ≈ 346 (in the wall) |
| port gate | `min_port_distance = 200 mm` → wall arc **θ ≈ −30° … +6°** |
| gate-off window | `min_port_distance = 0` on **2026_07_21, 07_22, 07_23, all three rats** (9 sessions) |
| train_state | varies 0/1/2 within rat across sessions — must be a covariate |
| templates in play | R4: 1, 2, 3, refined · R5: 1–5 · R6: 1–3 (non-monotone; a session can revert) |

**Consequences.** The single calibration removes any recalibration natural experiment for this
cohort (R1/R2/R3 span several and could supply one as an extension). The 6-vs-3 camera asymmetry is
what makes DANNCE a usable referee. Camera 2's low mean confidence is the leading mechanistic
suspect for H1.

---

## 5. Code architecture

The structural problem: the rear detector, matcher, gating and attempt logic are **copy-pasted
across three notebooks** (`single_session`, `multi_session`, `interactive_rat`) and kept in sync by
hand. This analysis needs all of it plus new machinery, so extract it once.

```
clirb/                          # importable package; removes the sys.path hacks
  __init__.py
  config.py                     # moved as-is
  skeleton.py                   # moved as-is
  session_cache.py              # moved as-is; CACHE_VERSION stays at 2 (see §6.3)
  templates.py                  # NEW — template loading + provenance
  events.py                     # ★ SINGLE SOURCE OF TRUTH (lifted verbatim)
  quality_cache.py              # NEW — derived per-frame cache, own version
  spatial/
    geometry.py
    quality.py
    dual.py
    maps.py
    models.py
    simulate.py
  plotting.py
scripts/
  build_event_table.py
  build_quality_cache.py
  run_spatial_confound.py
notebooks/
  spatial_confound.ipynb        # thin: loads parquet, renders figures
```

### 5.1 `clirb/events.py` — the extraction

Lifted **verbatim** from `multi_session.ipynb` cells 7/8/9/24 and `single_session.ipynb` cells 3/4,
so behavior is unchanged and the notebooks can be repointed with confidence:

```python
FPS = 20; TEMPLATE_LEN = 30; N_COMPONENTS = 2
REAR_RISE = 200.0; REAR_RESET = 150.0; REAR_MIN_FRAMES = int(0.5 * FPS)
LICK_BOUT_GAP_S = 2.0; ATTEMPT_WINDOW_S = 10.0; ALIGN_BUFFER_S = 0.75

def causal_median_5_kp(triang)                      -> (T,23,3) f32
def project_session_to_pcs(triang, template, *, flip_z=True)  -> (T, n_pcs)
def check_template_match(pc_buffer, tmpl, bounds)   -> (bool, int)
def get_template_match_indices(keys_pcs, tmpl, bounds, refractory_frames=None)
def correct_matches(pcs, tmpl, matches, buffer_frames=30)
def replay_rule_offline(pcs_full, template, accepted_errors=3) -> (matches, num_outside)
def detect_rears(pc0, rise=REAR_RISE, reset=REAR_RESET, min_frames=REAR_MIN_FRAMES)
def lick_frames_from_idx(lick_idx, n_frames); def lick_bouts(lick_frames, gap_fr=...)
def attempts(rear_end, lick_bout_frames, att_win_fr=...)
def rear_matched(rear_start, rear_end, match_cam0, template_len=TEMPLATE_LEN)
def best_align_end(rs, re, pcs, template, buf=ALIGN_BUF_FR) -> (end_frame, mse)
def port_gate_frame(rear_end, match_end)            # MATCH-END, never rear-start
```

Two invariants to assert in tests, both already documented conventions:
- rear duration is `(end - start)/FPS`; `n_above` is used **only** by the duration filter;
- the port gate is evaluated at the **match-end** frame for matched rears, else the rear-end frame —
  so matched-outside rears are a strict subset of matches-outside-port.

**Regression test:** on `R6/2026_07_08_1` the detector must yield **60 rears / 20 matched-rears**.
Add a test that reproduces the pilot's totals (15,642 rears, P = 0.403) so a refactor can't silently
change the detector.

### 5.2 `clirb/templates.py` — provenance (new, needed for H2)

The template `.npz` records where it came from. Fields observed in the cohort:

```
template (30,2)            bounds_to_use (30,2)       pc_weights (10,69)
feature_means (69,)        feature_stds (69,)         pcs_to_use (2,) = [0,1]
bounds () = 1.0            temp_origin_file, temp_origin_idx
n_matches, match_sessions (n,), match_starts (n,)              <- BUILD set
refined_from_sessions, refined_match_sessions, refined_match_ends/starts,
refined_bounds_mode ('max_dev'), original_template, original_bounds_to_use,
orig_session_names/templates/bounds/template_files
dannce_template (0,)       dannce_template_bounds (0,)          <- EMPTY in this cohort
```

```python
def load_template(rat, session)      -> dict + '_file', '_sha'
def template_epochs(rat)             -> DataFrame[session, template_file, epoch_id]
def build_provenance(template)       -> DataFrame[session, end_frame, kind={'built','refined'}]
```

The builders that write these fields are `~/CLIRB/refine_template.py` (829 lines, single-session)
and `~/CLIRB/refine_templates_mult_session.py` (697 lines, multi-session); seeds come from
`~/CLIRB/identify_templates_clean_kh.ipynb`. Read the multi-session one before Stage 2 — its
`CONFIG` block pins every parameter needed to reproduce the clustering exactly.

`build_provenance` is what makes H2 testable: resolve each build/refine match to its COM position
and hence to a θ. Caveat found in the pilot: `refined_match_sessions` exists on some templates
(R4_refined, R5_template_4) but not others (R4_template_3, R5_template_3, R6_template_2/3) — for
those, only the session *list* is stored, so per-match attribution must be re-derived by replaying
the rule on the listed sessions and matching frame indices. Handle both paths; mark the re-derived
ones `provenance_exact=False`. Also note `R5_template_5` stores a `refined_match_ends` of length 68
against 24 build matches (inherited from template_4) — do not assume the arrays align.

**`dannce_template` is empty in every R4/R5/R6 template**, which is why `analyse_dual_session`'s
DANNCE-template branch is commented out. §9 builds the DANNCE reference from scratch instead.

### 5.3 `clirb/spatial/geometry.py`

```python
ARENA = ArenaFrame(center=(-66.0, -98.0), radius=332.0)   # fitted, see §4

def fit_arena(com_xy, n_bins=24) -> ArenaFrame
    # robust: minimise across-angle std of the per-bin 99th-pct radius
    # (Nelder-Mead; converged to std = 4.0 mm on 504k pooled frames)
def theta(xy, arena)        -> degrees in (-180, 180]
def radius(xy, arena)       -> mm from centre
def wall_distance(xy, arena)-> arena.radius - radius
def sectors(theta, width=30)-> categorical; default 12 sectors of 30 deg
def port_arc(arena, port_xy, min_port_distance) -> (theta_lo, theta_hi)
def apex_height(triang, rear_start, rear_end)   -> max z of head/spine during rear
```

Sector width is a real analysis choice. Default **30°** (12 sectors, ~174 mm of wall each) as a
compromise between resolution and the ~15.6k rears available; **report every spatial result at 20°,
30° and 45°** to show it is not a binning artifact. The pilot used 45° and a provisional centre of
(−80,−80); all pilot sector numbers must be recomputed on the fitted frame.

### 5.4 `clirb/spatial/quality.py` — truth-free, all 151 sessions

```python
def reprojection_residual(keys_2d, cal_dir, conf_thresh=0.5) -> (T,23) px   # VECTORIZED
def conf_stats(keys_2d)      -> dict(mean_conf (T,), per_cam (T,3), n_cams_conf50 (T,))
def bone_length_cv(triang, edges)  -> (T,) coefficient of variation vs session median
def jitter(triang_rot)       -> (T,) mean |2nd difference| in egocentric coords
def nan_fraction(triang)     -> (T,)
def fov_margin(keys_2d, cal_dir, triang) -> (T,3) signed px distance to image edge
```

`fov_margin` is the cheap high-value one: rears are tall, the cameras look down, so a rear apex may
leave the calibrated volume at some angles and not others. Compute at the rear apex specifically.

### 5.5 `clirb/spatial/dual.py`, `maps.py`, `models.py`, `simulate.py`

```python
# dual.py
def load_dual(rat, session)              -> sleap_3d, dannce_3d (aligned, 20 Hz)
def common_pc_basis(rat, sessions, strategy='stratified')  -> basis, means
def dannce_reference_template(rat, basis, strategy='stratified_rears') -> tmpl, bounds
def d_dannce(dannce_pcs, rear_windows, tmpl)  -> (n_rears,) min-MSE alignment distance
def residual_field(sleap_pcs, dannce_pcs, theta) -> mu(theta), Sigma(theta), snippets

# maps.py
def rate_map(events, value, by='sector', min_n=50, cluster='session') -> DataFrame + CIs
def occupancy_normalize(map_, occupancy)
def spatial_permutation(map_, labels, n=10_000, stat='variance') -> p

# models.py
def fit_conditional(df, formula, groups)  -> statsmodels result
def matched_pairs(df, on='d_dannce', tol=..., across='sector') -> DataFrame
def mediation(df, treat='sector', mediator='d_dannce', outcome='matched')

# simulate.py
def block_bootstrap_residuals(snippets, sector, n, win=30) -> (n,30,2)
def counterfactual_map(behavior_library, residuals_by_sector, template, bounds,
                       accepted_errors=3) -> DataFrame[sector, p_match, ci_lo, ci_hi]
```

---

## 6. Schemas

### 6.1 Per-rear event table — `results/spatial_confound/rears.parquet` (~15.6k rows)

The central artifact. Every downstream analysis is a groupby on this.

| column | dtype | notes |
|---|---|---|
| `rat`, `session` | str | |
| `date`, `session_idx` | date, int | `session_idx` = chronological rank within rat, for learning curves |
| `train_state` | float | 0/1/2, NaN if unparsed |
| `min_port_distance` | float | 200 or 0 — the gate-off window |
| `template_file`, `template_epoch`, `template_sha` | str, int, str | H2 needs this; templates are non-monotone |
| `rear_idx` | int | index within session |
| `start_frame`, `end_frame`, `gate_frame`, `align_end` | int | gate at **match-end**, else rear-end |
| `duration_s`, `n_above` | float, int | duration = `(end-start)/FPS` |
| `x`, `y`, `z` | float | COM at `gate_frame` |
| `theta`, `radius`, `wall_dist` | float | fitted arena frame |
| `sector_20`, `sector_30`, `sector_45` | category | three binnings |
| `apex_z` | float | max height during the rear |
| `port_dist`, `in_gate_zone` | float, bool | |
| `matched` | bool | offline replay of the online rule |
| `online_triggered` | bool | from `behavior_log`, task sessions only |
| `align_mse` | float | SLEAP; template-dependent → **within-rat ranks only** |
| `d_dannce` | float | DANNCE; NaN on non-dual sessions |
| `pc_resid_0`, `pc_resid_1` | float | mean SLEAP−DANNCE PC residual over the window |
| `is_attempt`, `lick_bout_frame`, `latency_s` | bool, int, float | |
| `n_cams_conf50`, `mean_conf`, `conf_cam0/1/2` | float | mean over the rear window |
| `reproj_px`, `bone_cv`, `jitter`, `nan_frac` | float | |
| `fov_margin_min` | float | min over cams at apex |

### 6.2 Per-frame quality cache — `data/cache_quality/<rat>/<session>.npz`

`reproj_px (T,23) f32`, `conf (T,3,23) f32`, `n_cams_conf50 (T,) i8`, `bone_cv (T,) f32`,
`jitter (T,) f32`, `nan_frac (T,) f32`, `fov_margin (T,3) f32`, `theta (T,) f32`, `radius (T,) f32`,
plus `QUALITY_VERSION` and the same `src_signature` mechanism as `session_cache`.
Estimated ~40 MB/session → ~6 GB. Needed so maps can be **occupancy-normalized**, not just
rear-normalized.

### 6.3 Two engineering constraints that will bite otherwise

1. **Do not bump `CACHE_VERSION`.** The 11 GB raw cache is built; a bump forces a full rebuild off
   SMB. The derived quality cache gets its **own** version constant in its own directory.
2. **`session_raw_reproj_residual` must be vectorized before it is usable per-frame.** Today
   (`multi_session` cell 50) it loops T frames × 23 joints with a Python-level `np.linalg.svd` each
   and caps at `max_frames=20_000` — ~800k SVDs for a 36k-frame session. Batch it: build the
   `(T*23, 6, 4)` design tensor and call `np.linalg.svd` once; `cv2.undistortPoints` is already
   batchable. Expect ~100× speedup, which is the difference between "one session" and "all 151."

---

## 7. Stage 1 — Geometry, occupancy, and truth-free quality *(all 151 sessions)*

Needs no DANNCE, so it covers every session and every part of the arena.

**1.1 Confirm the arena frame** with `fit_arena` on pooled occupancy; report the per-angle wall
radius (expect std ≈ 4 mm) and re-derive θ for everything. Locate the port arc.

**1.2 Occupancy and rear maps** per rat and per session block: P(θ) for all frames, for rears, for
attempts, for matches, for licks. Pooled occupancy is already known to be non-uniform — 37.5% in
(−45°,0°], which is the **port arc**, so much of it is licking, not rearing. Report rear-conditional
maps separately from raw occupancy throughout.

**1.3 Quality-vs-θ profiles.** For each metric in §5.4, an occupancy-normalized profile with
cluster-robust CIs, at all three sector widths, per rat and pooled:
- reprojection residual, pooled and per camera
- **`n_cams_conf50`** — the occlusion proxy and leading H1 suspect (camera 2 averages conf 0.31 vs
  0.49/0.49; if a sector loses it, triangulation is effectively 2-camera and degenerate along the
  epipolar direction)
- bone-length CV — truth-free rigidity violation, needs no reference at all
- jitter, NaN/dropped-frame rate (source D)
- **`fov_margin` at rear apex** — prioritize; the most likely purely geometric mechanism

**1.4 Height dependence.** Repeat 1.3 as a function of `apex_z`. A confound that only appears at
rear height is much more damaging to this task than one that appears at floor level.

**Acceptance criteria.** Stage 1 succeeds if it either (a) names a metric with a θ-profile whose
peak/trough coincides with the P(match) profile in ≥2 of 3 rats, or (b) rules out all of them, which
pushes weight onto H2. Both outcomes are publishable results about the rig.

**Failure mode.** All truth-free metrics come out flat. Then either the bias is real but invisible
to these proxies (proceed to Stage 2, which does not depend on them), or B is small and the answer
is H0/H2.

---

## 8. Stage 2 — Template provenance *(H2; cheap, do it early)*

Entirely from the template `.npz` files plus the cached COM — a few minutes of compute for a
potentially project-defining result.

**2.1 Map every build/refine match to θ.** Pilot result, sector bins of 45° on the provisional
frame:

| template | set | n | top sector | share |
|---|---|---:|---|---:|
| R5_template_3 | built | 40 | (0°,45°] | **95%** |
| R5_template_4 | built | 40 | (0°,45°] | **95%** |
| R5_template_4 | refined | 68 | (0°,45°] | 74% |
| R5_template_5 | built | 24 | (0°,45°] | **100%** |
| R4_template_refined | refined | 82 | (0°,45°] | 41% |
| R4_template_3 | built | 33 | (45°,90°] | 48% |
| R4_template_2 | built | 27 | (45°,90°] | 26% |
| R6_template_2 | built | 36 | (−180°,−135°] | 33% |
| R6_template_3 | built | 26 | (45°,90°] | 54% |

**2.2 The three-rat pattern.** Template spatial concentration tracks both the strength of the
spatial reward gradient and whether the rat's occupancy sits on it:
- **R5** — templates 95–100% from one sector; occupies that sector (2,943/5,618 rears); highest P
  there (0.533) and lowest median align-MSE (656 vs 980–1360).
- **R6** — template 3 built in sectors R6 does *not* occupy (it stands at −180°…−90°); lowest
  overall match rate (0.318) and low P in both of its occupied sectors (0.260, 0.247).
- **R4** — most spatially spread templates (26–48% top sector); flattest P profile; highest overall
  match rate (0.461).

n = 3 rats. This is a **hypothesis-generating pattern, not a test.** Treat it as such in writing.

**2.2b The decisive mechanism test — do the KMeans clusters segregate by sector?** *(cheap, and the
single best figure in the plan)* Re-run the builder's clustering **exactly** as
`refine_templates_mult_session.py` does it — widened matches at `WIDEN_FACTOR = 2.5` with
`WIDEN_MATCH_METHOD = 'greedy_best'`, z-score, flatten to 60-dim, `KMeans(n_clusters=10, n_init=10,
random_state=0)` — on each template's own listed sessions. Then compute the **θ distribution of each
cluster**.

- If clusters are spatially segregated, H2's mechanism is demonstrated directly: "pick the largest
  cluster" is literally "pick the most-occupied sector," and the resulting `max_dev` bounds encode
  that sector's tracking bias.
- If clusters are spatially mixed, the bounds absorb the position variance instead, H2's amplifier
  is weak, and weight shifts back to H1/H0.

Quantify with the mutual information between cluster label and sector, against a label-shuffled
null. Also report, per template, the **spatial homogeneity of the chosen cluster** — this is the
knob that controls how much spatial selectivity the template ends up with, and it is directly
actionable (§12.2).

**2.3 Break the chicken-and-egg with temporal ordering.** Template concentration partly *reflects*
prior occupancy, so concentration alone proves nothing. The discriminating test is precedence:
compute each rat's sector occupancy in the sessions **before** the template was built vs **after**
it came into use. H2 predicts the occupancy concentration *increases* after; H0 predicts it was
already there and the template merely inherited it. Concrete instance: R5_template_3 first appears
2026_06_26_2, built from 2026_06_25_1 — compare R5's rear-θ distribution over 2026_06_05…06_24 vs
2026_06_28…07_13.

**2.4 Template-swap events as within-rat natural experiments.** Templates are **non-monotone** — R4
runs 1 → 2 → refined → 2 → 1 → 2 → 3, and R6 reverts to template 3 after a gap. Each swap changes
the matching rule while the rat and the rig stay fixed. H2 predicts the P(θ) profile shifts toward
the *new* template's build sector within a session or two. This is the single strongest available
test of H2 and it needs no DANNCE at all. Model as an interrupted time series with rat and
train_state controls.

---

## 9. Stage 3 — The DANNCE referee *(136 dual sessions)*

**The one methodological trap.** `analyse_dual_session` currently rebuilds *both* templates from
`sleap_corr` matches (its "pass 2"). Those matches are spatially biased, so the resulting "truth"
inherits exactly the bias we are trying to measure. Any conditioning analysis built on it is
circular. Two independent constructions, cross-validated against each other:

**3.1 Stratified-rear reference.** Detect rears in the *DANNCE* PC stream independently of SLEAP.
Draw a **spatially stratified** sample — equal N per 30° sector, capped by the sparsest usable
sector — and fit the reference template and bounds from that sample. Balanced by construction.

**3.2 Procrustes-mapped reference.** Map the SLEAP template into DANNCE space through a
position-balanced 7-DoF Procrustes fit on paired frames (`corrector/world_alignment.py` already does
the per-session fit; re-fit it on a sector-balanced frame sample rather than the first 5 minutes).

**3.3 Agreement check.** The two references must agree within the noise; if they do not, the
disagreement itself is a measurement of B′ and must be reported before proceeding.

**3.4 Per-rear `d_dannce`** = min-MSE alignment distance of the DANNCE trajectory to the reference,
over the same `[rear_start, rear_end + 0.75 s]` search window as `best_align_end`. This is the
position-independent behavior-quality covariate.

**3.5 The residual field.** Per rear window, `r = z_sleap − z_dannce` in the common basis →
**μ(θ)** and **Σ(θ)**, and a library of 30-frame residual snippets tagged by sector for Stage 4.

**3.6 Validate the referee — do not skip this.** DANNCE is better, not perfect; it is a different
6-camera reconstruction with its own spatial error profile.
- DANNCE bone-length CV vs θ, and COM-network confidence vs θ.
- If DANNCE degrades in the *same* sectors as SLEAP, the decomposition is **attenuated** and every
  B estimate must be reported as a lower bound, not a point estimate.
- Cross-check the sign and location of μ(θ) against the Stage 1 truth-free proxies. Agreement is
  strong evidence; disagreement means one of the two is measuring something else.
- 15 sessions lack DANNCE. Check they are missing-at-random with respect to θ and session index; if
  not, weight accordingly.

---

## 10. Stage 4 — Conditioning and the counterfactual map

### 10.1 Conditioning: does sector still matter at fixed true behavior?

Three analyses of decreasing assumption load; they should agree.

**Hierarchical logistic / GAM**, fit within (rat, template-epoch) blocks:

```
matched ~ s(theta) + d_dannce + apex_z + wall_dist + session_time
          + train_state + (1 | rat) + (1 | rat:session)
```

The `s(θ)` term surviving conditioning on `d_dannce` is the tracking-attributable spatial effect.
Report as an odds ratio, best vs worst sector, with a session-level bootstrap CI.

**Matched pairs** — the most persuasive single number in the project, and nearly assumption-free.
Pair rears from *different* sectors whose `d_dannce` agrees within a tight tolerance (genuinely the
same behavior in different places) and compare match rates within pairs. Tolerance chosen so that
within-pair `d_dannce` difference is ≪ the between-sector `d_dannce` difference; report the
pair count and the tolerance sensitivity curve.

**Formal mediation** — decompose the total sector→reward effect into the path through `d_dannce`
(= A) and the direct path (= B + B′).

### 10.2 The counterfactual map — the headline figure

Literally simulate "the same behavior, somewhere else."

1. Fix a behavior library: all DANNCE rear trajectories, or a canonical subset.
2. For each sector s, corrupt each trajectory with noise drawn from the **measured** residual
   distribution in s — **block-bootstrap 30-frame residual snippets** so temporal autocorrelation is
   preserved. i.i.d. Gaussian noise would badly understate the effect, because the rule counts
   *consecutive* out-of-bounds samples.
3. Run the **exact** online rule — `replay_rule_offline` with the session's own `bounds_to_use` and
   `accepted_errors = 3`, and the same 30-frame refractory.
4. Output **P(match | identical behavior, sector s)**, per rat and per template epoch.
5. **Null:** shuffle the sector labels of the residual snippets → the map under position-independent
   noise. Permutation p-value (n = 10,000) on the **spatial variance** of the map, which handles
   multiple comparisons across sectors in one statistic.
6. **Sensitivity:** repeat at `accepted_errors ∈ {1,3,5}` and with bounds scaled ×0.8/1.0/1.2, to
   show whether the effect is a knife-edge property of the current threshold or robust to it.
7. **Cross-template counterfactual (H2):** run the same behavior library through *each rat's
   templates* — including the other rats' — and ask whether P(match | θ) peaks at the build sector
   of whichever template is used. This is the direct simulation test of the template-import loop.

---

## 11. Stage 5 — Natural settling vs induced bias

A confirmed B does not prove the rat's localization was *caused* by it. Six tests:

**5.1 Temporal precedence.** Estimate the sector reward-gain map from *early* sessions (occupancy is
broadest then) or from occupancy-independent truth-free proxies. Then test whether occupancy drift
across sessions is predicted by that early map's gradient, per rat. Induced ⇒ the rat migrates up
the gradient. This is the same logic as §8.3 applied to the tracking field rather than the template.

**5.2 R5-vs-R6 as a built-in contrast.** R5 sits on its best sector; R6 sits on its two worst. If
the gradient shaped occupancy, R6 should not do that. Chase R6's alternative explanation — it is
worse at *every* align-MSE quartile (0.71–0.92 in its best quartile vs 0.91–0.99 for R4/R5), which
smells like a template/bounds problem rather than a spatial one. Test whether R6's template epochs
account for it. If they do, R6 becomes evidence for H2 rather than a counterexample to H1.

**5.3 Spatial win-stay / lose-shift.** P(sector change on the next rear | rewarded vs not), with
session-time and lag controls, and a permutation null that preserves the rat's marginal movement
statistics. A direct behavioral signature of location being reinforced.

**5.4 Port-gate positive control.** The 200 mm gate is a *known* spatial reward asymmetry covering
θ ≈ −30°…+6°. Does in-zone rearing decline across sessions? If the rat demonstrably learns that
known gate, it is capable of learning spatial reward structure — which raises the prior for the
subtler tracking gradient being learnable too. If it never learns the gate, H1/H2 as *drivers of
occupancy* become much less plausible, even if the gradient itself is real.

**5.5 Gate-off natural experiment.** `min_port_distance` was set to 0 on **2026_07_21–23 for all
three rats** (9 sessions), then restored. Does in-zone rearing rebound during the window and recede
after? Short and underpowered alone, but it is a real manipulation with a within-subject
before/during/after design, and it directly calibrates *how fast* these animals respond to a change
in spatial reward rules — which bounds how quickly they could have responded to a tracking gradient.

**5.6 Practice vs geometry — splitting A.** Does align-MSE (and `d_dannce`) in a sector improve with
cumulative rears performed *in that sector*? H0's "settling on a solution" predicts yes: skill is
acquired locally. If instead sector quality is flat with practice and set by geometry from session
one, A is posture-shaped-by-geometry rather than skill.

**Final output:** an explicit variance decomposition of the between-sector reward-rate spread into
A / B / B′ / C / D, with CIs, per rat.

---

## 12. Stage 6 — Consequences and mitigation

**12.1 What this does to existing conclusions.** `multi_session` §1.8b/§1.8e split by port zone but
not by sector. If B or B′ is confirmed, per-session match-rate trends partly track *where the rat
stood*, and cross-session learning curves are contaminated by occupancy drift. Quantify the
contamination on the existing panels and re-render the key ones sector-stratified. Be specific about
which published//circulated numbers change.

**12.2 Mitigations, cheapest first.**
1. Sector-stratified reporting everywhere (free; changes only analysis).
2. **Spatially stratified template building** — if H2 holds, this is the fix, and it is a small,
   well-localized change to `~/CLIRB/refine_templates_mult_session.py`. Three options in increasing
   order of intrusiveness:
   - **(a) balance the pool** — subsample the widened-match pool to equal N per sector *before*
     `KMeans`, so the largest cluster can no longer just be the most-occupied sector;
   - **(b) balance the bounds** — keep the chosen cluster's mean as the template, but compute
     `max_dev` bounds over a sector-balanced sample, so the corridor is wide enough to admit the
     same behavior performed elsewhere;
   - **(c) select the cluster on spatial homogeneity, not size** — prefer a cluster that spans
     sectors over a larger one confined to a single sector.

   (b) is the minimal intervention and preserves the existing template shapes, so try it first.
   Retro-test all three by rebuilding R5's template (its build sets are 95–100% single-sector, so it
   is the most sensitive case) and re-running the counterfactual map of §10.2. Success criterion:
   the map flattens without a material loss of overall match rate.
3. Per-sector bound calibration in the online rule.
4. **The neural corrector as a mitigation, with a proper success criterion.** Does applying
   `corrector/` at inference *flatten the counterfactual map*? That is a far better target for the
   corrector than the current F1 tables — it measures the thing the corrector is actually for.
5. Rig-level: a 4th SLEAP camera, or repositioning camera 2, if Stage 1 localizes the loss to it.

---

## 13. Figures

| # | figure | stage |
|---|---|---|
| F1 | arena schematic — fitted circle, port, gate arc, sector grid, pooled occupancy | 1 |
| F2 | polar P(match \| rear) per rat, 3 sector widths, cluster-robust CIs | 1 |
| F3 | quality-vs-θ panel: reproj, `n_cams_conf50`, bone CV, jitter, FOV margin, NaN | 1 |
| F4 | quality vs apex height × θ heatmap | 1 |
| F5 | template provenance — build/refine match locations on the arena, per template | 2 |
| F5b | **KMeans cluster × sector segregation** — the H2 mechanism figure, with shuffled null | 2 |
| F6 | occupancy before vs after each template swap (interrupted time series) | 2 |
| F7 | DANNCE referee validation — DANNCE bone CV vs θ; μ(θ) vs truth-free proxies | 3 |
| F8 | P(match) vs sector, raw vs conditioned on `d_dannce`; matched-pairs forest plot | 4 |
| F9 | **counterfactual map** P(match \| fixed behavior, sector) + permutation null | 4 |
| F10 | cross-template counterfactual — P(θ) peak vs template build sector | 4 |
| F11 | occupancy drift vs early reward-gain gradient, per rat | 5 |
| F12 | gate-off window: in-zone rearing before/during/after | 5 |
| F13 | variance decomposition A/B/B′/C/D, per rat | 5 |

---

## 14. Statistical hazards

- **Rears are not independent.** Strongly autocorrelated in space and time within a session. Use
  session-level bootstrap and cluster-robust SEs; never naive binomial CIs.
- **Severe occupancy imbalance.** R5 has 2,943 rears in one sector and ~2 in another. The
  eye-catching extremes in the pilot's 2-D grid (P = 0.07, 0.68) are n < 60 and should not be
  trusted. Enforce a minimum n per bin, show CIs on every map, and never report a bare bin maximum.
- **`align_mse` is template-dependent.** Not comparable across template epochs or across rats. Use
  within-(rat, epoch) ranks for any pooled statement. This also means **the P(match) profiles in §1
  are not directly comparable across rats** — only their shapes are.
- **Templates are non-monotone.** A session can revert to an earlier template. `template_epoch` is a
  factor, not an ordinal.
- **`train_state` varies within rat** (0/1/2) and changes whether tone/water are delivered at all.
  It must be a covariate in every model, and state-0 sessions may need exclusion from reward
  analyses.
- **Multiple comparisons over sectors** → permutation on the whole map, not per-sector p-values.
- **Circularity** — the trap in §9. The reference behavior measure must never be selected using
  SLEAP matches.
- **n = 3 rats.** Every cross-rat pattern (including the striking one in §8.2) is descriptive.
  Within-rat, within-session designs carry the inferential weight; cross-rat patterns are
  corroboration only.

---

## 15. Sequencing

| | step | output | depends on | rough effort |
|---|---|---|---|---|
| 1 | extract `clirb/events.py`; repoint the 3 notebooks; regression tests | no behavior change; drift hazard gone | — | ~half a day |
| 2 | `templates.py` + Stage 2 provenance + **cluster×sector test (§8.2b)** | **H2 test, cheap and possibly decisive; F5, F5b** | 1 | ~1 day |
| 3 | `geometry.py` + `build_event_table.py` | `rears.parquet` (~15.6k rows) | 1 | ~half a day |
| 4 | vectorize reprojection; `quality.py`; `build_quality_cache.py` | per-frame quality cache | 1 | ~1 day |
| 5 | Stage 1 maps, all 151 sessions | mechanism shortlist, F1–F4 | 3, 4 | ~1 day |
| 6 | `dual.py` — unbiased DANNCE reference, μ(θ), Σ(θ) | bias field, F7 | 3 | ~1–2 days |
| 7 | Stage 4.1 conditioning + matched pairs | A/(B,B′) split, F8 | 6 | ~1 day |
| 8 | Stage 4.2 counterfactual map | **headline, F9–F10** | 6 | ~1 day |
| 9 | Stage 5 settling-vs-induced | causal read, F11–F13 | 3, 8 | ~1–2 days |
| 10 | Stage 6 + `spatial_confound.ipynb` | figures, mitigation recommendation | all | ~1 day |

**Reordering note.** Step 2 (template provenance) is promoted ahead of the heavy infrastructure
because it is cheap, needs no DANNCE, and could redirect the whole project. Steps 1–5 are worth
doing regardless of how the DANNCE comparison turns out — they are the infrastructure this project
has been missing, and Stage 1 alone will likely name the physical mechanism.

**A falsifiable prediction to hold ourselves to.** If H1 is the story, the sector with the highest
counterfactual match probability should coincide with the sector of highest `n_cams_conf50` and
lowest reprojection residual, **and that coincidence should hold in R4, R5 and R6 independently** —
even though only R5 chose to stand there. If instead the counterfactual peak tracks each rat's
*template build sector* and moves when the template is swapped, the story is H2, the fix is in the
template-building procedure, and the tracking field is a secondary contributor.

---

## 16. Open questions for the user

1. **Scope** — R4/R5/R6 only, or extend to R1/R2/R3? The older cohort spans multiple calibrations,
   which would supply a recalibration natural experiment this cohort cannot (§4), at the cost of
   different templates, task versions and the older `processed_path` cache layout.
2. **Rear definition** — the hysteresis detector (200/150) is tuned for the current PC0 scale, which
   is template-dependent. Should the spatial analysis use it as-is for continuity with the existing
   notebooks, or a template-independent rear definition (e.g. absolute apex height) so that
   cross-template comparisons in §8.4 are cleaner? Recommend: **compute both**, use the height-based
   one for cross-template contrasts.
3. **Behavior library for the counterfactual** (§10.2) — pooled DANNCE rears across rats, or per-rat?
   Per-rat is cleaner; pooled has more power. Recommend per-rat, pooled as a sensitivity check.
4. **How were the `SESSIONS` lists for each refinement chosen?** They are hand-typed into the
   `CONFIG` block of `refine_templates_mult_session.py` and not recorded anywhere else, so the
   `.npz` provenance fields are the only surviving record. If the choice was itself influenced by
   "which sessions had good match rates," that adds a second selection step to the H2 loop worth
   modelling. If it was just "the three most recent," the loop is simpler.
5. **Was `PROMPT_FOR_CLUSTER` used interactively?** The default offers the largest cluster but a
   human can override it. If overrides happened, which cluster was chosen is not recoverable from
   the `.npz` — but the *saved* template can be matched back against the recomputed clusters to
   identify it. Worth doing as part of Stage 2.

*(Resolved while planning: the template-building procedure is fully specified in
`~/CLIRB/refine_templates_mult_session.py` and `~/CLIRB/refine_template.py` — see §2/H2 and §8.2b.)*
