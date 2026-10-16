# Interpretable online-matching features — project status

**What this project is.** Find a small set of hand-designed, *interpretable* features that (a) are
reliable between SLEAP and DANNCE, (b) can drive the online template-matching reward rule at
≥80% precision / ≥80% recall, and (c) give flexible control over more than one behavior
(rearing + turn direction + crouch/stretch).

**Where it lives.** It is **not** a separate project directory. It is the tail of the spatial-confound
investigation: `SPATIAL_CONFOUND_RESULTS.md` **§8l → §8y**, driven by `scripts/stage19…stage25`.
Everything before §8l is the *other* question (does arena position bias the match rule) — related
background, not this project.

**Status as of the last run: 2026-10-16** (§8z; everything before it 2026-09-14). All scripts, all
result CSVs, and all figures (F21–F37) are present on disk and consistent. Nothing needs re-running
to recover the state.

---

## The answer so far

| role | feature | definition | verdict |
|---|---|---|---|
| **height** | `ear_mean_z` | mean Z of {EarL, EarR}, no snout | **validated.** 5.41 mm SLEAP-vs-DANNCE RMSE (best of 16). Alone: **93–94% precision / 96% recall** on rearing. Clears 80/80 in every rat. |
| **crouch/stretch** | `dist_earmean_hipmean` | 3D dist, mean(EarL,EarR) ↔ mean(HipL,HipR) | **validated-ish.** 8.15 mm RMSE (best distance). With height: 93%/81–84%, but R6 recall dips to ~78%. |
| **turn** | `angle_skip_xy` | signed XY angle, (SpineF−SpineL)→(Snout−SpineF) | **marginal but the only survivor.** With height: 91%/82%. R6 recall mid-70s. |
| turn | `angle_torsion_xy`, `angle_body_head_xy` | hip-axis↔shoulder-axis; body-axis↔head | **ruled out.** 67% recall paired with height, and still 67% under a ground truth built from their own signal (§8v). Their self-consistent corridor swallows 600–830 events/session — not behaviorally selective. |

Three findings that constrain everything else:

1. **Snout is near-pure noise.** Adding it to a 2-ear height average makes the feature *worse*
   (5.41 → 6.93 mm). Every feature it touches degrades. Design around it.
2. **Angles are structurally noisy** (~17–24° RMSE, 19–27% sign-flip rate) — geometric amplification
   on short spine segments, not a tracking-quality problem. Longer lever arms help.
3. **The corridor-matching paradigm only works for behaviors with a consistent postural shape.**
   Rearing has one; *pure turning and locomotion do not* (§8x). Their rebuilt ground truth explodes
   to ~1,200 events/session — the theoretical max — for **every** feature, at every threshold tried.
   This is not tunable. Posture-independent turn/locomotion detection needs a different
   algorithm (variable-length event matching, or a peak-anchored single-frame classifier).

**So: for rewarding oriented rears, the template must be `rear_turn` (rears that are also turned),
not free-floating "turning."** `ear_mean_z + angle_skip_xy` on `rear_turn` gives 89.8%/80.2% pooled.

**But the direction split is weaker than §8x implied (§8z, 2026-10-16).** F35's per-direction
precision/recall measured SLEAP-vs-DANNCE *agreement*, not direction discrimination. A direct
separation test shows:
- **Height alone does not separate directions at all** — the two "direction-specific" `ear_mean_z`
  detectors fire on 81–83% of the same events. F35's height-only ~94%/95% per direction is not a
  direction-split result.
- **`ear_mean_z + angle_skip_xy` does carry real directional information** — own-pool hit rate ~2×
  other-pool (0.55 vs 0.24), consistent in all 3 rats. Not vacuous, but weak.
- **It recovers only ~55% of its own pool**, so bounds tuning faces a ~45-point gap, not the
  3–6 points F35 suggested.
- **Ceiling: SLEAP and DANNCE agree on turn *sign* only 84.7% of the time.** No direction-split
  validation can exceed this, whatever the features or tuning.

**PC2/PC3 are not a "turnedness" axis** (§8y). Joint R² with any hand-designed angle ≤0.25. PC2 is
one-sided limb+tail placement, PC3 is snout pitch + forelimbs. The PCA basis is fixed per rat (3
definitions total, not one per template file). Hand-designed angles remain the more direct route.

---

## The three open threads (in priority order)

1. **Decide whether direction-split rear+turn is viable at all, before tuning anything.** §8z
   reframed this: ~55% own-pool recovery against an 84.7% cross-modality turn-sign ceiling. The
   limiting factor may be turn-sign *measurement* rather than the matching corridor — in which case
   bounds tuning (§8x recommendation 2) cannot fix it. Worth settling before more feature search.
2. **Validate outside the corrector-OFF / new-SLEAP-model era.** Every precision/recall number in
   §8t–§8x comes from **33 sessions** in the most-accurate stratum, by explicit request. RMSEs roughly
   double under corrector-ON, so the numbers would drop — unknown how far. Rerunning `stage22` on the
   full cohort is mechanical.
3. **A genuinely posture-independent turn detector**, if still wanted — requires a different matching
   approach, not more features (see finding 3 above).

---

## Fragility to fix before building on this

- **None of it is committed.** The 811 new lines of `SPATIAL_CONFOUND_RESULTS.md` and all of
  `scripts/stage19…stage25*.py` are uncommitted/untracked. One `git clean` loses the project.
- **`compute_all_features` is copy-pasted into 4 scripts** (`stage21` as `compute_features`,
  plus `stage22`/`stage23`/`stage24`, and again in `render_sleap_dannce_feature_video.py`).
  Checked: the definitions agree — they differ only in which features had been added yet
  (stage22 has 14, stage21/23/24 have 16). But there is no single source of truth, and the next
  edit will silently diverge. Promote it to `clirb/features.py`.
- **F35's data and code are lost — and it mattered.** `make_figure_stage24.py:84` reads
  `directional_rear_turn.csv` from a since-deleted `/tmp` scratchpad; the PNG survives, the CSV and
  script do not. `stage26_direction_separation.py` now reimplements the split, but **its pool sizes
  do not reconcile with §8x's** (§8x: median 26 left / 15 right, total 41, against
  `stage24_seed_counts.csv`'s own rear+turn median of 22.5; the reconstruction gives 9/11, total 25).
  F35 was evidently computed on a different pool, and the left/right labels may be flipped. F35
  should be treated as unverifiable, not as a result.
- **The session set has grown.** 56 sessions now qualify as corrector-OFF/new-model (47 usable)
  versus 32 when §8t–§8y were computed. §8z is on the larger set; §8t–§8y have not been recomputed.

---

## Map of the code

```
SPATIAL_CONFOUND_RESULTS.md §8s   scripts/stage21_feature_reliability.py   -> stage21_feature_reliability.csv, F27
                            §8t/u scripts/stage22_feature_subset_matching.py -> stage22_combo_{results,summary}.csv, F28-F30
                            §8v/w scripts/stage23_feature_specific_ground_truth.py -> stage23_{combo_results,gt_sizes,combo_summary}.csv, F31-F32
                            §8x   scripts/stage24_multi_template_features.py -> stage24_{seed_counts,results,summary}.csv, F33-F35
                            §8y   scripts/stage25_pc23_correlation.py      -> stage25_pc_{correlations,loadings}.csv, F36-F37
                            §8z   scripts/stage26_direction_separation.py  -> stage26_separation_{per_session,summary}.csv  (no figure yet)
                            §8p-r scripts/stage19_keypoint_mse.py, stage20_keypoint_mse_worldspace.py,
                                  stage20b_stratify_by_pipeline_version.py  (per-keypoint groundwork, F21-F26)
```
Results CSVs: `results/spatial_confound/`. Figures: `results/figures/spatial_confound/`.
Shared infrastructure: `clirb/events.py` (rear detector, matcher, `project_session_to_pcs`,
`causal_median_5_kp`), `clirb/templates.py`, `clirb/spatial/`.

**Not part of this project:** the Sep 18 work — `clirb/sleap_variability.py`,
`scripts/build_variability_summary.py`, `multi_session_variability.ipynb`,
`r5_variability_over_time.ipynb`, `r5_before_after_lesion.ipynb` — is a separate thread on rear
*variability* across training and around lesions.

*Written 2026-10-16 as a revival map, updated the same day with §8z; the authoritative detail is in `SPATIAL_CONFOUND_RESULTS.md` §8l–§8y.*
