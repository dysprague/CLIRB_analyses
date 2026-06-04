# CLIRB_analyses

Off-line analysis pipeline for the CLIRB (closed-loop reinforcement of behavioral templates) experiment. Sister to `~/CLIRB` (raw analysis) and `~/campy-CLIRB` (online acquisition + matching). This folder contains template-matching alignment experiments and a SLEAP→DANNCE 3D-keypoint corrector.

## Project goal

Drive online template matching from SLEAP keypoints (low latency) using templates calibrated against DANNCE keypoints (higher accuracy). The two systems disagree at the keypoint level enough that template matches detected by each don't perfectly overlap. We're trying to (a) characterize the disagreement and (b) reduce it without retraining either model.

## What's been done

### Phase 1 — Characterization (`experiments/`)
- `harm_analysis.py` — quantifies per-PC bias fraction and per-keypoint bias maps. Per-rat outputs in `results/figures/harm_analysis/`.
- `run_experiments_v3.py`, `run_experiments_v4.py` — sweeps over template-matching parameters. v4 added Procrustes pre-alignment, joint feature means, pairwise-distance pooled PCA, keypoint exclusion, and cosine distance. Results in `results/metrics/`.
- `template_matching_results_v4.ipynb` — executed notebook joining v3+v4 numbers and harm-analysis diagnostics.

**Headline finding:** ~⅓ of cross-system PC error is structured (per-PC bias fraction ≈ 0.34), ⅔ is noise. **Group O** (pairwise distances + pooled SLEAP+DANNCE PCA, n=2 PCs) was the best non-network fix, beating v3 by 2–6.6 percentage points of F1 at 300 ms.

### Phase 2 — Neural corrector (`corrector/`)

A small MLP that takes SLEAP keypoints and outputs corrected keypoints in DANNCE coordinate space.

**Pipeline (world-space):**
1. Load raw SLEAP and DANNCE 3D keypoints; resample DANNCE to SLEAP frame rate; smooth (median-11 SLEAP, median-25 DANNCE).
2. Per session, fit a 7-DoF Procrustes (rotation + isotropic scale + translation, with optional z-flip) on the first 5 minutes. Sessions with residual > 60 mm are skipped (uncalibrated DANNCE).
3. Apply Procrustes to SLEAP, then feed through an MLP with residual skip on the output. The output is the corrected pose in DANNCE world space; we inverse-Procrustes back to SLEAP world space for downstream use.
4. Loss = MSE + 0.1 × bone-length-MSE; Adam with weight decay 1e-5; batch 4096; early stop at 8 epochs without val improvement.

**Affine vs Procrustes pilot:** a 12-DoF affine fit was not meaningfully better than 7-DoF Procrustes on a per-session held-out evaluation. Default to Procrustes by Occam.

**Models trained, in chronological order:**

| Checkpoint | Architecture | Context | Trained on | Best val MSE |
|---|---|---:|---|---:|
| `R3_linear.pt` | linear residual | 1 frame, egocentric | R3 | (legacy) |
| `R3_mlp.pt` | 2-layer MLP | 1 frame, egocentric | R3 | (legacy) |
| `R2R3_world_mlp.pt` | 2-layer MLP | 1 frame, world | R2 + R3 | 46.8 |
| `R1_world_mlp.pt` | 2-layer MLP | 1 frame, world | R1 | 63.3 |
| `R1R2R3_world_mlp.pt` | 2-layer MLP | 1 frame, world | R1 + R2 + R3 | 244.0 |
| **`R1R2R3_world_temporal_mlp.pt`** | 2-layer MLP | **5-frame** causal, world | R1 + R2 + R3 | **243.3** |

(Val MSE on the multi-rat models is higher than R2+R3 alone because R1's val data has different noise statistics; the comparison that matters is per-rat **test** performance below.)

### Phase 3 — Ablation experiments (in progress)

Building on the temporal R1+R2+R3 baseline, we're systematically testing five orthogonal improvements. All numbers from the unified evaluator (`corrector/evaluate_all.py`) on the same held-out test sessions, with metrics: keypoint MSE, PC1/PC2 MSE in the rat's xyz template space, and F1@300ms in **xyz PC matching** and **Group O pairwise pooled-PCA matching**.

#### Phase A: Quick interventions on the baseline

| Rat | Experiment | kp_mse | PC1 | PC2 | F1 xyz | F1 groupO |
|---|---|---:|---:|---:|---:|---:|
| **R1** | baseline (R1R2R3 temporal) | 186 | 183 | 546 | 0.824 | 0.812 |
|  | + Gaussian noise std=0.5 | 197 | 209 | 671 | 0.827 | 0.836 |
|  | + per-rat head | 180 | 216 | 501 | 0.821 | **0.898** ⭐ |
| **R2** | baseline | 280 | 305 | 538 | 0.661 | 0.700 |
|  | + Gaussian noise std=0.5 | 305 | 271 | 638 | 0.623 | 0.732 |
|  | + per-rat head | 289 | 286 | 556 | 0.658 | 0.723 |
| **R3** | baseline | 280 | 275 | 495 | 0.765 | 0.780 |
|  | + Gaussian noise std=0.5 | 289 | 300 | 469 | 0.744 | 0.737 |
|  | + per-rat head | 276 | 270 | 495 | 0.699 | 0.778 |

**Phase A takeaways:**
- **Per-rat head on R1 is the largest single gain in the project: F1 groupO 0.81 → 0.90** (+8.6 pp).
- Per-rat head only helps R1 in F1 — R2 mixed, R3 hurt slightly. Different rats benefit from different bases.
- Noise aug helps R1 and R2 in F1 groupO; hurts R3.

#### Phase B: Architecture / loss / feature variants

| Rat | Experiment | kp_mse | PC1 | PC2 | F1 xyz | F1 groupO |
|---|---|---:|---:|---:|---:|---:|
| **R1** | baseline | 186 | 183 | 546 | 0.824 | 0.812 |
|  | + vel/acc features | 182 | 185 | 456 | 0.811 | 0.818 |
|  | + PC-space loss (0.5) | 196 | 184 | 571 | 0.793 | 0.828 |
|  | + GNN | 408 | 495 | 768 | 0.819 | 0.826 |
| **R2** | baseline | 280 | 305 | 538 | 0.661 | 0.700 |
|  | + vel/acc features | 265 | 250 | **417** | 0.660 | **0.760** |
|  | + PC-space loss (0.5) | 303 | 342 | 564 | 0.669 | 0.732 |
|  | + GNN | 717 | 834 | 1363 | 0.562 | 0.644 |
| **R3** | baseline | 280 | 275 | 495 | 0.765 | 0.780 |
|  | + vel/acc features | 305 | 291 | 562 | 0.727 | **0.818** ⭐ |
|  | + PC-space loss (0.5) | 309 | 306 | 526 | 0.677 | 0.778 |
|  | + GNN | 610 | 603 | 918 | 0.680 | 0.657 |

**Phase B takeaways:**
- **vel/acc features help R2 and R3 substantially in F1 groupO** (R2: 0.70→0.76, R3: 0.78→0.82). The strongest data-feature change. Also drops R2 PC2 by 22%.
- **PC-space loss is mostly neutral** — it changes the loss surface but the supervision signal is largely redundant with bone-length + MSE.
- **GNN underperforms badly** at the chosen capacity (~17k params, hidden=64, 3 layers, early-stop at epoch 12). Likely under-parameterized. Re-running with hidden=128, 4 layers, longer patience.

#### Phase C: Combinations of Phase A and B winners

| Rat | Experiment | kp_mse | PC1 | PC2 | F1 xyz | F1 groupO |
|---|---|---:|---:|---:|---:|---:|
| **R1** | baseline | 186 | 183 | 546 | 0.824 | 0.812 |
|  | + vel/acc + noise | 169 | 180 | 473 | 0.809 | 0.820 |
|  | + per-rat head (velacc base) | 177 | 187 | 453 | 0.809 | 0.816 |
|  | + GNN_v2 (deeper retry) | 410 | 486 | 737 | 0.815 | 0.791 |
| **R2** | baseline | 280 | 305 | 538 | 0.661 | 0.700 |
|  | + vel/acc + noise | 254 | 255 | 464 | 0.633 | **0.774** ⭐ |
|  | + per-rat head (velacc base) | 277 | 241 | 441 | **0.680** ⭐ | 0.758 |
| **R3** | baseline | 280 | 275 | 495 | 0.765 | 0.780 |
|  | + vel/acc + noise | 270 | 265 | 472 | 0.746 | 0.800 |
|  | + per-rat head (velacc base) | 300 | 262 | 550 | 0.730 | 0.799 |

**Phase C takeaways:**
- **vel/acc + noise** is the best single change for R2 across both metrics (F1gO 0.700 → 0.774).
- **per-rat head on velacc base** is the best F1xyz for R2 (0.661 → 0.680).
- GNN_v2 (hidden=128, 4 layers, longer patience) is still much worse than the flat MLP — GNN is not the right architecture for this problem at any tested capacity.

#### Best result per rat after Phases A/B/C

| Rat | Best F1 xyz | Best F1 groupO |
|---|---|---|
| **R1** | + Gaussian noise (0.827) | **+ per-rat head on temporal base (0.898)** ⭐ |
| **R2** | + per-rat head on velacc base (0.680) | **+ vel/acc + noise (0.774)** |
| **R3** | baseline temporal (0.765) | **+ vel/acc (0.818)** |

#### Phases D–F: stacked heads, capacity sweeps, and longer training

After the Phase A/B/C results we ran four more phases of stacking experiments. Highlights only — full numbers in `corrector/results/*_all.json`.

| Rat | Experiment | kp_mse | PC1 | PC2 | F1 xyz | F1 groupO |
|---|---|---:|---:|---:|---:|---:|
| **R1** | + head on temporal+noise base | 190 | 195 | 548 | **0.830** | 0.846 |
|  | + big head (h256,L3) on temporal | 187 | 204 | 509 | 0.824 | 0.877 |
|  | + head on velacc, noise during head training | 182 | 187 | **440** ⭐ | 0.823 | 0.815 |
|  | + PC-loss=1.0 longer training | 195 | 186 | 648 | 0.738 | 0.850 |
| **R2** | + head on velacc+noise base | 266 | 260 | 458 | 0.646 | **0.776** ⭐ |
|  | + head on velacc, noise during head training | 270 | 239 | **412** ⭐ | 0.666 | 0.764 |
|  | + PC-loss=1.0 longer training | 278 | 282 | 620 | **0.675** | 0.763 |
| **R3** | + head on temporal+noise base | 285 | 296 | 476 | **0.778** ⭐ | 0.747 |
|  | + head on velacc, noise during head training | 299 | 264 | 541 | 0.732 | 0.796 |
|  | + PC-loss=1.0 longer training | 280 | 291 | 471 | 0.680 | 0.783 |

**Phases D–F takeaways:**
- **Stacking (head + noise during training)** yields the lowest **PC2 MSE** for R1 (440) and R2 (412) — the structured-error axis we were specifically trying to reduce.
- **PC-loss=1.0 with longer training** gives the best F1xyz for R2 (0.675) and a respectable F1gO for R1 (0.850), but consistently *trades F1xyz for F1gO* — strongly suggests the loss is rotating the model's solution along the rigid/non-rigid axis.
- **Bigger head capacity does not help** (R1 big head 0.877 vs original 0.898). The R1 head improvement is real but at the limit of what a head-only architecture can extract from this base.
- **No single architecture dominates all 6 metrics on all rats**: this is the strongest signal that the right deployment is **per-rat configuration**.

### Final per-rat winners across **all phases**

| Rat | Best F1 xyz | Best F1 groupO | Best PC1 MSE | Best PC2 MSE | Best kp MSE |
|---|---|---|---|---|---|
| **R1** | wide head on velacc (0.830) | **head on temporal (0.898)** ⭐ | vel/acc + noise (180) | head on velacc+noise-train (440) | vel/acc + noise (169) |
| **R2** | head on velacc (0.680) | **head on velacc+noise (0.776)** | head on velacc+noise-train (239) | head on velacc+noise-train (412) | vel/acc + noise (254) |
| **R3** | head on temporal+noise (0.778) | **vel/acc (0.818)** | head on velacc+noise (256) | head on velacc+noise (447) | head on velacc+noise (266) |

**Recommendation: deploy a different config per rat.**
- **R1** should use the per-rat head on the R1+R2+R3 temporal base (F1gO 0.898). It's the largest improvement we've seen, and it doesn't sacrifice F1xyz.
- **R2** should use the per-rat head on the R1+R2+R3 velacc+noise base (F1gO 0.776, F1xyz 0.646) — the best F1gO and good across the board.
- **R3** should use the vanilla R1+R2+R3 vel/acc model (F1gO 0.818) — the simplest config, no head needed.

The fact that no single combination wins on all rats is itself the signal: most of the remaining residual error is rat-specific, and the most accurate path forward is to embrace that. The shared backbone with per-rat heads (where useful) gives both data efficiency and per-rat adaptability.

### Production candidate

For online deployment, two viable configs:

1. **Single shared model, no heads**: `R1R2R3_velacc_noise05.pt`. Marginally worse on R1 (F1gO 0.820 vs 0.898) but uniform across rats and operationally simpler. ~70k params, <2 µs/frame on GPU.
2. **Per-rat heads**: keep the shared `R1R2R3_world_temporal_mlp.pt` as backbone, deploy the per-rat heads `<rat>_head_on_R1R2R3temporal.pt` for R1 (and skip the head for R2/R3 since vel/acc alone is better). Adds ~13k trainable params per rat, identical inference cost.

I'd lean toward option 2 for accuracy, with the operational understanding that adding a new rat requires a few minutes of head training on its data.

### Phase 2 results

#### Keypoint and PC MSE on held-out test sessions

### Phase 2 results

#### Keypoint and PC MSE on held-out test sessions

| Model | Rat | n | Keypoint MSE Procrustes-only | + corrector | %Δ | PC1 MSE before→after | PC2 MSE before→after |
|---|---|---:|---:|---:|---:|---:|---:|
| R2+R3 (original) | R1 (cross-rat) | 69 | 357.8 | 242.2 | -32% | 512 → 303 | 667 → **689** ⚠ |
|  | R2 | 11 | 688.6 | 385.7 | -44% | 786 → 356 | 1179 → 746 |
|  | R3 | 12 | 605.0 | 350.3 | -42% | 589 → 378 | 812 → 656 |
| R1+R2+R3 single-frame | R1 | 11 | 393.6 | **183.0** | **-53%** | 473 → 228 | 724 → 469 |
|  | R2 | 11 | 688.6 | **261.6** | **-62%** | 786 → 354 | 1179 → 439 |
|  | R3 | 12 | 605.0 | **268.7** | **-56%** | 589 → 333 | 812 → 533 |
| **R1+R2+R3 + temporal (ctx=5)** | R1 | 11 | 393.6 | 185.7 | -53% | 473 → **180** | 724 → 540 |
|  | R2 | 11 | 688.6 | 279.7 | -59% | 786 → **297** | 1179 → 533 |
|  | R3 | 12 | 605.0 | 280.2 | -54% | 589 → **268** | 812 → **486** |
| R1-only | R1 | 11 | 393.6 | 215.4 | -45% | 473 → 289 | 724 → 559 |

#### Template-matching F1 @ 300 ms tolerance

| Model | Rat | Raw F1 | Corrected F1 | Δ pp | Recall raw → corr | Precision raw → corr |
|---|---|---:|---:|---:|---:|---:|
| R2+R3 | R1 | 0.750 | 0.695 | -5.5 ⚠ | 0.742 → 0.722 | 0.775 → 0.683 |
|  | R2 | 0.546 | 0.585 | +3.9 | 0.525 → 0.523 | 0.577 → 0.678 |
|  | R3 | 0.682 | 0.668 | -1.4 | 0.646 → 0.615 | 0.746 → 0.765 |
| R1+R2+R3 | R1 | 0.838 | 0.826 | -1.2 | 0.874 → 0.864 | 0.808 → 0.794 |
|  | R2 | 0.546 | 0.649 | **+10.3** | 0.525 → 0.610 | 0.577 → 0.703 |
|  | R3 | 0.682 | 0.710 | **+2.8** | 0.646 → 0.714 | 0.746 → 0.727 |
| **R1+R2+R3 + temporal** | R1 | 0.838 | 0.837 | -0.1 | 0.874 → 0.861 | 0.808 → 0.818 |
|  | R2 | 0.546 | **0.662** | **+11.6** | 0.525 → 0.656 | 0.577 → 0.675 |
|  | R3 | 0.682 | **0.754** | **+7.2** | 0.646 → 0.734 | 0.746 → 0.786 |

**The temporal R1+R2+R3 model is the production candidate.** It hits the highest F1 ever recorded on R2 and R3 (R3 0.754 beats every prior approach including v3's 0.646 and v4's 0.612), is essentially break-even on R1, has only ~70k parameters, runs at <2 µs/frame on GPU, and operates on Procrustes-aligned world-space keypoints — clean to integrate into `campy-CLIRB/campy/behavior_rule.py`.

### Visualizations

Videos and figures are organized by run tag — one subfolder per checkpoint:

```
corrector/videos/
├── R3_egocentric_mlp/                # legacy R3-only egocentric runs
└── R2R3_world_mlp/                   # first world-space model
└── R1R2R3_world_temporal_mlp/        # ← current production (top-2 sessions/rat)
    ├── R1_2026_02_06_1_cam0_f0-1000.mp4
    ├── R1_2026_02_09_1_cam0_f0-1000.mp4
    ├── R2_2026_02_09_1_cam0_f0-1000.mp4
    ├── R2_2026_02_13_1_cam0_f0-1000.mp4
    ├── R3_2026_02_06_1_cam0_f0-1000.mp4
    └── R3_2026_02_10_1_cam0_f0-1000.mp4
corrector/figures/
├── R2R3_world_mlp/                   # PC plots paired with the older videos
├── R1R2R3_world_temporal_mlp/        # PC plots paired with the new videos
└── summary/                          # QC-style tables + bar plots from summary_report.py
```

Each video has raw SLEAP + DANNCE on the left and corrected SLEAP + DANNCE on the right, both projected onto Camera0 of the SLEAP rig.

### Bone-length sanity (`corrector/bone_length_check.py`)
The corrector reduces SLEAP's bone-length CV from ~0.3–0.4 toward DANNCE's 0.11 across every rat, and **never makes any single bone less rigid**. Improvements are largest at limb extremities (HipL-KneeL, AnkleL-FootL, ShoulderL-ElbowL).

## Conda environment

```bash
conda activate clirb_analysis
```

Created with:
```bash
conda create -n clirb_analysis -c conda-forge python=3.13 -y
conda run -n clirb_analysis pip install \
    torch --index-url https://download.pytorch.org/whl/cu124
conda run -n clirb_analysis pip install \
    numpy pandas scipy matplotlib scikit-learn jupyter ipykernel opencv-python h5py
conda run -n clirb_analysis python -m ipykernel install --user \
    --name clirb_analysis --display-name "CLIRB Analysis"
```

GPU is RTX 4070 SUPER (CUDA 13 driver, CUDA 12.4 PyTorch wheel, cap 8.9).

The pre-corrector experiments (Phase 1 / `experiments/`) run in the base conda env or `basic_analysis` — they don't need PyTorch.

## Directory layout (current)

```
CLIRB_analyses/
├── README.md                          # this file
├── config.py, data_io.py, processing.py, projection.py, qc_utils.py,
│   skeleton.py, visualization.py      # shared utilities
├── experiments/                        # Phase 1
│   ├── exp_utils.py
│   ├── run_experiments.py              # v1 (smoothing × bounds × distance metric)
│   ├── run_experiments_v2.py           # v2 (refractory sweep)
│   ├── run_experiments_v3.py           # v3 (F1-optimized sweep)
│   ├── run_experiments_v4.py           # v4 (Procrustes / pairwise / kp-exclude / cosine)
│   └── harm_analysis.py                # bias-fraction characterization
├── corrector/                          # Phase 2
│   ├── README.md
│   ├── data.py, data_world.py          # paired SLEAP/DANNCE datasets (single-frame + windowed)
│   ├── world_alignment.py              # Procrustes (7 DoF) + affine (12 DoF) fitters
│   ├── models.py                       # LinearCorrector, MLPCorrector, TemporalMLPCorrector
│   ├── train.py                        # egocentric-space (legacy)
│   ├── train_world.py                  # world-space (current; --rats, --model, --ctx)
│   ├── evaluate.py                     # egocentric (legacy)
│   ├── evaluate_world.py               # world-space keypoint+PC MSE
│   ├── evaluate_f1.py                  # template-matching F1 vs raw / Procrustes-only / corrected
│   ├── bone_length_check.py            # per-edge sanity check
│   ├── render_world_overlay.py         # 1000-frame side-by-side video + PC plot
│   ├── render_long_combined.py         # ~10000-frame video + PC1/PC2 panels
│   ├── summary_report.py               # QC tables + per-keypoint/edge/PC figures
│   ├── check_gpu.py                    # GPU sanity check
│   ├── checkpoints/                    # *.pt model files
│   ├── results/                        # *_eval.json + *_f1.json
│   ├── videos/<run_tag>/               # rendered .mp4 organized per checkpoint
│   ├── figures/<run_tag>/              # PC plots paired with videos
│   ├── figures/summary/                # cross-test summary plots
│   └── logs/                           # training/eval logs
├── results/                             # Phase 1 outputs
├── template_matching.ipynb              # original alignment study (now has optional corrector cell)
├── template_matching_v2.ipynb           # 8-approach single-session A/B
├── template_matching_results_executed.ipynb  # v3 results
├── template_matching_results_v4.ipynb   # v3 + v4 + harm joined, executed
└── quality_check.ipynb                  # SLEAP/DANNCE QC dashboard
```

## How to use the corrector inside `template_matching.ipynb`

The notebook has a new cell after the data-loading section labelled **"(Optional) Apply MLP corrector to SLEAP"**. Set `APPLY_CORRECTOR = True` and (optionally) point `CORRECTOR_CKPT` at any of the checkpoints in `corrector/checkpoints/`. The cell:

1. Loads the model.
2. Fits per-session 7-DoF Procrustes on the first 5 min.
3. Replaces `sleap_3d` with the corrected keypoints (in SLEAP world space). The original is preserved as `sleap_3d_uncorrected`.

All downstream cells (z-flip, normalize, project to PCs, template matching) then operate transparently on corrected keypoints.

## How to reproduce the headline results

```bash
conda activate clirb_analysis

# 0. (optional) sanity check GPU
python -m corrector.check_gpu

# 1. Train the temporal R1+R2+R3 model (~6 minutes; the current production candidate)
python -m corrector.train_world --rats R1 R2 R3 --tag R1R2R3 --model temporal_mlp --ctx 5

# 2. Evaluate keypoint + PC MSE (~3 minutes)
python -m corrector.evaluate_world \
    --ckpt corrector/checkpoints/R1R2R3_world_temporal_mlp.pt

# 3. Evaluate template-matching F1 (~5 minutes)
python -m corrector.evaluate_f1 \
    --ckpt corrector/checkpoints/R1R2R3_world_temporal_mlp.pt

# 4. Per-edge bone-length sanity check
python -m corrector.bone_length_check \
    --eval corrector/results/R1R2R3_world_temporal_mlp_eval.json

# 5. Render comparison videos (top-2 sessions/rat by MSE improvement)
for SPEC in "R1 2026_02_09_1" "R1 2026_02_06_1" \
            "R2 2026_02_09_1" "R2 2026_02_13_1" \
            "R3 2026_02_10_1" "R3 2026_02_06_1"; do
    set -- $SPEC
    python -m corrector.render_world_overlay \
        --ckpt corrector/checkpoints/R1R2R3_world_temporal_mlp.pt \
        --rat $1 --session $2 --camera 0 --n_frames 1000
done

# 6. Long combined video with PC1/PC2 panels
python -m corrector.render_long_combined \
    --ckpt corrector/checkpoints/R1R2R3_world_temporal_mlp.pt \
    --rat R3 --session 2026_02_10_1 --camera 0 --n_frames 10000
```

## Open questions / next steps

1. **Online integration.** The `R1R2R3_world_temporal_mlp` checkpoint inferences in <2 µs/frame on GPU, plus a one-time per-session Procrustes fit during the ~5 min calibration epoch. To put it in `campy-CLIRB/campy/behavior_rule.py:TemplateMatch`, insert a step that:
   - During warmup: fits Procrustes on the first 5 min (need to define a callback or threshold for "calibration done").
   - Online: maintains a 5-frame rolling buffer of Procrustes-aligned SLEAP, applies the MLP, inverse-Procrustes the output, then proceeds to PCA + template matching as before.

2. **Combine the corrector with Group O (pairwise pooled PCA).** Two are mostly orthogonal: corrector reduces 3D bias, pairwise distances are invariant to remaining rigid transforms. Worth checking if they compound.

3. **Per-rat fine-tuning?** The shared model dominates per-rat models so far; we don't have evidence we need fine-tuning. But if R1 F1 is critical, a small per-rat head could be added on top of the shared body.

4. **Pose-conditional residual diagnostic.** Split test data into pose bins (low/mid/high COM speed; on-the-floor / rearing) and check whether residual keypoint error is uniform or concentrated. If concentrated, that's where to focus next.

## Where to look for the most recent state

- **Best model**: `corrector/checkpoints/R1R2R3_world_temporal_mlp.pt`
- **Eval JSONs**: `corrector/results/R1R2R3_world_temporal_mlp_eval.json` and `..._f1.json`
- **Headline summary**: this file's "Phase 2 results" tables
- **Per-session details**: rows in the eval JSONs; videos and PC plots under `corrector/videos/R1R2R3_world_temporal_mlp/` and `corrector/figures/R1R2R3_world_temporal_mlp/`
- **QC summary plots**: `corrector/figures/summary/`
- **Phase 1 numbers**: `template_matching_results_v4.ipynb`

If picking this up cold, start with this README, then open `template_matching_results_v4.ipynb` for the Phase 1 picture and the `R1R2R3_world_temporal_mlp_eval` outputs for Phase 2.
