# 2D-input corrector — overnight regularization + per-rat sweep

**Date**: 2026-05-21 → 2026-05-22 (overnight session)
**Author**: Claude (instructed run)

## TL;DR

1. **Inference-time outlier guard fixes the R1 kp_mse blowup.** R1/2026_02_13_1
   went from 23,082 → 411 kp_mse. R1 aggregate kp_mse dropped from 2413 → 352.5.
   No degradation on any other session. Production-quality fix; lives in
   `evaluate_all.py:correct_triangulation_refiner`. Renderer inherits it.
2. **Regularization (`dropout=0.1, weight_decay=1e-4`) helps R2 specifically,
   hurts R1/R3 slightly.** This is the rat-capacity asymmetry we already knew
   about from the velacc story; the 2D-input pipeline shows the same pattern.
3. **Per-rat fine-tuning did not beat the v2 baseline on R1 or R3 and did not
   beat v3_reg on R2.** Two sessions of per-rat ft (bone-loss on, bone-loss off,
   lr=5e-5..1e-4) all landed essentially flat. R3 also exposes a trainer bug
   where the val_mse exponentially diverges across fine-tune epochs even though
   train_mse improves cleanly — see "open issues."
4. **Recommended 2D-input production picks** (each evaluated with the new
   inference guard applied):
   - R1: `R1R2R3_2d_v2.pt`   F1xyz=0.835   kp_mse=352.5
   - R2: `R1R2R3_2d_v3_reg.pt` F1xyz=0.528 kp_mse=585.1
   - R3: `R1R2R3_2d_v2.pt`   F1xyz=0.671   kp_mse=428.2

## Full result table

All numbers are post-eval on the same test split, with the new inference-time
outlier guard applied. Best per-rat-per-metric in **bold**.

| Rat | Metric | velacc baseline | v2 (guard) | v3_reg drop=0.1 | v4 drop=0.05 | R{x}_ft bone | R{x}_ft nobone |
|-----|--------|-----------------|------------|-----------------|--------------|--------------|----------------|
| R1 | F1xyz | ~0.83 | **0.835** | 0.820 | 0.829 | 0.825 | 0.826 |
| R1 | kp_mse |  | **352.5** | 387.4 | 388.3 | 370.9 | 379.4 |
| R2 | F1xyz | 0.680 | 0.452 | **0.528** | 0.470 | 0.487 | 0.489 |
| R2 | kp_mse |  | **477.4** | 585.1 | 557.5 | 709.9 | 701.3 |
| R3 | F1xyz | 0.727 | **0.671** | 0.605 | 0.612 | 0.628 | 0.633 |
| R3 | kp_mse |  | **428.2** | 521.4 | 495.2 | 523.6 | 527.8 |

velacc figures are from `handoff.md` section 3.

## What changed in the code

- `corrector/evaluate_all.py:correct_triangulation_refiner` — added the inference
  outlier guard (drops `>1000 mm` raw-triang frames back to Procrustes-only,
  clips per-keypoint residual to ±200 mm). Inherited by the renderer.
- `corrector/train_2d_input.py` — added `--dropout`, `--init_ckpt` (per-rat
  fine-tune from a pretrained checkpoint). Checkpoint metadata now includes
  `dropout`, `weight_decay`, `init_ckpt`, `lr`.
- `corrector/render_world_overlay.py` — passes `dropout` through to the model.
- `corrector/evaluate_all.py` build path — passes `dropout` through to the model.

## Checkpoints produced

```
corrector/checkpoints/R1R2R3_2d_v3_reg.pt          dropout=0.1   wd=1e-4   best_val_mse=119.6
corrector/checkpoints/R1R2R3_2d_v4_reg_light.pt    dropout=0.05  wd=1e-4   best_val_mse=121.6
corrector/checkpoints/R1_2d_ft_from_v3.pt          per-rat ft + bone loss   val=105.0
corrector/checkpoints/R2_2d_ft_from_v3.pt          per-rat ft + bone loss   val=67.1
corrector/checkpoints/R3_2d_ft_from_v3.pt          per-rat ft + bone loss   val=723K (broken)
corrector/checkpoints/R1_2d_ft_v3_nobone.pt        per-rat ft, no bone loss  val=104.6
corrector/checkpoints/R2_2d_ft_v3_nobone.pt        per-rat ft, no bone loss  val=68.7
corrector/checkpoints/R3_2d_ft_v3_nobone.pt        per-rat ft, no bone loss  val=789K (broken)
```

## Videos rendered

6 sessions × `R1R2R3_2d_v3_reg.pt` at `corrector/videos/R1R2R3_2d_v3_reg/`,
matching the v2 set at `corrector/videos/R1R2R3_2d_v2/` for side-by-side
comparison.

## Open issues

1. **R3 fine-tune val_mse diverges exponentially across epochs even though
   train_mse improves cleanly.** Both R3 fine-tunes hit this (bone loss on AND
   off). The model is producing huge predictions on R3 val sessions that aren't
   filtered by the trainer's `evaluate()` loop because the outlier guard isn't
   applied at the per-sample level there. The saved checkpoint picks "best val"
   which is just "least-broken val" — an early epoch from a sequence that is
   monotonically getting worse on the in-trainer metric, even though the
   eval-pipeline kp_mse is fine. The trainer val signal is unusable for R3 ft.
   Two fixes worth trying:
    - Apply the inference-time outlier guard inside the trainer's `evaluate()`.
    - Or evaluate the trainer's val MSE on Procrustes-aligned + clip-residual
      output instead of raw model output.
2. **R2 kp_mse regression on every per-rat fine-tune** (709 vs 477 baseline)
   despite F1xyz holding. The model is making xyz more *template-like* but less
   *DANNCE-like*. Bone loss is irrelevant — happens without it. This is the
   classic "improve the operational metric while degrading the keypoint metric"
   pattern; safe to ignore if you treat F1xyz as the gold metric, but worth
   investigating.
3. **Per-rat fine-tuning gave roughly zero gain.** This is somewhat surprising
   given the velacc story (per-rat heads were the single biggest 3D-input win).
   The 2D-input model already has more input channels per rat (calibration is
   per-session and varies across the calibration cutover), so the global model
   may already be doing per-rat adaptation implicitly via the global pool.

## Recommended next steps (not done this session)

1. **Patch trainer's `evaluate()` to apply the inference-time outlier guard.**
   Once val_mse is honest, re-run R3 fine-tune and see if it actually wins.
2. **Train v3_reg longer.** It plateaued near 120 val_mse but v2's 112 was hit
   in 11 epochs without dropout — possibly v3_reg with `--epochs 120` would
   undercut v2. Was not tried this session due to overnight constraints.
3. **Aux reprojection loss** is still untried and was item #6 on the original
   priority list. Worth ~5 lines in `train_2d_input.py`.
4. **Deploy rebuilt template_1** (Phase H.1) — completely independent of any
   2D-input work and is the largest unrealized F1 win sitting in the project
   (R2 +0.17).
