# corrector

Tiny network that maps SLEAP egocentric keypoints → DANNCE egocentric keypoints.

## Files

- `data.py` — paired (SLEAP, DANNCE) dataset, deterministic session-level split.
- `models.py` — `LinearCorrector` and `MLPCorrector` (both add a residual to the input).
- `train.py` — training loop with MSE + bone-length regularizer.
- `evaluate.py` — keypoint MSE + per-PC bias fraction + template-match F1 + inference time.
- `check_gpu.py` — standalone GPU sanity check.

## Usage

```bash
# 1. Verify GPU
python -m corrector.check_gpu

# 2. Train one model per rat
python -m corrector.train --rat R1 --model linear      # baseline
python -m corrector.train --rat R1 --model mlp         # main

# 3. Evaluate on held-out test sessions
python -m corrector.evaluate --rat R1 --model mlp
```

Checkpoints land in `corrector/checkpoints/<rat>_<model>.pt`; eval reports in `corrector/results/<rat>_<model>_eval.json`.

## Splits

Sessions are sorted chronologically, the **last 15%** become the test set, the rest are shuffled (seed=0) and split 70/15 train/val. So the test set always tests for temporal drift in addition to within-session generalization.

## Loss

`MSE(pred, dannce) + bone_weight * MSE(bone_lengths_pred, bone_lengths_dannce)` with weight decay on the optimizer. The bone-length term keeps the model from creating skeletons that don't match a rat.

## Architecture choices

Both models output `x + delta(x)`. The output layer is initialized to zero so training starts at the identity (no correction) and only moves away when the data demands it. This makes early training stable.

`LinearCorrector` has 4,830 parameters; `MLPCorrector` (default 2×128) has ~42k. Both inference in single-digit microseconds per frame on either CPU or GPU.
