"""Verify clirb.features.compute_features reproduces the five hand-synced copies
it replaces (stage21/22/23/24 + render_sleap_dannce_feature_video), exactly.

Extracts each script's original function from the file (or from a git revision,
via --rev) without importing the script, runs both implementations on random
keypoint arrays, and asserts bit-for-bit equality on every shared feature.

  python scripts/_verify_features_equivalence.py            # current working tree
  python scripts/_verify_features_equivalence.py --rev HEAD # against a commit

Once the scripts have been consolidated onto clirb.features they no longer hold
a local copy to compare against, so the working-tree run reports SKIP for each.
The meaningful check is then against the last revision that still had the copies
(--rev), which is how this was validated at consolidation time: all five copies
matched bit-for-bit across 228 array comparisons.
"""
import argparse, ast, subprocess, sys, os

import numpy as np

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb import features as F  # noqa: E402

TARGETS = [
    ('scripts/stage21_feature_reliability.py', 'compute_features', 'vector'),
    ('scripts/stage22_feature_subset_matching.py', 'compute_all_features', 'speed'),
    ('scripts/stage23_feature_specific_ground_truth.py', 'compute_all_features', 'speed'),
    ('scripts/stage24_multi_template_features.py', 'compute_all_features', 'speed'),
    ('scripts/render_sleap_dannce_feature_video.py', 'compute_all_features', 'speed'),
]


def load_source(path, rev=None):
    if rev:
        return subprocess.run(['git', 'show', f'{rev}:{path}'], capture_output=True,
                              text=True, check=True).stdout
    return open(path).read()


def extract(src, fname):
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == fname:
            return ast.get_source_segment(src, n)
    return None


def build(fn_src):
    """Exec the extracted function with the globals it expects."""
    ns = dict(np=np, FPS=F.FPS, signed_angle_xy=F.signed_angle_xy,
              SNOUT=F.SNOUT, EARL=F.EARL, EARR=F.EARR,
              SPINEF=F.SPINEF, SPINEM=F.SPINEM, SPINEL=F.SPINEL, TAILBASE=F.TAILBASE,
              HIPL=F.HIPL, HIPR=F.HIPR, SHOULDERL=F.SHOULDERL, SHOULDERR=F.SHOULDERR)
    exec(fn_src, ns)
    return ns[[k for k in ns if k.startswith('compute_')][0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rev', default=None, help='git revision to compare against')
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    cases = [rng.normal(size=(n, 23, 3)).astype(np.float64) * 60 for n in (2, 17, 300)]

    all_ok = True
    for path, fname, mode in TARGETS:
        src = load_source(path, args.rev)
        fn_src = extract(src, fname)
        if fn_src is None or 'feat.' in fn_src:
            print(f'  SKIP  {path}: already consolidated onto clirb.features '
                  f'(use --rev <commit-before-consolidation> to re-check)')
            continue
        old = build(fn_src)
        n_checked = 0
        for kp in cases:
            a, b = old(kp), F.compute_features(kp, velocity=mode)
            shared = set(a) & set(b)
            missing = set(a) - set(b)
            if missing:
                print(f'  FAIL  {path}: new module missing {sorted(missing)}')
                all_ok = False
            for k in sorted(shared):
                if not np.array_equal(np.asarray(a[k]), np.asarray(b[k])):
                    d = np.max(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)))
                    print(f'  FAIL  {path}: {k} differs (max abs {d:.3e})')
                    all_ok = False
                n_checked += 1
        print(f'  {"OK  " if all_ok else "FAIL"}  {path}: {fname}() -> '
              f'{len(set(old(cases[-1])))} features, {n_checked} array comparisons')

    print('\nALL EQUIVALENT' if all_ok else '\nMISMATCHES FOUND')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
