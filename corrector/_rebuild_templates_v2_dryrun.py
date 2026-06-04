"""One-rat dry-run for _rebuild_templates_v2.py.

Runs the rebuild only on R1/template_1 to sanity-check end-to-end before the
full job. Writes to results/template_rebuild_v2/_dryrun/.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/home/yutaka-sprague/CLIRB_analyses")
sys.path.insert(0, str(REPO))

from corrector import _rebuild_templates_v2 as v2

# Re-point output to a dryrun subfolder so we don't pollute the real outputs.
v2.OUT_DIR = REPO / "results" / "template_rebuild_v2" / "_dryrun"
v2.OUT_DIR.mkdir(parents=True, exist_ok=True)

rat = "R1"
template_file = f"{rat}_template_1.npz"
print(f"\n=== DRYRUN  {rat} / {template_file} ===", flush=True)
result = v2.rebuild_one_template(rat, template_file)
v2.save_template(rat, "template_1", result)
print("\nDryrun complete.")
print(f"  N windows: {result['n_windows']}")
print(f"  Sessions used: {len(result['sessions_used'])}")
print(f"  PCs used: {result['pcu']}")
print(f"  Template shape: {result['new_template'].shape}")
print(f"  pc_template_bounds shape: {result['pc_template_bounds'].shape}")
print(f"  pc_template_bounds min/max on used PCs: "
      f"{result['pc_template_bounds'][:, result['pcu']].min():.3f} / "
      f"{result['pc_template_bounds'][:, result['pcu']].max():.3f}")
