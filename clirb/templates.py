"""Template loading + provenance.

The template .npz records where it came from. Fields seen in the R4/R5/R6 cohort:

  template (30,2)        bounds_to_use (30,2)     pc_weights (10,69)
  feature_means (69,)    feature_stds (69,)       pcs_to_use (2,) = [0,1]
  bounds () = 1.0        temp_origin_file, temp_origin_idx
  n_matches, match_sessions (n,), match_starts (n,)        <- BUILD set
  refined_from_sessions, refined_match_sessions, refined_match_ends/starts,
  refined_bounds_mode ('max_dev'), original_template, original_bounds_to_use,
  orig_session_names/templates/bounds/template_files
  dannce_template (0,)   dannce_template_bounds (0,)        <- EMPTY in this cohort

Builders that write these: ~/CLIRB/refine_template.py (single-session) and
~/CLIRB/refine_templates_mult_session.py (multi-session).
"""
import os
import glob
import hashlib
import numpy as np
import pandas as pd

DATA_ROOT = "/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data"


def find_template_file(rat, session):
    sd = os.path.join(DATA_ROOT, rat, session, 'sleap')
    cand = sorted(glob.glob(os.path.join(sd, f'{rat}_template_*.npz')))
    if cand:
        return cand[0]
    fallback = os.path.join(DATA_ROOT, rat, 'templates', f'{rat}_template_1.npz')
    return fallback if os.path.isfile(fallback) else None


def load_template(rat, session):
    p = find_template_file(rat, session)
    if p is None:
        return None
    z = np.load(p, allow_pickle=True)
    t = {k: z[k] for k in z.files}
    t['_file'] = os.path.basename(p)
    t['_path'] = p
    t['_sha'] = hashlib.sha1(
        np.ascontiguousarray(t['template']).tobytes() +
        np.ascontiguousarray(t['bounds_to_use']).tobytes()).hexdigest()[:10]
    return t


def _clean_session(s):
    return str(s).replace('\\', '/').split('/')[-1]


def build_provenance(template):
    """Return DataFrame[session, frame, kind, exact] for the matches that built
    (and refined) this template.

    kind='built'  -> match_sessions / match_starts
    kind='refined'-> refined_match_sessions / refined_match_ends when the
                     per-match session labels exist, else NaN session with
                     exact=False (must be re-derived by replaying the rule).
    """
    rows = []
    # --- BUILD set: per-match sessions are always stored.
    if 'match_starts' in template and np.asarray(template['match_starts']).size:
        starts = np.asarray(template['match_starts']).ravel()
        sess = np.asarray(template.get('match_sessions', [])).ravel()
        if sess.size == starts.size:
            for s, f in zip(sess, starts):
                rows.append((_clean_session(s), int(f), 'built', True))

    # --- REFINED set: per-match sessions exist on some templates only.
    ends = np.asarray(template.get('refined_match_ends', [])).ravel()
    if ends.size:
        rsess = np.asarray(template.get('refined_match_sessions', [])).ravel()
        if rsess.size == ends.size:
            for s, f in zip(rsess, ends):
                rows.append((_clean_session(s), int(f), 'refined', True))
        else:
            # Only the session LIST survives. Mark inexact; caller re-derives.
            for s in np.asarray(template.get('refined_from_sessions', [])).ravel():
                rows.append((_clean_session(s), -1, 'refined', False))
    return pd.DataFrame(rows, columns=['session', 'frame', 'kind', 'exact'])


def template_epochs(rat, sessions):
    """DataFrame[session, template_file, template_sha, epoch_id].

    Templates are NON-MONOTONE (a session can revert to an earlier file), so
    epoch_id is a factor keyed on the file+sha, not an ordinal.
    """
    rows = []
    for s in sessions:
        t = load_template(rat, s)
        rows.append((s, t['_file'] if t else None, t['_sha'] if t else None))
    df = pd.DataFrame(rows, columns=['session', 'template_file', 'template_sha'])
    keys = {k: i for i, k in enumerate(sorted(df['template_sha'].dropna().unique()))}
    df['epoch_id'] = df['template_sha'].map(keys)
    return df
