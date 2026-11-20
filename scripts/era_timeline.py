"""Per-session rear / match / reward counts over the training timeline, split
into baseline, initial-training and post-lesion eras for R4, R5 and R6.

Eras (per the lab-meeting definition; post-lesion cutoff is PER RAT because R6
was lesioned ~5 weeks before R4/R5):

  baseline       2026_06_05 .. 2026_06_08     (no task)
  training       2026_06_11 .. 2026_09_01     (minus any post-lesion sessions)
  post_lesion    R4, R5: after 2026_09_02
                 R6:     after 2026_08_01

Sessions outside every window (2026_06_09/10, and anything after 2026_09_24)
are labelled `unassigned` and plotted in grey rather than silently dropped.

Three quantities per session, all on the SLEAP cam0 frame grid at 20 Hz:

  n_rears      hysteresis rear detector on PC0 (rise 200, reset 150, >0.5 s
               above rise) -- `clirb.events.detect_rears`, the same detector
               multi_session.ipynb and the spatial-confound work use.
  n_matched    rears whose span overlaps the 30-frame window of at least one
               offline replay of THAT SESSION'S OWN template rule. This is the
               behaviour the rig was trying to reward, recomputed offline, so
               it is defined for task and no-task sessions alike.
  n_rewarded   online triggers (behavior_log.csv `triggered == 1`) that are
               followed by a lick bout within LICK_WINDOW_S. A trigger alone
               does not mean water was collected -- it is logged even outside
               the task context and outside the port gate -- so this is the
               lick-confirmed subset: rewards the animal actually drank.

Also recorded for context: raw online trigger count, lick-bout count, attempts
(lick bouts attributable to a preceding rear), train_state, and the template
file in force, so template refits can be marked on the timeline.

Usage
  .../envs/clirb_analysis/bin/python scripts/era_timeline.py            # compute + figure
  .../envs/clirb_analysis/bin/python scripts/era_timeline.py --figure-only

Reads only the local session cache (`session_cache.py`) plus `~/olveczky_lab`
read-only; writes `results/era_timeline/`.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import session_cache as sc                      # noqa: E402
from clirb import events as ev                  # noqa: E402
from clirb import templates as tmpl             # noqa: E402

DATA_ROOT = "/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data"
OUT = "results/era_timeline"
RATS = ["R4", "R5", "R6"]
FPS = 20.0
LICK_WINDOW_S = 10.0                 # multi_session.ipynb's LICK_WINDOW_MS
LICK_WINDOW_FR = int(round(LICK_WINDOW_S * FPS))

BASELINE = ("2026_06_05", "2026_06_08")
TRAINING = ("2026_06_11", "2026_09_01")
LESION_AFTER = {"R4": "2026_09_02", "R5": "2026_09_02", "R6": "2026_08_01"}


# --------------------------------------------------------------------------- #
def session_date(session: str) -> str:
    return session[:10]


def assign_era(rat: str, session: str) -> str:
    d = session_date(session)
    if d > LESION_AFTER[rat]:
        return "post_lesion"
    if BASELINE[0] <= d <= BASELINE[1]:
        return "baseline"
    if TRAINING[0] <= d <= TRAINING[1]:
        return "training"
    return "unassigned"


def discover(rats=RATS):
    out = []
    for rat in rats:
        rat_dir = os.path.join(DATA_ROOT, rat)
        if not os.path.isdir(rat_dir):
            continue
        for s in sorted(os.listdir(rat_dir)):
            sd = os.path.join(rat_dir, s, "sleap")
            if not s[:4].isdigit() or not os.path.isdir(sd):
                continue
            if not (os.path.isfile(os.path.join(sd, "triang_keys_3D.npy"))
                    and os.path.isfile(os.path.join(sd, "frame_mapping.csv"))):
                continue
            out.append((rat, s))
    return out


def rewarded_triggers(trig_cam0, lick_bout_frames, window_fr=LICK_WINDOW_FR):
    """Triggers followed by a lick bout within `window_fr` frames.

    Each lick bout can confirm at most one trigger (the most recent one it
    could belong to), so two triggers in quick succession followed by a single
    bout count as one collected reward, not two.
    """
    if not len(trig_cam0) or not len(lick_bout_frames):
        return np.zeros(len(trig_cam0), dtype=bool)
    trig = np.asarray(trig_cam0, dtype=int)
    bouts = np.sort(np.asarray(lick_bout_frames, dtype=int))
    order = np.argsort(trig)
    claimed = np.zeros(len(trig), dtype=bool)
    for b in bouts:
        cand = order[(trig[order] <= b) & (trig[order] >= b - window_fr)
                     & ~claimed[order]]
        if cand.size:
            claimed[cand[-1]] = True        # most recent unclaimed trigger
    return claimed


def analyse(rat, session):
    row = {"rat": rat, "session": session, "date": session_date(session),
           "era": assign_era(rat, session), "ok": False, "err": ""}
    try:
        c = sc.get_raw_session(rat, session, want_dual=False)
        if c is None:
            row["err"] = "raw inputs missing/corrupt"
            return row
        triang = np.asarray(c["triang_3d"], dtype=np.float32)
        n_frames = len(triang)
        row["n_frames"] = n_frames
        row["duration_min"] = n_frames / FPS / 60.0
        row["train_state"] = (float(c["train_state"])
                              if np.isfinite(np.asarray(c["train_state"],
                                                        dtype=float)) else np.nan)

        template = tmpl.load_template(rat, session)
        row["template_file"] = os.path.basename(tmpl.find_template_file(rat, session) or "")

        pcs_full, _ = ev.project_session_to_pcs(triang, template, flip_z=True)
        pcs_to_use = np.asarray(template["pcs_to_use"]).ravel().astype(int)
        pc0 = pcs_full[:, pcs_to_use[0]]

        rears = ev.detect_rears(pc0)
        rear_start = np.array([r[0] for r in rears], dtype=int)
        rear_end = np.array([r[1] for r in rears], dtype=int)
        row["n_rears"] = len(rears)

        match_frames, _ = ev.replay_rule_offline(pcs_full, template)
        row["n_offline_matches"] = int(len(match_frames))
        hit, _ = ev.rear_matched(rear_start, rear_end, match_frames)
        row["n_matched"] = int(hit.sum())

        # --- online triggers, on the cam0 grid ---
        pf_to_cam0 = np.asarray(c["frame_mapping_cam0"], dtype=int)
        has_log = bool(c["has_task_log"])
        if has_log:
            trig_pf = np.asarray(c["behavior_log_frames"], dtype=int).ravel()
            trig_pf = trig_pf[(trig_pf >= 0) & (trig_pf < len(pf_to_cam0))]
            trig_cam0 = pf_to_cam0[trig_pf]
        else:
            trig_cam0 = np.array([], dtype=int)
        row["has_task_log"] = has_log
        row["n_triggers"] = int(len(trig_cam0))

        # --- licking ---
        lick_f = ev.lick_frames_from_idx(
            np.asarray(c["lick_idx_sleap_cams"], dtype=int).ravel(), n_frames)
        bouts = ev.lick_bouts(lick_f)
        row["n_licks"] = int(len(lick_f))
        row["n_lick_bouts"] = int(len(bouts))

        a_ri, _ = ev.attempts(rear_end, bouts)
        row["n_attempts"] = int(len(a_ri))

        claimed = rewarded_triggers(trig_cam0, bouts)
        row["n_rewarded"] = int(claimed.sum())
        row["ok"] = True
    except Exception as e:                       # noqa: BLE001
        row["err"] = repr(e)
    return row


def build():
    sessions = discover()
    print(f"discovered {len(sessions)} sessions across {RATS}")
    rows = []
    for i, (rat, s) in enumerate(sessions, 1):
        r = analyse(rat, s)
        rows.append(r)
        flag = "" if r["ok"] else f"  !! {r['err']}"
        print(f"[{i:3d}/{len(sessions)}] {rat} {s:<14} {r['era']:<12} "
              f"rears={r.get('n_rears', '-'):>4} matched={r.get('n_matched', '-'):>4} "
              f"trig={r.get('n_triggers', '-'):>4} rewarded={r.get('n_rewarded', '-'):>4}"
              f"{flag}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "session_counts.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(df)} rows, {int(df.ok.sum())} ok)")
    return df




# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8880', '#e4e3df'
C_REARS, C_MATCHED, C_REWARD = '#8a8880', '#2a78d6', '#eb6834'
ERA_BAND = {'baseline': '#dde6f0', 'training': '#f1f4ee',
            'post_lesion': '#fbe4d8', 'unassigned': '#ebeae6'}
ERA_LABEL = {'baseline': 'baseline\n(no task)', 'training': 'initial training',
             'post_lesion': 'post-lesion', 'unassigned': 'no task'}
# Rig-level change points worth marking: after this date SLEAP tracking is a
# different animal (new arena + 2026_10_14 calibration), so the zero matches on
# R5's last two sessions are a tracking failure, not a behavioural one.
# See results/newcal/REPORT.md.
RIG_CHANGES = {'2026-10-14': 'new arena + recalibration'}


def _era_runs(sub):
    """Contiguous (era, first_date, last_date) runs down a rat's timeline."""
    runs, cur, start = [], None, None
    prev = None
    for _, r in sub.iterrows():
        if r.era != cur:
            if cur is not None:
                runs.append((cur, start, prev))
            cur, start = r.era, r.date_dt
        prev = r.date_dt
    if cur is not None:
        runs.append((cur, start, prev))
    return runs


def figure(df, per_30min=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.transforms as mtransforms

    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': GRID,
        'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'lines.linewidth': 1.6,
        'grid.color': GRID, 'grid.linewidth': 0.7,
        'legend.frameon': False, 'figure.dpi': 170,
    })

    d = df[df.ok].copy()
    d['date_dt'] = pd.to_datetime(d['date'], format='%Y_%m_%d')
    d = d.sort_values(['rat', 'date_dt', 'session'])

    scale = (30.0 / d['duration_min']) if per_30min else 1.0
    for col in ('n_rears', 'n_matched', 'n_rewarded'):
        d[col + '_p'] = d[col] * scale

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.4), sharex=True,
                             constrained_layout=True)
    xlo = d.date_dt.min() - pd.Timedelta(days=3)
    xhi = d.date_dt.max() + pd.Timedelta(days=3)
    MIN_LABEL_DAYS = 9
    GAP_DAYS = 6

    for ax, rat in zip(axes, RATS):
        sub = d[d.rat == rat].reset_index(drop=True)
        ymax = max(1.0, float(sub['n_rears_p'].max())) * 1.30
        blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

        # --- era bands ---
        runs = _era_runs(sub)
        edges = [xlo]
        for i in range(len(runs) - 1):
            edges.append(runs[i][2] + (runs[i + 1][1] - runs[i][2]) / 2)
        edges.append(xhi)
        for (era, _, _), lo, hi in zip(runs, edges[:-1], edges[1:]):
            ax.axvspan(lo, hi, color=ERA_BAND.get(era, '#eee'), lw=0, zorder=0)
            if (hi - lo).days >= MIN_LABEL_DAYS:
                ax.text(lo + (hi - lo) / 2, 0.955, ERA_LABEL.get(era, era),
                        transform=blend, ha='center', va='top', fontsize=8.5,
                        color=INK2, zorder=5)

        # --- boundary rules: task onset and the lesion gap ---
        for i, (era, _, _) in enumerate(runs[1:], start=1):
            if era == 'post_lesion':
                ax.axvline(edges[i], color='#c0392b', lw=2.0, ls='--', zorder=4)
                ax.text(edges[i], 0.995, '  lesion', transform=blend, ha='left',
                        va='top', fontsize=8.5, color='#c0392b',
                        fontweight='bold', zorder=6)
            elif era == 'training':
                ax.axvline(edges[i], color=INK2, lw=1.0, ls=':', zorder=4)
                ax.text(edges[i], 0.995, '  task on', transform=blend, ha='left',
                        va='top', fontsize=8.5, color=INK2, zorder=6)

        # --- rig-level change points (only on panels with data past them) ---
        for cdate, clabel in RIG_CHANGES.items():
            cx = pd.Timestamp(cdate)
            if (sub.date_dt >= cx).any():
                ax.axvline(cx, color='#7a5fb0', lw=1.2, ls='-.', zorder=4)
                ax.text(cx, 0.995, f'  {clabel}', transform=blend, ha='left',
                        va='top', fontsize=8, color='#7a5fb0', zorder=6)

        # --- template-content changes, on the bottom edge ---
        shas = sub['template_sha'].to_numpy()
        changed = np.where(shas[1:] != shas[:-1])[0] + 1
        if len(changed):
            ax.plot(sub.date_dt.iloc[changed], np.full(len(changed), -0.028),
                    transform=blend, marker='^', ms=4.6, ls='none',
                    color='#7a5fb0', zorder=6, clip_on=False,
                    label='template changed' if rat == RATS[0] else None)

        # --- the three series; lines break across gaps > GAP_DAYS so the
        #     surgery/recovery break is not drawn as a straight run of data ---
        gap = sub.date_dt.diff().dt.days.to_numpy()
        brk = np.isfinite(gap) & (gap > GAP_DAYS)
        def _broken(col):
            y = sub[col].astype(float).to_numpy().copy()
            y[brk] = np.nan
            return y
        for col, color, ms, lw, lab in (
                ('n_rears_p', C_REARS, 2.8, 1.3, 'rears detected'),
                ('n_matched_p', C_MATCHED, 3.2, 1.6, 'matched rears (offline replay)'),
                ('n_rewarded_p', C_REWARD, 3.2, 1.6, 'rewards collected (lick-confirmed)')):
            ax.plot(sub.date_dt, _broken(col), color=color, lw=lw, zorder=3, label=lab)
            ax.plot(sub.date_dt, sub[col], color=color, marker='o', ms=ms,
                    ls='none', zorder=3)

        notask = sub[~sub.has_task_log]
        if len(notask):
            ax.plot(notask.date_dt, np.zeros(len(notask)), marker='x', ms=6,
                    ls='none', color=INK, mew=1.3, zorder=5,
                    label='no task log (rig not rewarding)')

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0, ymax)
        ax.set_ylabel('events per 30 min' if per_30min else 'events per session')
        ax.grid(axis='y', alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        lesion = sub.loc[sub.era == 'post_lesion', 'date_dt']
        note = f'   first post-lesion session {lesion.min():%b %d}' if len(lesion) else ''
        ax.set_title(f'{rat}{note}', loc='left', fontsize=12, fontweight='bold',
                     color=INK, pad=9)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[-1].set_xlabel('session date (2026)')

    handles, labels = [], []
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc='outside lower center', ncol=5, fontsize=9)

    sub_t = ('Counts normalised to a 30-minute session.' if per_30min else
             'Raw per-session counts. Baseline sessions are 20 min, task sessions 30-35 min.')
    fig.suptitle('Rears, template matches and collected rewards across training\n' + sub_t,
                 fontsize=11.5, ha='left', x=0.012, color=INK2, linespacing=1.5)

    os.makedirs(OUT, exist_ok=True)
    stem = 'era_timeline_rates' if per_30min else 'era_timeline'
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{stem}.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT}/{stem}.png / .pdf')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure-only", action="store_true",
                    help="skip recomputation; read the existing CSV")
    args = ap.parse_args()
    if args.figure_only:
        df = pd.read_csv(os.path.join(OUT, "session_counts.csv"))
    else:
        df = build()
    print(df.groupby(["rat", "era"]).size().to_string())
    figure(df)
    figure(df, per_30min=True)
