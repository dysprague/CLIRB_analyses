"""
Shared visualization utilities for plotting skeletons, PCs, and events.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from config import EDGES, NODES
from datetime import datetime


# ---------------------------------------------------------------------------
# Skeleton plotting
# ---------------------------------------------------------------------------
def plot_skeleton_3d(ax, keypoints, edges=EDGES, color="blue", alpha=0.8,
                     marker_size=30, linewidth=2):
    """
    Plot a single-frame 3D skeleton on a matplotlib 3D axis.

    Parameters
    ----------
    ax : Axes3D
    keypoints : (n_keypoints, 3)
    """
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
               c=color, s=marker_size, alpha=alpha, depthshade=True)
    for e in edges:
        pts = keypoints[e, :]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=color, linewidth=linewidth, alpha=alpha * 0.8)


def set_skeleton_axes(ax, keypoints, pad=50):
    """Set equal-aspect 3D axes limits from keypoint data."""
    for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        lo = keypoints[:, dim].min() - pad
        hi = keypoints[:, dim].max() + pad
        setter(lo, hi)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


# ---------------------------------------------------------------------------
# Event plots
# ---------------------------------------------------------------------------
def plot_event_raster(ax, event_dict, session_duration_s=None):
    """
    Plot a raster of events (licks, tones, rewards).

    Parameters
    ----------
    ax : matplotlib Axes
    event_dict : dict mapping event names to arrays of times in ms
    """
    colors = {"lick": "blue", "tone": "red", "reward": "green"}
    y_positions = {"lick": 1, "tone": 2, "reward": 3}

    for name, times_ms in event_dict.items():
        if name not in y_positions:
            continue
        y = y_positions[name]
        times_s = np.array(times_ms) / 1000.0
        ax.vlines(times_s, y - 0.4, y + 0.4, colors=colors.get(name, "gray"),
                  linewidth=0.8, label=name)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel("Time (s)")
    if session_duration_s:
        ax.set_xlim(0, session_duration_s)


def plot_peri_event_psth(ax, event_times_ms, signal_times_ms,
                         pre_ms=5000, post_ms=5000, bin_ms=500,
                         label="Events"):
    """
    Plot peri-event time histogram.

    Parameters
    ----------
    ax : matplotlib Axes
    event_times_ms : array of trigger event times
    signal_times_ms : array of signal event times (e.g., licks)
    """
    bin_edges = np.arange(-pre_ms, post_ms + bin_ms, bin_ms)
    all_counts = np.zeros(len(bin_edges) - 1)

    for ev_t in event_times_ms:
        relative = signal_times_ms - ev_t
        counts, _ = np.histogram(relative, bins=bin_edges)
        all_counts += counts

    if len(event_times_ms) > 0:
        all_counts /= len(event_times_ms)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / 1000.0
    ax.bar(bin_centers, all_counts, width=bin_ms / 1000.0 * 0.9,
           alpha=0.7, label=label)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time relative to event (s)")
    ax.set_ylabel("Avg count per bin")


# ---------------------------------------------------------------------------
# Session date parsing
# ---------------------------------------------------------------------------
def session_to_datetime(session_id):
    """Parse 'YYYY_MM_DD_N' session ID to datetime."""
    parts = session_id.split("_")
    return datetime(int(parts[0]), int(parts[1]), int(parts[2]))


# ---------------------------------------------------------------------------
# Multi-session feature plots
# ---------------------------------------------------------------------------
def plot_feature_over_sessions(sessions, feature_values, feature_name,
                               ax=None, color="blue", label=None):
    """
    Plot a feature (mean +/- std) over sessions.

    Parameters
    ----------
    sessions : list of session ID strings
    feature_values : list of arrays, one per session
    feature_name : str
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    dates = [session_to_datetime(s) for s in sessions]
    means = [np.mean(v) if len(v) > 0 else np.nan for v in feature_values]
    stds = [np.std(v) if len(v) > 0 else 0 for v in feature_values]

    ax.errorbar(dates, means, yerr=stds, fmt="o-", color=color,
                capsize=3, label=label)
    ax.set_ylabel(feature_name)
    ax.set_xlabel("Date")
    ax.tick_params(axis="x", rotation=45)
    return ax
