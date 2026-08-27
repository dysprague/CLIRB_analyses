"""Arena geometry: fitted circular frame, perimeter angle, sectors.

Fitted on 504k pooled COM frames across R4/R5/R6:
    centre = (-66, -98) mm, radius = 332 mm, across-angle wall-radius std 4.0 mm
The animals are essentially always at the wall (median COM radius 310 mm), so
perimeter angle THETA -- not a 2-D grid -- is the spatial coordinate.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ArenaFrame:
    cx: float
    cy: float
    radius: float

    def theta(self, xy):
        xy = np.asarray(xy)
        return np.degrees(np.arctan2(xy[..., 1] - self.cy, xy[..., 0] - self.cx))

    def radius_of(self, xy):
        xy = np.asarray(xy)
        return np.hypot(xy[..., 0] - self.cx, xy[..., 1] - self.cy)

    def wall_distance(self, xy):
        return self.radius - self.radius_of(xy)


ARENA = ArenaFrame(cx=-66.0, cy=-98.0, radius=332.0)


def fit_arena(com_xy, n_bins=24, pct=99.0):
    """Robust circle fit: minimise the across-angle std of the per-bin `pct`
    radius. Returns ArenaFrame."""
    from scipy.optimize import minimize
    xy = np.asarray(com_xy)
    xy = xy[np.isfinite(xy).all(axis=1)]

    def cost(c):
        th = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
        r = np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1])
        b = np.digitize(th, np.linspace(-np.pi, np.pi, n_bins + 1)) - 1
        rm = np.array([np.percentile(r[b == k], pct) if (b == k).sum() > 50 else np.nan
                       for k in range(n_bins)])
        return np.nanstd(rm)

    c0 = np.median(xy, axis=0)
    res = minimize(cost, c0, method='Nelder-Mead')
    cx, cy = res.x
    th = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
    r = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    b = np.digitize(th, np.linspace(-np.pi, np.pi, n_bins + 1)) - 1
    rm = np.array([np.percentile(r[b == k], pct) if (b == k).sum() > 50 else np.nan
                   for k in range(n_bins)])
    return ArenaFrame(float(cx), float(cy), float(np.nanmean(rm))), float(res.fun)


def sector_edges(width_deg):
    n = int(round(360 / width_deg))
    return np.linspace(-180, 180, n + 1)


def sector_index(theta, width_deg=30):
    """0-based sector index; theta in degrees (-180, 180]."""
    e = sector_edges(width_deg)
    return np.clip(np.digitize(np.asarray(theta), e) - 1, 0, len(e) - 2)


def sector_center(idx, width_deg=30):
    e = sector_edges(width_deg)
    return (e[:-1] + e[1:])[np.asarray(idx)] / 2.0


def port_arc(arena, port_xy, min_port_distance):
    """Wall arc (theta_lo, theta_hi) within min_port_distance of the port.

    Half-angle solved on the wall circle: the set of wall points within
    `min_port_distance` of the port, using the law of cosines.
    """
    px, py = np.asarray(port_xy).ravel()[:2]
    th_p = np.degrees(np.arctan2(py - arena.cy, px - arena.cx))
    d_p = np.hypot(px - arena.cx, py - arena.cy)
    R = arena.radius
    # |wall_point - port|^2 = R^2 + d_p^2 - 2 R d_p cos(dtheta) <= min_port^2
    c = (R ** 2 + d_p ** 2 - min_port_distance ** 2) / (2 * R * d_p)
    if c >= 1:
        return (th_p, th_p)
    if c <= -1:
        return (-180.0, 180.0)
    half = np.degrees(np.arccos(c))
    return (th_p - half, th_p + half)


def apex_height(triang, start, end, node_idx=None):
    """Max height during [start, end]. Uses the max over all keypoints' z if no
    node given (the head is the tallest during a rear)."""
    seg = triang[start:end + 1]
    if seg.size == 0:
        return np.nan
    z = seg[:, node_idx, 2] if node_idx is not None else seg[:, :, 2].max(axis=1)
    return float(np.nanmax(z))
