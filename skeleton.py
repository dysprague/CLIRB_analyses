"""
Skeleton normalization and visualization utilities.
"""
import numpy as np
from config import NODES, EDGES, NODE_IDX


def normalize_skeleton_batch(points_3d):
    """
    Translate and rotate 3D skeleton data to egocentric coordinates.

    Centers on SpineM (index 4) and rotates so SpineF-SpineM axis
    points along +X.

    Parameters
    ----------
    points_3d : (n_frames, n_keypoints, 3) array

    Returns
    -------
    markers_rotated : (n_frames, n_keypoints, 3) — egocentric coordinates
    global_rotmat   : (n_frames, 2, 2) — per-frame rotation matrices
    spine_m         : (n_frames, 3) — original SpineM positions
    """
    SpineF = points_3d[:, NODE_IDX["SpineF"], :]  # (n_frames, 3)
    SpineM = points_3d[:, NODE_IDX["SpineM"], :]  # (n_frames, 3)

    rotangle = np.arctan2(
        -(SpineF[:, 1] - SpineM[:, 1]),
        (SpineF[:, 0] - SpineM[:, 0]),
    )

    cos_a = np.cos(rotangle)
    sin_a = np.sin(rotangle)
    global_rotmat = np.zeros((points_3d.shape[0], 2, 2))
    global_rotmat[:, 0, 0] = cos_a
    global_rotmat[:, 0, 1] = -sin_a
    global_rotmat[:, 1, 0] = sin_a
    global_rotmat[:, 1, 1] = cos_a

    markers_centered = points_3d - SpineM[:, np.newaxis, :]
    markers_rotated = markers_centered.copy()
    markers_rotated[:, :, :2] = np.einsum(
        "fij,fkj->fki", global_rotmat, markers_centered[:, :, :2]
    )

    return markers_rotated, global_rotmat, SpineM


def normalize_skeleton_single(points_3d):
    """
    Single-frame version of normalize_skeleton_batch.

    Parameters
    ----------
    points_3d : (n_keypoints, 3) array

    Returns
    -------
    markers_rotated : (n_keypoints, 3)
    global_rotmat   : (2, 2)
    spine_m         : (3,)
    """
    SpineF = points_3d[NODE_IDX["SpineF"], :]
    SpineM = points_3d[NODE_IDX["SpineM"], :]

    rotangle = np.arctan2(-(SpineF[1] - SpineM[1]), (SpineF[0] - SpineM[0]))
    global_rotmat = np.array([
        [np.cos(rotangle), -np.sin(rotangle)],
        [np.sin(rotangle),  np.cos(rotangle)],
    ])

    markers_centered = points_3d - SpineM
    markers_rotated = markers_centered.copy()
    markers_rotated[:, :2] = (global_rotmat @ markers_centered[:, :2].T).T

    return markers_rotated, global_rotmat, SpineM


def project_to_pcs(keys_3d_rotated, pc_weights, feature_means):
    """
    Project egocentric keypoints into PC space.

    Parameters
    ----------
    keys_3d_rotated : (n_frames, n_keypoints, 3) — egocentric coordinates
    pc_weights      : (n_pcs, n_features) — PCA components
    feature_means   : (n_features,) — mean used during PCA fit

    Returns
    -------
    projected : (n_frames, n_pcs)
    """
    flat = keys_3d_rotated.reshape(keys_3d_rotated.shape[0], -1)
    return (flat - feature_means) @ pc_weights.T
