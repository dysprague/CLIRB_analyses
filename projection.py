"""
Camera projection utilities: project 3D keypoints onto 2D camera views.
Uses calibration parameters (intrinsics, extrinsics, distortion).
"""
import numpy as np
import cv2


def undistort_points(points_2d, K, dist_coeffs):
    """
    Undistort 2D pixel coordinates.

    Parameters
    ----------
    points_2d : (N, 2) array
    K : (3, 3) intrinsic matrix
    dist_coeffs : [k1, k2, p1, p2]

    Returns
    -------
    (N, 2) normalized undistorted coordinates
    """
    pts = points_2d.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, np.array(dist_coeffs))
    return undist.reshape(-1, 2)


def project_3d_to_2d(points_3d, K, r, t, dist_coeffs):
    """
    Project 3D points onto a 2D camera image plane.

    Parameters
    ----------
    points_3d : (N, 3) array of 3D world coordinates
    K : (3, 3) intrinsic matrix
    r : (3, 3) rotation matrix
    t : (3,) translation vector (note: negated from calibration file)
    dist_coeffs : [k1, k2, p1, p2]

    Returns
    -------
    points_2d : (N, 2) array of pixel coordinates
    """
    # Transform to camera coordinates
    # Note: calibration stores t such that we use -t, but load_calibration
    # already negates it. Here we use the values as-is.
    pts_cam = (r @ points_3d.T).T + t.reshape(1, 3)

    # Normalize
    pts_norm = pts_cam[:, :2] / pts_cam[:, 2:3]

    # Apply distortion
    x, y = pts_norm[:, 0], pts_norm[:, 1]
    r2 = x**2 + y**2
    k1, k2, p1, p2 = dist_coeffs

    # Radial distortion
    radial = 1 + k1 * r2 + k2 * r2**2
    x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_dist = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    # Apply intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * x_dist + cx
    v = fy * y_dist + cy

    return np.column_stack([u, v])


def project_3d_to_2d_for_camera(points_3d, calibration_folder, camera_idx=1):
    """
    Convenience: load calibration and project 3D points for a specific camera.

    Parameters
    ----------
    points_3d : (N, 3) or (n_frames, N, 3)
    calibration_folder : str
    camera_idx : int — which camera (0, 1, 2)

    Returns
    -------
    points_2d : (N, 2) or (n_frames, N, 2)
    """
    from data_io import load_calibration
    import scipy.io as sio
    import os

    cam_files = sorted(
        f for f in os.listdir(calibration_folder)
        if f.startswith("hires_cam") and f.endswith("_params.mat")
    )
    fname = cam_files[camera_idx]
    params = sio.loadmat(os.path.join(calibration_folder, fname), simplify_cells=True)

    K = np.transpose(params["K"])
    r = np.transpose(params["r"])
    t = -params["t"]
    Rdist = params["RDistort"]
    Tdist = params["TDistort"]
    dist_coeffs = [Rdist[0], Rdist[1], Tdist[0], Tdist[1]]

    if points_3d.ndim == 3:
        # Batch: (n_frames, n_keypoints, 3)
        n_frames = points_3d.shape[0]
        result = np.zeros((n_frames, points_3d.shape[1], 2))
        for i in range(n_frames):
            result[i] = project_3d_to_2d(points_3d[i], K, r, t, dist_coeffs)
        return result
    else:
        return project_3d_to_2d(points_3d, K, r, t, dist_coeffs)
