import os

import cv2

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from itertools import islice
from collections import deque

from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

from sklearn.decomposition import PCA
import time

from scipy.ndimage import median_filter

import PyQt6

matfile = 'data/example_keypoints/ARID1B_WK1_2022_10_10_M1_points.mat'

skel_label = sio.loadmat(matfile, simplify_cells=True)
labels = skel_label['RP2']

skeleton = skel_label['skeleton']
# skeleton
nodes = skeleton['joint_names']
nodes = list(map(str, nodes))

edges = skeleton['joints_idx']-1 # python indexings



def normalize_skeleton(points_3d):

    SpineF = points_3d[3,:]  # shape: (n_frames, 3)
    SpineM = points_3d[4, :]  # shape: (n_frames, 3)

    rotangle = np.arctan2( -(SpineF[1] - SpineM[1]), (SpineF[0] - SpineM[0]) )

    global_rotmat = np.zeros((2, 2))

    global_rotmat[0, 0] = np.cos(rotangle)
    global_rotmat[0, 1] = -np.sin(rotangle)
    global_rotmat[1, 0] = np.sin(rotangle)
    global_rotmat[1, 1] = np.cos(rotangle) 

    markers_centered = points_3d - points_3d[4,:] #23x3

    markers_rotated = markers_centered 
    markers_rotated[:,:2] = np.transpose(global_rotmat @ np.transpose(markers_rotated[:,:2]))

    return markers_rotated, global_rotmat, SpineM

def save_template_matches_video(match_indices, pc_template, template_bounds, 
                                keypoints, signal, skeleton_edges, 
                                video_path, savepath, n_components=2, max_matches=5):
    """
    Create a concatenated video of template matches showing:
    - Camera1 video
    - 3D skeleton
    - PC trajectories with template overlay
    
    Parameters:
    - match_indices: List of frame indices where matches were found
    - pc_template: Template PC trajectory, shape (template_length, n_components)
    - template_bounds: Tolerance bounds, shape (template_length, n_components)
    - keypoints: Full keypoint data, shape (n_frames, n_keypoints, 3)
    - signal: Full PC data, shape (n_frames, n_components)
    - skeleton_edges: Skeleton connections
    - video_path: Path to Camera1 video file
    - savepath: Output video path
    - n_components: Number of PC components (default 2)
    - max_matches: Maximum number of matches to include in video
    """
    
    template_length = pc_template.shape[0]
    
    # Limit number of matches
    if len(match_indices) > max_matches:
        print(f"Limiting to first {max_matches} matches for video")
        match_indices = match_indices[:max_matches]
    
    # Pre-load all video frames and skeleton data for all matches
    all_match_data = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    for match_idx in match_indices:
        start_idx = match_idx - template_length + 1
        end_idx = match_idx + 1
        
        if start_idx < 0:
            print(f"Skipping match at {match_idx} (starts before frame 0)")
            continue
        
        video_frames = []
        skeleton_frames = []
        pc_frames = []
        
        for frame_idx in range(start_idx, end_idx):
            # Read video frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, video_frame = cap.read()
            if ret:
                video_frames.append(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
            else:
                # Placeholder black frame
                video_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Get skeleton frame
            skeleton_frames.append(keypoints[frame_idx, :, :])
            
            # Get PC values
            pc_frames.append(signal[frame_idx, :n_components])
        
        all_match_data.append({
            'video_frames': video_frames,
            'skeleton_frames': skeleton_frames,
            'pc_frames': np.array(pc_frames),
            'match_idx': match_idx
        })
    
    cap.release()
    
    if len(all_match_data) == 0:
        print("No valid match data could be loaded")
        return False
    
    # Calculate total number of frames
    total_frames = len(all_match_data) * template_length
    
    # Set up figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    ax_video = plt.subplot(2, 2, 1)
    ax_skeleton = plt.subplot(2, 2, 2, projection='3d')
    ax_pc1 = plt.subplot(2, 2, 3)
    ax_pc2 = plt.subplot(2, 2, 4)
    
    # Calculate skeleton axis limits from all matches
    all_skeletons = np.concatenate([m['skeleton_frames'] for m in all_match_data], axis=0)
    x_range = [all_skeletons[:, :, 0].min() - 10, all_skeletons[:, :, 0].max() + 10]
    y_range = [all_skeletons[:, :, 1].min() - 10, all_skeletons[:, :, 1].max() + 10]
    z_range = [all_skeletons[:, :, 2].min() - 10, all_skeletons[:, :, 2].max() + 10]
    
    # Calculate PC axis limits
    pc1_range = [signal[:, 0].min() - 5, signal[:, 0].max() + 5]
    pc2_range = [signal[:, 1].min() - 5, signal[:, 1].max() + 5]
    
    def update(global_frame):
        # Determine which match and local frame
        match_num = global_frame // template_length
        local_frame = global_frame % template_length
        
        match_data = all_match_data[match_num]
        
        # Clear all axes
        ax_video.clear()
        ax_skeleton.clear()
        ax_pc1.clear()
        ax_pc2.clear()
        
        # Plot video frame
        ax_video.imshow(match_data['video_frames'][local_frame]*2)
        ax_video.set_title(f"Camera1\nMatch {match_num + 1}/{len(all_match_data)} (Frame {match_data['match_idx']})")
        ax_video.axis('off')
        
        # Plot 3D skeleton
        ax_skeleton.set_xlim(x_range)
        ax_skeleton.set_ylim(y_range)
        ax_skeleton.set_zlim(z_range)
        ax_skeleton.set_xlabel('X')
        ax_skeleton.set_ylabel('Y')
        ax_skeleton.set_zlabel('Z')
        ax_skeleton.set_title('3D Skeleton')
        
        keypoints_frame = match_data['skeleton_frames'][local_frame]
        ax_skeleton.scatter(keypoints_frame[:, 0], keypoints_frame[:, 1], keypoints_frame[:, 2],
                          c='red', s=50, alpha=0.8)
        
        if skeleton_edges is not None:
            for edge in skeleton_edges:
                pt1 = keypoints_frame[edge[0], :]
                pt2 = keypoints_frame[edge[1], :]
                ax_skeleton.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]],
                               'b-', linewidth=2, alpha=0.7)
        
        ax_skeleton.view_init(elev=20, azim=45)
        
        # Plot PC1 with template
        ax_pc1.set_xlim([0, template_length - 1])
        ax_pc1.set_ylim(pc1_range)
        
        time_axis = np.arange(template_length)
        ax_pc1.plot(time_axis, pc_template[:, 0], 'k--', linewidth=2, label='Template', alpha=0.5)
        ax_pc1.fill_between(time_axis,
                           pc_template[:, 0] - template_bounds[:, 0],
                           pc_template[:, 0] + template_bounds[:, 0],
                           alpha=0.2, color='gray', label='Bounds')
        
        ax_pc1.plot(time_axis[:local_frame + 1], match_data['pc_frames'][:local_frame + 1, 0],
                   'b-', linewidth=2, label='Current Match')
        ax_pc1.scatter(local_frame, match_data['pc_frames'][local_frame, 0],
                      c='red', s=100, zorder=5)
        
        ax_pc1.set_xlabel('Frame')
        ax_pc1.set_ylabel('PC1 Score')
        ax_pc1.set_title('PC1 Trajectory')
        ax_pc1.legend(loc='upper right')
        ax_pc1.grid(True, alpha=0.3)
        
        # Plot PC2 with template
        ax_pc2.set_xlim([0, template_length - 1])
        ax_pc2.set_ylim(pc2_range)
        
        ax_pc2.plot(time_axis, pc_template[:, 1], 'k--', linewidth=2, label='Template', alpha=0.5)
        ax_pc2.fill_between(time_axis,
                           pc_template[:, 1] - template_bounds[:, 1],
                           pc_template[:, 1] + template_bounds[:, 1],
                           alpha=0.2, color='gray', label='Bounds')
        
        ax_pc2.plot(time_axis[:local_frame + 1], match_data['pc_frames'][:local_frame + 1, 1],
                   'b-', linewidth=2, label='Current Match')
        ax_pc2.scatter(local_frame, match_data['pc_frames'][local_frame, 1],
                      c='red', s=100, zorder=5)
        
        ax_pc2.set_xlabel('Frame')
        ax_pc2.set_ylabel('PC2 Score')
        ax_pc2.set_title('PC2 Trajectory')
        ax_pc2.legend(loc='upper right')
        ax_pc2.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'Template Matches - Frame {local_frame + 1}/{template_length}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return ax_video, ax_skeleton, ax_pc1, ax_pc2
    
    # Create animation
    print(f"Creating animation with {total_frames} total frames...")
    anim = FuncAnimation(fig, update, frames=total_frames,
                                  interval=50, blit=False, repeat=False)
    
    # Save animation
    print(f"Saving video to {savepath}...")
    anim.save(savepath, writer='ffmpeg', fps=20, dpi=100)
    plt.close(fig)
    
    print(f"Successfully saved video!")
    return True

def animate_template(keys_3D, keys_pca, pc_stds, pc_means, fps, num_frames, save_folder, fname, normalize=True, window_seconds=20, figsize=(10,8), flip_skel=False):
    T, K, D = keys_3D.shape

    num_frames = min(num_frames, T)

    t_all = np.arange(num_frames, dtype=float) / float(fps)
    win_frames = int(round(window_seconds * float(fps)))
    win_frames = max(win_frames, 1)

    pcs_to_plot = 6
    pc_min = []
    pc_max = []
    for pc in range(pcs_to_plot):
        y = keys_pca[:num_frames, pc]
        ylo, yhi = np.min(y), np.max(y)
        if np.isclose(ylo, yhi):
            # avoid zero-height axes
            pad = 1.0
        else:
            pad = 0.05 * (yhi - ylo)
        pc_min.append(ylo - pad)
        pc_max.append(yhi + pad)
    pc_min = np.array(pc_min)
    pc_max = np.array(pc_max)

    # --------------------------
    # Figure & layout (50/50)
    # --------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.25, figure=fig)

    # Top: 3D skeleton (takes upper half)
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.set_xlim(-400, 400)
    ax3d.set_ylim(-400, 400)
    ax3d.set_zlim(0, 200)
    ax3d.set_title("3D Skeleton")

    # Bottom: 2x3 grid of PC time series
    gs_bottom = gs[1, 0].subgridspec(2, 3, wspace=0.25, hspace=0.35)
    pc_axes = []
    pc_lines = []
    for r in range(2):
        for c in range(3):
            pc_idx = r * 3 + c  # 0..5
            ax = fig.add_subplot(gs_bottom[r, c])
            ax.set_title(f"PC {pc_idx + 1}")
            ax.set_xlim(0, num_frames/fps)
            ax.set_ylim(pc_min[pc_idx]-2*pc_stds[pc_idx], [pc_idx]+2*pc_stds[pc_idx])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")
            (line,) = ax.plot([], [], lw=1.75, c='red')
            pc_axes.append(ax)
            pc_lines.append(line)

    pc_axes[4].set_title('COM x velocity')
    pc_axes[5].set_title('COM y velocity')

    # --- PCs ---
    x = np.linspace(0, num_frames/fps, num_frames)

    for pc_idx in range(pcs_to_plot):
        y = keys_pca[:, pc_idx]
        std = pc_stds[pc_idx]
        pc_lines[pc_idx].set_data(x, y)
        pc_axes[pc_idx].fill_between(x, y+std, y-std, alpha=0.3, color='grey')
        # roll the x-window to show last window_seconds (or from 0)

    # --------------------------
    # 3D artists
    # --------------------------
    sc = ax3d.scatter([], [], [], c='r', s=30)
    bone_lines = [ax3d.plot([], [], [], 'k-', lw=2)[0] for _ in edges]

    # --------------------------
    # Frame update
    # --------------------------
    def _normalize_frame(frame_points):
        """Be robust to different normalize_skeleton signatures."""
        out = frame_points
        try:
            # if your normalize_skeleton returns (markers, rot, center)
            res = normalize_skeleton(frame_points)
            if isinstance(res, tuple):
                out = res[0]
            else:
                out = res
        except Exception:
            # fall back to raw
            out = frame_points
        return out

    def update(frame):
        # --- 3D skeleton ---
        if normalize:
            markers = _normalize_frame(keys_3D[frame])
        else:
            markers = keys_3D[frame]

        if flip_skel:
            sc._offsets3d = (
            markers[:, 0],
            markers[:, 1],
            100-markers[:, 2]  # flip Z for your viewing convention
            )
            for i, (a, b) in enumerate(edges):
                start, end = markers[a], markers[b]
                bone_lines[i].set_data([start[0], end[0]], [start[1], end[1]])
                bone_lines[i].set_3d_properties([100-start[2], 100-end[2]])

        else:

            sc._offsets3d = (
                markers[:, 0],
                markers[:, 1],
                markers[:, 2]  # flip Z for your viewing convention
            )
            for i, (a, b) in enumerate(edges):
                start, end = markers[a], markers[b]
                bone_lines[i].set_data([start[0], end[0]], [start[1], end[1]])
                bone_lines[i].set_3d_properties([start[2], end[2]])

        # Return all artists we touched (no blit for 3D)
        return [sc, *bone_lines, *pc_lines]

    # --------------------------
    # Animate & save
    # --------------------------
    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000.0 / float(fps),
        blit=False,
    )

    os.makedirs(save_folder, exist_ok=True)
    writer = FFMpegWriter(fps=fps)
    out_path = os.path.join(save_folder, f"{fname}.mp4")
    ani.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


def animate_with_PCs(keys_3D, keys_pca, fps, num_frames, save_folder, fname, normalize=True, window_seconds=20, figsize=(10,8)):
    T, K, D = keys_3D.shape

    num_frames = min(num_frames, T)

    t_all = np.arange(num_frames, dtype=float) / float(fps)
    win_frames = int(round(window_seconds * float(fps)))
    win_frames = max(win_frames, 1)

    pcs_to_plot = 6
    pc_min = []
    pc_max = []
    for pc in range(pcs_to_plot):
        y = keys_pca[:num_frames, pc]
        ylo, yhi = np.min(y), np.max(y)
        if np.isclose(ylo, yhi):
            # avoid zero-height axes
            pad = 1.0
        else:
            pad = 0.05 * (yhi - ylo)
        pc_min.append(ylo - pad)
        pc_max.append(yhi + pad)
    pc_min = np.array(pc_min)
    pc_max = np.array(pc_max)

    # --------------------------
    # Figure & layout (50/50)
    # --------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.25, figure=fig)

    # Top: 3D skeleton (takes upper half)
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.set_xlim(-600, 600)
    ax3d.set_ylim(-600, 600)
    ax3d.set_zlim(0, 200)
    ax3d.set_title("3D Skeleton")

    # Bottom: 2x3 grid of PC time series
    gs_bottom = gs[1, 0].subgridspec(2, 3, wspace=0.25, hspace=0.35)
    pc_axes = []
    pc_lines = []
    for r in range(2):
        for c in range(3):
            pc_idx = r * 3 + c  # 0..5
            ax = fig.add_subplot(gs_bottom[r, c])
            ax.set_title(f"PC {pc_idx + 1}")
            ax.set_xlim(0, max(window_seconds, t_all[-1] if len(t_all) else window_seconds))
            ax.set_ylim(pc_min[pc_idx], pc_max[pc_idx])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")
            (line,) = ax.plot([], [], lw=1.75)
            pc_axes.append(ax)
            pc_lines.append(line)

    pc_axes[4].set_title('COM x velocity')
    pc_axes[5].set_title('COM y velocity')

    # --------------------------
    # 3D artists
    # --------------------------
    sc = ax3d.scatter([], [], [], c='r', s=30)
    bone_lines = [ax3d.plot([], [], [], 'k-', lw=2)[0] for _ in edges]

    # --------------------------
    # Frame update
    # --------------------------
    def _normalize_frame(frame_points):
        """Be robust to different normalize_skeleton signatures."""
        out = frame_points
        try:
            # if your normalize_skeleton returns (markers, rot, center)
            res = normalize_skeleton(frame_points)
            if isinstance(res, tuple):
                out = res[0]
            else:
                out = res
        except Exception:
            # fall back to raw
            out = frame_points
        return out

    def update(frame):
        # --- 3D skeleton ---
        if normalize:
            markers = _normalize_frame(keys_3D[frame])
        else:
            markers = keys_3D[frame]

        sc._offsets3d = (
            markers[:, 0],
            markers[:, 1],
            100 - markers[:, 2]  # flip Z for your viewing convention
        )
        for i, (a, b) in enumerate(edges):
            start, end = markers[a], markers[b]
            bone_lines[i].set_data([start[0], end[0]], [start[1], end[1]])
            bone_lines[i].set_3d_properties([100 - start[2], 100 - end[2]])

        # --- PCs ---
        t_now = t_all[frame]
        start_idx = max(0, frame - win_frames + 1)
        x = t_all[start_idx:frame + 1]

        for pc_idx in range(pcs_to_plot):
            y = keys_pca[start_idx:frame + 1, pc_idx]
            pc_lines[pc_idx].set_data(x, y)
            # roll the x-window to show last window_seconds (or from 0)
            xmin = max(0.0, t_now - window_seconds)
            xmax = max(window_seconds, t_now) if t_now < window_seconds else t_now
            pc_axes[pc_idx].set_xlim(xmin, xmax)

        # Return all artists we touched (no blit for 3D)
        return [sc, *bone_lines, *pc_lines]

    # --------------------------
    # Animate & save
    # --------------------------
    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000.0 / float(fps),
        blit=False,
    )

    os.makedirs(save_folder, exist_ok=True)
    writer = FFMpegWriter(fps=fps)
    out_path = os.path.join(save_folder, f"{fname}.mp4")
    ani.save(out_path, writer=writer)
    plt.close(fig)
    return out_path

def animate_skeleton_with_pca_segments(
    keys_3D,
    pca_scores,
    match_idxs,
    fps,
    save_folder,
    fname,
    edges,
    normalize=True,
    pre_frames=40,
    pcs_to_plot=6,
    figsize=(10, 8),
):
    """
    Animate only the windows preceding each match index, and show PCs (1..6) for
    the *entire current segment* while the 3D skeleton animates.

    Parameters
    ----------
    keys_3D : ndarray, shape (T, K, 3)
        3D keypoints over time.
    pca_scores : ndarray, shape (T, M>=6)
        Per-frame PCA embedding; we will display first 6 PCs.
    match_idxs : list[int]
        Indices in [0, T-1]. For each idx, we animate frames [idx-pre_frames, idx].
    fps : int or float
        Frames per second for the output video.
    save_folder : str
        Directory to save the MP4.
    fname : str
        Filename stem (no extension).
    edges : list[tuple[int,int]]
        Bone connections for drawing lines between keypoints.
    normalize : bool
        If True, apply normalize_skeleton() per frame before plotting (robust to both
        return signatures: markers only OR (markers, rot, center)).
    pre_frames : int
        Number of frames before each match index to include in each segment.
    pcs_to_plot : int
        Number of PCs to display (first N columns of pca_scores); default 6.
    figsize : tuple
        Matplotlib figure size.

    Returns
    -------
    out_path : str
        Path to the saved MP4 file.
    """

    T, K, D = keys_3D.shape
    assert D == 3, "keys_3D must be (T, K, 3)"
    assert pca_scores.shape[0] == T, "pca_scores must have same T as keys_3D"
    assert pca_scores.shape[1] >= pcs_to_plot, f"Need at least {pcs_to_plot} PCs"

    # --------------------------
    # Build segments
    # --------------------------
    # For each match idx, segment = [start, end] inclusive.
    segs = []
    for idx in match_idxs:
        end = int(np.clip(idx, 0, T - 1))
        start = int(max(0, end - pre_frames))
        if start <= end:
            segs.append((start, end))
    if not segs:
        raise ValueError("No valid segments after applying match_idxs and pre_frames.")

    # Concatenate indices for animation timeline
    seg_frames = [np.arange(s, e + 1, dtype=int) for (s, e) in segs]
    concat_idx = np.concatenate(seg_frames, axis=0)  # shape (total_frames,)
    total_frames = concat_idx.shape[0]

    # Per-segment cumulative lengths to locate which segment a global frame belongs to
    seg_lengths = np.array([len(fr) for fr in seg_frames], dtype=int)
    seg_cum = np.cumsum(seg_lengths)  # end positions (1-based)
    seg_cum_start = np.concatenate(([0], seg_cum[:-1]))  # starts in concatenated timeline

    # Precompute per-segment time vectors and PC arrays (static per segment)
    seg_times = [np.arange(L, dtype=float) / float(fps) for L in seg_lengths]
    seg_pcs = [pca_scores[frames][:, :pcs_to_plot] for frames in seg_frames]  # list of (L, pcs)
    # y-limits per PC per segment (with a little padding)
    seg_pc_lims = []
    for s_idx, pcs in enumerate(seg_pcs):
        lims = []
        for pc in range(pcs_to_plot):
            y = pcs[:, pc]
            ylo, yhi = np.min(y), np.max(y)
            if np.isclose(ylo, yhi):
                pad = 1.0
            else:
                pad = 0.05 * (yhi - ylo)
            lims.append((ylo - pad, yhi + pad))
        seg_pc_lims.append(lims)  # list of list[(ymin, ymax)]

    # --------------------------
    # Figure layout: 50% 3D top, 50% PCs bottom
    # --------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.25, figure=fig)

    # Top: 3D skeleton
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.set_title("3D Skeleton")
    ax3d.set_xlim(-600, 600)
    ax3d.set_ylim(-600, 600)
    ax3d.set_zlim(0, 200)

    # Bottom: 2x3 grid for PCs (first 6 by default)
    rows = 2
    cols = 3
    assert pcs_to_plot <= rows * cols, "pcs_to_plot exceeds panel capacity (2x3)."
    gs_bottom = gs[1, 0].subgridspec(rows, cols, wspace=0.25, hspace=0.35)
    pc_axes, pc_lines = [], []
    for r in range(rows):
        for c in range(cols):
            pc_idx = r * cols + c
            if pc_idx >= pcs_to_plot:
                break
            ax = fig.add_subplot(gs_bottom[r, c])
            ax.set_title(f"PC {pc_idx + 1}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Score")
            (line,) = ax.plot([], [], lw=1.75)
            pc_axes.append(ax)
            pc_lines.append(line)

    # 3D artists
    sc = ax3d.scatter([], [], [], c='r', s=30)
    bone_lines = [ax3d.plot([], [], [], 'k-', lw=2)[0] for _ in edges]

    # Helper to normalize one frame (compatible with both return signatures)
    def _normalize_frame(frame_points):
        if not normalize:
            return frame_points
        try:
            res = normalize_skeleton(frame_points)
            return res[0] if isinstance(res, tuple) else res
        except Exception:
            return frame_points

    # Track segment changes so PC panels only update when the segment switches
    last_seg_idx = None

    def _locate_segment(global_frame_idx):
        """Return (seg_idx, local_idx) for concatenated frame index."""
        # seg_idx is first index where seg_cum > global_frame_idx
        seg_idx = int(np.searchsorted(seg_cum, global_frame_idx, side="right"))
        local_idx = global_frame_idx - seg_cum_start[seg_idx]
        return seg_idx, local_idx

    # Update function
    def update(global_f):
        nonlocal last_seg_idx

        seg_idx, local_idx = _locate_segment(global_f)
        # Current absolute frame in original sequence
        frame = concat_idx[global_f]

        # 3D skeleton update
        markers = _normalize_frame(keys_3D[frame])
        sc._offsets3d = (markers[:, 0], markers[:, 1], 100 - markers[:, 2])
        for i, (a, b) in enumerate(edges):
            start, end = markers[a], markers[b]
            bone_lines[i].set_data([start[0], end[0]], [start[1], end[1]])
            bone_lines[i].set_3d_properties([100 - start[2], 100 - end[2]])

        # PC panels: refresh **only when the segment changes**
        if last_seg_idx != seg_idx:
            times = seg_times[seg_idx]
            pcs = seg_pcs[seg_idx]  # (L, pcs_to_plot)
            lims = seg_pc_lims[seg_idx]
            # Update each PC plot with the full segment trace and fixed limits
            for pc_idx in range(pcs_to_plot):
                pc_lines[pc_idx].set_data(times, pcs[:, pc_idx])
                ymin, ymax = lims[pc_idx]
                pc_axes[pc_idx].set_xlim(0.0, times[-1] if len(times) else 1.0 / float(fps))
                pc_axes[pc_idx].set_ylim(ymin, ymax)
            last_seg_idx = seg_idx

        return [sc, *bone_lines, *pc_lines]

    # Animate & save
    ani = FuncAnimation(
        fig,
        update,
        frames=int(total_frames),
        interval=1000.0 / float(fps),
        blit=False,
    )

    os.makedirs(save_folder, exist_ok=True)
    writer = FFMpegWriter(fps=fps)
    out_path = os.path.join(save_folder, f"{fname}.mp4")
    ani.save(out_path, writer=writer)
    plt.close(fig)
    return out_path