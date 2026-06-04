"""
Two corrector models, both predicting a residual added to the SLEAP input.

  LinearCorrector  — single dense layer over the flattened (23,3) keypoints.
                     Equivalent to "per-keypoint debias plus all linear couplings."
  MLPCorrector     — small MLP with ReLU. Can learn pose-conditional corrections.

Both take and return shape (B, 23, 3).
"""
import torch
import torch.nn as nn

N_KP = 23
N_DIM = 3
IN_DIM = N_KP * N_DIM  # 69


class LinearCorrector(nn.Module):
    """y = x + W @ flatten(x) + b   reshaped back to (23, 3)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(IN_DIM, IN_DIM)
        # Init small so the model starts near identity
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):  # x: (B, 23, 3)
        B = x.shape[0]
        flat = x.reshape(B, -1)
        delta = self.linear(flat).reshape(B, N_KP, N_DIM)
        return x + delta


class MLPCorrector(nn.Module):
    """y = x + MLP(flatten(x)).reshape(23, 3).

    Default: 2 hidden layers of 128 units, ReLU. Residual skip on the output.
    """

    def __init__(self, hidden: int = 128, n_hidden_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        d_in = IN_DIM
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        layers.append(nn.Linear(d_in, IN_DIM))
        self.mlp = nn.Sequential(*layers)
        # Init the last layer to zero so the model starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):  # x: (B, 23, 3)
        B = x.shape[0]
        flat = x.reshape(B, -1)
        delta = self.mlp(flat).reshape(B, N_KP, N_DIM)
        return x + delta


class TemporalMLPCorrector(nn.Module):
    """y_t = x_t + MLP(flatten(x_{t-T+1:t+1})).reshape(23, 3).

    Takes a (B, T_ctx, 23, 3) causal window, applies an MLP over the flattened
    window (T_ctx * 69 inputs), and predicts a (23, 3) residual that is added
    to the LAST frame in the window. Output is the corrected pose at time t.

    Default: T_ctx=5 frames, 2 hidden layers of 128 units.
    """

    def __init__(self, ctx: int = 5, hidden: int = 128,
                 n_hidden_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ctx = ctx
        in_dim = ctx * IN_DIM
        layers = []
        d_in = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        layers.append(nn.Linear(d_in, IN_DIM))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):  # x: (B, T_ctx, 23, 3)
        if x.dim() == 3:
            # Single-frame input — broadcast to a window of 1 (no context).
            # Mostly here so the same code path can run with ctx=1.
            x = x.unsqueeze(1)
        B, T, _, _ = x.shape
        flat = x.reshape(B, -1)
        delta = self.mlp(flat).reshape(B, N_KP, N_DIM)
        # Residual is added to the LAST frame
        return x[:, -1, :, :] + delta


class VelAccMLPCorrector(nn.Module):
    """y = x + MLP(flatten([pose, velocity, acceleration])).reshape(23, 3).

    Input: (B, 3, 23, 3) — channel 0 = pose, channel 1 = velocity, channel 2 = acceleration.
    The first frame has zero velocity/acceleration; the second has zero acceleration
    (these are computed in the data loader / inference adapter).
    """

    def __init__(self, hidden: int = 128, n_hidden_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        in_dim = 3 * IN_DIM   # pose + vel + acc, each 69
        layers = []
        d_in = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        layers.append(nn.Linear(d_in, IN_DIM))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):  # x: (B, 3, 23, 3)
        B = x.shape[0]
        flat = x.reshape(B, -1)
        delta = self.mlp(flat).reshape(B, N_KP, N_DIM)
        return x[:, 0, :, :] + delta  # residual added to pose channel


class GNNCorrector(nn.Module):
    """Skeleton-aware GNN over the 23-keypoint graph.

    Each node aggregates its neighbors via mean-pooling, applies a linear
    update, and outputs a per-node residual. We do `n_layers` rounds of
    message passing, then a per-node MLP head produces the residual.

    The graph is built once from `EDGES` (anatomical bones, undirected) — we
    pre-compute the (23, 23) symmetric adjacency including self-loops for
    efficient batched matmul.
    """

    def __init__(self, hidden: int = 64, n_layers: int = 3,
                 edges=None, dropout: float = 0.0):
        super().__init__()
        if edges is None:
            from config import EDGES as _EDGES
            edges = _EDGES
        # Symmetric adjacency with self-loops, row-normalized
        A = torch.zeros(N_KP, N_KP)
        for (i, j) in edges:
            A[i, j] = 1.0; A[j, i] = 1.0
        A += torch.eye(N_KP)
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        A = A / deg
        self.register_buffer("A", A)

        # Per-node MLP layers. Each layer: aggregate neighbors -> linear -> ReLU
        self.proj_in = nn.Linear(N_DIM, hidden)
        self.layers = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, N_DIM),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):  # x: (B, 23, 3)
        h = self.proj_in(x)              # (B, 23, hidden)
        for lin in self.layers:
            # Aggregate neighbors: A @ h
            agg = torch.einsum("ij,bjd->bid", self.A, h)
            h = nn.functional.relu(lin(agg))
            if self.dropout is not None:
                h = self.dropout(h)
        delta = self.head(h)             # (B, 23, 3)
        return x + delta


class PerRatHeadCorrector(nn.Module):
    """A small MLP head trained on top of a frozen base corrector.

    Pipeline at inference:
        y = base(x)               # produces a corrected pose; frozen
        y_final = y + head(y)     # head learns the residual *that base missed*

    The base checkpoint is frozen on construction. Optimizer is only over
    head parameters.
    """

    def __init__(self, base_ckpt: str, hidden: int = 64, n_hidden_layers: int = 2):
        super().__init__()
        # Load and freeze the base
        import torch as _torch
        ck = _torch.load(base_ckpt, map_location="cpu", weights_only=False)
        kw = dict(hidden=ck.get("hidden", 128),
                  n_hidden_layers=ck.get("n_hidden_layers", 2))
        if ck["model_name"] == "temporal_mlp":
            kw["ctx"] = ck.get("ctx", 5)
        if ck["model_name"] == "velacc_mlp":
            pass
        if ck["model_name"] == "gnn":
            kw = {"hidden": ck.get("hidden", 64), "n_layers": ck.get("n_layers", 3)}
        self.base = build_model(ck["model_name"], **kw)
        self.base.load_state_dict(ck["state_dict"])
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        # The head sees the base output (B, 23, 3)
        layers = []
        d_in = IN_DIM
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ReLU(inplace=True))
            d_in = hidden
        layers.append(nn.Linear(d_in, IN_DIM))
        self.head = nn.Sequential(*layers)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self._base_ctx = ck.get("ctx", 1)
        self._base_kind = ck["model_name"]

    def forward(self, x):
        # Base forward path. The temporal_mlp's forward already handles the
        # window dimension internally; for ctx>1 we expect x of shape (B, ctx, 23, 3).
        with torch.no_grad():
            y = self.base(x)
        delta = self.head(y.reshape(y.shape[0], -1)).reshape(y.shape)
        return y + delta


class TriangulationRefiner(nn.Module):
    """PointNet-style 2D-input corrector.

    Per-keypoint feature dim 21:
        xyz_triang            (3)   triangulated 3D (Procrustes-aligned into DANNCE world)
        per_cam_xy_normalized (6)   3 cams x 2, pixel coords normalized to [-1, 1]
        per_cam_conf          (3)   SLEAP detection confidence per camera
        per_cam_reproj_resid  (6)   3 cams x 2, normalized (detected - reprojected)
        per_cam_visibility    (3)   1 if detection is finite and conf > 0, else 0

    Forward: x (B, 23, 21).  The first 3 channels of each keypoint are the
    triangulated xyz; the residual is added back to those after a per-kp MLP
    that also sees a max-pooled global summary.

    Inputs that should be NaN (wrong-side-of-camera, missing detection) must be
    zeroed by the trainer; the visibility channel carries the mask.
    """

    PER_KP_IN_DIM = 21

    def __init__(self, hidden: int = 128, n_per_kp_layers: int = 3,
                 global_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        # Per-keypoint shared MLP -> (B, 23, hidden)
        layers = []
        d_in = self.PER_KP_IN_DIM
        for _ in range(n_per_kp_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        self.per_kp = nn.Sequential(*layers)

        # Global summary projection: hidden -> global_dim, then max-pool over kp.
        self.global_proj = nn.Linear(hidden, global_dim)

        # Per-kp head: takes [per_kp_feat (hidden), global_feat (global_dim)]
        # and outputs a 3-vec residual.
        head_in = hidden + global_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):  # x: (B, 23, 21)
        B, K, F = x.shape
        assert F == self.PER_KP_IN_DIM, (
            f"TriangulationRefiner expects per-kp dim {self.PER_KP_IN_DIM}, got {F}"
        )
        h = self.per_kp(x)                     # (B, 23, hidden)
        g = self.global_proj(h)                # (B, 23, global_dim)
        g_pooled, _ = g.max(dim=1)             # (B, global_dim)
        g_broadcast = g_pooled.unsqueeze(1).expand(-1, K, -1)  # (B, 23, global_dim)
        feat = torch.cat([h, g_broadcast], dim=-1)             # (B, 23, hidden+gd)
        delta = self.head(feat)                                # (B, 23, 3)
        xyz_triang = x[..., :3]                                # (B, 23, 3)
        return xyz_triang + delta


class TemporalTriangulationRefiner(nn.Module):
    """T_ctx-frame 2D-input corrector.

    Per-frame per-kp feature dim is the same 21 as TriangulationRefiner. The
    network sees a (B, T_ctx, 23, 21) window and outputs a single (B, 23, 3)
    pose for the LAST frame in the window (causal).

    Pipeline:
      1. Per-frame, per-kp MLP shared across (T_ctx, 23): (B,T,23,21) -> (B,T,23,H).
      2. Per-frame max-pool over keypoints + Linear -> per-frame global (B,T,G).
      3. Concat per-frame globals across time, project to (B, G).
      4. Per-kp head: [last-frame per-kp embedding (H), temporal global broadcast
         to kp (G), last-frame xyz_triang (3)] -> Linear->ReLU->Linear -> (B,23,3)
         residual. Zero-init last layer (identity start).
      5. Output = xyz_triang[:, -1] + residual.
    """

    PER_KP_IN_DIM = 21

    def __init__(self, ctx: int = 5, hidden: int = 128,
                 n_per_kp_layers: int = 3, global_dim: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.ctx = ctx

        # 1) Per-kp MLP (shared across frames and keypoints).
        per_kp = []
        d_in = self.PER_KP_IN_DIM
        for _ in range(n_per_kp_layers):
            per_kp.append(nn.Linear(d_in, hidden))
            per_kp.append(nn.ReLU(inplace=True))
            if dropout > 0:
                per_kp.append(nn.Dropout(dropout))
            d_in = hidden
        self.per_kp = nn.Sequential(*per_kp)

        # 2) Per-frame global projection.
        self.global_proj = nn.Linear(hidden, global_dim)

        # 3) Temporal fuse: (T_ctx, global_dim) -> (global_dim,)
        self.temporal_fuse = nn.Sequential(
            nn.Linear(ctx * global_dim, global_dim),
            nn.ReLU(inplace=True),
        )

        # 4) Per-kp head: in = hidden (last-frame per-kp) + global_dim (temporal)
        #                       + 3 (last-frame xyz_triang skip).
        head_in = hidden + global_dim + 3
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):  # x: (B, T_ctx, 23, 21)
        if x.dim() == 3:
            # Allow single-frame call; broadcast to a T_ctx window of identical frames.
            x = x.unsqueeze(1).expand(-1, self.ctx, -1, -1).contiguous()
        B, T, K, F = x.shape
        assert T == self.ctx, f"expected ctx={self.ctx}, got T={T}"
        assert F == self.PER_KP_IN_DIM, (
            f"TemporalTriangulationRefiner expects per-kp dim "
            f"{self.PER_KP_IN_DIM}, got {F}"
        )

        # 1) Per-frame per-kp embedding (shared MLP).
        h = self.per_kp(x)                              # (B, T, K, H)

        # 2) Per-frame global: max-pool over kp, then linear.
        per_frame_pool, _ = h.max(dim=2)                # (B, T, H)
        per_frame_g = self.global_proj(per_frame_pool)  # (B, T, G)

        # 3) Temporal fuse — keep ordering by flatten-and-project.
        temporal_g = self.temporal_fuse(per_frame_g.reshape(B, -1))  # (B, G)

        # 4) Per-kp head sees last-frame per-kp embedding + temporal global +
        #    last-frame xyz_triang skip.
        h_last = h[:, -1, :, :]                          # (B, K, H)
        g_broadcast = temporal_g.unsqueeze(1).expand(-1, K, -1)  # (B, K, G)
        xyz_last = x[:, -1, :, :3]                       # (B, K, 3)
        feat = torch.cat([h_last, g_broadcast, xyz_last], dim=-1)  # (B, K, H+G+3)
        delta = self.head(feat)                          # (B, K, 3)
        return xyz_last + delta


class TemporalMLPWith2D(nn.Module):
    """temporal_mlp + current-frame 2D inputs.

    Same flat-MLP-over-flattened-window architecture as TemporalMLPCorrector,
    with an additional input bundle for the *current* (last) frame in the
    window:
      - per-camera 2D pixel coords (normalized to [-1, 1] by (VIDEO_W, VIDEO_H))
      - per-camera SLEAP detection confidence
      - per-camera visibility (1 if detection finite AND conf > 0)

    Past-frame 2D info is intentionally NOT included; the temporal context
    comes from the 3D pose window. 2D for the current frame lets the model
    discount unreliable cameras.

    Inputs (forward) is a dict-like tuple to keep things simple at this layer:
        x_pose : (B, T_ctx, 23, 3)          aligned triangulated 3D window
        x_2d   : (B, 3, 23, 2)              current-frame 2D (NaNs zeroed)
        x_conf : (B, 3, 23)                 current-frame confidence (zeroed where invisible)
        x_vis  : (B, 3, 23)                 1/0 visibility mask

    The model flattens and concatenates these into a (B, pose_len + 2d_len +
    conf_len + vis_len) vector, runs an MLP, and returns
        pose[:, -1, :, :] + delta
    where delta is (B, 23, 3). Last linear layer is zero-init so the model
    starts at identity.
    """

    def __init__(self, ctx: int = 5, hidden: int = 128,
                 n_hidden_layers: int = 2, dropout: float = 0.0,
                 n_cam: int = 3, n_kp: int = 23):
        super().__init__()
        self.ctx = ctx
        self.n_cam = n_cam
        self.n_kp = n_kp
        pose_len = ctx * n_kp * 3
        xy_len = n_cam * n_kp * 2
        conf_len = n_cam * n_kp
        vis_len = n_cam * n_kp
        self.in_dim = pose_len + xy_len + conf_len + vis_len

        layers = []
        d_in = self.in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        layers.append(nn.Linear(d_in, n_kp * 3))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_pose, x_2d, x_conf, x_vis):
        # x_pose: (B, T_ctx, 23, 3); x_2d: (B, 3, 23, 2); x_conf/vis: (B, 3, 23)
        if x_pose.dim() == 3:
            x_pose = x_pose.unsqueeze(1)
        B = x_pose.shape[0]
        flat = torch.cat([
            x_pose.reshape(B, -1),
            x_2d.reshape(B, -1),
            x_conf.reshape(B, -1),
            x_vis.reshape(B, -1),
        ], dim=-1)
        delta = self.mlp(flat).reshape(B, self.n_kp, 3)
        return x_pose[:, -1, :, :] + delta


class TemporalMLPWith2DReproj(nn.Module):
    """temporal_mlp_2d + current-frame per-camera reprojection residuals.

    Inputs (current frame for 2D-side channels; T_ctx frames for pose):
        x_pose    : (B, T_ctx, 23, 3)        SLEAP-aligned triangulated 3D window
        x_2d      : (B, 3, 23, 2)            normalized 2D pixel coords
        x_conf    : (B, 3, 23)               SLEAP detection confidence
        x_vis     : (B, 3, 23)               visibility mask
        x_reproj  : (B, 3, 23, 2)            (detected - reprojected) / RESID_NORM_PX,
                                              zeroed where invisible

    Flat MLP. Last layer zero-init. Residual added to x_pose[:, -1, :, :].
    """

    def __init__(self, ctx: int = 5, hidden: int = 128,
                 n_hidden_layers: int = 2, dropout: float = 0.0,
                 n_cam: int = 3, n_kp: int = 23):
        super().__init__()
        self.ctx = ctx
        self.n_cam = n_cam
        self.n_kp = n_kp
        pose_len = ctx * n_kp * 3
        xy_len = n_cam * n_kp * 2
        conf_len = n_cam * n_kp
        vis_len = n_cam * n_kp
        reproj_len = n_cam * n_kp * 2
        self.in_dim = pose_len + xy_len + conf_len + vis_len + reproj_len

        layers = []
        d_in = self.in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden
        layers.append(nn.Linear(d_in, n_kp * 3))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x_pose, x_2d, x_conf, x_vis, x_reproj):
        if x_pose.dim() == 3:
            x_pose = x_pose.unsqueeze(1)
        B = x_pose.shape[0]
        flat = torch.cat([
            x_pose.reshape(B, -1),
            x_2d.reshape(B, -1),
            x_conf.reshape(B, -1),
            x_vis.reshape(B, -1),
            x_reproj.reshape(B, -1),
        ], dim=-1)
        delta = self.mlp(flat).reshape(B, self.n_kp, 3)
        return x_pose[:, -1, :, :] + delta


def build_model(name: str, **kw) -> nn.Module:
    if name == "linear":
        return LinearCorrector()
    if name == "mlp":
        return MLPCorrector(**kw)
    if name == "temporal_mlp":
        return TemporalMLPCorrector(**kw)
    if name == "velacc_mlp":
        return VelAccMLPCorrector(**kw)
    if name == "gnn":
        return GNNCorrector(**kw)
    if name == "perrat_head":
        return PerRatHeadCorrector(**kw)
    if name == "triangulation_refiner":
        return TriangulationRefiner(**kw)
    if name == "temporal_triangulation_refiner":
        return TemporalTriangulationRefiner(**kw)
    if name == "temporal_mlp_2d":
        return TemporalMLPWith2D(**kw)
    if name == "temporal_mlp_2d_reproj":
        return TemporalMLPWith2DReproj(**kw)
    raise ValueError(f"unknown model: {name}")
