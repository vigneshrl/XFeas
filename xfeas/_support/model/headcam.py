"""
SUPERSEDED (v4, 2026-10-19): every G1 / Spot path now uses the ONE shared module xfeas/sim/camera_mount.py (G1 default =
the 27 deg head_php mount below; XFEAS_CAM_MOUNT=head | legacy). Kept unchanged for gen3_headcam_check.py / gen3_hc_analyze.py.

XFEAS_CAM_MOUNT=head (additive 2026-10-19; analysis/GEN3_HEADCAM.md). The G1 decision camera attached to the robot's REAL
head camera frame instead of the virtual pelvis camera (XFeasDecider._snapshot: pelvis position + yaw + spec.cam
(fwd 0.10, up 0.55, pitch -0.5), gravity-aligned, ignores torso pitch / roll).

Mount = the Unitree G1 head RealSense D435 frame, a fixed joint on torso_link, from Unitree's own G1 description as shipped
in baselines/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf (lines 593-599):
    <joint name="d435_joint" type="fixed"> <origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"/>
    <parent link="torso_link"/> <child link="d435_link"/>
  (d435_link: x forward, y left, z up; pitch +0.8308 rad = optical axis 47.6 deg BELOW the torso's forward axis).
  The rev_1_0 file is the one whose head / logo offset (0.0039635, 0, -0.044) equals AMO's g1.xml head / logo geoms;
  the 23-dof file of the same release has the head 1 cm lower (-0.054) and d435 z 0.41987.
  Alternative (check sheet only, XFEAS_CAM_MOUNT=head_php): Holosoma / PHP D435i sim mount
  (holosoma config_values/wbt/g1/sensor.py: torso_link, pos (0.01, 0.01, 0.44), quat wxyz (0.6029, 0.3759, -0.3630,
  -0.6029) = 27.0 deg down; xfeas/sim/controllers/php/php_ref.py).
Intrinsics / resolution / clipping: UNCHANGED, the training camera's own prim /World/XFeasHeadCam (focal 12, aperture
20.955 x 20.955, 224 x 224, clip 0.05-60; XFeasDecider.setup) -- only the camera->world matrix changes.

Per snapshot: torso_link world pose from the PhysX articulation view (the pose the renderer draws) -> camera->world M
(USD row vectors: rows = right, up, back (= -optical axis), eye) -> the headlight is put 0.05 m along the camera's up axis
(the old rule put it 0.05 m above the spec.cam eye) -> the head shell mesh (the camera sits INSIDE it: d435 x 0.0576 vs
head front 0.078 m) is hidden for that render only (as the PHP bridge does; the real camera looks through the shell's
window) -> render -> self-body mask: every depth pixel whose back-projected point lies inside a robot link's visual
bounding box (link frame, from the link's own visual vertices, + MASK_MARGIN) is set to DEPTH_FAR (= the value the
pipeline already uses for "no return": posinf -> 60) -> cmap.set_mount(M, base) so every depth -> points step of this frame
(cmap.points, gap_estimate, feas, memory) uses the exact matrix.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os
import numpy as np
from xfeas_decider import XFeasDecider

URDF_SRC = "baselines/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf (d435_joint)"
D435_XYZ = (0.0576235, 0.01753, 0.42987)
D435_PITCH = 0.8307767239493009
PHP_POS = (0.01, 0.01, 0.44)
PHP_QUAT_WXYZ = (0.60290764, 0.37585401, -0.362958, -0.60290764)
HEAD_OFF = (0.0039635, 0.0, -0.044)  # head_link / logo geoms in torso_link (AMO g1.xml = rev_1_0 head_joint)
LIGHT_UP = 0.05
MASK_MARGIN = 0.02
MASK_DMAX = 2.5  # only pixels nearer than this can be the robot (feet are ~1.3 m below the eye)
DEPTH_FAR = 60.0


def mount_name():
    m = os.environ.get("XFEAS_CAM_MOUNT", "pelvis") or "pelvis"
    assert m in ("pelvis", "head", "head_php"), m
    return m


def _quat_xyzw_to_R(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mount_in_torso(mount):
    """-> (p (3,), R (3,3)) camera frame in torso_link; R columns = camera right, up, back (USD camera axes)"""
    if mount == "head":
        c, s = math.cos(D435_PITCH), math.sin(D435_PITCH)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # URDF rpy (0, p, 0)
        fwd, left, up = Ry[:, 0], Ry[:, 1], Ry[:, 2]
        return np.array(D435_XYZ, float), np.stack([-left, up, -fwd], 1)
    if mount == "head_php":
        w, x, y, z = PHP_QUAT_WXYZ
        return np.array(PHP_POS, float), _quat_xyzw_to_R([x, y, z, w])  # camera -Z forward, +Y up (USD convention)
    raise ValueError(mount)


def cam_matrix_from_torso(Rt, pt, mount):
    """torso_link world pose -> camera->world (4, 4) USD row-vector matrix"""
    p, Rm = mount_in_torso(mount)
    Rc = Rt @ Rm
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3], M[3, :3] = Rc[:, 0], Rc[:, 1], Rc[:, 2], pt + Rt @ p
    return M


def backproject224(depth, M):
    """full-resolution pixel-centre back-projection (no min-pool) -> (H*W, 3) world points (inf / far -> nan)"""
    d = np.asarray(depth, np.float64)
    N = d.shape[0]
    f = N * 12.0 / 20.955
    c = N / 2.0
    v, u = np.mgrid[0:N, 0:N]
    P = np.stack([(u + 0.5 - c) / f * d, -(v + 0.5 - c) / f * d, -d, np.ones_like(d)], -1).reshape(-1, 4)
    W = P @ np.asarray(M, np.float64)
    W = W[:, :3]
    W[~np.isfinite(d.ravel()) | (d.ravel() <= 0)] = np.nan
    return W


class HeadCamDecider(XFeasDecider):
    """XFeasDecider with the camera on the G1 head (same prim / intrinsics / annotators); _snapshot() is the only change.
    attach() after setup(): ctx, the gensim3 V3Mixin shim (link geometry) and the mount name."""

    def attach(self, ctx, shim, mount):
        from pxr import Usd, UsdGeom

        self.mount = mount
        self.shim = shim
        self.Usd, self.UsdGeom = Usd, UsdGeom
        root = ctx.get("robot_root") or "/World/" + str(ctx["kind"])
        self.robot_root = root
        self._inner = ctx["rob"]
        self._names = None  # resolved at the first pose query (articulation initialised after setup)
        # the head shell mesh under torso_link/visuals (hidden for the camera render only)
        st = ctx["stage"]
        vis = st.GetPrimAtPath(root + "/pelvis/torso_link/visuals")
        tree = []
        heads = []
        for p in Usd.PrimRange(vis, Usd.TraverseInstanceProxies()):
            tree.append((str(p.GetPath()), p.GetTypeName(), bool(p.IsInstanceProxy()), bool(p.IsInstance())))
            if "head" in p.GetName().lower() and p.IsA(UsdGeom.Imageable):
                heads.append(p)
        # outermost head prim that is not an instance proxy (visibility can be authored there)
        self._hide = None
        for p in heads:
            q = p
            while q.IsInstanceProxy():
                q = q.GetParent()
            self._hide = UsdGeom.Imageable(q)
            break
        self.vis_tree = tree
        self.hide_path = str(self._hide.GetPrim().GetPath()) if self._hide is not None else None
        self.hide_head = True
        self.last = None
        import scene_lighting as SL

        SL.HEAD_MOUNT_FN = self.light_matrix
        print(
            f"[headcam] mount {mount} ({URDF_SRC if mount == 'head' else 'holosoma d435i'}); "
            f"hidden for the camera render: {self.hide_path}; visuals tree {len(tree)} prims",
            flush=True,
        )

    # ---------------------------------------------------------------- poses
    def link_T(self):
        if self._names is None:
            self._names = list(self._inner._art._articulation_view.body_names)
            self._i_torso = self._names.index("torso_link")
            print(f"[headcam] {len(self._names)} bodies, torso_link index {self._i_torso}", flush=True)
        pv = self._inner._art._articulation_view._physics_view
        return np.asarray(pv.get_link_transforms(), np.float64).reshape(-1, len(self._names), 7)[0]

    def torso_pose(self, T=None):
        T = self.link_T() if T is None else T
        q = T[self._i_torso, 3:7]
        return _quat_xyzw_to_R(q), T[self._i_torso, :3].copy()

    def cam_M(self, T=None):
        Rt, pt = self.torso_pose(T)
        return cam_matrix_from_torso(Rt, pt, self.mount)

    def light_matrix(self, bs=None):
        M = self.cam_M().copy()
        M[3, :3] = M[3, :3] + LIGHT_UP * M[1, :3]
        return M

    def head_link_pose(self, T=None):
        Rt, pt = self.torso_pose(T)
        return Rt, pt + Rt @ np.array(HEAD_OFF)

    # ---------------------------------------------------------------- self-body mask
    def _boxes(self):
        if getattr(self, "_bx", None) is None:
            if self.shim._links is None:
                self.shim._links_init()
            L = self.shim._links
            bx = []
            for i, B in enumerate(L["boxes"]):
                if B is None:
                    continue
                bx.append(
                    (
                        self._names.index(L["names"][i]) if L["names"][i] in self._names else i,
                        np.asarray(B, float).min(0) - MASK_MARGIN,
                        np.asarray(B, float).max(0) + MASK_MARGIN,
                        L["names"][i],
                    )
                )
            self._bx = bx
        return self._bx

    def self_mask(self, depth, M, T=None):
        """(N, N) bool: pixels that see the robot's own body (link visual bounding boxes, link frame, + MASK_MARGIN)"""
        d = np.asarray(depth, np.float64)
        N = d.shape[0]
        W = backproject224(d, M)
        cand = np.isfinite(d.ravel()) & (d.ravel() > 0) & (d.ravel() < MASK_DMAX)
        out = np.zeros(N * N, bool)
        if not cand.any():
            return out.reshape(N, N), {}
        T = self.link_T() if T is None else T
        Wc = W[cand]
        hit = np.zeros(len(Wc), bool)
        per = {}
        for i, lo, hi, nm in self._boxes():
            R = _quat_xyzw_to_R(T[i, 3:7])
            t = T[i, :3]
            L = (Wc - t) @ R
            m = np.all((L >= lo) & (L <= hi), 1)
            if m.any():
                per[nm] = int(m.sum())
            hit |= m
        out[np.flatnonzero(cand)[hit]] = True
        return out.reshape(N, N), per

    # ---------------------------------------------------------------- render
    def render_raw(self, M, hide_head=True):
        Gf = self.Gf
        self.cam_xf.Set(Gf.Matrix4d(*[float(v) for v in np.asarray(M, float).ravel()]))
        if hide_head and self._hide is not None:
            self._hide.MakeInvisible()
        try:
            self.rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        finally:
            if hide_head and self._hide is not None:
                self._hide.MakeVisible()
        rgb = np.asarray(self.ann_rgb.get_data())[..., :3].astype(np.uint8)
        dep = np.nan_to_num(np.asarray(self.ann_dep.get_data()).astype(np.float32), posinf=60.0)
        return rgb, dep

    def _place_light(self, M):
        import scene_lighting as SL

        # the episode runner's Lighting instance owns /World/RobotHeadlight; set it to the mount directly (same rule as follow)
        try:
            from pxr import UsdGeom

            hl = self.stage.GetPrimAtPath("/World/RobotHeadlight")
            if hl.IsValid():
                L = np.asarray(M, float).copy()
                L[3, :3] = L[3, :3] + LIGHT_UP * L[1, :3]
                xf = UsdGeom.Xformable(hl)
                ops = xf.GetOrderedXformOps()
                if ops:
                    ops[0].Set(self.Gf.Matrix4d(*[float(v) for v in L.ravel()]))
        except Exception as e:
            print("[headcam] headlight placement failed:", repr(e), flush=True)

    def _snapshot(self, bs):
        import cmap as CMAP

        T = self.link_T()
        M = self.cam_M(T)
        self._place_light(M)
        rgb, dep = self.render_raw(M, hide_head=self.hide_head)
        mask, per = self.self_mask(dep, M, T)
        dep = dep.copy()
        dep[mask] = DEPTH_FAR
        base = (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"]))
        CMAP.set_mount(M, base)
        self.last = dict(M=M, base=base, mask_frac=float(mask.mean()), mask_links=per, mask=mask)
        return rgb, dep

    def snapshot_pelvis(self, bs):
        """the OLD camera (XFeasDecider._snapshot) for side-by-side checks; clears the cmap mount"""
        import cmap as CMAP

        CMAP.set_mount(None)
        return XFeasDecider._snapshot(self, bs)
