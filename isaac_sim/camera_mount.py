"""
THE shared decision-camera module (v4 rebuild, 2026-10-19; analysis/REBUILD_V4_PHASE01.md). Every path that renders the
layer's RGB-D frame -- collectors (gensim.GenSim.snapshot -> gensim_v1/v2/gensim3/run_gen_v3), closed-loop runners
(xfeas_decider.XFeasDecider._snapshot -> gen3_methods, course / terrain harness, gen3_diag), lighting (scene_lighting.follow)
and the check / video tools -- gets the camera pose, intrinsics, headlight and self-body mask from here.

Mounts (MOUNTS; the pose is a FIXED transform on a robot link, composed with that link's PhysX pose EVERY frame = full
6-DoF, pitch and roll included):
  g1_head_php   DEFAULT for every Unitree G1 controller (g1a/g1c AMO, g1g GR00T, g1l Isaac Lab, g1u URMA, g1s SONIC, g1p PHP).
                Holosoma / PHP D435i sim mount (holosoma config_values/wbt/g1/sensor.py d435i_front_depth_camera; also
                isaac_sim/controllers/php/php_ref.py): torso_link, pos (0.01, 0.01, 0.44), quat wxyz (0.60290764, 0.37585401, -0.362958,
                -0.60290764) (USD camera: -Z forward, +Y up) = optical axis 27.0 deg below the torso's forward axis.
  g1_head_d435  Unitree's own d435_joint (g1_29dof_with_hand_rev_1_0.urdf): torso_link (0.0576235, 0.01753, 0.42987),
                pitch 0.8308 rad (47.6 deg down). Kept for comparison (XFEAS_CAM_MOUNT=head); it sees a 1.10 m beam only
                within ~0.5-0.8 m (analysis/GEN3_HEADCAM.md).
  spot_body     DEFAULT for Spot (spotp ReLIC): body link, eye (0.415, 0.0, -0.02) in the body frame (1 cm ahead of the body
                mesh's nose face, measured on the check sheet; spec.cam's x 0.52 floated 11.5 cm in front of the nose), optical
                axis pitched 0.45 rad (25.8 deg) down about the body y axis (spec.cam), no roll; BODY-FIXED (follows body
                pitch / roll) instead of yaw-only gravity-aligned.
  legacy        the old rule of every pre-v4 collector / runner: root position + yaw only, spec.cam (fwd, up, pitch),
                gravity-aligned (SetLookAt(eye, eye + look, z-up)). ONLY with an explicit XFEAS_CAM_MOUNT=legacy (or the
                old spelling 'pelvis'). Other robot kinds (jackal, iris, ...) keep it by default (no link mount defined).
XFEAS_CAM_MOUNT: unset / '' / 'default' -> the robot's default above; 'head_php' | 'head' | 'spot_body' | 'legacy' | 'pelvis'.

Intrinsics (unchanged, the training camera): 224 x 224, focal 12 mm, aperture 20.955 x 20.955 mm (fx = fy = 128.27 px,
HFOV = VFOV = 82.3 deg), principal point at the image centre, clip 0.05-60 m; annotator distance_to_image_plane.
Matrix convention (as everywhere in the repo): camera->world 4x4 in USD ROW-vector form, rows = camera right, up, back
(= -optical axis), eye. Pixel (u, v) ray in the camera frame ((u + .5 - c) / f, -(v + .5 - c) / f, -1).

Per render (MountedCamera.render):
  1. M from the mount link's PhysX pose (articulation physics view, the pose the renderer draws);
  2. headlight /World/RobotHeadlight on the same mount, LIGHT_UP m along the camera's up axis, looking along the axis;
  3. G1: torso_link visual subtree hidden for this render only (the D435i sits INSIDE the head shell; PHP / Holosoma
     exclude torso_link from their camera raycast the same way);
  4. exact self-body mask: the instance-id segmentation of the same render product; every pixel whose rendered prim lies
     under the robot root is robot (falls back to link-geometry boxes if the annotator is unavailable); masked depth
     pixels are set to DEPTH_FAR (= the pipeline's "no return"); RGB is not altered;
  5. cmap.set_mount(M, base, cam_h) so every depth -> points step of this frame uses the exact matrix (cam_h = eye height
     above the robot's lowest body point = the floor prior under this mount); clearmap's Spot-arm heuristic self mask is
     switched off (clearmap.EXACT_SELF_MASK) because the robot pixels are already removed exactly.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os
import numpy as np

N_PX = 224
FOCAL_MM = 12.0
APERTURE_MM = 20.955
CLIP = (0.05, 60.0)
FX = N_PX * FOCAL_MM / APERTURE_MM
HFOV_DEG = math.degrees(2 * math.atan(APERTURE_MM / 2 / FOCAL_MM))
LIGHT_UP = 0.05
DEPTH_FAR = 60.0
MASK_MARGIN = 0.02
CAM_PRIM = "/World/XFeasHeadCam"
EXPO_D_REF = 2.0       # m: camera -> floor distance along the axis at which the headlight is used at full intensity
EXPO_MIN = 0.05

PHP_POS = (0.01, 0.01, 0.44)
PHP_QUAT_WXYZ = (0.60290764, 0.37585401, -0.362958, -0.60290764)
D435_XYZ = (0.0576235, 0.01753, 0.42987)
D435_PITCH = 0.8307767239493009
SPOT_EYE = (0.415, 0.0, -0.02)     # v4 check sheet: the ReLIC Spot body mesh's nose face is at x 0.405 m at this height (y 0,
#                                   z -0.02); spec.cam's 0.52 put the eye 11.5 cm IN FRONT of the robot. Now 1 cm ahead of the face.
SPOT_PITCH = 0.45
# gen5 (2026-10-25) wheeled / aerial bodies: body-fixed front cameras on the root link (measured on the check sheet
# analysis/camera_check_v5_xbody/, see that README). Eye = 1 cm ahead of the measured front face of the body at the eye's
# height, optical axis pitched down about the body y axis, no roll.
JACKAL_EYE = (0.2375, 0.0, 0.218)   # base_link: in front of the shipped Bumblebee stereo housing (the Jackal's front sensor)
JACKAL_PITCH = 0.26                # 15 deg down
IRIS_EYE = (0.1267, 0.0, -0.010)   # Iris body: 1 cm ahead of the fuselage nose, slightly below the body centre
IRIS_PITCH = 0.52                  # 30 deg down
HEAD_OFF = (0.0039635, 0.0, -0.044)        # G1 head_link / logo geoms in torso_link (AMO g1.xml = rev_1_0 head_joint)
# head shell (head_link.STL) bounding box in torso_link of the rev_1_0 model that the PHP / Holosoma mount refers to (identical
# in AMO g1_amo.usd, GR00T g1_gear_wbc.usd and PHP g1_php.usd; pxr, 2026-10-19). A model whose head shell sits elsewhere in
# its torso_link frame (SONIC: 1 cm lower; the Isaac Lab G1: different torso_link origin) gets the camera moved WITH the
# head shell (same size required, < 5 mm): the camera is fixed to the head, not to a frame convention.
HEAD_REF_MIN = np.array([-0.0624, -0.0778, 0.2808])
HEAD_REF_MAX = np.array([0.0779, 0.0778, 0.4866])

SRC = dict(g1_head_php="holosoma config_values/wbt/g1/sensor.py d435i_front_depth_camera (isaac_sim/controllers/php/php_ref.py CAM_POS/CAM_QUAT_WXYZ)",
           g1_head_d435="baselines/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf d435_joint",
           spot_body="spot_posture spec.cam pitch -0.45 / z -0.02, eye moved to the body nose face (x 0.415), body-fixed",
           legacy="robot spec.cam (fwd, up, pitch), root position + yaw only",
           jackal_front="Jackal base_link, eye 1 cm ahead of the Bumblebee stereo housing (USD bumblebee_stereo_camera_frame), 15 deg down, body-fixed",
           iris_front="Iris body, eye 1 cm ahead of the fuselage nose, 30 deg down, body-fixed")
ALIASES = {"head_php": "g1_head_php", "php": "g1_head_php", "head": "g1_head_d435", "d435": "g1_head_d435", "pelvis": "legacy",
           "legacy": "legacy", "spot_body": "spot_body", "g1_head_php": "g1_head_php", "g1_head_d435": "g1_head_d435",
           "jackal_front": "jackal_front", "iris_front": "iris_front"}


def family(kind):
    k = str(kind).lower()
    if k.startswith("g1"):
        return "g1"
    if k.startswith("spot"):
        return "spot"
    if k.startswith("jackal"):          # gen5 2026-10-25
        return "jackal"
    if k.startswith("iris"):
        return "iris"
    return "other"


def mount_name(kind="g1"):
    """the mount for a robot kind under XFEAS_CAM_MOUNT (see module doc)"""
    m = (os.environ.get("XFEAS_CAM_MOUNT", "") or "").strip().lower()
    fam = family(kind)
    if m in ("", "default", "v4"):
        return {"g1": "g1_head_php", "spot": "spot_body", "jackal": "jackal_front", "iris": "iris_front"}.get(fam, "legacy")
    assert m in ALIASES, f"XFEAS_CAM_MOUNT={m!r}"
    m = ALIASES[m]
    if m != "legacy":
        assert (m.startswith("g1_") and fam == "g1") or (m == "spot_body" and fam == "spot") or (m == f"{fam}_front"), f"mount {m} on robot {kind}"
    return m


def is_legacy(kind="g1"):
    return mount_name(kind) == "legacy"


def _q_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w); x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mount_link(mount):
    return {"g1_head_php": "torso_link", "g1_head_d435": "torso_link", "spot_body": "body", "jackal_front": "base_link",
            "iris_front": "body"}.get(mount)


def mount_in_link(mount):
    """-> (p (3,), R (3, 3)) camera in the mount link frame; R columns = camera right, up, back (USD camera axes)"""
    if mount == "g1_head_php":
        w, x, y, z = PHP_QUAT_WXYZ
        return np.array(PHP_POS, float), _q_to_R(x, y, z, w)
    if mount == "g1_head_d435":
        c, s = math.cos(D435_PITCH), math.sin(D435_PITCH)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        fwd, left, up = Ry[:, 0], Ry[:, 1], Ry[:, 2]
        return np.array(D435_XYZ, float), np.stack([-left, up, -fwd], 1)
    if mount == "spot_body":
        c, s = math.cos(SPOT_PITCH), math.sin(SPOT_PITCH)
        fwd = np.array([c, 0.0, -s]); up = np.array([s, 0.0, c]); right = np.array([0.0, -1.0, 0.0])
        return np.array(SPOT_EYE, float), np.stack([right, up, -fwd], 1)
    if mount in ("jackal_front", "iris_front"):
        eye, pit = (JACKAL_EYE, JACKAL_PITCH) if mount == "jackal_front" else (IRIS_EYE, IRIS_PITCH)
        c, s = math.cos(pit), math.sin(pit)
        fwd = np.array([c, 0.0, -s]); up = np.array([s, 0.0, c]); right = np.array([0.0, -1.0, 0.0])
        return np.array(eye, float), np.stack([right, up, -fwd], 1)
    raise ValueError(mount)


def cam_matrix_from_link(R_link, p_link, mount):
    """mount link world pose (R (3, 3), p (3,)) -> camera->world 4x4 (USD row vectors)"""
    p, Rm = mount_in_link(mount)
    Rc = np.asarray(R_link, float) @ Rm
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3], M[3, :3] = Rc[:, 0], Rc[:, 1], Rc[:, 2], np.asarray(p_link, float) + np.asarray(R_link, float) @ p
    return M


def legacy_matrix(bs, cam):
    """the pre-v4 rule (XFeasDecider._snapshot / GenSim.snapshot: SetLookAt(eye, eye + look, z-up).GetInverse())"""
    fwd, up, pitch = cam
    c, s = math.cos(bs["yaw"]), math.sin(bs["yaw"])
    eye = np.asarray(bs["pos"], float) + np.array([fwd * c, fwd * s, up])
    f = np.array([c * math.cos(pitch), s * math.cos(pitch), math.sin(pitch)])
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    M = np.eye(4); M[0, :3], M[1, :3], M[2, :3], M[3, :3] = r, u, -f, eye
    return M


def legacy_light_matrix(bs, cam):
    fwd, up, pitch = cam
    return legacy_matrix(bs, (fwd, up + LIGHT_UP, pitch))


def light_from_cam(M):
    L = np.asarray(M, float).copy(); L[3, :3] = L[3, :3] + LIGHT_UP * L[1, :3]
    return L


def axis_elev_deg(M):
    f = -np.asarray(M, float)[2, :3]
    return math.degrees(math.asin(float(np.clip(f[2], -1, 1))))


def backproject224(depth, M):
    """full-resolution pixel-centre back-projection -> (H*W, 3) world points (non-finite / <= 0 -> nan)"""
    d = np.asarray(depth, np.float64); N = d.shape[0]
    f = N * FOCAL_MM / APERTURE_MM; c = N / 2.0
    v, u = np.mgrid[0:N, 0:N]
    P = np.stack([(u + 0.5 - c) / f * d, -(v + 0.5 - c) / f * d, -d, np.ones_like(d)], -1).reshape(-1, 4)
    W = (P @ np.asarray(M, np.float64))[:, :3]
    W[~np.isfinite(d.ravel()) | (d.ravel() <= 0)] = np.nan
    return W


def define_camera(stage, rep, path=CAM_PRIM):
    """the training camera prim + render product + rgb / depth annotators (IDENTICAL to every collector)"""
    from pxr import UsdGeom, Gf
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(FOCAL_MM); cam.CreateHorizontalApertureAttr(APERTURE_MM)
    cam.CreateVerticalApertureAttr(APERTURE_MM); cam.CreateClippingRangeAttr(Gf.Vec2f(*CLIP))
    xf = UsdGeom.Xformable(cam.GetPrim()).MakeMatrixXform()
    rp = rep.create.render_product(path, (N_PX, N_PX))
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); a_rgb.attach([rp])
    a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); a_dep.attach([rp])
    return xf, rp, a_rgb, a_dep


ACTIVE = None          # the MountedCamera of this process (scene_lighting.follow / tools read it)


class MountedCamera:
    """mount + exact self mask for ONE robot. Construct after the robot exists; link data is resolved lazily (the
    articulation physics view exists only after world.reset)."""

    def __init__(self, stage, rob, kind, root=None, spec=None, mount=None, rp=None, rep=None, log=print):
        self.stage, self.kind, self.spec = stage, str(kind), spec
        self.inner = getattr(rob, "_v_inner", rob)            # VRob proxy -> the adapter
        self.root = root or getattr(self.inner, "prim_path", None) or f"/World/{kind}"
        self.mount = mount or mount_name(kind)
        self.fam = family(kind)
        self.link = mount_link(self.mount)
        self.log = log
        self._names = None; self._i_link = None
        self._geo = None
        self.seg = None
        self.hide = []
        self.last = None
        self.mount_delta = np.zeros(3); self.head_bbox = None; self.head_size_err = None
        if self.mount != "legacy":
            self._find_hide()
            if rp is not None and rep is not None:
                self.attach_seg(rp, rep)
            try:
                import clearmap as CLM
                CLM.EXACT_SELF_MASK = True
            except Exception:
                pass
        global ACTIVE
        ACTIVE = self
        print(f"[camera_mount] robot {kind} root {self.root}: mount {self.mount} ({SRC[self.mount]}); link {self.link}; "
              f"hidden for camera renders: {[str(h.GetPrim().GetPath()) for h in self.hide]}; exact mask "
              f"{'instance segmentation' if self.seg is not None else 'link boxes (until attach_seg)'}", flush=True)

    # ---------------------------------------------------------------- setup helpers
    def _find_hide(self):
        """G1: the torso_link visual subtree (torso + logo + head shell) -- hidden for camera renders only"""
        from pxr import Usd, UsdGeom, UsdPhysics
        self.hide = []
        if self.fam != "g1":
            return
        rootp = self.stage.GetPrimAtPath(self.root)
        tl = None
        for p in Usd.PrimRange(rootp):
            if p.GetName() == "torso_link" and p.HasAPI(UsdPhysics.RigidBodyAPI):
                tl = p; break
        if tl is None:
            for p in Usd.PrimRange(rootp):
                if p.GetName() == "torso_link":
                    tl = p; break
        if tl is None:
            raise RuntimeError(f"[camera_mount] no torso_link under {self.root}")
        self.torso_prim = tl
        vis = [c for c in tl.GetChildren() if "visual" in c.GetName().lower()]
        if not vis:                                        # meshes directly under the body (no 'visuals' group)
            vis = [c for c in tl.GetChildren() if c.IsA(UsdGeom.Imageable) and "collision" not in c.GetName().lower()
                   and not c.HasAPI(UsdPhysics.RigidBodyAPI) and c.GetName() not in ("sites",)]
        for v in vis:
            q = v
            while q.IsInstanceProxy():
                q = q.GetParent()
            self.hide.append(UsdGeom.Imageable(q))
        # head shell meshes wherever they live (rev_1_0 USDs: torso_link/visuals/head_link; the Isaac Lab G1: a separate
        # head_link body) -> hidden too, and the mount is re-anchored to the head shell (below)
        tc = Usd.TimeCode.Default()
        Tinv = np.array(UsdGeom.Xformable(tl).ComputeLocalToWorldTransform(tc).GetInverse(), float)
        P = []
        hidden = {str(h.GetPrim().GetPath()) for h in self.hide}
        for m in Usd.PrimRange(rootp, Usd.TraverseInstanceProxies()):
            if not m.IsA(UsdGeom.Mesh):
                continue
            ps = str(m.GetPath())
            comps = ps[len(self.root):].split("/")
            if not any(c.lower().startswith("head") for c in comps) or "collision" in ps.lower():
                continue
            if UsdGeom.Imageable(m).ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                continue
            v = UsdGeom.Mesh(m).GetPointsAttr().Get()
            if v is None or len(v) == 0:
                continue
            M = np.array(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(tc), float) @ Tinv
            P.append(np.asarray(v, float) @ M[:3, :3] + M[3, :3])
            if not any(ps.startswith(h + "/") or ps == h for h in hidden):       # its body's direct 'visuals' child
                q = m
                while q.GetParent().IsValid() and not q.GetParent().HasAPI(UsdPhysics.RigidBodyAPI) and str(q.GetParent().GetPath()) != self.root:
                    q = q.GetParent()
                while q.IsInstanceProxy():
                    q = q.GetParent()
                self.hide.append(UsdGeom.Imageable(q)); hidden.add(str(q.GetPath()))
        self.head_bbox = None; self.mount_delta = np.zeros(3)
        if P:
            P = np.concatenate(P, 0)
            lo, hi = P.min(0), P.max(0)
            self.head_bbox = (lo, hi)
            # anchors: the head shell's FRONT (max x), lateral centre and TOP (max z) -- the camera sits behind the front
            # window and below the crown; with an identical shell this equals the centre shift
            d = np.array([hi[0] - HEAD_REF_MAX[0], 0.5 * (lo[1] + hi[1]) - 0.5 * (HEAD_REF_MIN[1] + HEAD_REF_MAX[1]), hi[2] - HEAD_REF_MAX[2]])
            size_err = float(np.abs((hi - lo) - (HEAD_REF_MAX - HEAD_REF_MIN)).max())
            self.head_size_err = size_err
            if np.abs(d).max() > 5e-4:
                self.mount_delta = d
            print(f"[camera_mount] head shell bbox in torso_link: min {np.round(lo, 4).tolist()} max {np.round(hi, 4).tolist()}; "
                  f"front/top vs the reference rev_1_0 head {np.round(d, 4).tolist()} (shell size diff {size_err:.4f} m) -> mount offset "
                  f"{np.round(self.mount_delta, 4).tolist()}", flush=True)
        else:
            print("[camera_mount] WARNING: no head shell mesh found; mount not re-anchored", flush=True)
        # an INSTANCEABLE visuals prim does not take a visibility override in the renderer (Isaac Lab G1 head_link/visuals:
        # MakeInvisible -> still 34210 px; check sheet 2026-10-19); hide its body prim instead when that body has no child
        # bodies (flat link hierarchy: only its own visuals / collisions / sites are under it)
        out = []
        for h in self.hide:
            q = h.GetPrim()
            if q.IsInstance():
                body = q.GetParent()
                nested = any(d != body and d.HasAPI(UsdPhysics.RigidBodyAPI) for d in Usd.PrimRange(body))
                if body.HasAPI(UsdPhysics.RigidBodyAPI) and not nested:
                    out.append(UsdGeom.Imageable(body)); continue
                print(f"[camera_mount] WARNING: {q.GetPath()} is instanceable and its body has child bodies; hiding may not take", flush=True)
            out.append(h)
        self.hide = out

    def attach_seg(self, rp, rep):
        try:
            a = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast", init_params={"colorize": False})
            a.attach([rp])
            self.seg = a
        except Exception as e:
            print(f"[camera_mount] instance segmentation unavailable ({e!r}); link-box mask", flush=True)
            self.seg = None

    def _names_init(self):
        if self._names is None:
            av = self.inner._art._articulation_view
            self._names = list(av.body_names)
            if self.link is not None:
                self._i_link = self._names.index(self.link)

    def link_T(self):
        """(n_bodies, 7) [x y z qx qy qz qw] PhysX link poses"""
        self._names_init()
        pv = self.inner._art._articulation_view._physics_view
        return np.asarray(pv.get_link_transforms(), np.float64).reshape(-1, len(self._names), 7)[0]

    def link_pose(self, name=None, T=None):
        T = self.link_T() if T is None else T
        i = self._i_link if name is None else self._names.index(name)
        q = T[i, 3:7]
        return _q_to_R(*q), T[i, :3].copy()

    # ---------------------------------------------------------------- body geometry (mask fallback, floor prior, checks)
    def geometry(self):
        """per body: <= 600 visual-mesh vertices in the LINK frame (constant)"""
        if self._geo is not None:
            return self._geo
        from pxr import Usd, UsdGeom, UsdPhysics
        self._names_init()
        by = {}
        for p in Usd.PrimRange(self.stage.GetPrimAtPath(self.root)):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                by.setdefault(p.GetName(), p)
        rng = np.random.default_rng(0)
        tc = Usd.TimeCode.Default()
        G = []
        for nm in self._names:
            pr = by.get(nm)
            if pr is None:
                G.append(None); continue
            Linv = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(tc).GetInverse()
            P = []
            it = iter(Usd.PrimRange(pr, Usd.TraverseInstanceProxies()))
            for m in it:
                if m != pr and m.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                if not m.IsA(UsdGeom.Mesh):
                    continue
                im = UsdGeom.Imageable(m)
                if im.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                    continue
                if "collision" in str(m.GetPath()).lower():
                    continue
                v = UsdGeom.Mesh(m).GetPointsAttr().Get()
                if v is None or len(v) == 0:
                    continue
                Mm = np.array(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(tc) * Linv, float)
                v = np.asarray(v, float)
                if len(v) > 6000:
                    v = v[rng.choice(len(v), 6000, replace=False)]
                P.append(v @ Mm[:3, :3] + Mm[3, :3])
            if not P:
                G.append(None); continue
            P = np.concatenate(P, 0)
            if len(P) > 600:
                ext = np.unique(np.concatenate([np.argmin(P, 0), np.argmax(P, 0)]))
                P = np.concatenate([P[ext], P[rng.choice(len(P), 600 - len(ext), replace=False)]], 0)
            G.append(P)
        self._geo = G
        return G

    def body_points_world(self, T=None):
        T = self.link_T() if T is None else T
        out = []
        for i, B in enumerate(self.geometry()):
            if B is None:
                continue
            out.append(B @ _q_to_R(*T[i, 3:7]).T + T[i, :3])
        return np.concatenate(out, 0)

    def lowest_body_z(self, T=None):
        return float(self.body_points_world(T)[:, 2].min())

    def box_mask(self, depth, M, T=None, dmax=2.5):
        """link-box mask (visual bounds in each link frame + MASK_MARGIN): the fallback / cross-check of the exact mask"""
        d = np.asarray(depth, np.float64); N = d.shape[0]
        W = backproject224(d, M)
        cand = np.isfinite(d.ravel()) & (d.ravel() > 0) & (d.ravel() < dmax)
        out = np.zeros(N * N, bool)
        if not cand.any():
            return out.reshape(N, N)
        T = self.link_T() if T is None else T
        Wc = W[cand]; hit = np.zeros(len(Wc), bool)
        for i, B in enumerate(self.geometry()):
            if B is None:
                continue
            lo, hi = B.min(0) - MASK_MARGIN, B.max(0) + MASK_MARGIN
            L = (Wc - T[i, :3]) @ _q_to_R(*T[i, 3:7])
            hit |= np.all((L >= lo) & (L <= hi), 1)
        out[np.flatnonzero(cand)[hit]] = True
        return out.reshape(N, N)

    def seg_mask(self):
        """exact robot-pixel mask from the instance-id segmentation of the LAST render (None if unavailable)"""
        if self.seg is None:
            return None
        try:
            D = self.seg.get_data()
            ids = np.asarray(D["data"]).reshape(N_PX, N_PX)
            lab = D["info"].get("idToLabels", {})
            rob = [int(k) for k, v in lab.items() if str(v).startswith(self.root + "/") or str(v) == self.root]
            return np.isin(ids, np.array(rob, ids.dtype)) if rob else np.zeros_like(ids, bool)
        except Exception as e:
            print(f"[camera_mount] segmentation read failed ({e!r}); link-box mask", flush=True)
            self.seg = None
            return None

    # ---------------------------------------------------------------- pose
    def matrix(self, bs=None, T=None):
        if self.mount == "legacy":
            return legacy_matrix(bs, self.spec.cam)
        if self.fam == "spot" and bs is not None and T is None and "quat" in bs:
            w, x, y, z = [float(v) for v in bs["quat"]]                 # base = body: the articulation root pose
            return cam_matrix_from_link(_q_to_R(x, y, z, w), np.asarray(bs["pos"], float), self.mount)
        R, p = self.link_pose(T=T)
        M = cam_matrix_from_link(R, p, self.mount)
        if self.fam == "g1" and np.any(self.mount_delta):
            M[3, :3] = M[3, :3] + R @ self.mount_delta
        return M

    def light_matrix(self, bs=None):
        if self.mount == "legacy":
            return legacy_light_matrix(bs, self.spec.cam)
        try:
            return light_from_cam(self.matrix(bs))
        except Exception:                      # articulation not initialised yet -> scene_lighting falls back for this tick
            return None

    def place_light(self, M, Gf=None):
        from pxr import UsdGeom, Gf as _Gf
        hl = self.stage.GetPrimAtPath("/World/RobotHeadlight")
        if not hl.IsValid():
            return
        L = light_from_cam(M) if self.mount != "legacy" else None
        if L is None:
            return
        ops = UsdGeom.Xformable(hl).GetOrderedXformOps()
        if ops:
            ops[0].Set(_Gf.Matrix4d(*[float(v) for v in L.ravel()]))

    # ---------------------------------------------------------------- render
    def render(self, cam_xf, rep, ann_rgb, ann_dep, bs=None, hide=True, mask=True, rt_subframes=4):
        """-> rgb (224, 224, 3) u8, depth (224, 224) f32 with robot pixels = DEPTH_FAR, M, info"""
        from pxr import Gf
        T = self.link_T() if self.mount != "legacy" else None
        M = self.matrix(bs, T)
        cam_xf.Set(Gf.Matrix4d(*[float(v) for v in np.asarray(M, float).ravel()]))
        cam_h = None
        if self.mount != "legacy":
            self.place_light(M)
            try:
                cam_h = float(M[3, 2] - self.lowest_body_z(T))
            except Exception:
                cam_h = None
        expo = self.expo_scale(M, cam_h)
        hl_attr, hl_I0 = None, None
        if expo < 1.0:                          # exposure on the mount: headlight scaled for THIS render only, restored after
            hl = self.stage.GetPrimAtPath("/World/RobotHeadlight")
            if hl.IsValid() and hl.GetAttribute("inputs:intensity"):
                hl_attr = hl.GetAttribute("inputs:intensity"); hl_I0 = hl_attr.Get()
                if hl_I0 and hl_I0 > 0:
                    hl_attr.Set(float(hl_I0) * expo)
                else:
                    hl_attr = None
        hid = []
        if hide:
            for h in self.hide:
                h.MakeInvisible(); hid.append(h)
        try:
            rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0, pause_timeline=False)
        finally:
            for h in hid:
                h.MakeVisible()
            if hl_attr is not None:
                hl_attr.Set(float(hl_I0))
        rgb = np.asarray(ann_rgb.get_data())
        dep = np.asarray(ann_dep.get_data())
        if rgb.ndim != 3 or dep.ndim != 2:
            return None
        rgb = rgb[..., :3].astype(np.uint8)
        dep = np.nan_to_num(dep.astype(np.float32), posinf=DEPTH_FAR)
        info = dict(mount=self.mount, expo=round(expo, 4))
        if self.mount != "legacy" and mask:
            m = self.seg_mask()
            src = "seg"
            if m is None:
                m = self.box_mask(dep, M, T); src = "box"
            dep = dep.copy(); dep[m] = DEPTH_FAR
            info.update(mask=m, mask_frac=float(m.mean()), mask_src=src)
            info["cam_h"] = cam_h
        self.last = dict(M=M, T=T, **info)
        return rgb, dep, M, info

    def expo_scale(self, M, cam_h):
        """EXPOSURE ON THE MOUNT (v4, 2026-10-19; XFEAS_CAM_EXPO, default 1 = on, 0 = off). The night headlight sits on the
        camera, so the floor's irradiance -- and the image brightness -- grows as 1 / d^2 with the distance d from the
        camera to the floor along the optical axis. Deep crawls (AMO h0.30 p1.57: eye 0.57 m, axis -84 deg -> d 0.57 m) were
        saturated white. The headlight intensity used for the CAMERA render is scaled by (d / EXPO_D_REF)^2, clipped to
        [EXPO_MIN, 1]: identical rule for every controller (pure geometry of the mount: eye height above the robot's lowest
        body point and the axis elevation; no image statistics, no history). Never brightens; axis above -5 deg -> 1.
        The recording / scene lighting is unchanged (the intensity is restored right after the render)."""
        if self.mount == "legacy" or os.environ.get("XFEAS_CAM_EXPO", "1") != "1" or cam_h is None or not np.isfinite(cam_h):
            return 1.0
        el = -math.radians(axis_elev_deg(M))
        if el < math.radians(5.0):
            return 1.0
        d = max(float(cam_h), 0.05) / math.sin(el)
        return float(np.clip((d / EXPO_D_REF) ** 2, EXPO_MIN, 1.0))

    def set_cmap(self, bs, info=None):
        """cmap.set_mount for the frame just rendered (the closed-loop points() path)"""
        import cmap as CMAP
        if self.mount == "legacy":
            CMAP.set_mount(None); return
        info = info or self.last or {}
        CMAP.set_mount(self.last["M"], (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])), cam_h=info.get("cam_h"))

    def base_of(self, bs):
        return (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"]))

    def eye_ahead(self, bs):
        """camera eye ahead of the root along the base yaw (m)"""
        M = self.matrix(bs)
        c, s = math.cos(float(bs["yaw"])), math.sin(float(bs["yaw"]))
        return float((M[3, 0] - float(bs["pos"][0])) * c + (M[3, 1] - float(bs["pos"][1])) * s)


class SimCamera:
    """adapter for a collector harness (gensim.GenSim / envsim.EnvSim and subclasses, e.g. scripts/gen4/gensim4 via
    XFEAS_CAM4_MODULE=camera_mount:sim_camera): render(bs) -> (rgb, depth, M, info) on the harness' own /World/HeadCam
    render product, light_matrix(bs). Uses the harness' MountedCamera (GenSim._mount_init) when it exists."""

    def __init__(self, sim):
        self.sim = sim
        mc = getattr(sim, "mcam", None)
        if mc is None:
            mc = MountedCamera(sim.stage, sim.rob, sim.kind, root=sim.root, spec=sim.spec)
        self.mc = mc
        self.mount = mc.mount

    def render(self, bs):
        s = self.sim
        s.LIGHT.follow(bs, s.spec.cam)
        return self.mc.render(s.cam_xf, s.rep, s.ann_rgb, s.ann_dep, bs)

    def light_matrix(self, bs=None):
        return self.mc.light_matrix(bs)


def sim_camera(sim):
    return SimCamera(sim)
