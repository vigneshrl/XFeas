"""
Scripted RECORDING camera for comparison videos (episode_runner.py, env REC_CAM=<name>). Never seen by any method.

  REC_CAM=d1side   scene D1: a side camera perpendicular to the lane (looking +x at the corridor wall), dollying
                   along with the robot. Far from the door it stands 1.8 m from the lane at 1.05 m; as the robot
                   reaches the door it moves in to 0.85 m from the lane at 0.80 m above the floor, i.e. just under the
                   door's underside (0.87-0.94 m there), so the gap between Spot's back and the door reads as a line.
                   Placement checked with PhysX raycasts at install (printed: '[reccam] check ...').

Night: the robot's headlight lights only the area ahead of the robot, so the recording camera gets a FILL light
(/World/RecFill, a sphere light next to the recording camera) that is switched ON only for the recording camera's own
render and OFF again before any method renders. Methods render their head cameras in their own act() with the fill
OFF (and >= 1 fresh subframe); scripts/xembody/rec_cam_verify.py checks that a head-camera image rendered right after
a fill-lit recording pass equals one rendered without any recorder (up to render noise).

A small head-camera picture-in-picture (same mount and intrinsics as the methods' head camera, 256x256) is rendered
in a separate pass with the fill light OFF and saved as pip_<n>.png next to the frames (episode_video.py only reads
.jpg, so it ignores them).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os
import numpy as np


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class D1Side:
    X0, FL = -4.0, 0.681

    def __init__(self):
        self.ys = None
        self.focal = 12.0

    def pose(self, bs):
        ry = float(bs["pos"][1])
        self.ys = ry if self.ys is None else self.ys + 0.12 * (ry - self.ys)
        s = _smooth((self.ys - (-3.9)) / 0.9)                          # 0 approach .. 1 at the door
        ex, ez = -5.8 + 0.95 * s, self.FL + 1.05 - 0.25 * s
        self.focal = 12.0 - 5.0 * s                                   # wider lens close in (82 -> 113 deg HFOV)
        ey = min(max(self.ys + 0.10, -5.0), -0.5)
        eye = np.array([ex, ey, ez])
        tgt = np.array([-3.6, ey, self.FL + 0.45 + 0.15 * (1 - s)])
        return eye, tgt

    def samples(self):
        out = []
        for ry in np.arange(-6.0, 1.51, 0.25):
            self.ys = None
            out.append((ry,) + tuple(self.pose(dict(pos=np.array([self.X0, ry, 0.0])))))
        self.ys = None
        return out


class LaneSide:
    """Generic side camera for any straight lane (G1 overhang study, scene D1g): same dolly as D1Side, in the lane
    frame. Config from env REC_LANE (json): A [x, y] lane start, yaw, floor, s_door [s0, s1] (object span along the
    lane), side (+1 = left of the heading), lat [far, near], z [far, near] (above floor), z_tgt (above floor)."""

    def __init__(self, cfg=None):
        import json as _j
        c = cfg or _j.loads(os.environ["REC_LANE"])
        self.A = np.asarray(c["A"], float); self.yaw = float(c["yaw"]); self.FL = float(c["floor"])
        self.d = np.array([math.cos(self.yaw), math.sin(self.yaw)]); self.n = np.array([-self.d[1], self.d[0]]) * float(c.get("side", 1))
        self.s0, self.s1 = c["s_door"]; self.lat = c.get("lat", [1.8, 0.85]); self.z = c.get("z", [1.25, 0.95])
        self.zt = float(c.get("z_tgt", 0.6)); self.len = float(c.get("len", 6.0))
        self.ys = None; self.focal = 12.0

    def lane_origin(self, eye):
        s = float((np.asarray(eye[:2]) - self.A) @ self.d)
        p = self.A + s * self.d
        return np.array([p[0], p[1], eye[2]])

    def pose(self, bs):
        rs = float((np.asarray(bs["pos"][:2], float) - self.A) @ self.d)
        self.ys = rs if self.ys is None else self.ys + 0.12 * (rs - self.ys)
        u = _smooth((self.ys - (self.s0 - 1.9)) / 0.9)
        lat = self.lat[0] + (self.lat[1] - self.lat[0]) * u
        ez = self.FL + self.z[0] + (self.z[1] - self.z[0]) * u
        self.focal = 12.0 - 5.0 * u
        se = min(max(self.ys + 0.10, 1.0), self.len - 1.0)
        p = self.A + se * self.d
        eye = np.array([p[0] + lat * self.n[0], p[1] + lat * self.n[1], ez])
        tg = p - 0.4 * self.n
        tgt = np.array([tg[0], tg[1], self.FL + self.zt + 0.15 * (1 - u)])
        return eye, tgt

    def samples(self):
        out = []
        for rs in np.arange(0.0, self.len + 0.01, 0.25):
            self.ys = None
            p = self.A + rs * self.d
            out.append((rs,) + tuple(self.pose(dict(pos=np.array([p[0], p[1], 0.0])))))
        self.ys = None
        return out


CAMS = dict(d1side=D1Side, laneside=LaneSide)


def scene_side_cfg(start, goal, floor, hazard, tall=False):
    """REC_CAM=sceneside: LaneSide for any straight-lane scene (episode_runner SCENES): object span = hazard s +- 0.45,
    the side with the clearer view is picked at the first frame (PhysX rays), eye lower for quadrupeds; the near eye
    stays >= 1.25 m from the lane centre, i.e. outside the support columns of raised objects (<= 1.15 m)."""
    A = np.asarray(start, float)[:2]; g = np.asarray(goal, float)[:2]
    d = (g - A) / max(np.linalg.norm(g - A), 1e-6)
    sh = float((np.asarray(hazard, float)[:2] - A) @ d)
    return dict(A=A.tolist(), yaw=math.atan2(d[1], d[0]), floor=float(floor), s_door=[sh - 0.45, sh + 0.45], side=1,
                lat=[1.8, 1.35] if tall else [1.7, 1.25], z=[1.6, 1.3] if tall else [1.2, 1.0], z_tgt=0.9 if tall else 0.5,
                len=float(np.linalg.norm(g - A)))


def install(rec, stage, rep, pxq, carb, spec, robot_root, name):
    from pxr import UsdGeom, UsdLux, Gf
    if name == "d1side" and robot_root.endswith("/g1a"):   # G1 overhang study: humanoid needs a farther, taller view of D1
        cam = LaneSide(dict(A=[-4.0, -6.0], yaw=math.pi / 2, floor=0.681, s_door=[3.55, 4.30], side=1, lat=[1.8, 1.45],
                            z=[1.25, 0.85], z_tgt=0.6, len=7.5))
        cam.X0 = -4.0
    elif name == "sceneside":
        import json as _j
        cam = LaneSide(_j.loads(os.environ["REC_SCENE_LANE"])); cam.auto_side = True
    else:
        cam = CAMS[name]()
    # ---- fill light (recording camera only)
    fill = UsdLux.SphereLight.Define(stage, "/World/RecFill")
    fill.CreateRadiusAttr(0.15); fill.CreateIntensityAttr(0.0)
    fill.CreateEnableColorTemperatureAttr(True); fill.CreateColorTemperatureAttr(5000.0)
    from pxr import Sdf
    fill.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
    fxf = UsdGeom.Xformable(fill.GetPrim()).MakeMatrixXform()
    cam.fill_i = float(os.environ.get("REC_FILL", "1.2e5"))
    # ---- head-camera PiP (fill OFF pass)
    pc = UsdGeom.Camera.Define(stage, "/World/RecHeadPiP")
    pc.CreateFocalLengthAttr(12.0); pc.CreateHorizontalApertureAttr(20.955); pc.CreateVerticalApertureAttr(20.955)
    pc.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
    pxf = UsdGeom.Xformable(pc.GetPrim()).MakeMatrixXform()
    prp = rep.create.render_product("/World/RecHeadPiP", (256, 256))
    pann = rep.AnnotatorRegistry.get_annotator("rgb"); pann.attach([prp])
    PIP_EVERY = int(os.environ.get("REC_PIP_EVERY", "2"))
    state = dict(pip=None)
    recam = UsdGeom.Camera(stage.GetPrimAtPath("/World/ChaseHQ"))

    def check():
        """PhysX raycasts (run at the first grab, when physics is live): the eye must be reachable from the lane"""
        bad = 0
        for ry, eye, tgt in cam.samples():
            o = cam.lane_origin(eye) if hasattr(cam, "lane_origin") else np.array([cam.X0, eye[1], eye[2]])
            dv = eye - o; n = float(np.linalg.norm(dv))
            h = pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, dv / n)), n + 0.15)
            hit = bool(h and h.get("hit")) and not str(h.get("collision", "")).startswith(robot_root)
            if hit:
                bad += 1
            print(f"[reccam] check robot_y={ry:+.2f} eye=({eye[0]:.2f},{eye[1]:.2f},{eye[2]:.2f}) lane->eye "
                  f"{'BLOCKED by ' + str(h.get('collision')) + ' at %.2f m' % float(h['distance']) if hit else 'clear'} (+0.15 m)", flush=True)
        print(f"[reccam] {name}: {bad} blocked samples", flush=True)
        return bad

    def grab(bs, tag=0, _rec=rec):
        import cv2
        if not state.get("checked"):
            state["checked"] = True
            if getattr(cam, "auto_side", False):      # sceneside: take the lane side with the clearer view
                b1 = check(); cam.n = -cam.n; b2 = check()
                if b1 <= b2:
                    cam.n = -cam.n
                print(f"[reccam] auto side: blocked {b1} vs {b2} -> side {'first' if b1 <= b2 else 'second'}", flush=True)
            else:
                check()
        G = Gf
        eye, tgt = cam.pose(bs)
        recam.GetFocalLengthAttr().Set(float(cam.focal))
        _rec.xf.Set(G.Matrix4d().SetLookAt(G.Vec3d(*map(float, eye)), G.Vec3d(*map(float, tgt)), G.Vec3d(0, 0, 1)).GetInverse())
        fwd, up, pitch = spec.cam
        c, s_ = math.cos(bs["yaw"]), math.sin(bs["yaw"])
        he = np.asarray(bs["pos"], float) + np.array([fwd * c, fwd * s_, up])
        ht_ = he + np.array([c * math.cos(pitch), s_ * math.cos(pitch), math.sin(pitch)])
        pxf.Set(G.Matrix4d().SetLookAt(G.Vec3d(*map(float, he)), G.Vec3d(*map(float, ht_)), G.Vec3d(0, 0, 1)).GetInverse())
        if _rec.n % PIP_EVERY == 0:                                # PiP pass: fill OFF
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
            p = np.asarray(pann.get_data())
            if p.ndim == 3 and p.size:
                state["pip"] = p[..., :3].copy()
        if state["pip"] is not None:
            cv2.imwrite(os.path.join(_rec.dir, f"pip_{_rec.n:06d}.png"), np.ascontiguousarray(state["pip"][..., ::-1]))
        # recording pass: fill ON (next to the recording camera, a little toward the lane), then OFF
        d = tgt - eye; d[2] = 0; d /= max(np.linalg.norm(d), 1e-6)
        fxf.Set(G.Matrix4d().SetTranslate(G.Vec3d(*map(float, eye + 0.30 * d + np.array([0, 0, -0.05])))))
        fill.GetIntensityAttr().Set(cam.fill_i)
        try:
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
            im = np.asarray(_rec.ann.get_data())
        finally:
            fill.GetIntensityAttr().Set(0.0)
        if im.ndim != 3 or not im.size:
            return
        cv2.imwrite(os.path.join(_rec.dir, f"{_rec.n:06d}.jpg"), np.ascontiguousarray(im[..., 2::-1]),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        _rec.tags.append(int(tag)); _rec.n += 1

    rec.grab = grab
    rec._pose = cam.pose
    return cam
