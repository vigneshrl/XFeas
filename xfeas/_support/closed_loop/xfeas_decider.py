#!/usr/bin/env python
"""
XFeas as a DECISION LAYER on top of the baseline navigation stack (nav_baseline.py --decider
xfeas_decider:XFeasDecider). The planner, local planner, recovery and the robot's stock controller
are unchanged; XFeas only decides HOW (and whether) to execute what the nav stack wants:

  every PERIOD s while the nav stack is driving, with >= 2 s of proprio history:
    head-camera RGB-D (same intrinsics/mount as training) + 2 s proprio + robot descriptor
      -> select_xe sidecar scores continuous forward speeds (and a back-off) for what is ahead
    'go' at v*   -> cap the nav stack's forward command at v* (yaw rate scaled with it, so the
                    path curvature is unchanged) until the next decision
    no safe go   -> tell the nav stack the area ahead is infeasible (avoid_pts) so it replans
                    around it; if there is no way round, the nav stack gives up safely

Checkpoint: env XFEAS_CKPT (default: data/xe_train/v2/xfeas_*.pt). Decisions are logged per run
(run["decisions"] in runs.jsonl) so every speed cap / avoid can be audited against the video.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import glob, io, json, math, os, struct, subprocess
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
PY = xfeas.paths.XFEAS_PY
PERIOD = 1.0  # s between decisions
AVOID_COOLDOWN = 4.0  # s before another avoid request


class XFeasDecider:
    name = "xfeas"

    def __init__(self):
        ck = os.environ.get("XFEAS_CKPT") or sorted(glob.glob(os.path.join(ROOT, "data/xe_train/v2/xfeas_*.pt")))[0]
        self.ckpt = ck
        self.log, self.srv = [], None

    # ---------------------------------------------------------------- lifecycle
    def setup(self, stage, rep, world, rob, spec, kind):
        from pxr import UsdGeom, Gf
        import robots as RB

        self.stage, self.rep, self.rob, self.spec, self.kind, self.Gf = stage, rep, rob, spec, kind, Gf
        cam = UsdGeom.Camera.Define(stage, "/World/XFeasHeadCam")  # IDENTICAL to collect_*.py
        cam.CreateFocalLengthAttr(12.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
        self.cam_xf = UsdGeom.Xformable(cam.GetPrim()).MakeMatrixXform()
        rp = rep.create.render_product("/World/XFeasHeadCam", (224, 224))
        self.ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann_rgb.attach([rp])
        self.ann_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.ann_dep.attach([rp])
        self._mount_setup(stage, rep, rp, rob, spec, kind)
        self.buf = RB.ProprioBuffer(seconds=2.0, hz=50, phys_hz=50)  # decide() is called at 50 Hz
        self.srv = subprocess.Popen(
            [PY, os.path.join(THIS, "select_xe.py"), "--ckpt", self.ckpt, "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        print(f"[xfeas] decider ready ({os.path.basename(self.ckpt)})", flush=True)

    def reset(self, ctx):
        self.buf.reset()
        self.v_nom = float(ctx["speed"])
        self.cap, self.t_next, self.t_avoid = None, 2.0, -99.0
        self.desc = None
        self.log = []

    # ---------------------------------------------------------------- sidecar
    def _rd(self, n):
        b = b""
        while len(b) < n:
            c = self.srv.stdout.read(n - len(b))
            if not c:
                raise RuntimeError("selector sidecar died")
            b += c
        return b

    def _ask(self, rgb, dep, pb, pj, task):
        req = io.BytesIO()
        np.savez(
            req, rgb=rgb, depth=dep, prop_base=pb, prop_joints=pj, desc=json.dumps(self.desc), task=json.dumps(task)
        )
        b = req.getvalue()
        self.srv.stdin.write(struct.pack("<I", len(b)))
        self.srv.stdin.write(b)
        self.srv.stdin.flush()
        r = np.load(io.BytesIO(self._rd(struct.unpack("<I", self._rd(4))[0])), allow_pickle=False)
        return json.loads(str(r["choice"]))

    def _mount_setup(self, stage, rep, rp, rob, spec, kind, root=None):
        """v4 (xfeas/sim/camera_mount.py): the shared mount (G1 default = the 27 deg PHP/Holosoma head D435i, Spot =
        body-fixed front camera; XFEAS_CAM_MOUNT=legacy = the old spec.cam rule below)"""
        import sys as _s

        _c = os.path.join(ROOT, "xfeas", "sim")
        if _c not in _s.path:
            _s.path.insert(0, _c)
        import camera_mount as CMT

        self.CMT = CMT
        self.mcam = None
        if not CMT.is_legacy(kind):
            self.mcam = CMT.MountedCamera(
                stage, rob, kind, root=root or getattr(self, "_root_hint", None), spec=spec, rp=rp, rep=rep
            )
            import scene_lighting as SL

            SL.HEAD_MOUNT_FN = self.mcam.light_matrix

    def _snapshot(self, bs):
        if getattr(self, "mcam", None) is not None:  # v4 shared mount (camera_mount.py)
            out = self.mcam.render(self.cam_xf, self.rep, self.ann_rgb, self.ann_dep, bs)
            if out is None:
                raise RuntimeError("[xfeas] decision camera returned no frame")
            rgb, dep, M, info = out
            self.mcam.set_cmap(bs, info)
            self.last = dict(
                M=M,
                base=self.mcam.base_of(bs),
                mask_frac=info.get("mask_frac", 0.0),
                mask_src=info.get("mask_src"),
                mask=info.get("mask"),
                cam_h=info.get("cam_h"),
            )
            return rgb, dep
        Gf = self.Gf
        fwd, up, pitch = self.spec.cam
        c, s = math.cos(bs["yaw"]), math.sin(bs["yaw"])
        eye = bs["pos"] + np.array([fwd * c, fwd * s, up])
        tgt = eye + np.array([c * math.cos(pitch), s * math.cos(pitch), math.sin(pitch)])
        self.cam_xf.Set(
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        self.rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        rgb = np.asarray(self.ann_rgb.get_data())[..., :3].astype(np.uint8)
        dep = np.nan_to_num(np.asarray(self.ann_dep.get_data()).astype(np.float32), posinf=60.0)
        return rgb, dep

    def _avoid_pts(self, bs):
        """Footprint-wide band 0.6..2.0 m ahead of the robot (what the head camera judged)."""
        c, s = math.cos(bs["yaw"]), math.sin(bs["yaw"])
        half = 0.5 * max(self.desc["bbox_xyz_m"][1], 0.4) + 0.15
        pts = []
        for d in np.arange(0.6, 2.01, 0.1):
            for l in np.arange(-half, half + 1e-6, 0.1):
                pts.append((float(bs["pos"][0] + d * c - l * s), float(bs["pos"][1] + d * s + l * c)))
        return pts

    # ---------------------------------------------------------------- decision
    def decide(self, obs, planned):
        rob, bs, t = obs["robot"], obs["bs"], float(obs["t"])
        self.buf.tick(rob, bs)
        if self.desc is None:
            self.desc = rob.descriptor(self.stage)
        tw = np.asarray(planned["twist"], float).copy()
        if obs.get("recovery") == "drive" and t >= self.t_next and self.buf.full():
            self.t_next = t + PERIOD
            rgb, dep = self._snapshot(bs)
            pb, pj = self.buf.arrays()
            sp = self.spec
            v_hi = max(min(self.v_nom, sp.max_cmd[0]), sp.v_min + 0.01)
            alts = [
                dict(skill="go", v=[sp.v_min, v_hi], task=True),
                dict(skill="back_off", v=[min(sp.v_min, sp.max_cmd[1]), sp.max_cmd[1]], task=False),
            ]
            ch = self._ask(rgb, dep, pb, pj, dict(alts=alts, v_ref=max(sp.max_cmd[0], 0.3), seed=int(t * 10)))
            entry = dict(
                t=round(t, 2),
                x=round(float(bs["pos"][0]), 2),
                y=round(float(bs["pos"][1]), 2),
                skill=ch.get("skill"),
                v=ch.get("v"),
                p=ch.get("p_success"),
                p_std=ch.get("p_std"),
            )
            if ch.get("skill") == "go":
                self.cap = float(ch["v"])
                entry["action"] = f"cap {self.cap:.2f}"
            else:
                self.cap = None
                if t - self.t_avoid > AVOID_COOLDOWN:
                    self.t_avoid = t
                    entry["action"] = "avoid ahead"
                    self.log.append(entry)
                    print(f"[xfeas] t={t:.1f} {entry}", flush=True)
                    return dict(twist=np.zeros(4), avoid=True, avoid_pts=self._avoid_pts(bs))
                entry["action"] = "hold (avoid cooldown)"
                self.cap = sp.v_min
            self.log.append(entry)
            print(f"[xfeas] t={t:.1f} {entry}", flush=True)
        if self.cap is not None and tw[0] > self.cap > 0:
            r = self.cap / tw[0]
            tw[0] = self.cap
            tw[3] *= r
        return dict(twist=tw)

    def __del__(self):
        try:
            if self.srv is not None:
                self.srv.stdin.close()
                self.srv.wait(timeout=10)
        except Exception:
            pass
