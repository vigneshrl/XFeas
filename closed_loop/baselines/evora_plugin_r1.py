#!/usr/bin/env python
"""
[baselines_r1 copy of baselines/travblox_eval/arl4robots/v2/evora_plugin.py, 2026-10-21] Identical camera, rate,
twist and sidecar protocol. Changes: (1) the sidecar is evora_mppi_server_r1.py (v2 planning + read-only map
logging); (2) the map log directory (env EVORA_R1_LOGDIR) is passed at reset; (3) robot kinds "spotp" (ReLIC Spot of
the terrain field) and "spot" share Spot's footprint half-width 0.25 m (v2 looked up FOOT_HALF[kind], which has no
"spotp" key and would have fallen back to 0.3); (4) flush() writes the last log chunk at episode end;
(5) SELF-FILTER: the ReLIC Spot carries its arm, which the head camera sees ~1 m ahead at 0.5-1 m height; v2's Spot
had no arm. Without a filter the bridge's obstacle layer marks the robot's own gripper as an obstacle that moves with
it (first r1 trial, dirt_bumpy t0, 2026-10-21). Pixels whose instance-segmentation prim path is under the robot's
root are sent as `selfmask` and excluded from the map back-projection only (TractionNet still sees the unmodified
RGB-D image). This is the standard robot-body filter of any real deployment, not a change to EVORA's planner.
""" """
EVORA (mppi_numba as released + previously trained TractionNet checkpoint) as an episode_runner method:

  $ISAACSIM_PYTHON closed_loop/episode_runner.py --robot spot --scene S1,...,S6 \
      --method evora_plugin:EvoraMPPI --trials 3 --tag _ep_evora      (PYTHONPATH=.../arl4robots/v2)

Thin interface only: a head RGB-D camera on the robot (same intrinsics / mount as every other camera
method in this project: 224x224, 12 mm / 20.955 mm, spec.cam), odometry pose and the relative goal go to
the sidecar (evora_mppi_server.py) every MPPI step (dt = 0.1 s, the planner's own dt); the sidecar builds
the map online and returns MPPI's (v, w), which is sent as the body twist [v, 0, 0, w] and held until the
next step. The robot adapter clips it to the robot's limits. No map, route or waypoints.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import io, json, math, os, struct, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = xfeas_paths.EVORA_PY
FOOT_HALF = dict(spot=0.25, anymal=0.30, h1=0.28, carter=0.32, jackal=0.22, iris=0.30)   # local_nav.FOOT[k][1]
DT = 0.1


class EvoraMPPI:
    name = "evora"
    label = "EVORA (mppi_numba MPPI as released + TractionNet ckpt), map-free"

    def setup(self, ctx):
        from pxr import UsdGeom, Gf
        self.ctx, self.Gf, self.rep = ctx, Gf, ctx["rep"]
        self.kind, self.spec = ctx["kind"], ctx["spec"]
        cam = UsdGeom.Camera.Define(ctx["stage"], "/World/EvoraHeadCam")
        cam.CreateFocalLengthAttr(12.0); cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955); cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
        self.cam_xf = UsdGeom.Xformable(cam.GetPrim()).MakeMatrixXform()
        rp = self.rep.create.render_product("/World/EvoraHeadCam", (224, 224))
        self.ann_rgb = self.rep.AnnotatorRegistry.get_annotator("rgb"); self.ann_rgb.attach([rp])
        self.ann_dep = self.rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); self.ann_dep.attach([rp])
        self.root = ctx.get("robot_root") or f"/World/{self.kind}"
        try:
            self.ann_seg = self.rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast", init_params={"colorize": False})
            self.ann_seg.attach([rp])
        except Exception as e:
            print(f"[evora] instance segmentation unavailable ({e!r}): no self-filter", flush=True); self.ann_seg = None
        self.self_px = []
        env = dict(os.environ, NUMBA_FORCE_CUDA_CC="9.0")       # numba-cuda/NVVM 12.9 rejects sm_120 atomics; PTX 9.0 is JIT'd
        for k in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):     # do not leak Isaac's python env into the sidecar
            env.pop(k, None)
        self.srv = subprocess.Popen([PY, os.path.join(HERE, "evora_mppi_server_r1.py")], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, bufsize=0, env=env)
        print("[evora] plugin ready", flush=True)

    def _rd(self, n):
        b = b""
        while len(b) < n:
            c = self.srv.stdout.read(n - len(b))
            if not c:
                raise RuntimeError("evora sidecar died")
            b += c
        return b

    def _ask(self, **kw):
        b = io.BytesIO(); np.savez(b, **kw); b = b.getvalue()
        self.srv.stdin.write(struct.pack("<I", len(b))); self.srv.stdin.write(b); self.srv.stdin.flush()
        return json.loads(str(np.load(io.BytesIO(self._rd(struct.unpack("<I", self._rd(4))[0])))["out"]))

    def reset(self, ep):
        self.ep = ep
        self.t_next, self.tw, self.log, self.k, self.info = 0.0, np.zeros(4), [], 0, "EVORA: starting"
        self.started = False

    def _snapshot(self, bs):
        fwd, up, pitch = self.spec.cam
        c, s = math.cos(bs["yaw"]), math.sin(bs["yaw"])
        eye = np.asarray(bs["pos"], float) + np.array([fwd * c, fwd * s, up])
        f = np.array([c * math.cos(pitch), s * math.cos(pitch), math.sin(pitch)])
        Gf = self.Gf
        self.cam_xf.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, eye + f)),
                                                Gf.Vec3d(0, 0, 1)).GetInverse())
        self.rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        rgb = np.asarray(self.ann_rgb.get_data())[..., :3].astype(np.uint8)
        dep = np.nan_to_num(np.asarray(self.ann_dep.get_data()).astype(np.float32), posinf=60.0)
        m = np.zeros(dep.shape[:2], np.uint8)
        if self.ann_seg is not None:
            try:
                sd = self.ann_seg.get_data(); ids = np.asarray(sd["data"]).reshape(dep.shape[:2])
                lab = sd["info"]["idToLabels"]
                own = [int(k) for k, v in lab.items() if str(v).startswith(self.root)]
                if own:
                    m = np.isin(ids, own).astype(np.uint8)
            except Exception as e:
                if not self.self_px:
                    print(f"[evora] self-mask failed ({e!r})", flush=True)
        self.self_px.append(int(m.sum()))
        ld = os.environ.get("EVORA_R1_LOGDIR")
        if ld and len(self.self_px) in (1, 50, 200):          # head-camera evidence: RGB, depth, self-mask
            try:
                os.makedirs(ld, exist_ok=True)
                np.savez_compressed(os.path.join(ld, f"head_{len(self.self_px):04d}.npz"), rgb=rgb, depth=dep.astype(np.float16), selfmask=m)
            except Exception:
                pass
        self.selfmask = m
        return rgb, dep, eye, f

    def act(self, obs):
        t, bs = float(obs["t"]), obs["bs"]
        if not self.started:
            pos = np.asarray(bs["pos"][:2], float)
            kk = "spot" if str(self.kind).startswith("spot") else self.kind
            kw = dict(logdir=os.environ["EVORA_R1_LOGDIR"]) if os.environ.get("EVORA_R1_LOGDIR") else {}
            self._ask(cmd="reset", start=pos, goal=np.asarray(obs["goal_xy"], float), floor=float(self.ep["floor_z0"]),
                      inflate=FOOT_HALF.get(kk, 0.3) + 0.1, **kw)
            self.started = True
        if t >= self.t_next:
            self.t_next = t + DT
            rgb, dep, eye, f = self._snapshot(bs)
            r = self._ask(cmd="step", pos=np.asarray(bs["pos"], float), yaw=float(bs["yaw"]), has_img=True,
                          rgb=rgb, depth=dep, eye=eye, f=f, selfmask=self.selfmask)
            self.tw = np.array([r.get("v", 0.0), 0.0, 0.0, r.get("w", 0.0)])
            self.info = (f"EVORA-MPPI v={self.tw[0]:.2f} w={self.tw[3]:.2f}  traction ahead {r.get('trac_ahead', -1):.2f}"
                         if r.get("ok") else f"EVORA: {r.get('info')}")
            self.k += 1
            if self.k % 50 == 1:                                  # heartbeat for the batch watchdog
                print(f"[evora] t={t:.1f} pos=({bs['pos'][0]:.2f},{bs['pos'][1]:.2f}) {self.info}", flush=True)
            if self.k % 5 == 1:
                self.log.append(dict(t=round(t, 2), x=round(float(bs["pos"][0]), 2), y=round(float(bs["pos"][1]), 2),
                                     **{k: r[k] for k in ("v", "w", "ok", "trac_ahead", "known", "n_obst", "solve_ms", "info") if k in r}))
        return dict(twist=self.tw.copy(), info=self.info, decisions=self.log)

    def flush(self):
        try:
            self._ask(cmd="flush")
        except Exception as e:
            print("[evora] flush failed", e, flush=True)

    def __del__(self):
        try:
            self.srv.stdin.close(); self.srv.wait(timeout=10)
        except Exception:
            pass
