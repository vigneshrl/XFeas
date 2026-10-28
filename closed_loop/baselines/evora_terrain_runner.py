#!/usr/bin/env python
"""
Baselines round 1 (experiments/BASELINE_PROTOCOL.md): EVORA on Spot, sets 1-2 (terrain field). ONE tile x ONE trial
per process (fresh process per trial).

Copied from scripts/baselines_r1/travblox_terrain_runner.py (itself copied from scripts/terrain/run_terrain.py feel
mode) so that every baseline sees the identical field, start, goal and outcome rules:
  * field = catalog.field_spec(seed 0, all 14 types, difficulty 0.6), build.py, run_terrain's lighting profile
    (sun 175, HDRI sky x0.2, headlight x0.2), night_white, XFEAS_FOOT_MATERIAL from the environment (fm1),
    robot spotp (stock ReLIC Spot, clips commands to |vx| <= 1.0 m/s, |wz| <= 1.0 rad/s);
  * trial-k start = run_terrain's (tile-local x = -1.6, y_off / yaw0 from rng(1000*seed + k));
  * goal = 0.5 m before the far tile edge on the lane centre (tile-local x = L - 0.5, y = 0), relative goal only;
  * outcomes: success = base within 0.5 m of the goal (strict), fall, stuck = no 0.2 m of progress toward the goal
    for 15 s, timeout T_MAX = 3 x straight / 0.26 + 40 s (same budget as the TravBlox runner);
  * method = evora_plugin_r1:EvoraMPPI (the v2 EVORA bridge: mppi_numba as released + TractionNet ckpt, map-free);
    its sidecar logs its own map + chosen trajectory every planning step to <frames>/<tile>_t<k>/evora_map/.
Outputs: <out>/<tile>_t<k>.json, <out>/traj_<tile>_t<k>.npy, <frames>/<tile>_t<k>/ (chase jpgs, meta.json, evora_map/).

  XFEAS_FOOT_MATERIAL=1 isaac_sim/isaaclab_python.sh closed_loop/baselines/evora_terrain_runner.py \
      --tile mud_wet --trial 0 --out data/baselines_r1/evora/2_slippery
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse, json, math, os, re, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
XE = os.path.join(ROOT, "closed_loop")
TER = os.path.join(ROOT, "scenes", "terrain")
for p in (TER, XE, os.path.join(XE, "controllers"), THIS):
    sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--tile", required=True)
ap.add_argument("--trial", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--difficulty", type=float, default=0.6)
ap.add_argument("--light", default="night_white", choices=["night_white", "night_yellow", "day"])
ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="", help="frame dir root (default <out>/frames)")
ap.add_argument("--video", type=int, default=1)
ap.add_argument("--vid-every", dest="vid_every", type=int, default=4, help="50 Hz / 4 = 12.5 fps")
ap.add_argument("--method", default="evora_plugin_r1:EvoraMPPI")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SUN, SKY_SCALE, HL_SCALE = 175.0, 0.2, 0.2
SKY_HDR = xfeas_paths.ISAAC_ASSETS + "/Assets/Skies/Cloudy/lakeside_4k.hdr"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": "RaytracedLighting"})
import carb
try:
    carb.settings.get_settings().set("/log/level", "error")
    carb.settings.get_settings().set("/rtx/raytracing/fractionalCutoutOpacity", True)
except Exception:
    pass
import omni.usd
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf
from isaacsim.core.utils.stage import create_new_stage
import catalog as CAT
import build as B
import foot_material as FMAT
FM_ON = FMAT.enabled()
import spot_posture as FSP
FAC, KIND = FSP.make_spotp, "spotp"
HZ = int(FAC.SPEC.phys_hz)

create_new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
spec = CAT.field_spec(a.seed, None, a.difficulty)
TILES, MATS = B.build_field(stage, spec)
TT = [t for t in TILES if t["type"] == a.tile]
assert TT, f"tile {a.tile} not in field"
TILE = TT[0]
print(f"[r1] built {len(TILES)} tiles; running {TILE['id']} origin {TILE['origin']}", flush=True)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(SUN); sun.CreateAngleAttr(0.8); sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 0.0, 30.0))
for _ in range(30):
    app.update()

from omni.isaac.core.world import World
world = World(physics_dt=1.0 / HZ, rendering_dt=1 / 50.0, stage_units_in_meters=1.0)
stage = world.stage
import scene_lighting
LIGHT = scene_lighting.Lighting(stage)
LIGHT.sky.CreateTextureFileAttr().Set(Sdf.AssetPath(SKY_HDR)); LIGHT.sky.CreateTextureFormatAttr().Set("latlong")
LIGHT.sky.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def set_light(mode, rng):                       # = run_terrain.set_light with its default profile
    info = LIGHT.set(mode, rng)
    if mode == "day":
        LIGHT.sky.GetIntensityAttr().Set(info["sky"] * SKY_SCALE); info["sky"] = round(info["sky"] * SKY_SCALE)
        info["sun_I"] = round(SUN * info["sun"])
    elif mode.startswith("night"):
        hi = info["headlight_I"] * HL_SCALE
        LIGHT.hl.GetIntensityAttr().Set(hi); info["headlight_I"] = round(hi)
    info["profile"] = dict(sun=SUN, sky_scale=SKY_SCALE, hl_scale=HL_SCALE)
    return info


def tile_height(t, x, y):
    return t["height_at"](x, y)


SPEC = FAC.SPEC
rob = FAC(KIND, f"/World/{KIND}", [-1.5, TILES[0]["origin"][1], SPEC.spawn_dz], yaw=0.0)
try:                                            # shared head-camera mount for the headlight (as run_terrain v4)
    sys.path.insert(0, os.path.join(ROOT, "isaac_sim"))
    import camera_mount as _CMT
    if not _CMT.is_legacy(KIND):
        scene_lighting.HEAD_MOUNT_FN = _CMT.MountedCamera(stage, rob, KIND, root=f"/World/{KIND}", spec=SPEC).light_matrix
except Exception as _e4:
    print("[camera_mount] headlight mount not set:", repr(_e4), flush=True)
world.reset()
world.add_physics_callback("drive", lambda dt: rob.step(dt))
for _ in range(int(0.5 * HZ)):
    world.step(render=False)

from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.replicator.core")
import omni.replicator.core as rep
from omni.physx import get_physx_scene_query_interface
from PIL import Image
import importlib
_mod, _cls = a.method.split(":")
METHOD = getattr(importlib.import_module(_mod), _cls)()
METHOD.setup(dict(stage=stage, rep=rep, world=world, app=app, rob=rob, spec=SPEC, kind=KIND, hz=HZ,
                  pxq=get_physx_scene_query_interface(), carb=carb, robot_root=f"/World/{KIND}"))
LIGHT_INFO = set_light(a.light, np.random.default_rng(7))
print(f"[r1] METHOD {a.method} ({getattr(METHOD, 'label', '')}); lighting {a.light} {LIGHT_INFO}; fm={int(FM_ON)}", flush=True)


class Chase:
    """behind-right chase camera + recording-only fill light (switched on only for this render)"""
    RES = (1280, 720)
    BACK, SIDE, UP, CLEAR = 2.2, 1.3, 1.25, 0.9

    def __init__(self, d):
        self.dir = d; os.makedirs(d, exist_ok=True)
        for fn in os.listdir(d):
            os.remove(os.path.join(d, fn))
        self.pxq = get_physx_scene_query_interface()
        c = UsdGeom.Camera.Define(stage, "/World/VidCam_chase")
        c.CreateFocalLengthAttr(14.0); c.CreateHorizontalApertureAttr(20.955)
        c.CreateVerticalApertureAttr(20.955 * self.RES[1] / self.RES[0]); c.CreateClippingRangeAttr(Gf.Vec2f(0.03, 200.0))
        self.xf = UsdGeom.Xformable(c.GetPrim()).MakeMatrixXform()
        rp = rep.create.render_product("/World/VidCam_chase", self.RES)
        self.an = rep.AnnotatorRegistry.get_annotator("rgb"); self.an.attach([rp])
        f = UsdLux.SphereLight.Define(stage, "/World/VidFill")
        f.CreateRadiusAttr(0.2); f.CreateIntensityAttr(0.0)
        f.CreateEnableColorTemperatureAttr(True); f.CreateColorTemperatureAttr(4800.0)
        f.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
        self.fill, self.fill_xf = f, UsdGeom.Xformable(f.GetPrim()).MakeMatrixXform()
        self.fill_i = 4.0e5 if a.light.startswith("night") else 0.0
        self.p, self.n = None, 0

    def _ray(self, o, d, n):
        h = self.pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, d)), float(n))
        if h and h.get("hit") and not str(h.get("collision", "")).startswith(f"/World/{KIND}"):
            return float(h["distance"])
        return None

    def _top(self, x, y):
        h = self._ray((x, y, 20.0), (0, 0, -1), 40.0)
        return 20.0 - h if h else 0.0

    def shot(self, bs):
        p = np.asarray(bs["pos"], float)
        al = 1.0 - math.exp(-(a.vid_every / 50.0) / 0.6)
        self.p = p.copy() if self.p is None else self.p + al * (p - self.p)
        q = self.p
        eye = np.array([q[0] - self.BACK, q[1] - self.SIDE, 0.0])
        eye[2] = max(q[2] + self.UP, self._top(eye[0], eye[1]) + self.CLEAR)
        tgt = np.array([q[0] + 1.2, q[1], q[2] - 0.25])
        o = p + np.array([0, 0, 0.3])
        for _ in range(4):
            dv = eye - o; n = float(np.linalg.norm(dv))
            if not self._ray(o, dv / n, n + 0.3):
                break
            eye = eye + np.array([0.0, 0.0, 0.5])
        self.xf.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse())
        dd = tgt - eye; dd[2] = 0; dd /= max(np.linalg.norm(dd), 1e-6)
        self.fill_xf.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, eye + 0.3 * dd + np.array([0, 0, 0.2])))))
        if self.fill_i > 0:
            self.fill.GetIntensityAttr().Set(self.fill_i)
        try:
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
            im = np.asarray(self.an.get_data())
        finally:
            self.fill.GetIntensityAttr().Set(0.0)
        if im.ndim != 3 or not im.size:
            return False
        Image.fromarray(im[..., :3].astype(np.uint8)).save(os.path.join(self.dir, f"{self.n:06d}.jpg"), quality=90)
        self.n += 1
        return True


tag = f"{a.tile}_t{a.trial}"
fdir = os.path.join(a.frames or os.path.join(a.out, "frames"), tag)
CH = Chase(fdir) if a.video else None
for _ in range(3):
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)

# ------------------------------------------------------------------ trial-k start (= run_terrain.place / feel loop)
t = TILE; ox, oy, _ = t["origin"]; L = float(t["size"][0]) if "size" in t else float(CAT.TILE_SIZE[0])
X0_START = -1.6
k = a.trial
rng = np.random.default_rng(1000 * a.seed + k)
y_off = 0.0 if k == 0 else float(rng.uniform(-0.15, 0.15))
yaw0 = 0.0 if k == 0 else float(rng.uniform(-0.05, 0.05))
rob.teleport(ox + X0_START, oy + y_off, yaw0, 0.0); rob.command(np.zeros(4))
for _ in range(int(1.5 * HZ)):
    world.step(render=False)
bs = rob.base_state()
start_ok = bs is not None and not rob.fallen(bs) and np.hypot(bs["pos"][0] - ox - X0_START, bs["pos"][1] - oy - y_off) < 0.5
goal = np.array([ox + L - 0.5, oy])
straight = float(np.hypot(*(goal - np.array([ox + X0_START, oy + y_off]))))
T_MAX = 3.0 * straight / 0.26 + 40.0
res = dict(method="evora", label=getattr(METHOD, "label", ""), robot=KIND, set="terrain_field", tile=a.tile,
           bucket=t["bucket"], trial=k, seed=a.seed, difficulty=a.difficulty, foot_material=int(FM_ON), light=a.light,
           light_info=LIGHT_INFO, start=[round(ox + X0_START, 3), round(oy + y_off, 3), round(yaw0, 4)],
           goal=np.round(goal, 3).tolist(), goal_local_x=round(L - 0.5, 2), straight_m=round(straight, 2), t_max=round(T_MAX, 1),
           map_given=False, success_rule="strict: base within 0.5 m of goal",
           obstacle_band=os.environ.get("EVORA_BAND", "local"))
if not start_ok:
    res.update(outcome="invalid_start", success=False)
    json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1, default=float)
    print(f"[r1] {tag}: invalid_start", flush=True); print("[r1] DONE", flush=True); app.close(); sys.exit(0)

os.environ["EVORA_R1_LOGDIR"] = os.path.join(fdir, "evora_map")
METHOD.reset(dict(speed_frac=0.0, floor_z0=0.0, scene=f"terrain_{a.tile}", trial=k))
ctl = max(1, HZ // 50); dt_c = ctl / HZ
t_sim, tick, outcome, info = 0.0, 0, None, ""
best, t_prog, traj, frames, viz, t_viz = 1e9, 0.0, [], [], [], -1.0
tw = np.zeros(4); wall0 = time.time(); max_tilt = 0.0
while True:
    for _ in range(ctl):
        world.step(render=False)
    t_sim += dt_c; tick += 1
    bs = rob.base_state()
    if bs is None or rob.fallen(bs):
        outcome = "fall"; break
    x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
    d = goal - np.array([x, y]); c, s_ = math.cos(yaw), math.sin(yaw)
    LIGHT.follow(bs, SPEC.cam)
    r = METHOD.act(dict(t=t_sim, dt=dt_c, bs=bs, rob=rob, goal_rel=np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]]),
                        goal_xy=goal.copy(), goal_dist=float(np.hypot(*d))))
    tw = np.asarray(r["twist"], float); info = r.get("info", "")
    rob.command(tw)
    dg = float(np.hypot(*d)); max_tilt = max(max_tilt, math.degrees(bs["tilt"]))
    if not traj or t_sim - traj[-1][0] >= 0.1:
        traj.append([round(t_sim, 2), round(x, 3), round(y, 3), round(float(bs["pos"][2]), 3), round(yaw, 3),
                     round(float(bs["v_b"][0]), 3), round(float(tw[0]), 3), round(float(tw[3]), 3), round(math.degrees(bs["tilt"]), 1)])
    if CH is not None and (tick - 1) % a.vid_every == 0:
        if CH.shot(bs):
            frames.append(dict(i=CH.n - 1, t=round(t_sim, 2), x=round(x, 3), y=round(y, 3), yaw=round(yaw, 3),
                               v=round(float(bs["v_b"][0]), 3), cmd_v=round(float(tw[0]), 3), cmd_w=round(float(tw[3]), 3),
                               dist=round(dg, 2), info=info, plan_k=int(getattr(METHOD, "k", 0)) - 1))
    if r.get("giveup"):
        outcome = "gave_up"; break
    if dg < best - 0.2:
        best, t_prog = dg, t_sim
    if dg < 0.5:
        outcome = "success"; break
    if t_sim - t_prog > 15.0:
        outcome = "stuck"; break
    if t_sim > T_MAX:
        outcome = "timeout"; break
rob.command(np.zeros(4))
METHOD.flush()
bs = rob.base_state()
fin = bs["pos"][:2] if bs is not None else np.array(traj[-1][1:3]) if traj else goal * np.nan
res.update(outcome=outcome, success=outcome == "success", t_end=round(t_sim, 2),
           dist_to_goal=round(float(np.hypot(*(goal - fin))), 2), x_reached_local=round(max(r_[1] for r_ in traj) - ox, 2) if traj else None,
           progress=round(float(np.clip((max(r_[1] for r_ in traj) - ox - X0_START) / (L - 0.5 - X0_START), 0, 1)), 3) if traj else 0,
           max_tilt_deg=round(max_tilt, 1), mean_v=round(float(np.mean([r_[5] for r_ in traj])), 3) if traj else 0,
           last_info=info, wall_s=round(time.time() - wall0, 1), decisions=list(getattr(METHOD, "log", []) or []),
           frames=CH.n if CH else 0, n_plan_steps=int(getattr(METHOD, "k", 0)),
           self_px_mean=round(float(np.mean(getattr(METHOD, "self_px", [0]) or [0])), 1),
           self_px_max=int(np.max(getattr(METHOD, "self_px", [0]) or [0])))
np.save(os.path.join(a.out, f"traj_{tag}.npy"), np.asarray(traj, np.float32))
json.dump(res, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1, default=float)
if CH is not None:
    json.dump(dict(result={k_: v_ for k_, v_ in res.items() if k_ != "decisions"}, tile_origin=[ox, oy], tile_len=L,
                   x0_local=X0_START, fps=50.0 / a.vid_every, frames=frames),
              open(os.path.join(fdir, "meta.json"), "w"), indent=1, default=float)
print(f"[r1] {tag}: {outcome} t={t_sim:.1f} dist={res['dist_to_goal']} frames={res['frames']} plan_steps={res['n_plan_steps']} wall={res['wall_s']}s", flush=True)
try:
    METHOD.srv.stdin.close(); METHOD.srv.wait(timeout=20)
except Exception:
    pass
print("[r1] DONE", flush=True)
app.close()
