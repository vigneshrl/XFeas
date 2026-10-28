#!/usr/bin/env python
"""
Baselines round 1 (experiments/BASELINE_PROTOCOL.md), sets 1-2: VAMOS (released full stack, cmp_methods.VAMOSCmp =
vamos_plugin.VAMOS with its sidecar on GPU 0) on Spot (spot_posture:make_spotp) on the Isaac Lab terrain field.

COPY of scripts/terrain/run_terrain.py (field build, lighting, robot, chase-video and method arms unchanged, lines up to
the FEEL section), with a new single-trial driver at the end:
  * ONE tile x ONE trial per process (fresh process per trial), start = run_terrain's trial-k start
    (tile-local x = -1.6, y_off / yaw0 from default_rng(1000*seed + k); trial 0 = lane centre, yaw 0).
  * goal = lane centre, 0.5 m before the far edge (tile-local x = L - 0.5), relative goal only, no map.
  * outcomes as episode_runner: success (base within 0.5 m of the goal, strict), fall, stuck (no 0.2 m progress toward
    the goal for 15 s), timeout (T_MAX = 3 * straight / v_nom + 40 s).
  * per tick log of the robot pose; per planning step the pose the plan was made at (to place VAMOS's value map, which
    the released sidecar already dumps to <dbg>/predNNN_maps.npz, in the world frame).  Planner untouched.
  isaac_sim/isaaclab_python.sh closed_loop/baselines/vamos_terrain.py --tile ice --trial 0 --out <dir>
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse, json, math, os, re, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
XE = os.path.join(ROOT, "closed_loop")
for p in (os.path.join(ROOT, "scenes", "terrain"), XE, os.path.join(XE, "controllers"), THIS):
    sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="feel", choices=["feel"])
ap.add_argument("--tile", required=True, help="tile type (catalog name)")
ap.add_argument("--trial", type=int, required=True, help="trial index k (start pose of run_terrain's trial k)")
ap.add_argument("--goal_margin", type=float, default=0.5, help="goal = this far before the far edge")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--difficulty", type=float, default=0.6)
ap.add_argument("--types", default="", help="comma list (default: all catalog types)")
ap.add_argument("--robot", default="spotp", choices=["spotp", "g1c", "jackal", "g1u", "anyu", "g1l", "g1g", "g1s", "g1sx"],
                help="spotp = spot_posture:make_spotp, g1c = sidestep_g1:make_g1c (AMO), jackal = robots.make_robot. "
                     "Iris is not offered: it flies at a fixed 1 m altitude and never touches the terrain")
ap.add_argument("--trials", type=int, default=3)
ap.add_argument("--speed", type=float, default=0.6, help="fraction of the robot's max forward command")
ap.add_argument("--res", type=int, default=896, help="head-camera render resolution (square)")
ap.add_argument("--headlight", type=float, default=0.0, help="fixed headlight intensity (0 = scene_lighting random, seed 7)")
ap.add_argument("--sun", type=float, default=175.0, help="base DistantLight intensity (scene_lighting day x1.5-4)")
ap.add_argument("--sky_scale", type=float, default=0.2, help="x scene_lighting's day dome intensity (HDRI dome)")
ap.add_argument("--hl_scale", type=float, default=0.2, help="x scene_lighting's night headlight intensity")
ap.add_argument("--sky_hdr", default=xfeas_paths.ISAAC_ASSETS + "/Assets/Skies/Cloudy/lakeside_4k.hdr")
ap.add_argument("--only_head", type=int, default=0, help="render the head camera only (calibration)")
ap.add_argument("--out", default=os.path.join(ROOT, "data", "terrain"))
ap.add_argument("--save_usd", default="", help="export the built field (no robot) to this .usd")
ap.add_argument("--zero_mean", type=int, default=0, help="1 = zero-mean height-field relief (no net entry rise); "
                "default 0 = the original seed-0 field")
ap.add_argument("--tag", default="", help="suffix of the feel json (default: _fm0 / _fm1 from XFEAS_FOOT_MATERIAL)")
# ---- opt-in video (feel mode only; additive 2026-10-10). With --video 0 (default) none of this code runs.
ap.add_argument("--video", type=int, default=0, help="1 = record a side chase camera + head-camera inset per trial (feel mode)")
ap.add_argument("--vid-out", dest="vid_out", default="", help="frame directory root (default <out>/video)")
ap.add_argument("--vid-light", dest="vid_light", default="night_yellow", choices=["night_yellow", "night_white", "day", "none"],
                help="lighting for the recording (scene_lighting via set_light, seed 7); visual only, the robots are blind")
ap.add_argument("--vid-fill", dest="vid_fill", type=float, default=-1.0,
                help="recording-only fill light intensity (switched on only for the chase render); -1 = 4e5 at night, 0 by day")
ap.add_argument("--vid-every", dest="vid_every", type=int, default=2, help="record every N control ticks (50 Hz / 2 = 25 fps)")
ap.add_argument("--vid-trials", dest="vid_trials", default="", help="comma list of trial indices to record (default all); "
                "unrecorded trials still run, so the trial sequence (and the physics history) equals the feel run")
ap.add_argument("--only_tiles", default="", help="comma list: run only these tile types (field spec unchanged); default all")
# ---- opt-in closed-loop METHOD arm (feel mode only; additive 2026-10-11, GEN1_CLOSED_LOOP). With --method "" (default)
# none of this code runs. A method = an episode_runner plugin (module:Class, e.g. door_methods:StraightStock,
# oh_methods:StraightXFeasHold, gen_methods:StraightXFeasGen) that drives the robot across the tile toward a goal 1 m past
# X_OUT, seeing through its own head camera under --method-light (robot headlight, this field's lighting profile).
ap.add_argument("--method", default="", help="module:Class plugin that commands the walk phase (default: fixed V_CMD)")
ap.add_argument("--trace", type=float, default=0.0, help="(additive 2026-10-16, GEN2_G1_DEBUG) > 0: store a per-trial trace "
                "(t, x, y, yaw, vfwd, commanded twist) every TRACE s in the feel json (r['trace']); 0 = off (as before)")
ap.add_argument("--method-light", dest="method_light", default="night_white", choices=["night_white", "night_yellow", "day"])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": "RaytracedLighting"})
import carb
try:
    carb.settings.get_settings().set("/log/level", "error")
    carb.settings.get_settings().set("/rtx/raytracing/fractionalCutoutOpacity", True)   # oil film opacity
except Exception:
    pass
import omni.usd
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf
from isaacsim.core.utils.stage import create_new_stage
import catalog as CAT
import build as B

sys.path.insert(0, XE)
import foot_material as FMAT
FM_ON = FMAT.enabled()
# physics rate = the robot's own trained rate (RobotSpec.phys_hz): G1/AMO 500 Hz, Spot relic 200 Hz, Jackal 200 Hz
def _robot_factory(name):
    if name == "spotp":
        import spot_posture as F; return F.make_spotp, "spotp"
    if name in ("g1u", "anyu"):                    # URMA baseline (isaac_sim/controllers/urma/urma_robot.py), additive 2026-10-10
        sys.path.insert(0, os.path.join(ROOT, "isaac_sim", "controllers", "urma")); import urma_robot as F
        return (F.make_urma_g1 if name == "g1u" else F.make_urma_anymal), name
    if name == "g1c":
        import sidestep_g1 as F; return F.make_g1c, "g1a"
    if name == "g1l":                              # (additive 2026-10-17) Isaac Lab's released rough-terrain G1 policy
        sys.path.insert(0, os.path.join(ROOT, "isaac_sim", "controllers", "isaaclab")); import il_g1_robot as F
        return F.make_il_g1, "g1l"
    if name == "g1g":                              # (additive 2026-10-17) GR00T Decoupled WBC, released (scripts/gr00t)
        sys.path.insert(0, os.path.join(ROOT, "isaac_sim", "controllers", "gr00t")); import g1_gr00t as F
        return F.make_g1g, "g1g"
    if name == "g1s":                              # (additive 2026-10-17) GEAR-SONIC planner + tracker, released (scripts/sonic)
        sys.path.insert(0, os.path.join(ROOT, "isaac_sim", "controllers", "sonic")); import g1_sonic as F
        return F.make_g1s, "g1s"
    if name == "g1sx":                             # (additive 2026-10-17) SONIC + zero-shot XFeas adapter (isaac_sim/controllers/sonic/sonic_xfeas.py)
        sys.path.insert(0, os.path.join(ROOT, "isaac_sim", "controllers", "sonic")); import sonic_xfeas as F
        return F.make_g1s_xf, "g1s"
    import robots as RB_
    fac = lambda kind, path, pos, yaw=0.0: RB_.make_robot(kind, path, pos, yaw=yaw)
    fac.SPEC = RB_.SPECS[name]
    return fac, name
if a.mode == "feel":
    FAC, KIND = _robot_factory(a.robot); HZ = int(FAC.SPEC.phys_hz)
else:
    FAC, KIND, HZ = None, None, 200
create_new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
types = a.types.split(",") if a.types else None
spec = CAT.field_spec(a.seed, types, a.difficulty)
if a.zero_mean:
    for _t in spec["tiles"]:
        _t["geometry"]["zero_mean"] = True
json.dump(spec, open(os.path.join(a.out, f"field_seed{a.seed}.json"), "w"), indent=1)
t0 = time.time()
TILES, MATS = B.build_field(stage, spec)
print(f"[terrain] built {len(TILES)} tiles in {time.time() - t0:.1f} s:",
      [(t['id'], t['n_tris'], t['z_max']) for t in TILES], flush=True)
if a.save_usd:
    stage.GetRootLayer().Export(a.save_usd); print("[terrain] exported", a.save_usd, flush=True)
# a sun for "day" (scene_lighting scales every authored DistantLight as the sun; 0 at night)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(float(a.sun)); sun.CreateAngleAttr(0.8)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 0.0, 30.0))
for _ in range(30):
    app.update()

from omni.isaac.core.world import World
world = World(physics_dt=1.0 / HZ, rendering_dt=1 / 50.0, stage_units_in_meters=1.0)
stage = world.stage
import scene_lighting
LIGHT = scene_lighting.Lighting(stage)
if a.sky_hdr:      # day sky = a real HDRI (lit + reflected), same dome prim scene_lighting drives
    LIGHT.sky.CreateTextureFileAttr().Set(Sdf.AssetPath(a.sky_hdr))
    LIGHT.sky.CreateTextureFormatAttr().Set("latlong")
    LIGHT.sky.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def set_light(mode, rng):
    """scene_lighting modes + this field's lighting profile (sun / sky / headlight scales, see TERRAIN_ISAACLAB.md)"""
    info = LIGHT.set(mode, rng)
    if mode == "day":
        LIGHT.sky.GetIntensityAttr().Set(info["sky"] * a.sky_scale); info["sky"] = round(info["sky"] * a.sky_scale)
        info["sun_I"] = round(a.sun * info["sun"])
    elif mode.startswith("night"):
        hi = (a.headlight if a.headlight > 0 else info["headlight_I"]) * a.hl_scale
        LIGHT.hl.GetIntensityAttr().Set(hi); info["headlight_I"] = round(hi)
    info["profile"] = dict(sun=a.sun, sky_scale=a.sky_scale, hl_scale=a.hl_scale)
    return info

if a.mode == "build":
    world.reset()
    for _ in range(20):
        world.step(render=False)
    print("[terrain] DONE build", flush=True); app.close(); sys.exit(0)

# ================================================================== RENDER (scene only: NO robot in the stage)
SPOT_CAM = (0.52, -0.02, -0.45); BASE_H = 0.52        # Spot head camera (spot_posture spec.cam) + nominal base height


def tile_height(t, x, y):
    return t["height_at"](x, y)


if a.mode == "render":
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep
    from PIL import Image
    world.reset()
    for _ in range(int(1.0 * HZ)):          # let loose pebbles settle
        world.step(render=False)
    RD = os.path.join(a.out, "renders"); os.makedirs(RD, exist_ok=True)
    fwd, up_, pitch_c = SPOT_CAM
    cams = {}
    for nm, (fl, res) in dict(head=(12.0, (a.res, a.res)), over=(16.0, (1280, 800))).items():
        c = UsdGeom.Camera.Define(stage, f"/World/Cam_{nm}")
        c.CreateFocalLengthAttr(fl); c.CreateHorizontalApertureAttr(20.955)
        c.CreateVerticalApertureAttr(20.955 * res[1] / res[0])
        c.CreateClippingRangeAttr(Gf.Vec2f(0.03, 400.0))
        xf = UsdGeom.Xformable(c.GetPrim()).MakeMatrixXform()
        rp = rep.create.render_product(f"/World/Cam_{nm}", res)
        an = rep.AnnotatorRegistry.get_annotator("rgb"); an.attach([rp])
        cams[nm] = (xf, an)

    def look(xf, eye, tgt):
        xf.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse())

    def grab(nm):
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=12, delta_time=0.0, pause_timeline=False)
        img = np.asarray(cams[nm][1].get_data())
        return img[..., :3].astype(np.uint8) if img.ndim == 3 else None

    for _ in range(40):
        app.update()
    stats = []
    for t in TILES:
        ox, oy, _ = t["origin"]
        x0 = ox + 2.4
        gx, gy = np.meshgrid(x0 + np.array([-0.45, 0.0, 0.45]), oy + np.array([-0.2, 0.0, 0.2]))
        fz = float(np.max(tile_height(t, gx.ravel(), gy.ravel())))
        bs = dict(pos=np.array([x0, oy, fz + BASE_H]), yaw=0.0)          # a virtual Spot base: camera pose only
        eye_h = bs["pos"] + np.array([fwd, 0.0, up_])
        look(cams["head"][0], eye_h, eye_h + np.array([math.cos(pitch_c), 0.0, math.sin(pitch_c)]))
        OV_EYE = np.array([ox - 0.6, oy - 1.2, 2.4])                    # standing at the tile's edge, looking down its length
        look(cams["over"][0], OV_EYE, np.array([ox + 6.0, oy + 0.5, float(tile_height(t, ox + 6.0, oy + 0.5))]))
        for mode in ("day", "night_white", "night_yellow"):
            info = set_light(mode, np.random.default_rng(7))
            LIGHT.follow(bs, SPOT_CAM)
            shots = [("headcam", "head")] + ([("overview", "over")] if mode == "day" and not a.only_head else [])
            for tag, nm in shots:
                img = grab(nm)
                if img is None:
                    continue
                fn = f"{t['type']}_{tag}_{mode}.png"
                Image.fromarray(img).save(os.path.join(RD, fn))
                lum = img.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])
                stats.append(dict(file=fn, tile=t["type"], bucket=t["bucket"], shot=tag, mode=mode, light=info,
                                  eye=[round(float(v), 3) for v in (eye_h if tag == "headcam" else OV_EYE)],
                                  mean=round(float(lum.mean()), 1), p95=round(float(np.percentile(lum, 95)), 1),
                                  frac_sat=round(float((img.max(-1) >= 250).mean()), 4),
                                  frac_black=round(float((lum < 8).mean()), 4)))
                print(f"[terrain] {fn} mean {lum.mean():.1f} sat {stats[-1]['frac_sat']:.3f} black {stats[-1]['frac_black']:.3f}", flush=True)
    json.dump(dict(spec_seed=a.seed, cam=SPOT_CAM, base_h=BASE_H, stats=stats),
              open(os.path.join(a.out, "render_stats.json"), "w"), indent=1)
    print("[terrain] DONE render", flush=True); app.close(); sys.exit(0)

# ------------------------------------------------------------------ robot (feel mode)
T0 = TILES[0]
SPEC = FAC.SPEC
rob = FAC(KIND, f"/World/{KIND}", [-1.5, T0["origin"][1], SPEC.spawn_dz], yaw=0.0)
try:                                          # v4 (2026-10-19): headlight on the shared camera mount for EVERY arm
    import sys as _s4, os as _o4                  # (stock arms too); a method with a decision camera re-sets it
    _s4.path.insert(0, _o4.path.join(ROOT, "scripts", "common"))
    import camera_mount as _CMT, scene_lighting as _SL4
    if not _CMT.is_legacy(KIND):
        _SL4.HEAD_MOUNT_FN = _CMT.MountedCamera(stage, rob, KIND, root=f"/World/{KIND}", spec=SPEC).light_matrix
except Exception as _e4:
    print("[camera_mount] headlight mount not set:", repr(_e4), flush=True)
# what the feet are made of (decides how the tile's combine mode resolves)
robot_mats = []
for p in Usd.PrimRange(stage.GetPrimAtPath(f"/World/{KIND}")):
    if p.HasAPI(UsdPhysics.MaterialAPI):
        m = UsdPhysics.MaterialAPI(p); x = PhysxSchema.PhysxMaterialAPI(p)
        robot_mats.append(dict(path=str(p.GetPath()), mu_s=m.GetStaticFrictionAttr().Get(), mu_d=m.GetDynamicFrictionAttr().Get(),
                               combine=x.GetFrictionCombineModeAttr().Get() if x else None))
foot_bind = {}
for p in Usd.PrimRange(stage.GetPrimAtPath(f"/World/{KIND}")):
    if p.HasAPI(UsdPhysics.RigidBodyAPI) and re.search(r"(_foot$|ankle_roll_link$|wheel)", p.GetName()):
        pm, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")
        foot_bind[str(p.GetPath())] = str(pm.GetPath()) if pm else None
print("[terrain] XFEAS_FOOT_MATERIAL =", int(FM_ON), "\n[terrain] robot physics materials:", robot_mats, "\n[terrain] foot physics bindings:", foot_bind, flush=True)

world.reset()
world.add_physics_callback("drive", lambda dt: rob.step(dt))
for _ in range(int(0.5 * HZ)):
    world.step(render=False)
av = rob._art._articulation_view
pv = av._physics_view
BODY = list(av.body_names)
WHEELED = SPEC.family == "wheeled"
FEET = [i for i, n in enumerate(BODY) if re.search(r"(_foot$|ankle_roll_link$|_SHANK$)" if not WHEELED else r"wheel", n)]
if WHEELED:
    DOF = list(rob.robot.dof_names); WJ = [i for i, n in enumerate(DOF) if "wheel" in n.lower()]
    W_R = float(SPEC.extra["wheel_radius"])
print("[terrain] bodies:", BODY, "\n[terrain] feet:", [BODY[i] for i in FEET], flush=True)
LIM = rob.limits()
V_CMD = a.speed * LIM["vx_fwd"]

# ================================================================== opt-in VIDEO (--video 1; nothing below runs otherwise)
VID = None
if a.mode == "feel" and a.video:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep
    from omni.physx import get_physx_scene_query_interface
    from PIL import Image

    class _Video:
        """Side chase camera (robot's right, slightly behind, ~1 m above the base, looking at the feet) + head-camera
        inset (SPEC.cam mount, 12 mm, fill OFF). At night a fill light next to the chase camera is switched ON only for
        the chase render and OFF again before the head-camera render. Rendering uses orchestrator.step(delta_time=0):
        physics does not advance, so trial physics is the same as with --video 0."""
        RES, PIP = (1280, 720), 256
        BACK, SIDE, UP, CLEAR = 0.9, 2.6, 0.95, 1.0       # metres; CLEAR = min eye height above the top surface

        def __init__(self):
            self.root = a.vid_out or os.path.join(a.out, "video"); os.makedirs(self.root, exist_ok=True)
            self.pxq = get_physx_scene_query_interface(); self.robot_root = f"/World/{KIND}"
            self.light = None
            if a.vid_light != "none":
                self.light = set_light(a.vid_light, np.random.default_rng(7))
            night = a.vid_light.startswith("night")
            self.fill_i = (4.0e5 if night else 0.0) if a.vid_fill < 0 else float(a.vid_fill)
            self.cams = {}
            for nm, (fl, res) in dict(chase=(14.0, self.RES), head=(12.0, (self.PIP, self.PIP))).items():
                c = UsdGeom.Camera.Define(stage, f"/World/VidCam_{nm}")
                c.CreateFocalLengthAttr(fl); c.CreateHorizontalApertureAttr(20.955)
                c.CreateVerticalApertureAttr(20.955 * res[1] / res[0]); c.CreateClippingRangeAttr(Gf.Vec2f(0.03, 200.0))
                xf = UsdGeom.Xformable(c.GetPrim()).MakeMatrixXform()
                rp = rep.create.render_product(f"/World/VidCam_{nm}", res)
                an = rep.AnnotatorRegistry.get_annotator("rgb"); an.attach([rp])
                self.cams[nm] = (xf, an)
            f = UsdLux.SphereLight.Define(stage, "/World/VidFill")
            f.CreateRadiusAttr(0.2); f.CreateIntensityAttr(0.0)
            f.CreateEnableColorTemperatureAttr(True); f.CreateColorTemperatureAttr(4800.0)
            f.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
            self.fill, self.fill_xf = f, UsdGeom.Xformable(f.GetPrim()).MakeMatrixXform()
            self.verified = None; self.adjust = []
            print(f"[video] on: light {a.vid_light} {self.light} fill {self.fill_i} -> {self.root}", flush=True)

        def _look(self, nm, eye, tgt):
            self.cams[nm][0].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)),
                                                         Gf.Vec3d(0, 0, 1)).GetInverse())

        def _render(self, nm, sub=2):
            rep.orchestrator.step(rt_subframes=sub, delta_time=0.0, pause_timeline=False)
            im = np.asarray(self.cams[nm][1].get_data())
            return im[..., :3].astype(np.uint8).copy() if im.ndim == 3 and im.size else None

        def _ray(self, o, d, n):
            h = self.pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, d)), float(n))
            if h and h.get("hit") and not str(h.get("collision", "")).startswith(self.robot_root):
                return float(h["distance"]), str(h.get("collision", ""))
            return None

        def _top(self, x, y):
            """top surface z at (x, y): downward PhysX ray (tile mesh, rocks, pebbles, ground) and the tile height field"""
            h = self._ray((x, y, 20.0), (0, 0, -1), 40.0)
            z = 20.0 - h[0] if h else 0.0
            try:
                z = max(z, float(tile_height(self.tile, x, y)))
            except Exception:
                pass
            return z

        def begin(self, t, k):
            self.tile, self.k = t, k; self.p = None; self.n = 0; self.tick_i = 0; self.meta = []
            self.active = not a.vid_trials or k in [int(v) for v in a.vid_trials.split(",")]
            if not self.active:
                return
            self.dir = os.path.join(self.root, "frames", f"{KIND}_{t['type']}_t{k}"); os.makedirs(self.dir, exist_ok=True)
            for fn in os.listdir(self.dir):
                if fn.endswith((".jpg", ".png")):
                    os.remove(os.path.join(self.dir, fn))

        def pose(self, bs):
            p = np.asarray(bs["pos"], float)
            alpha = 1.0 - math.exp(-(a.vid_every / 50.0) / 0.5)
            self.p = p.copy() if self.p is None else self.p + alpha * (p - self.p)
            q = self.p
            ex, ey = q[0] - self.BACK, q[1] - self.SIDE
            ez = max(q[2] + self.UP, self._top(ex, ey) + self.CLEAR)
            eye = np.array([ex, ey, ez]); tgt = np.array([q[0] + 0.15, q[1], q[2] - 0.3])
            o = p + np.array([0.0, 0.0, 0.3])                 # line of sight robot -> eye must be clear (no camera in a rock)
            for _ in range(4):
                dv = eye - o; n = float(np.linalg.norm(dv))
                if not self._ray(o, dv / n, n + 0.3):
                    break
                eye = eye + np.array([0.0, 0.0, 0.5]); self.adjust.append(1)
            return eye, tgt

        def head(self, bs):
            fwd, up_, pitch = SPEC.cam
            c, s_ = math.cos(bs["yaw"]), math.sin(bs["yaw"])
            he = np.asarray(bs["pos"], float) + np.array([fwd * c, fwd * s_, up_])
            return he, he + np.array([c * math.cos(pitch), s_ * math.cos(pitch), math.sin(pitch)])

        def _chase_pass(self, eye, tgt):
            d = tgt - eye; d[2] = 0.0; d /= max(np.linalg.norm(d), 1e-6)
            self.fill_xf.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, eye + 0.3 * d + np.array([0, 0, 0.2])))))
            self._look("chase", eye, tgt)
            if self.fill_i > 0:
                self.fill.GetIntensityAttr().Set(self.fill_i)
            try:
                return self._render("chase")
            finally:
                self.fill.GetIntensityAttr().Set(0.0)

        def verify_fill(self, bs, eye, tgt):
            """head image after a fill-lit chase pass must equal one with no chase pass at all (up to render noise)"""
            he, ht = self.head(bs); self._look("head", he, ht)
            A1 = self._render("head", 4).astype(np.float32); A2 = self._render("head", 4).astype(np.float32)
            self._chase_pass(eye, tgt); self._look("head", he, ht)
            B = self._render("head", 4).astype(np.float32)
            self.fill.GetIntensityAttr().Set(max(self.fill_i, 4.0e5))
            self.fill_xf.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, he + np.array([0.6, 0.0, 0.3])))))
            C = self._render("head", 4).astype(np.float32); self.fill.GetIntensityAttr().Set(0.0)
            self.verified = dict(noise_A1_A2=round(float(np.abs(A1 - A2).mean()), 3), after_fill_B_A2=round(float(np.abs(B - A2).mean()), 3),
                                 leak_reference_C_A2=round(float(np.abs(C - A2).mean()), 3), head_mean=round(float(A2.mean()), 1),
                                 fill_i=self.fill_i)
            print(f"[video] fill check (mean abs diff, 0-255): {self.verified}", flush=True)

        def tick(self, bs, t_sim, phase):
            self.tick_i += 1
            if not self.active or bs is None or (self.tick_i - 1) % max(1, a.vid_every):
                return
            if self.light is not None:
                LIGHT.follow(bs, SPEC.cam)
            eye, tgt = self.pose(bs)
            if self.verified is None:
                self.verify_fill(bs, eye, tgt)
            if self.n % 2 == 0:                               # head inset at half rate, fill OFF
                he, ht = self.head(bs); self._look("head", he, ht)
                pim = self._render("head")
                if pim is not None:
                    self.pip = pim
            im = self._chase_pass(eye, tgt)
            if im is None:
                return
            Image.fromarray(im).save(os.path.join(self.dir, f"{self.n:06d}.jpg"), quality=93)
            if getattr(self, "pip", None) is not None:
                Image.fromarray(self.pip).save(os.path.join(self.dir, f"pip_{self.n:06d}.jpg"), quality=90)
            v_w = np.asarray(rob.robot.get_linear_velocity(), float)
            self.meta.append(dict(i=self.n, t=round(t_sim, 3), x=round(float(bs["pos"][0] - self.tile["origin"][0]), 3),
                                  vfwd=round(float(v_w[0] * math.cos(bs["yaw"]) + v_w[1] * math.sin(bs["yaw"])), 3),
                                  tilt=round(float(bs["tilt"]), 3), phase=phase,
                                  eye=[round(float(v), 2) for v in eye]))
            self.n += 1

        def end(self, r):
            if not self.active:
                return
            json.dump(dict(robot=KIND, tile=self.tile["type"], trial=self.k, seed=a.seed, speed=a.speed, v_cmd=V_CMD,
                           light=a.vid_light, light_info=self.light, fill_i=self.fill_i, fill_check=self.verified,
                           fps=50.0 / max(1, a.vid_every), result=r, eye_raised=len(self.adjust), frames=self.meta),
                      open(os.path.join(self.dir, "meta.json"), "w"), indent=1, default=float)
            print(f"[video] {self.dir}: {self.n} frames, eye raised {len(self.adjust)}x", flush=True)
            self.adjust = []

    VID = _Video()
    for _ in range(3):        # prime the render products; delta_time 0 = no physics step (app.update would step it)
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)


METHOD = None
if a.mode == "feel" and a.method:
    import importlib
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep
    from omni.physx import get_physx_scene_query_interface
    for _p in ("overhang", "climb", "gen_train"):
        sys.path.insert(0, os.path.join(ROOT, "scripts", _p))
    import local_nav as _LN
    _mod, _cls = a.method.split(":")
    METHOD = getattr(importlib.import_module(_mod), _cls)()
    _DS = _LN.DepthSensor
    _LN.DepthSensor = lambda *a_, **k_: None            # straight-drive arms do not use the planner's depth sensor
    try:
        METHOD.setup(dict(stage=stage, rep=rep, world=world, app=app, rob=rob, spec=SPEC, kind=KIND, hz=HZ,
                          pxq=get_physx_scene_query_interface(), carb=carb, robot_root=f"/World/{KIND}"))
    finally:
        _LN.DepthSensor = _DS
    METHOD_LIGHT = set_light(a.method_light, np.random.default_rng(7))
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
    print(f"[terrain] METHOD {a.method} ({getattr(METHOD, 'label', '')}); lighting {a.method_light} {METHOD_LIGHT}", flush=True)


def feet_state():
    T = np.asarray(pv.get_link_transforms()).reshape(-1, len(BODY), 7)[0]
    Vv = np.asarray(pv.get_link_velocities()).reshape(-1, len(BODY), 6)[0]
    return T[FEET, :3].copy(), Vv[FEET, :3].copy()


def place(t, x_local, y_off=0.0, yaw=0.0, settle=2.0):
    ox, oy, _ = t["origin"]
    gx, gy = np.meshgrid(ox + x_local + np.array([-0.45, 0.0, 0.45]), oy + y_off + np.array([-0.2, 0.0, 0.2]))
    fz = float(np.max(tile_height(t, gx.ravel(), gy.ravel()))) if x_local > 0 else 0.0
    rob.teleport(ox + x_local, oy + y_off, yaw, fz)
    rob.command(np.zeros(4))
    for _ in range(int(settle * HZ)):
        world.step(render=False)
    return fz


# ================================================================== SINGLE-TRIAL DRIVER (baselines_r1, new)
assert METHOD is not None and VID is not None, "run with --method vamos_r1:VAMOSR1 --video 1"
TT = [T_ for T_ in TILES if T_["type"] == a.tile]
assert TT, f"tile {a.tile} not in field"
t = TT[0]
ox, oy, _ = t["origin"]; L, Wd = t["size"][0], t["size"][1]
X0_START = -1.6
k = a.trial
rng = np.random.default_rng(1000 * a.seed + k)               # == run_terrain feel trial k
y_off = 0.0 if k == 0 else float(rng.uniform(-0.15, 0.15))
yaw0 = 0.0 if k == 0 else float(rng.uniform(-0.05, 0.05))
fz0 = place(t, X0_START, y_off, yaw0, settle=1.5)
bs = rob.base_state()
start = np.array([ox + X0_START, oy + y_off])
valid_start = bs is not None and not rob.fallen(bs) and float(np.hypot(*(bs["pos"][:2] - start))) < 0.3
goal = np.array([ox + L - a.goal_margin, oy])
straight = float(np.hypot(*(goal - start)))
v_nom = max(a.speed * SPEC.max_cmd[0], SPEC.v_min)
T_MAX = 3.0 * straight / v_nom + 40.0
res = dict(method="vamos", robot=KIND, set=a.tile, tile=a.tile, bucket=t["bucket"], trial=k, seed=a.seed,
           difficulty=a.difficulty, foot_material=int(FM_ON), lighting=METHOD_LIGHT, start=start.round(3).tolist(),
           yaw0=round(yaw0, 4), goal=goal.round(3).tolist(), straight_m=round(straight, 2), t_max=round(T_MAX, 1),
           speed_frac=a.speed, map_given=False, tile_origin=[ox, oy], tile_size=[L, Wd], valid_start=bool(valid_start))
print(f"[r1] tile {a.tile} trial {k}: start {start} goal {goal} T_MAX {T_MAX:.1f} valid_start {valid_start}", flush=True)
outcome = None if valid_start else "invalid_start"
METHOD.reset(dict(speed_frac=a.speed, floor_z0=fz0, scene=f"terrain_{t['type']}", trial=k))
VID.begin(t, k)
ctl = max(1, HZ // 50); dt_c = ctl / HZ
t_sim, best, t_prog, traj, wall0 = 0.0, 1e9, 0.0, [], time.time()
tw = np.zeros(4); left_tile = False; max_tilt = 0.0
_FT_OVR = float(os.environ.get("TERRAIN_FALL_TILT", "0") or 0)
while outcome is None:
    for _ in range(ctl):
        world.step(render=False)
    t_sim += dt_c
    bs = rob.base_state()
    # (additive 2026-10-28, env-gated; unset = unchanged) TERRAIN_FALL_TILT: the course runners' fall tilt (SC2A fall_tilt=1.0)
    # instead of the robot spec's, so a commanded crouch-walk is scored as on the course; min_z rule unchanged
    if bs is None or (rob.fallen(bs) if not _FT_OVR else (bs["tilt"] > _FT_OVR or bs["pos"][2] - rob._floor_z < SPEC.min_z)):
        outcome = "fall"; break
    n0 = VID.n
    VID.tick(bs, t_sim, "walk")
    LIGHT.follow(bs, SPEC.cam)
    x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
    d = goal - np.array([x, y]); c_, s_ = math.cos(yaw), math.sin(yaw)
    r_ = METHOD.act(dict(t=t_sim, dt=dt_c, bs=bs, rob=rob, goal_rel=np.array([c_ * d[0] + s_ * d[1], -s_ * d[0] + c_ * d[1]]),
                         goal_xy=goal.copy(), goal_dist=float(np.hypot(*d))))
    tw = np.asarray(r_["twist"], float); rob.command(tw)
    if VID.n > n0 and VID.meta:
        VID.meta[-1].update(wx=round(x, 3), wy=round(y, 3), yaw=round(yaw, 4), state=getattr(METHOD, "state", ""),
                            n_plan=len(getattr(METHOD, "plans", [])), cmd=[round(float(v), 3) for v in tw])
    xl, yl = x - ox, y - oy
    if not (0.0 <= xl <= L and abs(yl) <= Wd / 2) and xl > 0.0:
        left_tile = True
    max_tilt = max(max_tilt, math.degrees(float(bs["tilt"])))
    if not traj or t_sim - traj[-1][0] >= 0.1 - 1e-9:
        traj.append([round(t_sim, 2), round(x, 3), round(y, 3), round(float(bs["pos"][2]), 3), round(yaw, 3),
                     round(float(bs["v_b"][0]), 3), round(float(tw[0]), 3), round(float(tw[1]), 3), round(float(tw[3]), 3),
                     round(math.degrees(float(bs["tilt"])), 1)])
    dg = float(np.hypot(*d))
    if dg < best - 0.2:
        best, t_prog = dg, t_sim
    if dg < 0.5:
        outcome = "success"; break
    if t_sim - t_prog > 15.0:
        outcome = "stuck"; break
    if t_sim > T_MAX:
        outcome = "timeout"; break
rob.command(np.zeros(4))
bs = rob.base_state()
res.update(outcome=outcome, success=outcome == "success", t_end=round(t_sim, 2),
           dist_to_goal=round(float(np.hypot(*(goal - bs["pos"][:2]))), 2) if bs is not None else None,
           min_dist_to_goal=round(best, 2), x_reached_local=round(max([p[1] for p in traj] or [ox]) - ox, 2),
           left_tile=bool(left_tile), max_tilt_deg=round(max_tilt, 1),
           mean_vx_cmd=round(float(np.mean([p[6] for p in traj])) if traj else 0.0, 3), wall_s=round(time.time() - wall0, 1),
           traj_cols=["t", "x", "y", "z", "yaw", "vx_b", "cmd_vx", "cmd_vy", "cmd_wz", "tilt_deg"], traj=traj,
           decisions=list(getattr(METHOD, "log", []) or []), plans=list(getattr(METHOD, "plans", []) or []),
           map_dir=getattr(METHOD, "dbg", None), frames_dir=VID.dir, n_frames=VID.n, fill_check=VID.verified)
VID.end(dict(outcome=outcome, t_end=round(t_sim, 2)))
json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1, default=float)
print(f"[r1] RESULT {a.tile} t{k}: {outcome} t={t_sim:.1f} d={res['dist_to_goal']} plans={len(res['plans'])} "
      f"left_tile={left_tile} frames={VID.n}", flush=True)
print("[r1] DONE", flush=True)
try:
    METHOD.srv.stdin.close(); METHOD.srv.wait(timeout=20)
except Exception:
    pass
app.close()
