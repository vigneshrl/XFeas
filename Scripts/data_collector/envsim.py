"""
gen_v1 harness for the non-school TRAINING environments (import AFTER SimulationApp exists). GPU 0 only.

  EnvSim(app, factory, kind, env_id, tiles=None)
    env_id in office | warehouse_shelves | full_warehouse | warehouse | rivermark | simple_room | dt_warehouse | terrain
  * loads the Isaac environment from the _mirror path into a new stage (/World/Env/Asset reference, unit / up-axis xform
    as scripts/envsurvey/survey_env.py; env PhysicsScenes disabled; Rivermark PointInstancers disabled), or builds the
    Isaac Lab terrain tiles of the job (scenes/terrain/build.py, needs the isaaclab launcher)
  * hazards are built ONLY from the environment's OWN prop assets (CATALOG: the env's own prop USD files, referenced with
    their own meshes and materials; instancing broken; exact triangle colliders; any rigid body made static). Raised
    slabs rest on stacked support columns (the env's own boxes / pallets / stands) outside the lane (|l| >= 0.65 m);
    the SUPPORTED check (rays straddling each column top) and real_assets.audit_no_primitives run on every build
  * same automatic checks as gensim: lane clutter / headroom rays, intended geometry by rays (overhang underside,
    step edge, gap width, curb rise), visible-pixel check at the decision snapshot (run_gen_v1), exposure check
  * inherits decision / rollout / save-restore / labels from gensim.GenSim (via gensim_v1.ExpoMixin)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json, math, os, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
sys.path.insert(0, THIS)
import gensim as GS
import gensim_v1 as GV
import envfam as EF

MIR = xfeas_paths.ISAAC_ASSETS + "/Assets/Isaac/5.1/Isaac/Environments"
ENV_USD = dict(office=f"{MIR}/Office/office.usd", warehouse_shelves=f"{MIR}/Simple_Warehouse/warehouse_multiple_shelves.usd",
               full_warehouse=f"{MIR}/Simple_Warehouse/full_warehouse.usd", warehouse=f"{MIR}/Simple_Warehouse/warehouse.usd",
               rivermark=f"{MIR}/Outdoor/Rivermark/rivermark.usd", simple_room=f"{MIR}/Simple_Room/simple_room.usd",
               dt_warehouse=f"{MIR}/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd")
HELD_OUT = ("gq", "interiors", "hospital")
OP, WP = f"{MIR}/Office/Props", f"{MIR}/Simple_Warehouse/Props"
CATALOG = dict(
    office=dict(door=f"{OP}/SM_DoorB_Door.usd", box=f"{OP}/SM_Box.usd", boxa=f"{OP}/SM_BoxA.usd", boxbig=f"{OP}/SM_BoxBig.usd",
                boxport=f"{OP}/SM_BoxPortableA.usd", binder=f"{OP}/SM_RingBinder.usd", papertray=f"{OP}/SM_PaperTray.usd",
                book=f"{OP}/SM_Book_08.usd", bottle=f"{OP}/SM_BottleB.usd", chair=f"{OP}/SM_Chair.usd",
                cupboard=f"{OP}/SM_Cupboard.usd", bookcase=f"{OP}/SM_BookcaseA.usd", rack=f"{OP}/SM_Rack.usd",
                filecab=f"{OP}/SM_FileCabinet_02.usd", standa=f"{OP}/SM_ReceptionStandA.usd", pc=f"{OP}/SM_PC.usd"),
    warehouse=dict(pallet=f"{WP}/SM_PaletteA_01.usd", cardbox_a=f"{WP}/SM_CardBoxA_01.usd", cardbox_b=f"{WP}/SM_CardBoxB_01.usd",
                   cardbox_c=f"{WP}/SM_CardBoxC_01.usd", cardbox_c2=f"{WP}/SM_CardBoxC_02.usd", cardbox_d=f"{WP}/SM_CardBoxD_03.usd",
                   cardbox_d2=f"{WP}/SM_CardBoxD_02.usd", rackshelf=f"{WP}/SM_RackShelf_01_226.usd", beam=f"{WP}/SM_BracketBeam_3m.usd",
                   rackshield=f"{WP}/SM_Rackshield_02.usd", cardbox_d1=f"{WP}/SM_CardBoxD_01_327.usd",
                   crate_c=f"{WP}/SM_CratePlastic_C_03.usd", crate_e=f"{WP}/SM_CratePlastic_E_03.usd",
                   crate_nb=f"{WP}/SM_CratePlasticNote_B_03.usd", bottle=f"{WP}/SM_BottlePlasticB_02.usd", bracket=f"{WP}/SM_BracketSlot_229.usd"))
# support-column pieces per pack (stacked bottom-up to reach the slab's underside)
SUPPORT = dict(office=["boxbig", "boxport", "box", "standa", "pc"], warehouse=["pallet", "cardbox_a", "cardbox_b", "cardbox_c", "cardbox_c2"])
L_IN = 0.60          # supports' inner face >= 0.60 m from the lane centre (Spot half-width 0.28 m, G1 ~0.23 m)
SCALE_LIM = (0.85, 1.15)
OBJ_ROOT, SUP_ROOT, CL_ROOT = "/World/Overhang/EnvObj", GS.SUP_ROOT + "/Env", "/World/Climb/Env"


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1.0]])


def _ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1.0]])


def _T(v):
    M = np.eye(4); M[3, :3] = v
    return M


def _S(s):
    M = np.eye(4); M[0, 0] = M[1, 1] = M[2, 2] = s
    return M


class EnvProps:
    """copies of the environment's own prop assets (reference to the asset file = own meshes + own materials)"""

    def __init__(self, stage, catalog, root):
        from pxr import Usd, UsdGeom, Gf
        self.stage, self.Usd, self.UsdGeom, self.Gf = stage, Usd, UsdGeom, Gf
        self.cat, self.root = catalog, root
        UsdGeom.Xform.Define(stage, root)
        self.pool, self.used, self.box = {}, {}, {}
        self.bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    def _make(self, kind):
        from pxr import UsdGeom, UsdPhysics, Usd, Gf
        lst = self.pool.setdefault(kind, [])
        path = f"{self.root}/{kind}_{len(lst):03d}"
        par = UsdGeom.Xform.Define(self.stage, path)
        ch = self.stage.DefinePrim(path + "/asset")
        ch.GetReferences().AddReference(self.cat[kind])
        for q in Usd.PrimRange(ch):
            if q.IsInstance():
                q.SetInstanceable(False)
        n_g = 0
        for q in Usd.PrimRange(ch):
            if q.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(q).CreateRigidBodyEnabledAttr().Set(False)
            if q.IsA(UsdGeom.Gprim):
                UsdPhysics.CollisionAPI.Apply(q); n_g += 1
                if q.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr().Set("none")
        xf = par.MakeMatrixXform(); xf.Set(Gf.Matrix4d(1.0))
        if kind not in self.box:
            self.bbc.Clear()
            r = self.bbc.ComputeWorldBound(par.GetPrim()).ComputeAlignedRange()
            lo, hi = np.array(r.GetMin(), float), np.array(r.GetMax(), float)
            if r.IsEmpty() or n_g == 0:
                raise GS.Reject(f"prop_empty {kind}")
            self.box[kind] = (lo, hi)
        c = dict(kind=kind, xf=xf, prim=par.GetPrim(), path=path)
        lst.append(c)
        return c

    def size(self, kind, orient="as_is"):
        if kind not in self.box:
            c = self._make(kind); self.pool[kind].pop(); self.pool[kind].append(c)
        lo, hi = self.box[kind]
        C = self._corners(lo, hi) @ self._R0(kind, orient)
        return np.ptp(C[:, :3], 0)

    @staticmethod
    def _corners(lo, hi):
        return np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    def _R0(self, kind, orient):
        if orient != "flat":
            return np.eye(4)
        lo, hi = self.box[kind]; t = int(np.argmin(hi - lo))
        return np.eye(4) if t == 2 else (_ry(math.pi / 2) if t == 0 else _rx(math.pi / 2))

    def get(self, kind):
        lst = self.pool.setdefault(kind, []); k = self.used.get(kind, 0)
        c = lst[k] if k < len(lst) else self._make(kind)
        self.used[kind] = k + 1
        return c

    def place(self, c, x, y, z_bottom, yaw=0.0, orient="as_is", scale=1.0):
        """bbox centre (xy) at (x, y), bottom at z_bottom; returns the world AABB (lo, hi) of the placed copy"""
        lo, hi = self.box[c["kind"]]
        ctr = 0.5 * (lo + hi)
        M = _T(-ctr) @ self._R0(c["kind"], orient) @ _S(scale) @ _rz(yaw)
        C = self._corners(lo, hi) @ M
        t = np.array([x, y, z_bottom - C[:, 2].min()])
        M = M @ _T(t)
        c["xf"].Set(self.Gf.Matrix4d(*[float(v) for v in M.reshape(-1)]))
        Cw = self._corners(lo, hi) @ M
        return Cw[:, :3].min(0), Cw[:, :3].max(0)

    def ctop(self, kind, scale=1.0):
        """collider top below the bbox top (m, measured by calibrate(); 0 if not calibrated)"""
        return float(getattr(self, "cal", {}).get(kind, {}).get("drop") or 0.0) * scale

    def calibrate(self, sim, kinds):
        """place one copy of every kind alone in empty space, restart physics, cast a 9x9 grid of down rays over its
        footprint: collider top vs bbox top ('drop') and the fraction of the footprint with a collider ('solid')"""
        self.cal = getattr(self, "cal", {})
        todo = [k for k in kinds if k not in self.cal]
        if not todo:
            return self.cal
        sim.world.stop(); self.park()
        pos = {}
        for i, k in enumerate(todo):
            c = self.get(k); x, y = 800.0 + 6.0 * i, 800.0
            lo, hi = self.place(c, x, y, -50.0)
            pos[k] = (lo, hi)
        sim._restart()
        for k, (lo, hi) in pos.items():
            zs = []; n = 0
            for x in np.linspace(lo[0] + 0.03, hi[0] - 0.03, 9):
                for y in np.linspace(lo[1] + 0.03, hi[1] - 0.03, 9):
                    n += 1
                    dist, who, _ = sim.ray((x, y, hi[2] + 1.0), (0, 0, -1), 3.0)
                    if dist is not None and who.startswith(self.root):
                        zs.append(hi[2] + 1.0 - dist)
            self.cal[k] = dict(drop=round(float(hi[2] - max(zs)), 4) if zs else None, solid=round(len(zs) / n, 3),
                               top_med=round(float(hi[2] - np.median(zs)), 4) if zs else None, h=round(float(hi[2] - lo[2]), 4))
        sim.world.stop(); self.park()
        return self.cal

    def park(self):
        for kind, lst in self.pool.items():
            for i, c in enumerate(lst):
                c["xf"].Set(self.Gf.Matrix4d().SetTranslate(self.Gf.Vec3d(float(i) * 5.0, -600.0 - 7.0 * len(kind), -120.0)))
        self.used = {}


class EnvSim(GV.ExpoMixin, GS.GenSim):
    def __init__(self, app, factory_spec, kind, env_id, seed=0, side_cam=False, tiles=None):
        assert env_id not in HELD_OUT, "held-out environment"
        import importlib
        import carb, carb.settings
        try:
            carb.settings.get_settings().set("/log/level", "error")
        except Exception:
            pass
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep
        from omni.isaac.core.world import World
        from omni.physx import get_physx_scene_query_interface
        from pxr import UsdGeom, Gf, Usd, UsdLux, Sdf, UsdPhysics
        import robots as RB
        self.env_id = env_id
        self.app, self.carb, self.rep, self.Gf, self.UsdGeom, self.RB = app, carb, rep, Gf, UsdGeom, RB
        self.kind = kind
        self.rng = np.random.default_rng(777000 + seed)
        rep.set_global_seed(int(self.rng.integers(1 << 30)))
        _fm, _ff = factory_spec.split(":")
        self.FMOD = importlib.import_module(_fm); self.FACTORY = getattr(self.FMOD, _ff)
        if not hasattr(self.FACTORY, "SPEC"):
            self.FACTORY.SPEC = RB.SPECS[kind]
        spec = self.FACTORY.SPEC; self.spec = spec; self.HZ = HZ = spec.phys_hz
        self.RANGES = spec.extra.get("setpoint_ranges", {})
        self.NOM = getattr(self.FMOD, "NOMINAL_H", None) or getattr(getattr(self.FMOD, "g1_amo", None), "NOMINAL_H", None)
        t0 = time.time()
        from isaacsim.core.utils.stage import create_new_stage
        create_new_stage()
        st0 = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(st0, 1.0); UsdGeom.SetStageUpAxis(st0, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(st0, "/World")
        self.tiles = None
        if env_id != "terrain":
            usd = ENV_USD[env_id]
            src = Usd.Stage.Open(usd, load=Usd.Stage.LoadNone)
            mpu, up = UsdGeom.GetStageMetersPerUnit(src), UsdGeom.GetStageUpAxis(src); del src
            e = UsdGeom.Xform.Define(st0, "/World/Env")
            UsdGeom.Xform.Define(st0, "/World/Env/Asset").GetPrim().GetReferences().AddReference(usd)
            xf = UsdGeom.Xformable(e.GetPrim())
            if up == "Y":
                xf.AddRotateXOp().Set(90.0)
            if abs(mpu - 1.0) > 1e-6:
                xf.AddScaleOp().Set(Gf.Vec3f(mpu, mpu, mpu))
        try:
            from isaacsim.core.utils.stage import is_stage_loading
        except Exception:
            is_stage_loading = lambda: False
        for i in range(3000):
            app.update()
            if i > 20 and not is_stage_loading():
                break
        print(f"[envsim] {env_id} loaded in {time.time() - t0:.0f} s", flush=True)
        self.world = World(physics_dt=1.0 / HZ, rendering_dt=1 / 50.0, stage_units_in_meters=1.0)
        stage = self.stage = self.world.stage
        if env_id != "terrain":
            for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Env")):
                if p.IsA(UsdPhysics.Scene):
                    p.SetActive(False)
            if env_id == "rivermark":
                paths = [p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Env")) if p.IsA(UsdGeom.PointInstancer) and not p.IsInstanceProxy()]
                for pth in paths:
                    pr = stage.GetPrimAtPath(pth)
                    if pr and pr.IsValid() and pr.IsActive():
                        pr.SetActive(False)
                print(f"[envsim] rivermark: {len(paths)} PointInstancers disabled", flush=True)
        else:
            sys.path.insert(0, os.path.join(ROOT, "scenes", "terrain"))
            import build as TB
            self.tiles, _ = TB.build_field(stage, dict(tiles=tiles or []))
            print(f"[envsim] terrain: built {len(self.tiles)} tiles", flush=True)
        for r_ in ("/World/Overhang", "/World/Climb", GS.SUP_ROOT):
            UsdGeom.Xform.Define(stage, r_)
        import scene_lighting
        self.LIGHT = scene_lighting.Lighting(stage)
        self.world.reset()
        self.pxq = get_physx_scene_query_interface()
        pk = EF.pack(env_id)
        self.P = EnvProps(stage, CATALOG[pk], OBJ_ROOT) if env_id in ("office", "warehouse_shelves", "full_warehouse", "warehouse") else None
        self.PS = EnvProps(stage, CATALOG[pk], SUP_ROOT) if self.P is not None else None
        self.PC = EnvProps(stage, CATALOG[pk], CL_ROOT) if self.P is not None else None
        self.pk = pk
        self.root = f"/World/{kind}"
        self.rob = self.FACTORY(kind, self.root, [0.0, 0.0, -20.0], yaw=0.0)
        cam = UsdGeom.Camera.Define(stage, "/World/HeadCam")
        cam.CreateFocalLengthAttr(12.0); cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955); cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
        self.cam_xf = UsdGeom.Xformable(cam.GetPrim()).MakeMatrixXform()
        rp = rep.create.render_product("/World/HeadCam", (224, 224))
        self.ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); self.ann_rgb.attach([rp])
        self.ann_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); self.ann_dep.attach([rp])
        self._mount_init(stage, rep, rp, kind, self.spec)          # v4 shared camera mount (camera_mount.py)
        self.side = None
        if side_cam:
            sc = UsdGeom.Camera.Define(stage, "/World/SideCam")
            sc.CreateFocalLengthAttr(9.0); sc.CreateHorizontalApertureAttr(20.955); sc.CreateVerticalApertureAttr(20.955 * 9 / 16)
            sc.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
            self.side_xf = UsdGeom.Xformable(sc.GetPrim()).MakeMatrixXform()
            rp2 = rep.create.render_product("/World/SideCam", (384, 216))
            self.side = rep.AnnotatorRegistry.get_annotator("rgb"); self.side.attach([rp2])
            self.AUD = UsdLux.SphereLight.Define(stage, "/World/AuditFill"); self.AUD.CreateRadiusAttr(0.3); self.AUD.CreateIntensityAttr(0.0)
            self.AUD.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
            self.AUD_XF = UsdGeom.Xformable(self.AUD.GetPrim()).MakeMatrixXform()
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        self.world.reset()
        self.world.add_physics_callback("drive", lambda dt: self.rob.step(dt))
        for _ in range(int(0.5 * HZ)):
            self.world.step(render=False)
        self.desc = self.rob.descriptor(stage)
        self.buf = RB.ProprioBuffer(seconds=2.0, hz=50, phys_hz=HZ)
        self.EVERY = max(1, HZ // 50)
        self.bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        self.ROBOT_PRIM = stage.GetPrimAtPath(self.root)
        self.v_nom = max(0.6 * spec.max_cmd[0], spec.v_min)
        legged_g1 = kind.startswith("g1")
        self.STALL_S = 8.0 if legged_g1 else 5.0
        self.V_FLOOR = 0.06 if legged_g1 else 0.25
        self.T_CAP = 70.0 if legged_g1 else 40.0
        self.BOX = self.full_box()
        self._lane_cache = {}
        self.stand_z = float(spec.spawn_dz) - 0.05
        if self.P is not None:
            pk = self.pk
            self.CAL = dict(sup=self.PS.calibrate(self, SUPPORT[pk]),
                            cl=self.PC.calibrate(self, sorted(set(list(EF.STEP[pk]) + EF.CLUT[pk] + EF.WALL[pk] + EF.CLIMB[pk]))))
            self.PC.cal.update({k: v for k, v in self.PS.cal.items() if k not in self.PC.cal})
            print(f"[envsim] prop calibration (collider top drop below bbox / solid footprint): {json.dumps(self.CAL)}", flush=True)
            if kind == "iris":
                # Pegasus' drone loses its state when prims are CREATED mid-run (every hazard situation ended in
                # 'approach_fell', nulls were fine): pre-create every copy a build can need, so builds only MOVE prims
                self.world.stop()
                for PP, kinds, nmax in ((self.PS, SUPPORT[pk], 10), (self.PC, sorted(set(list(EF.STEP[pk]) + EF.CLUT[pk] + EF.WALL[pk] + EF.CLIMB[pk])), 10),
                                        (self.P, list(EF.SLAB[pk]), 2)):
                    for k_ in kinds:
                        for _ in range(20 if k_ == "cardbox_a" else nmax):
                            PP.get(k_)
                    PP.park()
                self._restart()
                print("[envsim] iris: prop copies pre-created", flush=True)

    # ------------------------------------------------------------------ helpers
    def _park_all(self):
        for P in (self.P, self.PS, self.PC):
            if P is not None:
                P.park()

    def profile(self, P0, d, n, fz, s_a, s_b, lats=(-0.2, 0.0, 0.2), ds=0.05):
        """floor-relative height under the lane (down rays from fz + 2 m), every hit counted (ours or env)"""
        out = []
        for s in np.arange(s_a, s_b + 1e-6, ds):
            row = [round(float(s), 3)]
            for l in lats:
                q = P0 + d * s + n * l
                dist, who, _ = self.ray((q[0], q[1], fz + 2.0), (0, 0, -1), 3.0)
                row.append(None if dist is None or who.startswith(self.root) else round(2.0 - dist, 4))
            out.append(row)
        return out

    def _xy(self, P0, d, n, s, l):
        q = P0 + d * s + n * l
        return float(q[0]), float(q[1])

    def _stack_plan(self, T, rng, room):
        """support column reaching EXACTLY T: pieces (kind, scale) bottom-up; all pieces of a stack share one uniform
        scale in SCALE_LIM; the lateral footprint of every piece must fit in `room`"""
        cal = getattr(self.PS, "cal", {})
        kinds = [k for k in SUPPORT[self.pk] if min(self.PS.size(k)[:2]) <= room
                 and cal.get(k, {}).get("drop") is not None and cal[k]["solid"] >= 0.6 and cal[k]["top_med"] <= 0.02]
        if not kinds:
            return None
        H = {k: float(self.PS.size(k)[2]) - cal[k]["drop"] for k in kinds}
        best = None
        for _ in range(400):
            n = int(rng.integers(1, 6))
            ks = [str(rng.choice(kinds)) for _ in range(n)]
            ks.sort(key=lambda k: -min(self.PS.size(k)[:2]))            # widest at the bottom
            hsum = sum(H[k] for k in ks)
            s = T / hsum
            if SCALE_LIM[0] <= s <= SCALE_LIM[1]:
                sc = abs(s - 1.0) + 0.02 * n
                if best is None or sc < best[0]:
                    best = (sc, ks, s)
        return None if best is None else [(k, best[2]) for k in best[1]]

    def _climb_row(self, P0, yaw, d, n, fz, s0, h, rot, lat, rng, width=1.9, depth=0.0):
        """a row of the env's own LOW props across the lane whose collider top is h (kind / stack / uniform scale
        solved from the calibrated collider heights, scale within SCALE_LIM) -> (depth along lane, built top, plan)"""
        cal = self.PC.cal
        cands, near = [], []
        for k in EF.CLIMB[self.pk]:
            ck = cal.get(k, {})
            if ck.get("drop") is None or ck.get("solid", 0) < 0.6 or ck.get("top_med", 1) > 0.02:
                continue
            orient = "flat" if k in EF.CLIMB_FLAT else "as_is"
            H = float(self.PC.size(k, orient)[2]) - (ck["drop"] if orient == "as_is" else 0.0)
            for stack in (1, 2, 3):
                sc = h / (stack * H)
                if SCALE_LIM[0] <= sc <= SCALE_LIM[1]:
                    cands.append((k, stack, sc, orient))
                else:
                    sc_c = float(np.clip(sc, *SCALE_LIM))
                    near.append((abs(stack * H * sc_c - h), (k, stack, sc_c, orient)))
        if not cands:                     # nearest reachable height within 3.5 cm (the built edge is measured anyway)
            near = [x for x in near if x[0] <= 0.035]
            if not near:
                raise GS.Reject(f"climb_no_combo h={h:.3f}")
            cands = [min(near)[1]]
        k, stack, sc, orient = cands[int(rng.integers(len(cands)))]
        sz = self.PC.size(k, orient)
        dep, wid = min(sz[:2]), max(sz[:2])
        along_x = sz[0] <= sz[1]
        yaw_p = yaw + (0.0 if along_x else math.pi / 2) + float(rot)
        ncol = max(1, int(math.ceil(width / (wid * sc))))
        if width < 1.5:                 # compound: stay inside the support columns (|l| < L_IN)
            ncol = max(1, int(math.floor(2 * (L_IN - 0.03 - abs(float(lat))) / (wid * sc))))
        l0 = float(lat) - 0.5 * (ncol - 1) * wid * sc
        nrow = max(1, int(math.ceil(depth / (dep * sc)))) if depth > 0 else 1      # platform: rows along the lane
        top = 0.0
        for i_r in range(nrow):
            for j in range(ncol):
                z = fz
                for _ in range(stack):
                    c = self.PC.get(k)
                    lo, hi = self.PC.place(c, *self._xy(P0, d, n, s0 + (i_r + 0.5) * dep * sc, l0 + j * wid * sc), z, yaw=yaw_p,
                                           scale=sc, orient=orient)
                    z = hi[2] - (self.PC.ctop(k, sc) if orient == "as_is" else 0.0)
                top = max(top, z - fz)
        return nrow * dep * sc, top, dict(kind=k, stack=stack, scale=round(sc, 3), ncol=ncol, nrow=nrow, orient=orient)

    # ------------------------------------------------------------------ build
    def build(self, sit):
        import real_assets as RA
        fam, P = sit["family"], sit["params"]
        lane = sit["lane"]; P0, yaw, d, n, fz = self.lane_frame(lane)
        L = float(lane["len"]); s0 = float(sit["s_front"])
        rng = np.random.default_rng(int(sit["seed"]) + 4242)
        self.world.stop()
        self._park_all()
        chk = {}
        if fam != "terrain":
            # the lane floor from the survey grid must be the floor PhysX actually has here (down rays, our objects and
            # the robot skipped): otherwise objects placed on 'fz' would float or sink
            self._restart()
            zs = []
            for s in np.arange(0.2, min(L, s0 + 3.0), 0.4):
                for l in (-0.25, 0.0, 0.25):
                    q = P0 + d * s + n * l
                    o = np.array([q[0], q[1], fz + 1.0]); acc = 0.0
                    for _ in range(4):
                        dist, who, _ = self.ray(o - np.array([0, 0, acc]), (0, 0, -1), 3.0 - acc)
                        if dist is None:
                            break
                        if self.ours(who):
                            acc += dist + 0.01; continue
                        zs.append(fz + 1.0 - acc - dist); break
            self.world.stop()
            if fam == "env_curb":
                zs = [z for z in zs if abs(z - fz) < 0.04]
            if len(zs) < 6:
                raise GS.Reject("floor_unmeasured")
            zmed = float(np.median(zs))
            chk["floor"] = dict(survey=round(fz, 3), measured=round(zmed, 3), spread=round(float(np.ptp(zs)), 3), n=len(zs))
            if abs(zmed - fz) > 0.04 or (fam != "env_curb" and np.ptp(zs) > 0.05):
                raise GS.Reject(f"floor_mismatch {chk['floor']}")
            fz = zmed
        geo = dict(P0=P0, yaw=yaw, d=d, n=n, fz=fz, s_front=s0, z_top=None, u_track=False, value=None, rot=0.0)
        sup = None
        oh = None
        lat_free = None
        is_oh = fam in ("env_overhang", "env_compound")
        if fam in ("env_overhang", "env_gap", "env_compound") or (fam == "null" and P.get("decoy") == "beside"):
            self._restart()
            lat_free = self.free_lat(P0, d, n, fz, (s0, s0 + 0.5, s0 + 1.0))
            self.world.stop()
            chk["lat_free"] = lat_free
        if fam == "terrain":
            t = P["tile"]
            geo.update(s_end=s0 + float(t["geometry"]["size"][0]), s_goal=s0 + 5.0)
            chk["terrain"] = dict(type=P["type"], difficulty=P["difficulty"], bucket=t["bucket"])
            geo["value"] = float(P["difficulty"])
        elif fam == "env_curb":
            s_r = float(P["s_riser"])
            geo.update(s_end=s_r + 0.1, s_goal=min(s_r + 1.2, L - 0.1))
        elif fam == "null":
            geo.update(s_end=s0 + 0.6, s_goal=s0 + 1.5)
            if P.get("decoy") == "beside" and self.P is None:
                geo["decoy_skipped"] = "no_prop_pack"
            if P.get("decoy") == "beside" and self.P is not None:
                side = 1.0 if P["l"] > 0 else -1.0
                room = (lat_free[0] if side > 0 else lat_free[1]) if lat_free else 2.0
                sz = self.PC.size(P["kind"])
                lc = side * min(abs(P["l"]), room - 0.5 * max(sz[:2]) - 0.05)
                if abs(lc) - 0.5 * max(sz[:2]) < 0.75:
                    geo["decoy_skipped"] = "no_room"
                else:
                    c = self.PC.get(P["kind"]); self.PC.place(c, *self._xy(P0, d, n, s0 + 0.5, lc), fz, yaw=yaw + P["yaw"])
                    geo["aim"] = (s0 + 0.5, lc, 0.3)
        elif fam == "env_step":
            k, stack = P["kind"], int(P["stack"])
            if self.PC.cal.get(k, {}).get("solid", 1.0) < 0.5:
                raise GS.Reject(f"prop_not_solid {k} {self.PC.cal.get(k)}")
            sz = self.PC.size(k)
            dep, wid = (sz[0], sz[1]) if P.get("rot", 0) == 0 and sz[0] <= sz[1] else (min(sz[:2]), max(sz[:2]))
            along_x = sz[0] <= sz[1]                     # put the SHORT side along the lane
            yaw_p = yaw + (0.0 if along_x else math.pi / 2) + float(P.get("rot", 0.0))
            ncol = max(1, int(math.ceil(1.9 / (wid * P["scale"]))))
            l0 = float(P["lat"]) - 0.5 * (ncol - 1) * wid * P["scale"]
            top = 0.0
            for j in range(ncol):
                z = fz
                for h_ in range(stack):
                    c = self.PC.get(k)
                    lo, hi = self.PC.place(c, *self._xy(P0, d, n, s0 + 0.5 * dep * P["scale"], l0 + j * wid * P["scale"]), z,
                                           yaw=yaw_p, scale=P["scale"])
                    z = hi[2] - self.PC.ctop(k, P["scale"])
                top = max(top, z - fz)
            geo.update(s_end=s0 + dep * P["scale"], s_goal=min(s0 + dep * P["scale"] + 0.9, L - 0.1), step_top=round(top, 3))
            if geo["s_goal"] < geo["s_end"] + 0.5:
                raise GS.Reject("step_goal_off_lane")
        elif fam == "env_climb":
            plat = bool(P.get("platform", False))
            dep, top, plan = self._climb_row(P0, yaw, d, n, fz, s0, float(P["h"]), P.get("rot", 0.0), P.get("lat", 0.0), rng,
                                             depth=float(P.get("pdepth", 1.5)) if plat else 0.0)
            if plat:                                       # climb ONTO the platform: goal on top (as the school ledge)
                geo.update(s_end=s0 + dep, s_goal=s0 + max(0.6, min(1.1, dep - 0.35)), step_top=round(top, 3), climb_plan=plan,
                           on_top=True, z_top=None, climb_platform=True)
            else:
                geo.update(s_end=s0 + dep, s_goal=min(s0 + dep + 0.9, L - 0.1), step_top=round(top, 3), climb_plan=plan)
                if geo["s_goal"] < geo["s_end"] + 0.5:
                    raise GS.Reject("step_goal_off_lane")
        elif fam == "env_clutter":
            hmax = 0.0
            for it in P["items"]:
                c = self.PC.get(it["kind"])
                lo, hi = self.PC.place(c, *self._xy(P0, d, n, s0 + it["ds"], it["l"]), fz, yaw=yaw + it["yaw"], scale=it["scale"])
                hmax = max(hmax, hi[2] - fz)
            geo.update(s_end=s0 + 1.8, s_goal=min(s0 + 2.4, L - 0.1))
        elif fam == "env_gap":
            w, e = float(P["w"]), float(P["e"])
            wk = P["wall"]
            if self.PC.cal.get(wk, {}).get("solid", 1.0) < 0.5:
                raise GS.Reject(f"prop_not_solid {wk} {self.PC.cal.get(wk)}")
            sz = self.PC.size(wk)
            # the piece's thin side along the lane (wall depth), long side across
            along_x = sz[0] <= sz[1]
            dep, wid = (sz[0], sz[1]) if along_x else (sz[1], sz[0])
            yaw_p = yaw + (0.0 if along_x else math.pi / 2)
            rows = 2 if P.get("depth2") else 1
            lim = (lat_free if lat_free else (2.5, 2.5))
            for side in (1.0, -1.0):
                edge = e + side * w / 2
                room = (lim[0] if side > 0 else lim[1]) + 0.04 - abs(edge) if (edge * side) > 0 else 2.0
                if room < 0.15:
                    raise GS.Reject(f"gap_wider_than_corridor {lat_free}")
                if wid > room + 0.12:                     # a piece may sink <= 12 cm into the corridor wall, no more
                    raise GS.Reject(f"gap_wall_no_room {room:.2f} < {wid:.2f}")
                ncol = max(1, int(math.floor((min(room, 2.0) + 0.06) / wid)))
                for j in range(ncol):
                    lc = edge + side * (0.5 * wid + j * wid)
                    for r_ in range(rows):
                        z = fz
                        nh = 3 if wk == "cardbox_a" else 1
                        for h_ in range(nh):
                            c = self.PC.get(wk)
                            lo, hi = self.PC.place(c, *self._xy(P0, d, n, s0 + (r_ + 0.5) * dep, lc), z, yaw=yaw_p)
                            z = hi[2] - self.PC.ctop(wk)
            Lg = rows * dep
            geo.update(s_end=s0 + Lg, s_goal=min(s0 + Lg + 0.8, L - 0.1), gap_L=round(Lg, 3))
            if geo["s_goal"] < geo["s_end"] + 0.4:
                raise GS.Reject("gap_goal_off_lane")
        elif is_oh:
            slab = P["slab"]
            if slab == "door":                       # the 2.07 m door leaf needs its full length to reach both columns
                P = dict(P, scale=max(float(P["scale"]), 1.08))
            sz = self.P.size(slab, "flat")
            long_x = sz[0] >= sz[1]
            yaw_s = yaw + (math.pi / 2 if long_x else 0.0) + float(P["yaw_rel"])
            half = 0.5 * max(sz[:2]) * P["scale"]; depth = min(sz[:2]) * P["scale"]
            lat = float(P["lat"])
            room_l = [(lat_free[0] if lat_free else 2.0) - L_IN - 0.03, (lat_free[1] if lat_free else 2.0) - L_IN - 0.03]
            # supports: column under each slab end, inner face at L_IN, footprint inside the slab's reach
            T = float(P["u"])
            cols = []
            for side, room in ((1.0, room_l[0]), (-1.0, room_l[1])):
                reach = half * math.cos(float(P["yaw_rel"])) + side * lat - L_IN - 0.04
                room_c = min(room, reach)
                if room_c < 0.2:
                    raise GS.Reject(f"unsupported_no_room side {side:+.0f} room {room:.2f} reach {reach:.2f}")
                plan = self._stack_plan(T, rng, room_c)
                if plan is None:
                    raise GS.Reject(f"unsupported_no_column T={T:.2f} room={room_c:.2f}")
                cols.append((side, plan))
            s_c = s0 + 0.5 * depth
            tops = []
            for side, plan in cols:
                z = fz
                wmax = max(min(self.PS.size(k)[:2]) * s for k, s in plan)
                lc = side * (L_IN + 0.5 * wmax + 0.01)
                for k, s in plan:
                    szk = self.PS.size(k)
                    yaw_k = yaw + (math.pi / 2 if szk[0] < szk[1] else 0.0)      # thin side lateral
                    c = self.PS.get(k)
                    lo, hi = self.PS.place(c, *self._xy(P0, d, n, s_c, lc), z, yaw=yaw_k, scale=s)
                    z = hi[2] - self.PS.ctop(k, s)
                tops.append(z)
            Tb = max(tops)                                   # slab rests on the higher column; both checked by rays
            c = self.P.get(slab)
            lo, hi = self.P.place(c, *self._xy(P0, d, n, s_c, lat), Tb, yaw=yaw_s, orient="flat", scale=P["scale"])
            sup = dict(tops=[round(t - fz, 4) for t in tops], T_target=T, plans=[[(k, round(s, 3)) for k, s in pl] for _, pl in cols])
            # the slab's lane-direction extent (yaw) from its world AABB projected on d
            Cs = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) - P0
            proj = Cs @ d
            oh = dict(s0=float(proj.min()), s_end=float(proj.max()), thick=float(hi[2] - lo[2]))
            geo.update(s_end=oh["s_end"], s_goal=oh["s_end"] + 0.5, u_track=True, s_front=s0)
            if abs(tops[0] - tops[1]) > 0.012:
                raise GS.Reject(f"support_tops_differ {tops}")
            if fam == "env_compound":                     # overhang THEN a climbable step (H2-type) behind it
                s_r = oh["s_end"] + float(P["gap"])
                dep, top, plan = self._climb_row(P0, yaw, d, n, fz, s_r, float(P["h"]), 0.0, 0.0, rng, width=1.1)
                geo.update(s_end=s_r + dep, s_goal=s_r + dep + 0.6, step_top=round(top, 3), climb_plan=plan, s_step=round(s_r, 3),
                           oh_end=round(oh["s_end"], 3))
        if geo["s_goal"] > L + 0.3:
            raise GS.Reject("goal_off_lane")
        self._restart()
        # ---------------- no-primitive audit (our copies are referenced env assets with their own materials)
        if self.P is not None:
            npc = RA.audit_no_primitives(self.stage, GS.OURS)
            chk["no_primitive"] = dict(n_gprims=npc["n_gprims"], ok=npc["ok"])
            if not npc["ok"]:
                raise GS.Reject(f"primitive_geometry {npc['bad'][:3]}")
        # ---------------- lane clutter / headroom
        if fam not in ("terrain",):
            lc_ = self.lane_check(P0, d, n, fz, min(geo["s_goal"] + 0.3, L))
            if fam == "env_curb":                           # the env's own curb is the hazard: only hits off the riser count
                s_r = float(P["s_riser"]); rise = float(P["rise"])
                bad = [c for c in lc_["clutter"] if not (c[0] >= s_r - 0.6 and abs(c[2] - max(rise, 0.0)) < 0.05)]   # curbs may run diagonally
                lc_["n_clutter"] = len(bad); lc_["clutter"] = bad
            chk["lane"] = lc_
            if lc_["n_clutter"] > 0:
                raise GS.Reject(f"lane_clutter {lc_['clutter'][:2]}")
            if lc_["n_low_head"] > 0 and not is_oh:
                raise GS.Reject(f"lane_low_headroom {lc_['low_head'][:2]}")
        # ---------------- intended geometry by rays
        if is_oh:
            # the slab's lowest COLLIDER point can sit a few mm above its bbox bottom (handles, bevels): measure the gap
            # over each column top and lower the slab onto the higher column (it then rests on it)
            gaps = []; gap_dbg = [0, 0, 0]
            for side, top in zip((1.0, -1.0), sup["tops"]):
                gs = []
                for l in np.arange(L_IN + 0.03, L_IN + 0.5, 0.05):
                    for s in np.linspace(oh["s0"] + 0.05, oh["s_end"] - 0.05, 4):
                        q = P0 + d * s + n * side * l
                        up, wu, _ = self.ray((q[0], q[1], fz + top - 0.004), (0, 0, 1), 0.2)
                        dn, wd, _ = self.ray((q[0], q[1], fz + top + 0.002), (0, 0, -1), 0.6)
                        gap_dbg[0] += 1; gap_dbg[1] += int(up is not None and wu.startswith(OBJ_ROOT))
                        gap_dbg[2] += int(dn is not None and wd.startswith(SUP_ROOT))
                        if up is not None and wu.startswith(OBJ_ROOT) and dn is not None and wd.startswith(SUP_ROOT) and dn <= 0.008:
                            gs.append(up - 0.004)
                gaps.append(min(gs) if gs else None)
            g = min([x for x in gaps if x is not None] or [0.0])
            sup["gap_fix"] = round(g, 4); sup["gaps"] = gaps; sup["gap_dbg"] = gap_dbg
            if 0.001 < g < 0.06:
                self.world.stop()
                M = np.array(c["xf"].Get()); M[3, 2] -= g
                c["xf"].Set(self.Gf.Matrix4d(*[float(v) for v in M.reshape(-1)]))
                self._restart()
        if is_oh:
            S, U, hits, n_r = self.clearance(P0, d, n, fz, oh["s0"] - 0.3, oh["s_end"] + 0.3)
            m = (S >= oh["s0"] - 0.05) & (S <= oh["s_end"] + 0.05)
            u_min = float(U[m].min()) if m.any() else 9.9
            geo["S"], geo["U"] = S, U
            chk["oh"] = dict(u_target=P["u"], T_built=round(max(sup["tops"]), 3), u_min=round(u_min, 3), obj_hits=hits, slab=P["slab"],
                             depth=round(oh["s_end"] - oh["s0"], 3))
            if hits == 0:
                raise GS.Reject("overhang_not_over_lane")
            if u_min < max(sup["tops"]) - 0.06 or u_min > max(sup["tops"]) + 0.30:
                raise GS.Reject(f"overhang_u_mismatch {u_min:.3f} vs {max(sup['tops']):.3f}")
            geo["value"] = round(u_min, 3)
            geo["s_front"] = round(max(oh["s0"], 0.5), 3)
            # SUPPORTED: rays straddling each column top: down from top+2 mm hits the column within 8 mm AND up from
            # top-2 mm... (slab bottom sits ON the top) -> up from top+0.5 mm hits the slab within 8 mm
            for it_sup in range(2):
                res = {}; g_meas = []
                for side, top in zip((1.0, -1.0), sup["tops"]):
                    n_c = 0
                    for s in np.arange(oh["s0"] - 0.1, oh["s_end"] + 0.1, 0.04):
                        for l in np.arange(L_IN + 0.02, L_IN + 0.6, 0.04):
                            q = P0 + d * s + n * side * l
                            dn, wd, _ = self.ray((q[0], q[1], fz + top + 0.002), (0, 0, -1), 0.6)       # (short rays
                            up, wu, _ = self.ray((q[0], q[1], fz + top - 0.004), (0, 0, 1), 0.6)        # miss here)
                            n_c += int(dn is not None and wd.startswith(SUP_ROOT) and dn <= 0.008 and up is not None
                                       and wu.startswith(OBJ_ROOT) and up <= 0.012)
                            if dn is not None and wd.startswith(SUP_ROOT) and dn <= 0.008 and up is not None and wu.startswith(OBJ_ROOT):
                                g_meas.append(up - 0.004)
                    blk = 0
                    for s in np.arange(oh["s0"], oh["s_end"] + 1e-6, 0.1):
                        q = P0 + d * s + n * side * (L_IN - 0.05)
                        dd, ww, _ = self.ray((q[0], q[1], fz + top - 0.02), (0, 0, -1), top)
                        blk += int(dd is not None and ww.startswith(SUP_ROOT))
                    diag = []
                    for l in (L_IN + 0.08, L_IN + 0.2):
                        q = P0 + d * (0.5 * (oh["s0"] + oh["s_end"])) + n * side * l
                        dn, wd, _ = self.ray((q[0], q[1], fz + top + 0.002), (0, 0, -1), 0.6)
                        up, wu, _ = self.ray((q[0], q[1], fz + top - 0.004), (0, 0, 1), 0.6)
                        dn2, wd2, _ = self.ray((q[0], q[1], fz + top + 0.5), (0, 0, -1), 1.0)
                        diag.append([None if dn is None else round(dn, 4), wd[-30:], None if up is None else round(up, 4), wu[-30:],
                                     None if dn2 is None else round(0.5 - dn2, 4), wd2[-30:]])
                    res[f"{side:+.0f}"] = dict(contacts=n_c, lane_blocked=blk, top=round(top, 4), diag=diag)
                ok = all(v["contacts"] > 0 and v["lane_blocked"] == 0 for v in res.values())
                g = min(g_meas) if g_meas else 0.0
                chk["supported"] = dict(sides=res, ok=ok, plans=sup["plans"], gap_measured=round(g, 4), iteration=it_sup)
                if ok or it_sup == 1 or not (0.001 < g < 0.06):
                    break
                # the slab's lowest collider point sits g above the column tops (handle / bevel): lower it onto them, re-check
                self.world.stop()
                M = np.array(c["xf"].Get()); M[3, 2] -= g
                c["xf"].Set(self.Gf.Matrix4d(*[float(v) for v in M.reshape(-1)]))
                self._restart()
                S, U, hits, n_r = self.clearance(P0, d, n, fz, oh["s0"] - 0.3, oh["s_end"] + 0.3)
                m = (S >= oh["s0"] - 0.05) & (S <= oh["s_end"] + 0.05)
                u_min = float(U[m].min()) if m.any() else 9.9
                geo["S"], geo["U"] = S, U; geo["value"] = round(u_min, 3); chk["oh"]["u_min_after_fix"] = round(u_min, 3)
            if not ok:
                raise GS.Reject(f"unsupported gap_fix={sup.get('gap_fix')} gaps={sup.get('gaps')} dbg={sup.get('gap_dbg')} {res}")
        if fam in ("env_step", "env_climb", "env_compound", "env_clutter", "env_curb") or (fam == "null" and geo.get("aim") is not None):
            prof = self.profile(P0, d, n, fz, max(0.0, s0 - (1.5 if fam == "env_curb" else 0.3)), min(geo["s_goal"] + 0.3, L))
            hv = lambda r: max([v_ for v_ in r[1:] if v_ is not None] or [0.0])
            if fam == "env_compound":
                ss_ = geo["s_step"]
                edge = max([hv(r) for r in prof if ss_ - 0.1 <= r[0] <= geo["s_end"] + 0.05] or [0.0])
                chk["curb"] = dict(h=round(edge, 3), target=P["h"], top_built=geo["step_top"], plan=geo["climb_plan"])
                if abs(edge - geo["step_top"]) > 0.03 or edge < 0.03:
                    raise GS.Reject(f"curb_mismatch {edge:.3f} vs {geo['step_top']:.3f}")
            elif fam in ("env_step", "env_climb"):
                edge = max([hv(r) for r in prof if s0 - 0.1 <= r[0] <= geo["s_end"] + 0.05] or [0.0])
                chk["step"] = dict(edge=round(edge, 3), top_built=geo["step_top"], kind=P.get("kind"), stack=P.get("stack"),
                                   target=P.get("h"), plan=geo.get("climb_plan"))
                if abs(edge - geo["step_top"]) > 0.05 or edge < 0.02:
                    raise GS.Reject(f"step_edge_mismatch {edge:.3f} vs {geo['step_top']:.3f}")
                geo["value"] = round(edge, 3)
                if geo.get("climb_platform"):
                    geo["z_top"] = fz + edge
            elif fam == "env_clutter":
                hmax = max([hv(r) for r in prof if s0 - 0.3 <= r[0] <= s0 + 1.9] or [0.0])
                chk["clutter"] = dict(h_max=round(hmax, 3), n=len(P["items"]))
                if hmax < 0.02:
                    raise GS.Reject("clutter_missing")
                geo["value"] = round(hmax, 3)
            elif fam == "env_curb":
                s_r = float(P["s_riser"])
                pre = [np.mean([v_ for v_ in r[1:] if v_ is not None]) for r in prof if r[0] < s_r - 0.3 and any(r[1:])]
                post = [np.mean([v_ for v_ in r[1:] if v_ is not None]) for r in prof if r[0] > s_r + 0.3 and any(r[1:])]
                rise = float(np.median(post) - np.median(pre)) if pre and post else 0.0
                chk["curb"] = dict(rise=round(rise, 3), survey=P["rise"])
                if abs(rise - float(P["rise"])) > 0.05 or abs(rise) < 0.04:
                    raise GS.Reject(f"curb_mismatch {rise:.3f} vs {P['rise']}")
                geo["value"] = round(rise, 3)
                if rise > 0:
                    geo["z_top"] = fz + rise
            else:
                if max([hv(r) for r in prof] or [0.0]) > 0.03:
                    raise GS.Reject("decoy_in_lane")
        if fam == "env_gap":
            w, e = float(P["w"]), float(P["e"])
            Lg = geo["gap_L"]
            moved = []
            for side in (1.0, -1.0):                      # move each wall so its measured face is exactly at the spec edge
                ds_ = []
                for z in (0.25, 0.5, 0.9, 1.3):
                    for s in np.linspace(s0 + 0.1 * Lg, s0 + 0.9 * Lg, 3):
                        q = P0 + d * s + n * e
                        dist, who, _ = self.ray((q[0], q[1], fz + z), (n[0] * side, n[1] * side, 0.0), 3.0)
                        if dist is not None and who.startswith(GS.OURS):
                            ds_.append(dist)
                if ds_:
                    err = min(ds_) - w / 2
                    if abs(err) > 0.003:
                        self.world.stop()
                        for c in self.PC.pool.get(P["wall"], [])[: self.PC.used.get(P["wall"], 0)]:
                            M = np.array(c["xf"].Get())
                            pos = M[3, :3]
                            if (pos[:2] - P0) @ n * side > e:
                                M[3, :2] -= n * side * err
                                c["xf"].Set(self.Gf.Matrix4d(*[float(v) for v in M.reshape(-1)]))
                        moved.append(round(err, 4))
            if moved:
                self._restart()
            chk["gap_edge_correction"] = moved
            ws = []
            for z in (0.25, 0.5, 0.9, 1.3):
                for s in np.linspace(s0 + 0.1 * Lg, s0 + 0.9 * Lg, 3):
                    q = P0 + d * s + n * e
                    a_, wa, _ = self.ray((q[0], q[1], fz + z), (n[0], n[1], 0.0), 3.0)
                    b_, wb, _ = self.ray((q[0], q[1], fz + z), (-n[0], -n[1], 0.0), 3.0)
                    if a_ is not None and b_ is not None and wa.startswith(GS.OURS) and wb.startswith(GS.OURS):
                        ws.append(a_ + b_)
            chk["gap"] = dict(w_target=w, w_min=round(min(ws), 3) if ws else None, w_max=round(max(ws), 3) if ws else None, n=len(ws), wall=P["wall"])
            if len(ws) < 6 or abs(min(ws) - w) > 0.04 or max(ws) - w > 0.10:
                raise GS.Reject(f"gap_width_mismatch {chk['gap']}")
            geo["value"] = round(min(ws), 3)
        if fam == "null":
            geo["value"] = 0.0
        geo["checks"] = chk
        geo["hazard_band"] = (geo["s_front"] - 1.0, geo["s_end"] + 0.3)
        if is_oh:
            geo["aim"] = (0.5 * (oh["s0"] + oh["s_end"]), float(P["lat"]), geo["value"] + 0.5 * min(oh["thick"], 0.3))
        elif fam in ("env_step", "env_climb"):
            geo["aim"] = (s0 + 0.15, 0.0, 0.5 * max(geo["value"], 0.04))
        elif fam == "env_clutter":
            geo["aim"] = (s0 + 0.8, 0.0, 0.5 * max(geo["value"], 0.04))
        elif fam == "env_gap":
            geo["aim"] = (s0, float(P["e"]), 0.6)
        return geo

    # terrain: the spawn and the decision snapshot must use the local ground (flat run-up at z = 0)
