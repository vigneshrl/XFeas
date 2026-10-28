"""
XFeas-Gen Isaac harness (import AFTER SimulationApp exists). One process, one robot, the school stage (never written),
hazards built IN MEMORY from duplicated REAL school assets only (2026-10-10 rule; primitives only with
XFEAS_LEGACY_PRIMITIVES=1 to reproduce old suites):
  overhang_lib.OverhangLib  (school-asset copies laid flat: doors, planks, partitions, benches, gratings, shelves,
                             lamps, windows, washbasins)                          -> overhang / compound / null decoys
  climb_lib.ClimbBuilder    (real_assets.RealPool: real plank stacks / terraces, real furniture rows; school gym-mat
                             copies)                                               -> step / gap walls / compound curb
  every build also runs real_assets.audit_no_primitives on our roots (Reject 'primitive_geometry' otherwise)
Per situation:
  build(sit)            place the hazard, restart physics (colliders follow moved copies only after a restart), run the
                        AUTOMATIC CHECKS (lane clutter + headroom by PhysX rays, intended geometry by rays, then -- at the
                        decision snapshot -- a visible-pixel check: PhysX rays through a pixel grid must hit the hazard and
                        agree with the RENDERED depth there) -> geo, or raise Reject(reason)
  decision(sit, geo)    spawn, settle, approach at the stock posture to the decision point, RGB-D snapshot (headlight),
                        2 s proprio history, then SAVE the full sim + controller state
  rollout(geo, v, sp)   RESTORE that state and hold one setpoint (v, body height, pitch, roll) to the goal -> labels:
                        success / fell / collision / stall / progress / max_tilt + costs (t_goal, energy, posture_dev,
                        tilt margin, min clearance)
  run_method(m, ...)    closed-loop arm from the spawn (episode_runner-style method plugin, e.g. oh_methods.StraightXFeasHold)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import copy, json, math, os, sys, time
from collections import deque
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
XE = os.path.join(ROOT, "closed_loop")
for _p in (XE, os.path.join(XE, "controllers"), os.path.join(ROOT, "scenes"), os.path.join(ROOT, "scenes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OURS = ("/World/Overhang", "/World/Climb")
SUP_ROOT = "/World/Overhang/Supports"           # real support columns of raised objects (no floating objects rule)
FX = 12.0 / 20.955 * 224.0                       # head camera: focal 12 mm, aperture 20.955 mm, 224 px


class Reject(Exception):
    pass


class GenSim:
    def __init__(self, app, factory_spec="spot_posture:make_spotp", kind="spotp", seed=0, side_cam=False):
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
        from pxr import UsdGeom, Gf, Usd, UsdLux, Sdf
        import robots as RB
        self.app, self.carb, self.rep, self.Gf, self.UsdGeom, self.RB = app, carb, rep, Gf, UsdGeom, RB
        self.kind = kind
        self.rng = np.random.default_rng(777000 + seed)
        rep.set_global_seed(int(self.rng.integers(1 << 30)))
        _fm, _ff = factory_spec.split(":")
        self.FMOD = importlib.import_module(_fm); self.FACTORY = getattr(self.FMOD, _ff)
        spec = self.FACTORY.SPEC; self.spec = spec; self.HZ = HZ = spec.phys_hz
        self.RANGES = spec.extra.get("setpoint_ranges", {})
        self.NOM = getattr(self.FMOD, "NOMINAL_H", None) or getattr(getattr(self.FMOD, "g1_amo", None), "NOMINAL_H", None)
        omni.usd.get_context().open_stage(os.path.join(ROOT, "scenes", "assets", "school.usd"))
        for _ in range(20):
            app.update()
        self.world = World(physics_dt=1.0 / HZ, rendering_dt=1 / 50.0, stage_units_in_meters=1.0)
        stage = self.stage = self.world.stage
        for p in ("/World/Spot", "/World/Surfaces"):
            pr = stage.GetPrimAtPath(p)
            if pr.IsValid():
                pr.SetActive(False)
        import scene_clean
        scene_clean.apply(stage)
        import scene_lighting
        self.LIGHT = scene_lighting.Lighting(stage)
        self.world.reset()
        self.pxq = get_physx_scene_query_interface()
        import overhang_lib as OL, climb_lib as CL
        self.OBJ = OL.OverhangLib(stage, self.rng)
        self.B = CL.ClimbBuilder(stage, self.world, self.pxq, carb, n_mats=3, sources=CL.MAT_ALL)
        import real_assets as RA
        self.SUP = None if RA.LEGACY else RA.RealLib(stage, root=SUP_ROOT)
        self.root = f"/World/{kind}"
        self.rob = self.FACTORY(kind, self.root, [0.0, 0.0, -20.0], yaw=0.0)
        self.B.on_reset = lambda: setattr(self.rob, "ready", False)
        cam = UsdGeom.Camera.Define(stage, "/World/HeadCam")
        cam.CreateFocalLengthAttr(12.0); cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955); cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
        self.cam_xf = UsdGeom.Xformable(cam.GetPrim()).MakeMatrixXform()
        rp = rep.create.render_product("/World/HeadCam", (224, 224))
        self.ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); self.ann_rgb.attach([rp])
        self.ann_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); self.ann_dep.attach([rp])
        # v4 (2026-10-19): the shared decision-camera mount (isaac_sim/camera_mount.py). G1 = 27 deg PHP/Holosoma head
        # D435i on torso_link, Spot = body-fixed front camera, full 6-DoF every frame, exact self mask, headlight on the
        # mount. XFEAS_CAM_MOUNT=legacy (or pelvis) = the old spec.cam yaw-only rule of _cam_pose.
        self._mount_init(stage, rep, rp, kind, spec)
        self.side = None
        if side_cam:                                   # AUDIT view only (never a model input)
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
        # robot-dependent run constants (episode_runner speed 0.6 of max_cmd)
        self.v_nom = max(0.6 * spec.max_cmd[0], spec.v_min)
        legged_g1 = kind.startswith("g1")
        self.STALL_S = 8.0 if legged_g1 else 5.0
        self.V_FLOOR = 0.06 if legged_g1 else 0.25
        self.T_CAP = 70.0 if legged_g1 else 40.0
        self.BOX = self.full_box()
        self._lane_cache = {}
        self.stand_z = float(spec.spawn_dz) - 0.05      # updated from the measured standing height at every spawn

    # ------------------------------------------------------------------ setpoint box (robot's OWN spec)
    def full_box(self):
        """the controller's declared ranges: v [max(v_min, 0.1), max_cmd vx], body_height, pitch, roll"""
        R = self.RANGES
        box = dict(v=(max(self.spec.v_min, 0.1), float(self.spec.max_cmd[0])))
        for k in ("body_height", "pitch", "roll"):
            box[k] = tuple(map(float, R[k])) if k in R else (0.0, 0.0)
        if self.NOM is None or "body_height" not in R:
            box["body_height"] = (0.0, 0.0)
        return box

    def nominal(self):
        return dict(v=self.v_nom, body_height=self.NOM if self.NOM is not None else 0.0, pitch=0.0, roll=0.0)

    def posture_dev(self, sp):
        """mean |setpoint - nominal| / box width over the posture dims the robot has"""
        nom = self.nominal(); out = []
        for k in ("body_height", "pitch", "roll"):
            lo, hi = self.BOX[k]
            if hi - lo > 1e-6:
                out.append(abs(sp[k] - nom[k]) / (hi - lo))
        return float(np.mean(out)) if out else 0.0

    # ------------------------------------------------------------------ helpers
    def lane_frame(self, lane):
        yaw = float(lane["yaw"]); d = np.array([math.cos(yaw), math.sin(yaw)]); n = np.array([-d[1], d[0]])
        return np.array(lane["start"], float), yaw, d, n, float(lane["floor"])

    def ray(self, o, v, L):
        h = self.pxq.raycast_closest(self.carb.Float3(*map(float, o)), self.carb.Float3(*map(float, v)), float(L))
        if h and h.get("hit"):
            return float(h["distance"]), str(h.get("collision", "")), np.asarray(h["position"], float)
        return None, "", None

    def ours(self, path):
        return path.startswith(OURS) or path.startswith(self.root)

    def reset_policy_state(self):
        c = getattr(self.rob, "ctrl", None)
        for at in ("_previous_action", "_old_action", "action"):
            if c is not None and hasattr(c, at):
                try:
                    setattr(c, at, np.zeros_like(getattr(c, at)))
                except Exception:
                    pass

    def clear_buf(self):
        self.buf.base, self.buf.joints, self.buf.extras = [], [], []; self.buf.k = 0

    def settle(self, x, y, yaw, fz, secs):
        self.rob.teleport(x, y, yaw, fz); self.reset_policy_state()
        if hasattr(self.rob, "reset_setpoints"):
            self.rob.reset_setpoints()                     # every episode / situation starts at the nominal posture
        self.clear_buf()
        for _ in range(int(secs * self.HZ)):
            self.world.step(render=False); self.buf.tick(self.rob)

    def robot_top(self):
        self.bbc.Clear()
        return float(self.bbc.ComputeWorldBound(self.ROBOT_PRIM).ComputeAlignedRange().GetMax()[2])

    # ------------------------------------------------------------------ build + geometry checks
    def _oh_spec(self, o):
        import real_assets as RA
        o = dict(o)
        if o["cls"] == "prim" and not RA.LEGACY:          # legacy spec -> a real plank of (about) the same depth
            o.update(cls="plank", copy_idx=0, across=True, scale=float(np.clip(float(o["depth"]) / 0.5, *RA.SCALE_LIM)))
        if not RA.LEGACY:                                  # no floating objects: a slanted board would slide off flat supports
            o["slant"] = 0.0
        sp = dict(cls=o["cls"], u=float(o["u"]), yaw_rel=float(o["yaw_rel"]), slant=float(o["slant"]), roll=float(o["roll"]))
        if o["cls"] == "prim":
            sp.update(depth=o["depth"], width=o["width"], thick=o["thick"], color=tuple(o["color"]), mat=o.get("mat"))
            sp["name"] = "prim_box"
        else:
            lst = self.OBJ.copies[o["cls"]]
            c = lst[int(o["copy_idx"]) % len(lst)]
            sp.update(copy=c, name=c["name"], across=bool(o["across"]), scale=float(o["scale"]))
        return sp

    def _place_overhang(self, o, P0, yaw, fz, s0, lat, max_depth=None):
        import real_assets as RA
        if max_depth is None:          # real objects may not be shrunk > 15 %: allow deeper ones (goal check still applies)
            max_depth = 1.3 if RA.LEGACY else 2.1
        sp = self._oh_spec(o)
        geo = self.OBJ.place(sp, P0, yaw, fz, s0, lat)
        if geo["depth"] > max_depth:                      # keep the goal on the checked lane (as collect_overhang)
            import real_assets as RA
            if sp["cls"] == "prim":
                sp["depth"] = min(sp["depth"], max_depth - 0.1)
            else:
                sc_new = sp["scale"] * (max_depth - 0.1) / geo["depth"]
                if sc_new < RA.SCALE_LIM[0] and not RA.LEGACY:     # real rule: no shrinking beyond 15 %; turn it instead
                    sp["across"] = not sp["across"]
                    geo2 = self.OBJ.place(sp, P0, yaw, fz, s0, lat)
                    if geo2["depth"] > max_depth:
                        sc_new = sp["scale"] * (max_depth - 0.1) / geo2["depth"]
                        if sc_new < RA.SCALE_LIM[0]:
                            raise Reject(f"overhang_too_deep {geo2['depth']:.2f}")
                    else:
                        sc_new = sp["scale"]
                sp["scale"] = sc_new
            geo = self.OBJ.place(sp, P0, yaw, fz, s0, lat)
        geo["name"] = sp["name"]; geo["scale_used"] = round(float(sp.get("scale", 1.0)), 3)
        return geo

    def _restart(self):
        self.world.stop(); self.world.reset(); self.rob.ready = False
        for _ in range(int(0.3 * self.HZ)):
            self.world.step(render=False)

    def clearance(self, P0, d, n, fz, s0, s1, lats=(-0.25, 0.0, 0.25), ds=0.05):
        """upward rays from 5 cm above the floor: u(s) = lowest hit above the floor over the laterals; hits on OUR objects"""
        S = np.arange(max(0.0, s0), s1 + 1e-6, ds); U = np.full(len(S), 9.9); hits = 0; n_r = 0
        for i, s in enumerate(S):
            for l in lats:
                q = P0 + d * s + n * l
                dist, who, _ = self.ray((q[0], q[1], fz + 0.05), (0, 0, 1), 4.0)
                n_r += 1
                if dist is not None and not who.startswith(self.root):
                    U[i] = min(U[i], dist + 0.05); hits += int(who.startswith(OURS))
        return S, U, hits, n_r

    def lane_check(self, P0, d, n, fz, s_end):
        """clutter in the lane (a non-hazard hit 6-150 cm above the floor along the lane) and headroom; hits on our
        hazard objects or the robot are skipped. Rays: down from floor+1.5 m, up from floor+5 cm."""
        bad, low_head, n_known = [], [], 0
        for s in np.arange(0.2, s_end + 1e-6, 0.2):
            for l in (-0.28, 0.0, 0.28):
                q = P0 + d * s + n * l
                dist, who, _ = self.ray((q[0], q[1], fz + 1.5), (0, 0, -1), 1.6)
                if dist is not None and not self.ours(who):
                    n_known += 1
                    z = 1.5 - dist
                    if z > 0.06:
                        bad.append((round(float(s), 2), l, round(z, 3), who[-60:]))
                dist, who, _ = self.ray((q[0], q[1], fz + 0.05), (0, 0, 1), 3.0)
                if dist is not None and not self.ours(who) and dist + 0.05 < 1.6:
                    low_head.append((round(float(s), 2), l, round(dist + 0.05, 3), who[-60:]))
        return dict(clutter=bad[:5], n_clutter=len(bad), low_head=low_head[:5], n_low_head=len(low_head), n_known=n_known)

    def free_lat(self, P0, d, n, fz, ss, zs=(0.3, 0.9, 1.4), Lmax=3.0):
        """free room (left = +n, right = -n) from the lane centre at lane positions ss, skipping our objects / robot
        (rays restart behind every hit on them). Physics must be running."""
        out = []
        for sgn in (1.0, -1.0):
            best = Lmax
            for s in ss:
                for z in zs:
                    o = np.r_[P0 + d * s, fz + z]; v = np.r_[n * sgn, 0.0]; acc = 0.0
                    for _ in range(8):
                        dist, who, _ = self.ray(o + v * acc, v, Lmax - acc)
                        if dist is None:
                            break
                        if self.ours(who):
                            acc += dist + 0.01; continue
                        best = min(best, acc + dist); break
            out.append(round(float(best), 3))
        return tuple(out)

    def _is_obj(self, who):
        return who.startswith("/World/Overhang/") and not who.startswith(SUP_ROOT)

    def _support_overhang(self, P0, yaw, d, n, fz, s_a, s_b, lat_free):
        """NO FLOATING OBJECTS: measure the placed object's underside beside the lane (upward PhysX rays, |l| >= L_IN),
        then build one real support column per side (furniture / plank pile, top exactly at the object's lowest point
        over the column's footprint) outside the lane. Raises Reject when the object cannot be supported."""
        import real_assets as RA
        self._restart()                                            # colliders of the moved object
        out = dict(sides={})
        cols = []
        for side in (1.0, -1.0):
            room = (lat_free[0] if side > 0 else lat_free[1]) - RA.L_IN - 0.03 if lat_free else 1.0
            if room < 0.25:
                raise Reject(f"unsupported_no_room side {side:+.0f} {lat_free}")
            hits = []
            for s in np.arange(s_a - 0.3, s_b + 0.3 + 1e-6, 0.05):
                for l in np.arange(RA.L_IN, RA.L_IN + min(room, 0.8) + 1e-6, 0.05):
                    q = P0 + d * s + n * side * l
                    dist, who, _ = self.ray((q[0], q[1], fz + 0.03), (0, 0, 1), 3.5)
                    if dist is not None and self._is_obj(who):
                        hits.append((float(s), float(l), 0.03 + dist))
            if not any(h[1] >= RA.L_IN + 0.15 for h in hits):
                raise Reject(f"unsupported_object_too_narrow side {side:+.0f}")
            H = np.array(hits)
            s_c = float(np.median(H[H[:, 1] <= RA.L_IN + 0.4, 0])) if (H[:, 1] <= RA.L_IN + 0.4).any() else float(np.median(H[:, 0]))
            win = lambda half: H[(np.abs(H[:, 0] - s_c) <= half), 2].min()
            T = float(win(0.6))
            pl = self.SUP.support_plan(T, max_lat=room)
            if pl is not None and pl["kind"] is None:              # a bare plank pile is 3 m long: lowest point over its length
                T2 = float(win(1.6))
                if T2 < T - 0.005:
                    T = T2; pl = self.SUP.support_plan(T, max_lat=room)
            if pl is None:
                raise Reject(f"unsupported_no_column T={T:.2f} room={room:.2f}")
            # the column's site must be free floor (no school object in its way)
            for ds in (-0.4, 0.0, 0.4):
                for dl in (0.05, 0.5 * pl["lat"], pl["lat"] - 0.05):
                    q = P0 + d * (s_c + ds) + n * side * (RA.L_IN + dl)
                    dist, who, _ = self.ray((q[0], q[1], fz + T - 0.01), (0, 0, -1), T + 0.2)
                    if dist is not None and not self.ours(who) and (T - 0.01 - dist) > 0.03:
                        raise Reject(f"support_site_cluttered {who[-40:]}")
            cols.append((s_c, side, T, room))
        self.world.stop()
        for s_c, side, T, room in cols:
            c = self.SUP.support_column(P0, yaw, fz, s_c, side, T, max_lat=room)
            out["sides"][f"{side:+.0f}"] = c
        return out

    def _check_supported(self, P0, d, n, fz, sup, s_a, s_b):
        """rays straddling each column's top plane: down from T+2 mm must hit the SUPPORT within 6 mm and up from
        T-2 mm must hit the OBJECT within 6 mm (they touch) at >= 1 point per side; and nothing of the supports may
        stand in the lane (|l| < L_IN - 0.05)."""
        import real_assets as RA
        res = {}
        for key, c in sup["sides"].items():
            side = float(key); T = c["top"]; n_c = 0
            for s in np.arange(c["s_c"] - 0.6, c["s_c"] + 0.6 + 1e-6, 0.05):
                for l in np.arange(RA.L_IN + 0.02, abs(c["l"][1]) - 0.01, 0.04):
                    q = P0 + d * s + n * side * l
                    dn, wd, _ = self.ray((q[0], q[1], fz + T + 0.002), (0, 0, -1), 0.02)
                    up, wu, _ = self.ray((q[0], q[1], fz + T - 0.002), (0, 0, 1), 0.02)
                    n_c += int(dn is not None and wd.startswith(SUP_ROOT) and up is not None and self._is_obj(wu)
                               and dn <= 0.008 and up <= 0.008)
            blk = 0
            for s in np.arange(s_a, s_b + 1e-6, 0.1):
                q = P0 + d * s + n * side * (RA.L_IN - 0.05)
                dd, ww, _ = self.ray((q[0], q[1], fz + T - 0.02), (0, 0, -1), T)
                blk += int(dd is not None and ww.startswith(SUP_ROOT))
            res[key] = dict(contacts=n_c, lane_blocked=blk, top=T, kind=c["kind"], pile=c["pile"])
        return dict(sides=res, ok=all(v["contacts"] > 0 and v["lane_blocked"] == 0 for v in res.values()))

    def build(self, sit):
        """place the situation's hazard; restart physics; automatic geometry checks -> geo (raises Reject)"""
        import real_assets as RA
        fam, P = sit["family"], sit["params"]
        lane = sit["lane"]; P0, yaw, d, n, fz = self.lane_frame(lane)
        L = float(lane["len"]); s0 = float(sit["s_front"])
        lat_free = None
        supported = (not RA.LEGACY) and (fam in ("overhang", "compound") or (fam == "null" and P["decoy"] == "high"))
        if not RA.LEGACY and (fam in ("step", "compound", "gap") or supported):
            lat_free = self.free_lat(P0, d, n, fz, (s0, s0 + 0.5, s0 + 1.0))
        if hasattr(self.B.pool, "lat_free"):
            self.B.pool.lat_free = lat_free if fam in ("step", "compound") else None
        self.world.stop()
        self.OBJ.park_all(); self.B.park()
        if self.SUP is not None:
            self.SUP.park()
        geo = dict(P0=P0, yaw=yaw, d=d, n=n, fz=fz, s_front=s0, z_top=None, u_track=False, value=None)
        boxes, oh = [], None
        col = tuple(P.get("color", (0.45, 0.36, 0.28)))
        sup_log = None
        if fam in ("overhang", "compound") or (fam == "null" and P["decoy"] in ("high", "beside_obj")):
            o_ = P["obj"]
            if fam == "null" and P["decoy"] == "beside_obj" and not RA.LEGACY:
                o_ = dict(o_, u=0.0, roll=0.0)              # no floating objects: the decoy lies on the floor beside the lane
            oh = self._place_overhang(o_, P0, yaw, fz, s0, float(P.get("lat", 0.0)))
            if supported:
                sup_log = self._support_overhang(P0, yaw, d, n, fz, s0, oh["s_end"], lat_free)
        if fam == "overhang":
            geo.update(s_end=oh["s_end"], s_goal=oh["s_end"] + 0.5, u_track=True)
        elif fam == "compound":
            s_c = oh["s_end"] + float(P["curb_gap"])
            goal = s_c + 0.25 + 0.7
            if goal > L - 0.1:
                s_c = L - 0.1 - 0.95
                if s_c - oh["s_end"] < 0.4:
                    raise Reject("compound_too_long")
            boxes = [dict(kind="box", s0=s_c, s1=s_c + 0.25, l0=-1.2, l1=1.2, z0=-0.05, z1=float(P["curb_h"]))]
            geo.update(s_end=s_c + 0.25, s_goal=s_c + 0.95, u_track=True, s_curb=round(s_c, 3), oh_end=oh["s_end"])
        elif fam == "step":
            k = P["kind"]
            if k == "ledge":
                boxes = [dict(kind="box", s0=s0, s1=s0 + 3.0, l0=P["lc"] - P["width"] / 2, l1=P["lc"] + P["width"] / 2, z0=-0.05, z1=P["h"])]
                geo.update(s_end=s0 + 3.0, s_goal=s0 + 1.3, on_top=True)
            elif k == "curb":
                boxes = [dict(kind="box", s0=s0, s1=s0 + 0.25, l0=P["lc"] - P["width"] / 2, l1=P["lc"] + P["width"] / 2, z0=-0.05, z1=P["h"])]
                geo.update(s_end=s0 + 0.25, s_goal=s0 + 1.25, on_top=False)
            elif k == "steps":
                nn, tr, r = int(P["n"]), float(P["tread"]), float(P["h"])
                boxes = [dict(kind="box", s0=s0 + i * tr, s1=s0 + nn * tr + 2.0, l0=-1.1, l1=1.1, z0=-0.05, z1=(i + 1) * r) for i in range(nn)]
                g = min(s0 + nn * tr + 0.8, L - 0.15)
                if g < s0 + nn * tr + 0.35:
                    raise Reject("steps_too_long")
                geo.update(s_end=s0 + nn * tr, s_goal=g, on_top=True)
            else:                                          # stacked school mats (copies)
                dep = 2.0 if P["short"] else 2.93
                boxes = [dict(kind="mat", src=P["srcs"][i], s0=s0 + (P["jit"][i][0] if i else 0.0),
                              l=P["lat"] + (P["jit"][i][1] if i else 0.0), layer=i, short=P["short"], yaw=0.0)
                         for i in range(int(P["layers"]))]
                geo.update(s_end=s0 + dep, s_goal=s0 + min(1.45, dep - 0.45), on_top=True)
        elif fam == "gap":
            w, Lg, e = float(P["w"]), float(P["L"]), float(P["e"])
            lo_, hi_ = 2.6, 2.6
            if lat_free is not None:                        # real furniture stops at the corridor walls (<= 12 cm into them)
                lo_, hi_ = min(2.6, lat_free[1] + 0.04), min(2.6, lat_free[0] + 0.04)
                if hi_ - (e + w / 2) < 0.15 or lo_ + (e - w / 2) < 0.15:
                    raise Reject(f"gap_wider_than_corridor {lat_free}")
            boxes = [dict(kind="box", s0=s0, s1=s0 + Lg, l0=e + w / 2, l1=hi_, z0=-0.05, z1=1.6),
                     dict(kind="box", s0=s0, s1=s0 + Lg, l0=-lo_, l1=e - w / 2, z0=-0.05, z1=1.6)]
            geo.update(s_end=s0 + Lg, s_goal=s0 + Lg + 0.8)
        elif fam == "null":
            if P["decoy"] == "beside_box":
                boxes = [dict(kind="box", s0=s0, s1=s0 + 1.5, l0=P["lc"] - 0.4, l1=P["lc"] + 0.4, z0=-0.05, z1=P["h"])]
            geo.update(s_end=s0 + 0.6, s_goal=s0 + 1.5)
        if geo["s_goal"] > L + 0.3:
            raise Reject("goal_off_lane")
        rot = float(P.get("rot", 0.0)) if fam == "step" else 0.0
        if boxes:
            self.B.build(boxes, P0, yaw, fz, rot=rot, piv_s=s0, color=col)   # restarts physics itself
            bx = [b for b in boxes if b["kind"] == "box"]
            if bx and P.get("mat") is not None and RA.LEGACY:     # real assets carry their own materials
                do = np.array([math.cos(yaw + rot), math.sin(yaw + rot)])
                self.B.pool.place(bx, P0 + d * s0 - do * s0, yaw + rot, fz, color=col, mat=int(P["mat"]))
            self._restart()
        else:
            self._restart()
        geo["rot"] = rot
        # ---------------- SUPPORTED check (no floating objects): the raised object rests on real supports on both sides
        if sup_log is not None:
            chk_sup = self._check_supported(P0, d, n, fz, sup_log, s0, oh["s_end"])
            if not chk_sup["ok"]:
                raise Reject(f"unsupported {chk_sup}")
            geo["supports"] = dict(sup_log, check=chk_sup)
        # ---------------- no-primitive check: every Gprim we created must be a mesh of a referenced real asset
        npc = RA.audit_no_primitives(self.stage, OURS)
        if not npc["ok"]:
            raise Reject(f"primitive_geometry {npc['bad'][:3]}")
        built = list(getattr(self.B.pool, "log", []) or []) if boxes else []
        if fam == "gap" and built and not RA.LEGACY:        # passage length = depth of the two pieces at the gap
            fl = [b for b in built if b.get("kind") == "furniture_line"]
            if len(fl) == 2:
                Lg_b = min(fl[0]["pieces"][0][3], fl[1]["pieces"][0][3])
                geo.update(s_end=s0 + Lg_b, s_goal=s0 + Lg_b + 0.8)
                P = dict(P, L=round(Lg_b, 3))
        chk = dict(lane=self.lane_check(P0, d, n, fz, min(geo["s_goal"] + 0.3, L)), supports=geo.get("supports"),
                   no_primitive=dict(n_gprims=npc["n_gprims"], ok=npc["ok"]), built=[{k: v for k, v in b.items() if k != "recs"} for b in built],
                   lat_free=lat_free)
        if fam in ("overhang", "compound") or (fam == "null" and oh is not None):
            S, U, hits, n_r = self.clearance(P0, d, n, fz, s0 - 0.3, oh["s_end"] + 0.3)
            m = (S >= s0 - 0.05) & (S <= oh["s_end"] + 0.05)
            u_min = float(U[m].min()) if m.any() else 9.9
            chk.update(oh=dict(u_target=P["obj"]["u"], u_min=round(u_min, 3), obj_hits=hits, n_rays=n_r, depth=oh["depth"],
                               width=oh["width"], name=oh["name"], s_end=oh["s_end"]))
            geo["S"], geo["U"] = S, U
            if fam != "null":
                geo["value"] = round(u_min, 3)
                if hits == 0:
                    raise Reject("overhang_not_over_lane")
                # u target = the rotated bbox bottom, so the lowest point under the lane can only be HIGHER (slant / yaw /
                # roll / lateral offset); lower than the target = a collider where no geometry was intended
                tol_lo = 0.03 if P["obj"]["cls"] == "prim" else 0.06
                if u_min < P["obj"]["u"] - tol_lo or u_min > P["obj"]["u"] + 0.30:
                    raise Reject(f"overhang_u_mismatch {u_min:.3f} vs {P['obj']['u']:.3f}")
            elif u_min < 1.9 and P["decoy"] != "beside_obj":
                raise Reject(f"decoy_low {u_min:.2f}")
            elif P["decoy"] == "beside_obj" and hits > 0 and not RA.LEGACY:
                raise Reject("decoy_in_lane")
        if fam in ("step", "compound", "gap") or (fam == "null" and P["decoy"] == "beside_box"):
            prof = self.B.profile(P0, yaw, fz, s0 - 0.3, min(geo["s_goal"] + 0.4, L + 0.3), lats=(-0.2, 0.0, 0.2), ds=0.05)
            hv = lambda r: max([v_ for v_ in r[1:] if v_ is not None] or [0.0])
            if fam == "step":
                win = 0.15 if P["kind"] == "steps" else 0.35          # steps: only the first riser
                edge = max([hv(r) for r in prof if s0 - 0.1 <= r[0] <= s0 + win] or [0.0])
                tg = [r for r in prof if abs(r[0] - geo["s_goal"]) < 0.03]
                top_goal = float(np.mean([v_ for v_ in tg[0][1:] if v_ is not None])) if tg else 0.0
                chk["step"] = dict(edge=round(edge, 3), top_goal=round(top_goal, 3), kind=P["kind"])
                geo["value"] = round(edge, 3)
                if geo["on_top"]:
                    if top_goal < 0.02:
                        raise Reject(f"step_top_missing {top_goal:.3f}")
                    geo["z_top"] = fz + top_goal
                if P["kind"] in ("ledge", "curb", "steps") and abs(rot) < 1e-6 and abs(edge - P["h"]) > 0.03:
                    raise Reject(f"step_edge_mismatch {edge:.3f} vs {P['h']:.3f}")
                if P["kind"] == "mats" and edge < 0.04:
                    raise Reject("mats_missing")
            if fam == "compound":
                c = [hv(r) for r in prof if geo["s_curb"] <= r[0] <= geo["s_curb"] + 0.25]
                chk["curb"] = dict(h=round(max(c or [0.0]), 3), target=P["curb_h"])
                if abs(max(c or [0.0]) - P["curb_h"]) > 0.03:
                    raise Reject("curb_mismatch")
            if fam == "null":
                inlane = [hv(r) for r in prof]
                if max(inlane or [0.0]) > 0.03:
                    raise Reject("decoy_box_in_lane")
        if fam == "gap" and not RA.LEGACY and len(getattr(self.B.pool, "lines", [])) == 2:
            # real furniture: side panels are not exactly at the bounding box (handles, cornices) -> measure each gap
            # edge by rays and move that row so the edge is exactly where the spec puts it, then restart physics
            w_, Lg_, e_ = float(P["w"]), float(P["L"]), float(P["e"])
            moved = []
            for ln in self.B.pool.lines:
                u3 = np.r_[ln["u"], 0.0]; ds = []
                for z in (0.25, 0.5, 0.9, 1.3):
                    for s in np.linspace(s0 + max(0.10, 0.3 * Lg_), s0 + Lg_ - 0.05, 3):
                        q = P0 + d * s + n * e_
                        dist, who, _ = self.ray((q[0], q[1], fz + z), u3, 3.0)
                        if dist is not None and who.startswith(OURS):
                            ds.append(dist)
                if ds:
                    err = min(ds) - w_ / 2
                    if abs(err) > 0.003:
                        self.B.pool.lib.shift(ln["recs"], -np.asarray(ln["u"]) * err); moved.append(round(err, 4))
            if moved:
                self._restart()
            chk["gap_edge_correction"] = moved
        if fam == "gap":
            w, Lg, e = float(P["w"]), float(P["L"]), float(P["e"])
            ws = []
            s_lo = s0 + (max(0.10, 0.3 * Lg) if not RA.LEGACY else 0.05)   # real furniture: doors/frames recess the front
            for z in ((0.1, 0.5, 0.9, 1.3) if RA.LEGACY else (0.25, 0.5, 0.9, 1.3)):     # real lockers stand on feet
                for s in np.linspace(s_lo, s0 + Lg - 0.05, 3):
                    q = P0 + d * s + n * e
                    a, wa, _ = self.ray((q[0], q[1], fz + z), (n[0], n[1], 0.0), 3.0)
                    b, wb, _ = self.ray((q[0], q[1], fz + z), (-n[0], -n[1], 0.0), 3.0)
                    if a is not None and b is not None and wa.startswith(OURS) and wb.startswith(OURS):
                        ws.append(a + b)
                        if abs(a + b - w) > 0.03:
                            chk.setdefault("gap_bad_rays", []).append((z, round(float(s - s0), 2), round(a + b, 3), wa[-45:], wb[-45:]))
            chk["gap"] = dict(w_target=w, w_min=round(min(ws), 3) if ws else None, w_max=round(max(ws), 3) if ws else None, n=len(ws))
            tol_hi = 0.03 if RA.LEGACY else 0.06               # real side panels are not perfectly flat: the spec width
            if len(ws) < 10 or abs(min(ws) - w) > 0.03 or max(ws) - w > tol_hi or w - max(ws) > 0.03:   # = narrowest
                raise Reject(f"gap_width_mismatch {chk['gap']} {chk.get('gap_bad_rays', [])[:4]} built {[b.get('pieces') for b in chk.get('built', [])]}")
            geo["value"] = round(min(ws), 3)
        if fam == "null":
            geo["value"] = 0.0
        if chk["lane"]["n_clutter"] > 0:
            raise Reject(f"lane_clutter {chk['lane']['clutter'][:2]}")
        if chk["lane"]["n_low_head"] > 0 and fam not in ("overhang", "compound"):
            raise Reject(f"lane_low_headroom {chk['lane']['low_head'][:2]}")
        geo["checks"] = chk
        geo["hazard_band"] = (s0 - 1.0, geo["s_end"] + 0.3)
        # where the AUDIT view looks (lane frame s, l, z above the floor)
        if oh is not None and fam != "null":
            geo["aim"] = (s0 + 0.5 * oh["depth"], float(P.get("lat", 0.0)), geo["value"] + 0.5 * min(oh["thick"], 0.3))
        elif oh is not None:
            geo["aim"] = (s0 + 0.5 * oh["depth"], float(P.get("lat", 0.0)), float(P["obj"]["u"]) + 0.1)
        elif fam == "step":
            geo["aim"] = (s0 + 0.15, 0.0, 0.5 * max(geo["value"], 0.04))
        elif fam == "gap":
            geo["aim"] = (s0, float(P["e"]), 0.6)
        elif fam == "null" and boxes:
            geo["aim"] = (s0 + 0.75, float(P["lc"]), 0.5 * float(P["h"]))
        return geo

    # ------------------------------------------------------------------ camera + visible-pixel check
    def _mount_init(self, stage, rep, rp, kind, spec):
        """v4: shared decision-camera mount (isaac_sim/camera_mount.py) for this harness' /World/HeadCam"""
        _cm = os.path.join(ROOT, "isaac_sim")
        if _cm not in sys.path:
            sys.path.insert(0, _cm)
        import camera_mount as CMT
        import scene_lighting as _SL
        self.CMT = CMT
        self.mcam = None if CMT.is_legacy(kind) else CMT.MountedCamera(stage, self.rob, kind, root=self.root, spec=spec, rp=rp, rep=rep)
        self.snap_info = None
        _SL.HEAD_MOUNT_FN = self.mcam.light_matrix if self.mcam is not None else None

    def _cam_pose(self, bs):
        fwd, up, pitch = self.spec.cam
        c, s = math.cos(bs["yaw"]), math.sin(bs["yaw"])
        eye = np.asarray(bs["pos"], float) + np.array([fwd * c, fwd * s, up])
        tgt = eye + np.array([c * math.cos(pitch), s * math.cos(pitch), math.sin(pitch)])
        return eye, tgt

    def snapshot(self, bs):
        if getattr(self, "mcam", None) is not None:          # v4 shared mount (camera_mount.py)
            self.LIGHT.follow(bs, self.spec.cam)
            out = self.mcam.render(self.cam_xf, self.rep, self.ann_rgb, self.ann_dep, bs)
            if out is None:
                return None
            rgb, dep, M, info = out
            self.snap_info = {k: v for k, v in info.items() if k != "mask"}
            return rgb, dep, np.array(M, float)
        self.LIGHT.follow(bs, self.spec.cam)
        eye, tgt = self._cam_pose(bs)
        Gf = self.Gf
        M = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse()
        self.cam_xf.Set(M)
        self.rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        rgb, dep = np.asarray(self.ann_rgb.get_data()), np.asarray(self.ann_dep.get_data())
        if rgb.ndim != 3 or dep.ndim != 2:
            return None
        return rgb[..., :3].astype(np.uint8), np.nan_to_num(dep.astype(np.float32), posinf=60.0), np.array(M, float)

    def visible(self, M, dep, step=8):
        """PhysX rays through a pixel grid from the camera: pixels whose first hit is OUR hazard, and how many of them
        the RENDERED depth agrees with (|depth - ray depth| < 5 cm + 3 %): catches invisible copies / phantom colliders"""
        eye = M[3, :3]
        n_h, n_ok, n_all = 0, 0, 0
        for v in range(step // 2, 224, step):
            for u in range(step // 2, 224, step):
                dc = np.array([(u + 0.5 - 112) / FX, -(v + 0.5 - 112) / FX, -1.0, 0.0])
                dw = (dc @ M)[:3]; nrm = np.linalg.norm(dw); dw = dw / nrm
                dist, who, _ = self.ray(eye, dw, 12.0)
                n_all += 1
                if dist is None or not who.startswith(OURS):
                    continue
                n_h += 1
                z_ray = dist / nrm                        # distance to the image plane
                if abs(float(dep[v, u]) - z_ray) < 0.05 + 0.03 * z_ray:
                    n_ok += 1
        return dict(n_haz=n_h, n_agree=n_ok, n_grid=n_all)

    def check_view(self, geo, dist=2.5, h=1.0):
        """AUDIT view (not a model input): the head camera prim placed dist m before the front edge, h m above the floor,
        AIMED at the hazard (geo['aim']); visible-pixel + rendered-depth agreement on the hazard. Verifies that what the
        colliders say is there is also rendered there, even for hazards outside the head camera's field of view."""
        if geo.get("aim") is None:
            return None
        P0, d, n, fz = geo["P0"], geo["d"], geo["n"], geo["fz"]
        sa, la, za = geo["aim"]
        q = P0 + d * max(0.1, geo["s_front"] - dist)
        eye = np.array([q[0], q[1], fz + h]); tgt = np.r_[P0 + d * sa + n * la, fz + za]
        Gf = self.Gf
        M = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse()
        self.cam_xf.Set(M)
        self.rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        rgb, dep = np.asarray(self.ann_rgb.get_data()), np.asarray(self.ann_dep.get_data())
        if rgb.ndim != 3 or dep.ndim != 2:
            return None
        dep = np.nan_to_num(dep.astype(np.float32), posinf=60.0)
        return dict(self.visible(np.array(M, float), dep), mean_rgb=round(float(rgb[..., :3].mean()), 1), dist=dist)

    # ------------------------------------------------------------------ state save / restore (branching)
    _SKIP = ("_art", "robot", "policy", "RB", "AM", "torch", "spec", "C", "kn_q", "kn_tau", "ctrl")

    @staticmethod
    def _plain(v):
        return isinstance(v, (np.ndarray, list, dict, deque, tuple, int, float, bool, str, type(None), np.generic))

    def save_state(self):
        a = self.rob._art
        st = dict(pose=[np.array(x, float) for x in a.get_world_pose()], lin=np.array(a.get_linear_velocity(), float),
                  ang=np.array(a.get_angular_velocity(), float), q=np.array(a.get_joint_positions(), float),
                  dq=np.array(a.get_joint_velocities(), float))
        st["rob"] = {k: copy.deepcopy(v) for k, v in vars(self.rob).items() if k not in self._SKIP and self._plain(v)}
        C = getattr(self.rob, "C", None)              # G1/AMO controller object: history buffers, gait clock, last action
        if C is not None:
            st["C"] = {k: copy.deepcopy(v) for k, v in vars(C).items() if self._plain(v)}
        return st

    def restore_state(self, st):
        a = self.rob._art
        for k, v in st["rob"].items():
            setattr(self.rob, k, copy.deepcopy(v))
        if "C" in st:
            for k, v in st["C"].items():
                setattr(self.rob.C, k, copy.deepcopy(v))
        a.set_world_pose(st["pose"][0], st["pose"][1])
        a.set_linear_velocity(st["lin"]); a.set_angular_velocity(st["ang"])
        a.set_joint_positions(st["q"]); a.set_joint_velocities(st["dq"])

    # ------------------------------------------------------------------ lane driving
    def lane_coords(self, bs, geo):
        p = np.asarray(bs["pos"][:2], float) - geo["P0"]
        return float(p @ geo["d"]), float(p @ geo["n"])

    def goal_reached(self, bs, geo, s, e):
        if s < geo["s_goal"] or abs(e) > 0.5:
            return None
        if geo.get("z_top") is not None and float(bs["pos"][2]) - geo["z_top"] < 0.25:
            return "crossed_low"
        return "crossed"

    def u_of(self, geo, s):
        S, U = geo.get("S"), geo.get("U")
        if S is None or not (S[0] - 0.05 <= s <= S[-1] + 0.05):
            return None
        return float(U[np.argmin(np.abs(S - s))])

    def drive(self, geo, v, stop_fn, t_max, method=None, track=True, vid=None):
        """lane-following straight drive at v (or a method plugin's twist). Labels + cost signals."""
        HZ, EVERY, rob = self.HZ, self.EVERY, self.rob
        best, t_best, maxtilt, energy, min_clear, top_max = -1e9, 0.0, 0.0, 0.0, 9.9, None
        goal_xy = geo["P0"] + geo["d"] * geo["s_goal"]
        idx = getattr(rob, "idx", None)
        s = e = 0.0; t = 0.0; bs = None; k = 0
        for step in range(int(t_max * HZ)):
            t = step / HZ
            self.world.step(render=False); self.buf.tick(rob)
            if step % EVERY:
                continue
            k += 1
            bs = rob.base_state()
            if bs is None or rob.fallen(bs):
                return dict(reason="fell", t=t, s=s, e=e, max_tilt=maxtilt, energy=energy, min_clear=min_clear, top=top_max, bs=bs)
            maxtilt = max(maxtilt, float(bs["tilt"]))
            s, e = self.lane_coords(bs, geo)
            tau = getattr(rob, "last_tau", None)
            if tau is not None:
                dq = np.asarray(rob._art.get_joint_velocities(), float).reshape(-1)
                if idx is not None and len(tau) == len(idx):
                    dq = dq[idx]
                if len(dq) == len(tau):
                    energy += float(np.abs(np.asarray(tau) * dq).sum()) * EVERY / HZ
            if method is None:
                err = (geo["yaw"] - bs["yaw"] + math.pi) % (2 * math.pi) - math.pi
                rob.command([v, 0.0, 0.0, float(np.clip(2.0 * err - 1.5 * e, -1.0, 1.0))])
            else:
                x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
                dd = goal_xy - np.array([x, y]); c, s_ = math.cos(yaw), math.sin(yaw)
                obs = dict(t=t, dt=EVERY / HZ, bs=bs, rob=rob, goal_rel=np.array([c * dd[0] + s_ * dd[1], -s_ * dd[0] + c * dd[1]]),
                           goal_xy=goal_xy.copy(), goal_dist=float(np.hypot(*dd)))
                self.LIGHT.follow(bs, self.spec.cam)
                r = method.act(obs)
                rob.command(np.asarray(r["twist"], float))
            if s > best + 0.10:
                best, t_best = s, t
            if track and geo.get("u_track") and k % 5 == 0 and geo["s_front"] - 0.7 <= s <= geo["s_end"] + 0.7:
                top = self.robot_top() - geo["fz"]
                u = self.u_of(geo, s)
                top_max = top if top_max is None else max(top_max, top)
                if u is not None:
                    min_clear = min(min_clear, u - top)
            if vid is not None and k % 6 == 0:
                self.side_frame(vid, bs, geo)
            r_ = stop_fn(s, e, t, (t - t_best) > self.STALL_S, bs)
            if r_:
                return dict(reason=r_, t=t, s=s, e=e, max_tilt=maxtilt, energy=energy, min_clear=min_clear, top=top_max, bs=bs)
        return dict(reason="timeout", t=t_max, s=s, e=e, max_tilt=maxtilt, energy=energy, min_clear=min_clear, top=top_max, bs=bs)

    def side_frame(self, vid, bs, geo):
        if self.side is None:
            return
        import cv2
        P0, d, n, fz = geo["P0"], geo["d"], geo["n"], geo["fz"]
        s_r = float((np.asarray(bs["pos"][:2]) - P0) @ d)
        eye = np.r_[P0 + d * (s_r - 2.2) + n * 0.45, fz + 1.4]
        tgt = np.r_[P0 + d * (s_r + 1.4), fz + 0.4]
        Gf = self.Gf
        self.side_xf.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse())
        self.AUD_XF.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, np.r_[P0 + d * (s_r - 1.0), fz + 1.7]))))
        self.AUD.GetIntensityAttr().Set(6.0e4)
        self.rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
        self.AUD.GetIntensityAttr().Set(0.0)
        im = np.asarray(self.side.get_data())
        if im.ndim == 3 and im.size:
            cv2.imwrite(os.path.join(vid["dir"], f"{vid['k']:05d}.jpg"), np.ascontiguousarray(im[..., 2::-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])
            vid["k"] += 1

    # ------------------------------------------------------------------ the decision point
    def set_lighting(self, sit):
        Lt = sit["lighting"]
        info = self.LIGHT.set(Lt["mode"], np.random.default_rng(sit["seed"] + 991))
        if Lt.get("dark") and Lt["mode"].startswith("night"):
            for p, a, i0, is_sun in self.LIGHT.base:
                a.Set(0.0)
            info = dict(info, room=0.0, dark=True)
        return info

    def spawn_s(self, sit):
        """spawn distance along the lane: Spot from the lane start (>= 2 s of driving before the decision, like the
        closed-loop arms); G1 (slow) 0.6 m before the decision point"""
        if self.kind.startswith("g1"):
            return max(0.0, sit["s_front"] - sit["d_dec"] - 0.6)
        return 0.2

    def decision(self, sit, geo, light):
        P0, d, n, fz = geo["P0"], geo["d"], geo["n"], geo["fz"]
        s_sp = self.spawn_s(sit)
        q0 = P0 + d * s_sp + n * float(sit["lat0"])
        vis_check = None
        if geo.get("aim") is not None:
            # check view with the robot parked out of the way (30 m below the lane) so it cannot occlude the hazard
            self.rob.teleport(q0[0], q0[1], geo["yaw"], fz - 30.0)
            self.world.step(render=False)
            low = geo.get("family") in ("overhang", "compound") or (geo.get("family") == "null" and geo["aim"][2] > 0.6)
            vis_check = self.check_view(geo, 1.6, h=0.25) if low else self.check_view(geo, 2.5)
        self.settle(q0[0], q0[1], geo["yaw"] + float(sit["yaw0"]), fz, 1.0 if not self.kind.startswith("g1") else 2.0)
        bs = self.rob.base_state()
        if bs is None or self.rob.fallen(bs):
            raise Reject("bad_spawn")
        self.stand_z = float(bs["pos"][2]) - fz
        s_dec_t = geo["s_front"] - float(sit["d_dec"])
        stop = lambda s, e, t, st, bs_: ("dec" if (s >= s_dec_t and self.buf.full()) else ("stall" if st else None))
        a = self.drive(geo, self.v_nom, stop, 30.0, track=False)
        if a["reason"] != "dec":
            raise Reject(f"approach_{a['reason']}")
        bs = a["bs"]
        snap = self.snapshot(bs)
        if snap is None:
            raise Reject("render_failed")
        relit = None
        if float(snap[0].mean()) < 10.0:                  # near-black: re-light (no effect on physics), as the collectors
            src = "dark" if light.get("dark") else light["mode"]
            if not light["mode"].startswith("night"):
                light = dict(self.LIGHT.set("day", self.rng), relit_from=src); snap = self.snapshot(bs)
            if snap is not None and float(snap[0].mean()) < 10.0:
                m_ = light["mode"] if light["mode"].startswith("night") else "night_white"
                light = dict(self.LIGHT.set(m_, self.rng), dark=False, relit_from=src); snap = self.snapshot(bs)
            relit = light.get("relit_from")
        if snap is None or float(snap[0].mean()) < 10.0:
            raise Reject("near_black_snapshot")
        vis = self.visible(snap[2], snap[1])
        pb, pj = self.buf.arrays()
        s_dec, e_dec = self.lane_coords(bs, geo)
        st = self.save_state()
        return dict(rgb=snap[0], depth=snap[1], cam=snap[2], pb=pb, pj=pj, s_dec=s_dec, e_dec=e_dec, state=st, light=light, vis_check=vis_check,
                    vis=vis, relit=relit, mean_rgb=round(float(snap[0].mean()), 1), t_app=a["t"])

    # ------------------------------------------------------------------ one branch
    def t_max(self, geo, s_from):
        return float(min(self.T_CAP, max(10.0, (geo["s_goal"] - s_from) / self.V_FLOOR + 6.0)))

    def outcome(self, res, geo, s_from):
        fell = res["reason"] == "fell"; success = res["reason"] == "crossed"
        stall = res["reason"] in ("stuck", "timeout")
        s_fin = res.get("s", s_from)
        a, b = geo["hazard_band"]
        collision = bool((stall and a <= s_fin <= b and geo.get("family") != "null") or res["min_clear"] < -0.03)
        progress = float(np.clip((s_fin - s_from) / max(geo["s_goal"] - s_from, 1e-3), -1.0, 1.0))
        return dict(success=success, fell=fell, stall=stall, collision=collision, progress=round(progress, 3),
                    reason=res["reason"], t=round(res["t"], 2), max_tilt=round(res["max_tilt"], 3),
                    tilt_margin=round(1.0 - res["max_tilt"] / self.spec.fall_tilt, 3),
                    energy=round(res["energy"], 1), energy_per_m=round(res["energy"] / max(s_fin - s_from, 0.1), 1),
                    min_clear=round(res["min_clear"], 3) if res["min_clear"] < 9 else None,
                    top_max=round(res["top"], 3) if res["top"] is not None else None, s_final=round(s_fin, 3),
                    e_final=round(res.get("e", 0.0), 3))

    def rollout(self, geo, dec, sp, vid=None):
        """restore the decision state, hold setpoint sp = dict(v, body_height, pitch, roll) to the goal"""
        self.restore_state(dec["state"])
        if self.NOM is not None and "body_height" in self.RANGES:
            self.rob.command(dict(body_height=sp["body_height"], pitch=sp["pitch"], roll=sp["roll"], vx=sp["v"]))
        goal = lambda s, e, t, st, bs_: (self.goal_reached(bs_, geo, s, e) or ("stuck" if st else None))
        res = self.drive(geo, sp["v"], goal, self.t_max(geo, dec["s_dec"]), vid=vid)
        self.rob.stop()
        out = self.outcome(res, geo, dec["s_dec"])
        out["t_goal"] = out["t"] if out["success"] else None
        out["posture_dev"] = round(self.posture_dev(sp), 3)
        return out

    def run_method(self, m, sit, geo, vid=None):
        """closed-loop arm from the spawn, episode_runner protocol (teleport, 2 s stand, method.reset, 50 Hz act)"""
        P0, d, n, fz = geo["P0"], geo["d"], geo["n"], geo["fz"]
        s_sp = self.spawn_s(sit)
        q0 = P0 + d * s_sp + n * float(sit["lat0"])
        self.settle(q0[0], q0[1], geo["yaw"] + float(sit["yaw0"]), fz, 2.0)
        self.rob.command(np.zeros(4))
        m.reset(dict(speed_frac=0.6, floor_z0=fz, scene="gen", trial=0))
        goal = lambda s, e, t, st, bs_: (self.goal_reached(bs_, geo, s, e) or ("stuck" if st else None))
        res = self.drive(geo, None, goal, self.t_max(geo, s_sp), method=m, vid=vid)
        self.rob.stop()
        if hasattr(self.rob, "reset_setpoints"):
            self.rob.reset_setpoints()
        out = self.outcome(res, geo, s_sp)
        out["t_goal"] = out["t"] if out["success"] else None
        log = getattr(m, "log", [])
        out["decisions"] = log
        out["n_dec"] = len(log)
        if log and "dh" in log[0]:
            dev = [self.posture_dev(dict(body_height=(self.NOM or 0) + x["dh"], pitch=x["pitch"], roll=x["roll"])) for x in log]
            out["posture_dev"] = round(float(np.mean(dev)), 3)
            out["depart_frac"] = round(float(np.mean([abs(x["dh"]) > 0.03 or abs(x["pitch"]) > 0.05 or abs(x["roll"]) > 0.05 for x in log])), 3)
        return out

    def make_method(self, spec_str, env=None):
        """episode_runner-style plugin, built in THIS process (ctx as the runner's). The planner's depth sensor is
        not needed by the straight-drive arms, so its render product is not created (no behavioural difference)."""
        import importlib
        import local_nav as LN
        for k, v in (env or {}).items():
            os.environ[k] = str(v)
        mod, cls = spec_str.split(":")
        m = getattr(importlib.import_module(mod), cls)()
        DS = LN.DepthSensor
        LN.DepthSensor = lambda *a, **k: None
        try:
            m.setup(dict(stage=self.stage, rep=self.rep, world=self.world, app=self.app, rob=self.rob, spec=self.spec,
                         kind=self.kind, hz=self.HZ, pxq=self.pxq, carb=self.carb, robot_root=self.root))
        finally:
            LN.DepthSensor = DS
        return m
