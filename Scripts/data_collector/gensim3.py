"""
gen_v3 harness (import AFTER SimulationApp exists). analysis/GEN3_SPEC.md.

V3Mixin (MRO: V3Mixin, gensim_v2.V2Mixin, GenSimV1 | EnvSim):
  setup_v3(base_spec)       wrap the harness robot in a virtual.VRob proxy (identity = the real controller)
  use_controller(spec, w)   switch the controller the layer talks to (real or virtual wrapper of the same base)
  save_state/restore_state  + the adapter's policy state AND the wrapper's own state (latency queue, filters, history)
  link_extents()            oriented body extents in the base-yaw frame from per-link visual bounds + physics poses
  run_probe(lane)           the <= 30 s self-test (probe.plan_segments) on the lane start, hazard objects parked
  rollout_v3(geo, dec, prog) execute a driver program (constant / two_phase / detour / turn_side_turn) from the decision
GenSimV3 = school; make_envsim_v3() = Isaac environments / terrain.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import copy, math, os, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(os.path.dirname(THIS), "gen")
for _p in (THIS, GEN):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gensim as GS
import gensim_v1 as GV
import gensim_v2 as G2
import ctrl_spec as CS
import virtual as VC
import probe as PR

wrap = G2.wrap


class ProgramCtl3:
    """constant / two_phase / detour executor of a driver program (posture by spec names, speed by the driver's v_cmd)"""

    def __init__(self, sim, geo, dec, prog, act):
        self.G, self.geo, self.P, self.A = sim, geo, prog, act
        self.mac = prog["macro"]
        self.s_front, self.s_end = float(geo["s_front"]), float(geo["s_end"])
        self.s_dec = float(dec["s_dec"])
        self.phase = "A" if self.mac == "two_phase" else "B"
        self.t_switch = None
        self.posture = None
        self.L = max(float(prog.get("L_ramp", 1.0)), 0.3)
        self.s_out0 = self.s_dec + 0.1
        self.s_back0 = self.s_end + float(prog.get("s_back", 0.3))
        self.lat = float(prog.get("lat_off", 0.0))
        self.wz_max = float(prog.get("wz_max", 1.0))
        self.t_in = dict(A=0.0, B=0.0, L=0.0)
        self._t_last = None

    def _apply(self, ph):
        G, P, A = self.G, self.P, self.A
        if ph == "L":
            post = dict(G.driver.nominal_posture); v = float(P["phases"]["B"]["v_cmd"])
            G.rob.set_env_intent(A.get("v_b", 0.3), VC_UNC, VC_UNC, 0.0)
        else:
            p = P["phases"][ph]; post = dict(p["posture"]); v = float(p["v_cmd"])
            sfx = "a" if ph == "A" else "b"
            G.rob.set_env_intent(A.get(f"v_{sfx}", A.get("v_b", 0.3)), A.get(f"top_{sfx}", A.get("top_b", VC_UNC)),
                                 A.get(f"wid_{sfx}", A.get("wid_b", VC_UNC)), A.get(f"pitch_{sfx}", A.get("pitch_b", 0.0)))
        if post:
            G.rob.command(post)
        self.posture = post; self.v = v

    def e_ref(self, s):
        if self.mac != "detour":
            return 0.0, 0.0
        if s < self.s_back0:
            x = (s - self.s_out0) / self.L
            return self.lat * G2.ramp(x), (self.lat / self.L if 0.0 <= x <= 1.0 else 0.0)
        x = (s - self.s_back0) / self.L
        return self.lat * (1.0 - G2.ramp(x)), (-self.lat / self.L if 0.0 <= x <= 1.0 else 0.0)

    def act(self, obs):
        G, geo, P = self.G, self.geo, self.P
        bs = obs["bs"]; s, e = G.lane_coords(bs, geo)
        if self._t_last is not None:
            self.t_in[self.phase] = self.t_in.get(self.phase, 0.0) + (obs["t"] - self._t_last)
        self._t_last = obs["t"]
        if self.posture is None:
            self._apply(self.phase)
        if self.mac == "two_phase":
            if self.phase == "A" and s >= self.s_front - float(P["d_commit"]):
                self.phase = "B"; self.t_switch = obs["t"]; self._apply("B")
            if self.phase == "B" and float(P.get("d_level", 0.0)) > 0 and s >= self.s_front + float(P["d_level"]):
                self.phase = "L"; self._apply("L")
        er, slope = self.e_ref(s)
        href = geo["yaw"] + math.atan(slope)
        kl = 1.5
        yo = float(P.get("yaw_off", 0.0) or 0.0)
        if yo and self.phase == "B" and s < self.s_front + float(P.get("d_yaw", 0.8)):
            href += yo; kl = 0.0
        err = wrap(href - float(bs["yaw"]))
        wz = float(np.clip(2.0 * err - kl * (e - er), -self.wz_max, self.wz_max))
        return dict(twist=np.array([self.v, 0.0, 0.0, wz]))


VC_UNC = 9.9


class TurnSideCtl3:
    """turn_side_turn: approach (stock posture + speed), turn to lane - side*pi/2, walk sideways (vy) through the passage
    centring on e_gap, turn back, go. Turning / lateral walking use vx = turn_vx / +-lat_vx when the probe found that the
    controller needs forward speed for them (gensim_v1.SidestepCtl generalised)."""

    def __init__(self, sim, geo, dec, prog, act):
        self.G, self.geo, self.P, self.A = sim, geo, prog, act
        self.side = float(prog["side"]); self.e_gap = float(prog.get("e_gap", 0.0))
        self.s0, self.s1 = float(geo["s_front"]), float(geo["s_end"])
        self.d_trig = float(prog.get("d_commit", 0.6)); self.past = 0.45
        self.phase = "approach"; self.posture = None
        self.cap = float(prog.get("wz_max", 1.0))
        self.tvx = float(prog.get("turn_vx", 0.0)); self.lvx = float(prog.get("lat_vx", 0.0))
        S = prog["phases"]["S"]; self.vy = float(S.get("vy_cmd", 0.3)); self.S = S
        self.v_nom = float(sim.driver.v_nom)
        self.v_go = float(prog["phases"]["B"]["v_cmd"]) if "B" in prog["phases"] else self.v_nom
        self.t_switch = None

    def _post(self, post, intent):
        if post:
            self.G.rob.command(dict(post))
        self.posture = post
        self.G.rob.set_env_intent(*intent)

    def act(self, obs):
        G, geo = self.G, self.geo
        bs = obs["bs"]; yaw = float(bs["yaw"]); h = geo["yaw"]
        s, e = G.lane_coords(bs, geo)
        cap = self.cap
        if self.posture is None:
            self._post(dict(G.driver.nominal_posture), (self.v_nom, VC_UNC, VC_UNC, 0.0))
        if self.phase == "approach":
            if s >= self.s0 - self.d_trig - 0.05:
                self.phase = "turn"; self.t_switch = obs["t"]
                self._post(self.S["posture"], (0.0, self.A.get("top_b", VC_UNC), self.A.get("wid_b", VC_UNC), 0.0))
            else:
                err = wrap(h - yaw)
                return dict(twist=np.array([self.v_nom, 0.0, 0.0, float(np.clip(2.0 * err - 1.5 * (e - self.e_gap), -cap, cap))]))
        if self.P.get("pure_lateral"):                 # driver honours a probe that measured "turn does not work"
            return self._pure_lateral(obs, s, e, yaw, h, cap)
        target = h - self.side * math.pi / 2
        if self.phase == "turn":
            err = wrap(target - yaw)
            if abs(err) < math.radians(5):
                self.phase = "side"
            else:
                return dict(twist=np.array([self.tvx, 0.0, 0.0, float(np.clip(1.5 * err, -cap, cap))]))
        if self.phase == "side":
            if s >= self.s1 + self.past:
                self.phase = "back"; self._post(dict(G.driver.nominal_posture), (self.v_go, VC_UNC, VC_UNC, 0.0))
            else:
                err = wrap(target - yaw)
                want = -self.side * 1.5 * (self.e_gap - e)
                if self.lvx > 0:
                    vx = math.copysign(self.lvx, want) if abs(self.e_gap - e) > 0.03 else (self.lvx if int(obs["t"]) % 2 == 0 else -self.lvx)
                else:
                    vx = float(np.clip(want, -0.2, 0.2))
                return dict(twist=np.array([vx, self.side * self.vy, 0.0, float(np.clip(1.5 * err, -cap, cap))]))
        if self.phase == "back":
            err = wrap(h - yaw)
            if abs(err) < math.radians(8):
                self.phase = "go"
            else:
                return dict(twist=np.array([self.tvx, 0.0, 0.0, float(np.clip(1.5 * err, -cap, cap))]))
        err = wrap(h - yaw)
        return dict(twist=np.array([self.v_go, 0.0, 0.0, float(np.clip(2.0 * err - 1.5 * e, -cap, cap))]))


    def _pure_lateral(self, obs, s, e, yaw, h, cap):
        """no turns: hold the lane heading, walk sideways (vy only, measured lateral speed) onto the gap line e_gap,
        then go straight through at v_go (XFEAS_GEN3_DRV_PROBE=1, turn measured not working)"""
        err = wrap(h - yaw)
        if self.phase in ("turn", "side"):
            self.phase = "side"
            de = self.e_gap - e
            if abs(de) > 0.03 and s < self.s1 + self.past:
                return dict(twist=np.array([self.lvx, math.copysign(self.vy, de), 0.0, float(np.clip(1.5 * err, -cap, cap))]))
            self.phase = "go"; self._post(dict(self.G.driver.nominal_posture), (self.v_go, VC_UNC, VC_UNC, 0.0))
        return dict(twist=np.array([self.v_go, 0.0, 0.0, float(np.clip(2.0 * err - 1.5 * (e - self.e_gap), -cap, cap))]))


class V3Mixin:
    v3 = True

    # ------------------------------------------------------------------ setup
    def setup_v3(self, base_spec):
        if isinstance(self.rob, VC.VRob):
            return
        inner = self.rob
        self.rob = VC.VRob(inner, base_spec, top_fn=self._top_now)
        self.rob._v_foot_fn = self._foot_dz
        self.base_spec = base_spec
        self.v_nom = float(base_spec.get("nominal", {}).get("speed", 0.3))
        self._links = None
        self.driver = None

    def use_controller(self, spec, wcfg, probe=None):
        import driver as DR
        self.rob.set_controller(spec, wcfg)
        self.ctrl_spec = spec
        self.v_nom = float(spec.get("nominal", {}).get("speed", 0.3))
        self.driver = DR.Driver(spec, probe) if probe is not None else None

    # ------------------------------------------------------------------ body geometry (physics poses + visual bounds)
    def _links_init(self):
        """per physics body: a sub-sample (<= 400) of its visual mesh vertices in the LINK frame (constant), so that body extents
        follow the physics poses exactly (no USD xform sync needed, no box-corner inflation)"""
        from pxr import Usd, UsdGeom, UsdPhysics
        inner = self.rob._v_inner
        av = inner._art._articulation_view
        names = list(av.body_names)
        by = {}
        for p in Usd.PrimRange(self.stage.GetPrimAtPath(self.root)):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                by.setdefault(p.GetName(), p)
        rng = np.random.default_rng(0)
        pts_all = []
        tc = Usd.TimeCode.Default()
        for nm in names:
            pr = by.get(nm)
            if pr is None:
                pts_all.append(None); continue
            Linv = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(tc).GetInverse()
            P = []
            it = iter(Usd.PrimRange(pr, Usd.TraverseInstanceProxies()))
            for m in it:
                if m != pr and m.HasAPI(UsdPhysics.RigidBodyAPI):
                    it.PruneChildren() if hasattr(it, "PruneChildren") else None
                    continue
                if not m.IsA(UsdGeom.Mesh):
                    continue
                im = UsdGeom.Imageable(m)
                if im.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                    continue
                if im.ComputeVisibility(tc) == UsdGeom.Tokens.invisible:
                    continue
                v = UsdGeom.Mesh(m).GetPointsAttr().Get()
                if v is None or len(v) == 0:
                    continue
                M = np.array(UsdGeom.Xformable(m).ComputeLocalToWorldTransform(tc) * Linv, float)     # row-vector convention
                v = np.asarray(v, float)
                if len(v) > 4000:
                    v = v[rng.choice(len(v), 4000, replace=False)]
                P.append(v @ M[:3, :3] + M[3, :3])
            if not P:
                pts_all.append(None); continue
            P = np.concatenate(P, 0)
            if len(P) > 400:                       # keep the extreme points + a random sub-sample
                ext = np.unique(np.concatenate([np.argmin(P, 0), np.argmax(P, 0)]))
                P = np.concatenate([P[ext], P[rng.choice(len(P), 400 - len(ext), replace=False)]], 0)
            pts_all.append(P)
        self._links = dict(names=names, boxes=pts_all, nb=len(names))
        self._i_feet = [i for i, n in enumerate(names) if n.endswith("ankle_roll_link")]
        ti = [i for i, n in enumerate(names) if n == "torso_link"]
        self._i_torso = ti[0] if ti else 0
        print(f"[v3] link geometry: {sum(b is not None for b in pts_all)}/{len(names)} bodies with visual points "
              f"({sum(len(b) for b in pts_all if b is not None)} pts)", flush=True)

    def _link_T(self):
        inner = self.rob._v_inner
        pv = inner._art._articulation_view._physics_view
        return np.asarray(pv.get_link_transforms()).reshape(-1, self._links["nb"], 7)[0]

    def link_extents(self, bs=None):
        """-> dict(width, depth, top_abs, torso_lean): body extents in the base-yaw frame (visual bounds of every link)"""
        from scipy.spatial.transform import Rotation
        if self._links is None:
            self._links_init()
        bs = bs or self.rob.base_state()
        T = self._link_T()
        pts = []
        for i, B in enumerate(self._links["boxes"]):
            if B is None:
                continue
            R = Rotation.from_quat(T[i, 3:7]).as_matrix()
            pts.append(B @ R.T + T[i, :3])
        P = np.concatenate(pts, 0)
        c, s = math.cos(bs["yaw"]), math.sin(bs["yaw"])
        d = P[:, :2] - np.asarray(bs["pos"][:2])
        xb = c * d[:, 0] + s * d[:, 1]; yb = -s * d[:, 0] + c * d[:, 1]
        Rt = Rotation.from_quat(T[self._i_torso, 3:7]).as_matrix()
        lean = float(-math.asin(float(np.clip(Rt[2, 0], -1, 1))))
        return dict(width=float(yb.max() - yb.min()), depth=float(xb.max() - xb.min()), top_abs=float(P[:, 2].max()), torso_lean=lean)

    def _foot_dz(self):
        try:
            if self._links is None:
                self._links_init()
            if len(self._i_feet) != 2:
                return np.nan
            T = self._link_T()
            return float(abs(T[self._i_feet[0], 2] - T[self._i_feet[1], 2]))
        except Exception:
            return np.nan

    def _top_now(self):
        try:
            return self.link_extents()["top_abs"]
        except Exception:
            return np.nan

    # ------------------------------------------------------------------ branching state (adapter + wrapper)
    @staticmethod
    def _snap(obj):
        return {k: copy.deepcopy(v) for k, v in vars(obj).items() if k not in GS.GenSim._SKIP and GS.GenSim._plain(v)}

    def save_state(self):
        inner = self.rob._v_inner; a = inner._art
        st = dict(pose=[np.array(x, float) for x in a.get_world_pose()], lin=np.array(a.get_linear_velocity(), float),
                  ang=np.array(a.get_angular_velocity(), float), q=np.array(a.get_joint_positions(), float),
                  dq=np.array(a.get_joint_velocities(), float))
        st["rob"] = self._snap(inner)
        C = getattr(inner, "C", None)
        if C is not None:
            st["C"] = self._snap(C)
        st["vrob"] = self.rob.v_save()
        return st

    def restore_state(self, st):
        inner = self.rob._v_inner; a = inner._art
        for k, v in st["rob"].items():
            setattr(inner, k, copy.deepcopy(v))
        if "C" in st:
            for k, v in st["C"].items():
                setattr(inner.C, k, copy.deepcopy(v))
        a.set_world_pose(st["pose"][0], st["pose"][1])
        a.set_linear_velocity(st["lin"]); a.set_angular_velocity(st["ang"])
        a.set_joint_positions(st["q"]); a.set_joint_velocities(st["dq"])
        if hasattr(inner, "_apply") and getattr(inner, "q_tgt", None) is not None:
            inner._apply()                          # Isaac Lab bridge: PhysX drive targets are not part of vars()
        self.rob.v_restore(st["vrob"])

    # ------------------------------------------------------------------ scene parking (probe on an empty lane)
    def park_scene(self):
        self.world.stop()
        if hasattr(self, "_park_all"):
            self._park_all()
        else:
            self.OBJ.park_all(); self.B.park()
            if self.SUP is not None:
                self.SUP.park()
        self._restart()

    def floor_at(self, xy, fz_guess):
        o = np.array([xy[0], xy[1], fz_guess + 1.0]); acc = 0.0
        for _ in range(5):
            dist, who, _ = self.ray(o - np.array([0, 0, acc]), (0, 0, -1), 3.0 - acc)
            if dist is None:
                return fz_guess
            if who.startswith(self.root) or self.ours(who):
                acc += dist + 0.01; continue
            return float(fz_guess + 1.0 - acc - dist)
        return fz_guess

    # ------------------------------------------------------------------ PROBE
    def run_probe(self, lane, budget=PR.BUDGET_S, log=print):
        """GEN3_SPEC section 4. lane: a lane dict (start, yaw, len, floor). Returns the probe record (tests + summary + trace)."""
        spec = self.ctrl_spec
        P0, yaw0, d, n, fz0 = self.lane_frame(lane)
        L = float(lane.get("len", 8.0))
        s_start = 0.6
        fz = self.floor_at(P0 + d * 1.5, fz0)
        HZ, EVERY = self.HZ, self.EVERY
        rob = self.rob
        segs = PR.plan_segments(spec)
        names = {r: (CS.cmd(spec, r)["name"] if CS.has(spec, r) else None) for r in ("lin_vel_x", "lin_vel_y", "yaw_rate")}
        wz_cap = (CS.rng(spec, "yaw_rate") or (-1, 1))[1]
        clock = dict(t=0.0, falls=0)
        t_wall0 = time.time()
        geo = dict(P0=P0, d=d, n=n, yaw=yaw0, fz=fz)
        st = dict(heading=+1.0)

        def spawn():
            q0 = P0 + d * s_start
            self.settle(q0[0], q0[1], yaw0, fz, 1.0)
            clock["t"] += 1.0; st["heading"] = +1.0

        def vel_cmd(vx, vy, wz):
            c = {}
            if names["lin_vel_x"]:
                c[names["lin_vel_x"]] = float(vx)
            if names["lin_vel_y"]:
                c[names["lin_vel_y"]] = float(vy)
            if names["yaw_rate"]:
                c[names["yaw_rate"]] = float(np.clip(wz, -wz_cap, wz_cap))
            return c

        def hold(cmd, vx, vy, wz, dmin, dmax, settle_kind=False, measure=True):
            if cmd:
                rob.command(dict(cmd))
            samp = []
            t = 0.0; fell = False
            n_steps = int(round(dmax * HZ))
            for k in range(n_steps):
                self.world.step(render=False); self.buf.tick(rob)
                t = (k + 1) / HZ
                if (k + 1) % EVERY:
                    continue
                bs = rob.base_state()
                if bs is None or rob.fallen(bs):
                    fell = True; break
                s_, e_ = self.lane_coords(bs, geo)
                if wz is None:
                    href = yaw0 + (0.0 if st["heading"] > 0 else math.pi)
                    err = wrap(href - float(bs["yaw"]))
                    wz_ = 2.0 * err - 1.5 * (e_ if st["heading"] > 0 else -e_)
                else:
                    wz_ = wz
                rob.command(vel_cmd(vx, vy, wz_))
                if measure and (k + 1) % (2 * EVERY) == 0:
                    rec = dict(t=t, vx=float(bs["v_b"][0]), vy=float(bs["v_b"][1]), wz=float(bs["w_b"][2]), tilt=float(bs["tilt"]), s=s_, e=e_)
                    if (k + 1) % (5 * EVERY) == 0:
                        ex = self.link_extents(bs)
                        rec.update(top=ex["top_abs"] - fz, width=ex["width"], depth=ex["depth"], lean=ex["torso_lean"])
                    samp.append(rec)
                    if settle_kind and t >= dmin:
                        tops = [r["top"] for r in samp if "top" in r and r["t"] >= t - 0.3]
                        vxs = [r["vx"] for r in samp if r["t"] >= t - 0.3]
                        if len(tops) >= 3 and max(tops) - min(tops) < 0.015 and np.std(vxs) < 0.05:
                            break
                if not measure and wz is not None:
                    pass
            clock["t"] += t
            return samp, fell, t

        def measure(samp, fell, dur, win=0.5):
            if not samp:
                return dict(vx=0.0, vy=0.0, wz=0.0, top_min=np.nan, top_p50=np.nan, width=np.nan, depth=np.nan, pitch_t=0.0,
                            tilt_max=0.0, fell=bool(fell), settled=False, dur=round(dur, 2))
            t_end = samp[-1]["t"]
            W = [r for r in samp if r["t"] >= t_end - win] or samp[-2:]
            G_ = [r for r in W if "top" in r] or [r for r in samp if "top" in r][-2:]
            a = dict(vx=float(np.mean([r["vx"] for r in W])), vy=float(np.mean([r["vy"] for r in W])), wz=float(np.mean([r["wz"] for r in W])),
                     top_min=float(np.min([r["top"] for r in G_])) if G_ else np.nan, top_p50=float(np.median([r["top"] for r in G_])) if G_ else np.nan,
                     width=float(np.percentile([r["width"] for r in G_], 90)) if G_ else np.nan,
                     depth=float(np.percentile([r["depth"] for r in G_], 90)) if G_ else np.nan,
                     pitch_t=float(np.mean([r["lean"] for r in G_])) if G_ else 0.0,
                     tilt_max=float(max(r["tilt"] for r in samp)), fell=bool(fell), dur=round(dur, 2))
            tops = [r["top"] for r in G_]
            a["settled"] = bool(len(tops) >= 2 and max(tops) - min(tops) < 0.03)
            return a

        def set_heading_from_yaw():
            bs = rob.base_state()
            if bs is not None:
                st["heading"] = 1.0 if abs(wrap(float(bs["yaw"]) - yaw0)) < math.pi / 2 else -1.0

        def turn180(vx=None):
            """turn the robot around (heading flip, un-measured), with the forward speed the turn test found necessary"""
            wz = 0.8 * wz_cap
            vx = st.get("turn_vx", 0.0) if vx is None else vx
            target = yaw0 + (math.pi if st["heading"] > 0 else 0.0)
            fell = False; t_tot = 0.0
            for _ in range(14):
                bs = rob.base_state()
                if bs is None or rob.fallen(bs):
                    fell = True; break
                err = wrap(target - float(bs["yaw"]))
                if abs(err) < math.radians(12) or clock["t"] > budget:
                    break
                samp, fell, t = hold({}, vx, 0.0, float(np.sign(err)) * wz, 0.4, 0.4, measure=False)
                t_tot += t
                if fell:
                    break
            set_heading_from_yaw()
            return fell, t_tot

        rob._v_trace = []
        rob.set_env_intent(0.0, VC_UNC, VC_UNC, 0.0)
        spawn()
        tests = []
        log(f"[probe] {spec['ctrl_id']} start lane {lane.get('id')} fz {fz:.3f} ({len(segs)} planned segments)")
        for sg in segs:
            rem = budget - clock["t"]
            need_flip = 0.0
            if sg["kind"] in ("speed", "posture", "mode"):
                bs_ = rob.base_state(); s__, _ = self.lane_coords(bs_, geo); v_need = max(sg["v"], 0.3) * sg["dur_max"] + 0.4
                if (st["heading"] > 0 and s__ + v_need > L - 0.4) or (st["heading"] < 0 and s__ - v_need < 0.3):
                    need_flip = 4.0                 # an un-measured 180 deg turn would come first
            if rem < sg["dur_max"] + need_flip + (0.0 if sg["kind"] != "turn" else 4.0):
                tests.append(dict(sg, skipped=True, ach={})); continue
            # keep on the lane: flip direction near the ends (un-measured turn, counted robot time)
            bs = rob.base_state(); s_, _ = self.lane_coords(bs, geo)
            if sg["kind"] in ("speed", "posture", "mode"):
                v_need = max(sg["v"], 0.3) * sg["dur_max"] + 0.4
                if (st["heading"] > 0 and s_ + v_need > L - 0.4) or (st["heading"] < 0 and s_ - v_need < 0.3):
                    turn180()
            rob.set_env_intent(sg["v"], VC_UNC, VC_UNC, 0.0)
            if sg["kind"] == "turn":
                # measured part: 1.2 s of yaw_rate at vx 0; if the robot does not turn, 1.2 s more with vx = retry_vx
                sgn = 1.0
                samp, fell, dur = hold({}, 0.0, 0.0, sgn * sg["wz"], 1.2, 1.2)
                a = measure(samp, fell, dur, win=0.8)
                rec = dict(sg, ach=a, needs_vx=False)
                if abs(a["wz"]) < 0.3 * abs(sg["wz"]) and not fell:
                    samp, fell, dur = hold({}, sg.get("retry_vx", 0.1), 0.0, sgn * sg["wz"], 1.2, 1.2)
                    a2 = measure(samp, fell, dur, win=0.8)
                    rec = dict(sg, ach=a2, needs_vx=True, ach_no_vx=a, v_used=sg.get("retry_vx", 0.1))
                    if abs(a2["wz"]) >= 0.3 * abs(sg["wz"]):
                        st["turn_vx"] = sg.get("retry_vx", 0.1)
                tests.append(rec)
                if not tests[-1]["ach"].get("fell"):
                    turn180()                       # complete the reversal (un-measured)
            elif sg["kind"] == "lateral":
                # reset the posture to nominal first (the stand segment's posture)
                samp, fell, dur = hold({}, 0.0, sg["vy"], 0.0, sg["dur_min"], sg["dur_max"])
                a = measure(samp, fell, dur)
                rec = dict(sg, ach=a, needs_vx=False, vy_used=sg["vy"])
                if abs(a["vy"]) < 0.3 * abs(sg["vy"]) and not fell:
                    samp, fell, dur = hold({}, sg.get("retry_vx", 0.1), sg["vy"], 0.0, 1.0, 1.0)
                    a2 = measure(samp, fell, dur)
                    rec = dict(sg, ach=a2, needs_vx=True, ach_no_vx=a, vy_used=sg["vy"], v_used=sg.get("retry_vx", 0.1))
                tests.append(rec)
            else:
                post = dict(sg["cmd"])
                if sg["kind"] in ("speed", "stand"):
                    post = {c["name"]: c["nominal"] for c in spec["commands"] if c["role"] not in CS.VEL_ROLES}
                samp, fell, dur = hold(post, sg["v"], 0.0, 0.0 if sg["kind"] == "stand" else None, sg["dur_min"], sg["dur_max"],
                                       settle_kind=sg["kind"] in ("posture", "mode"))
                a = measure(samp, fell, dur)
                if sg["kind"] == "speed" and samp:
                    vf = a["vx"]
                    t63 = next((r["t"] for r in samp if vf > 0.05 and r["vx"] >= 0.63 * vf), np.nan)
                    a["lag63"] = float(t63)
                tests.append(dict(sg, ach=a))
            _a = tests[-1]["ach"]
            log(f"[probe]   {sg['label']:16s} t={clock['t']:5.1f} vx {_a.get('vx', 0):+.2f} vy {_a.get('vy', 0):+.2f} wz {_a.get('wz', 0):+.2f} "
                f"top {_a.get('top_min', np.nan):.3f} w {_a.get('width', np.nan):.2f} d {_a.get('depth', np.nan):.2f} lean {_a.get('pitch_t', 0):+.2f} "
                f"tilt {_a.get('tilt_max', 0):.2f} dur {_a.get('dur', 0)} fell {_a.get('fell')} {'(vx retry)' if tests[-1].get('needs_vx') else ''}")
            if tests[-1]["ach"].get("fell"):
                clock["falls"] += 1
                log(f"[probe]   {sg['label']} FELL -> respawn")
                spawn()
        # XFEAS_GEN3_PROBE_STEADY=1 (additive 2026-10-19, generic for every controller; default off = old probes reproduce):
        # the 1.5 s speed segments above do not reach steady state for slow-accelerating controllers (SONIC: cmd 0.76 ->
        # 0.15 m/s measured, while the same command reaches ~0.75 m/s on the course). Each forward speed level is re-measured
        # from a fresh standing spawn at the lane start with a long hold (3-6 s, as long as the lane allows) and the mean of
        # the last 1.5 s; plus a 'speed_max' level at the spec's maximum forward command. The short-window record is kept
        # as ach_short. Outside the 30 s budget (measurement only; nothing is decided here).
        if os.environ.get("XFEAS_GEN3_PROBE_STEADY", "0") == "1":
            nom_ = {c["name"]: c["nominal"] for c in spec["commands"] if c["role"] not in CS.VEL_ROLES}
            vxr_ = CS.rng(spec, "lin_vel_x")
            sp_tests = [t_ for t_ in tests if t_.get("kind") == "speed" and not t_.get("skipped")]
            vmax_ = float(vxr_[1]) if vxr_ is not None else None
            if vmax_ is not None and sp_tests and vmax_ > max(t_["v"] for t_ in sp_tests) + 0.02:
                mx = dict(sp_tests[-1], label="speed_max", v=round(vmax_, 3)); mx.pop("ach", None)
                tests.append(mx); sp_tests.append(mx)
            # the same steady re-measurement for every posture / mode op that did not fall (their 1.2-2 s windows have
            # the same problem: SONIC crouch at 0.3 m/s measured 0.055 m/s, below the 0.08 m/s 'op_not_walking' floor)
            sp_tests += [t_ for t_ in tests if t_.get("kind") in ("posture", "mode") and not t_.get("skipped")
                         and not (t_.get("ach") or {}).get("fell")]
            rob.set_env_intent(0.0, VC_UNC, VC_UNC, 0.0)
            for t_ in sp_tests:
                v_ = float(t_["v"])
                H = float(np.clip((L - 1.4) / max(v_, 0.2), 3.0, 6.0))
                spawn()
                rob.set_env_intent(v_, VC_UNC, VC_UNC, 0.0)
                post_ = dict(nom_) if t_["kind"] == "speed" else dict(nom_, **t_["cmd"])
                samp, fell, dur = hold(post_, v_, 0.0, None, H, H)
                a_ = measure(samp, fell, dur, win=1.5)
                vf = a_["vx"]
                a_["lag63"] = float(next((r["t"] for r in samp if vf > 0.05 and r["vx"] >= 0.63 * vf), np.nan))
                if "ach" in t_:
                    t_["ach_short"] = t_["ach"]
                t_["ach"] = a_; t_["steady_hold_s"] = round(H, 2); t_["dur_min"] = t_["dur_max"] = round(H, 2)
                log(f"[probe-steady] {t_['label']:10s} vx_cmd {v_:.2f} hold {H:.1f}s -> vx {vf:+.3f} (short window "
                    f"{(t_.get('ach_short') or {}).get('vx', float('nan')):+.3f}) top {a_['top_min']:.3f} fell {a_['fell']}")
                if fell:
                    clock["falls"] += 1
                rob.command(dict(nom_)); rob.stop()
            spawn()
        # XFEAS_GEN3_PROBE_DRVSWEEP=1 (additive 2026-10-19, generic for every controller): a DRIVER-TABLE addendum after the
        # 30 s probe. Steady-state (last 1.0 s of 2.5 s holds) lateral speed per stable walking posture (nominal + every
        # moving, upright posture/mode op) x 3 lateral command levels x both sides, plus steady-state forward speed of the
        # nominal posture at 3 levels. Stored as out['drv_ops']; the model's fingerprint / ops are NOT changed (so the model
        # input stays as in training); only the non-learned driver reads it (XFEAS_GEN3_DRV_LAT=1).
        drv_ops = []
        if os.environ.get("XFEAS_GEN3_PROBE_DRVSWEEP", "0") == "1":
            nom = {c["name"]: c["nominal"] for c in spec["commands"] if c["role"] not in CS.VEL_ROLES}
            posts = [("nominal", dict(nom), 0.0)]
            for t_ in tests:
                a_ = t_.get("ach") or {}
                if t_.get("kind") in ("posture", "mode") and not a_.get("fell") and a_.get("vx", 0.0) > PR.WALK_V \
                        and a_.get("tilt_max", 9.0) < 0.6 and abs(a_.get("pitch_t", 0.0)) < 0.5:
                    posts.append((t_["label"], dict(nom, **t_["cmd"]), float(a_.get("top_min", np.nan))))
            vyr = CS.rng(spec, "lin_vel_y") if CS.has(spec, "lin_vel_y") else None
            vxr = CS.rng(spec, "lin_vel_x")
            HOLD, WIN = 2.5, 1.0
            # XFEAS_GEN3_DRV_LAT_VX=1 (v4 probe protocol, 2026-10-19; default off = as before): a controller whose lateral
            # test needed forward speed (lateral_needs_vx: AMO stands still for |vx| < 0.1 and ignores vy) is measured with
            # that same forward speed (the main probe's retry vx) -- with vx 0 every AMO row read vy ~0.000, so the driver's
            # table said "no lateral" and the rule arm's sidestep crawled at vy 0.03 (latest_results/v4_phase0 V4R)
            lat_vx = 0.0
            if os.environ.get("XFEAS_GEN3_DRV_LAT_VX", "0") == "1":
                lat_vx = float(max([t_.get("v_used", 0.0) for t_ in tests if t_.get("kind") == "lateral" and t_.get("needs_vx")], default=0.0))
            rob.set_env_intent(0.0, VC_UNC, VC_UNC, 0.0)
            spawn()
            for lab, post, top_ in posts:
                if vyr is not None:
                    vmax = min(max(abs(vyr[0]), abs(vyr[1])), 0.8)
                    for f in (0.4, 0.65, 1.0):
                        for sgn in (+1.0, -1.0):
                            vy_c = round(sgn * f * vmax, 3)
                            if lat_vx > 0:                  # keep on the lane (forward drift lat_vx * HOLD per hold)
                                bs_ = rob.base_state(); s__, _ = self.lane_coords(bs_, geo)
                                if (st["heading"] > 0 and s__ + lat_vx * HOLD + 0.6 > L - 0.4) or (st["heading"] < 0 and s__ - lat_vx * HOLD - 0.6 < 0.3):
                                    turn180()
                            samp, fell, dur = hold(post, lat_vx, vy_c, 0.0, HOLD, HOLD)
                            a_ = measure(samp, fell, dur, win=WIN)
                            drv_ops.append(dict(kind="lateral", post_label=lab, posture=post, vx_cmd=lat_vx, vy_cmd=vy_c,
                                                vx_ach=round(a_["vx"], 3), vy_ach=round(a_["vy"], 3), top_min=round(a_["top_min"], 3),
                                                depth=round(a_["depth"], 3), width=round(a_["width"], 3), tilt_max=round(a_["tilt_max"], 3),
                                                fell=bool(a_["fell"]), dur=a_["dur"]))
                            log(f"[probe-drv] lat {lab:18s} vy_cmd {vy_c:+.2f} -> vy {a_['vy']:+.3f} vx {a_['vx']:+.3f} top {a_['top_min']:.3f} fell {a_['fell']}")
                            if fell:
                                clock["falls"] += 1; spawn()
                            rob.command(dict(nom))
                if lab == "nominal" and vxr is not None:
                    hi_ = min(vxr[1], 1.0)
                    for f in (0.35, 0.65, 1.0):
                        bs_ = rob.base_state(); s__, _ = self.lane_coords(bs_, geo)
                        if (st["heading"] > 0 and s__ + f * hi_ * HOLD + 0.6 > L - 0.4) or (st["heading"] < 0 and s__ - f * hi_ * HOLD - 0.6 < 0.3):
                            turn180()
                        vx_c = round(f * hi_, 3)
                        if os.environ.get("XFEAS_GEN3_PROBE_STEADY", "0") == "1":     # steady: fresh spawn, long hold
                            Hf = float(np.clip((L - 1.4) / max(vx_c, 0.2), 3.0, 6.0))
                            spawn(); samp, fell, dur = hold(post, vx_c, 0.0, None, Hf, Hf)
                            a_ = measure(samp, fell, dur, win=1.5)
                        else:
                            samp, fell, dur = hold(post, vx_c, 0.0, None, HOLD, HOLD)
                            a_ = measure(samp, fell, dur, win=WIN)
                        drv_ops.append(dict(kind="speed", post_label=lab, posture=post, vx_cmd=vx_c, vy_cmd=0.0, vx_ach=round(a_["vx"], 3),
                                            vy_ach=round(a_["vy"], 3), top_min=round(a_["top_min"], 3), tilt_max=round(a_["tilt_max"], 3),
                                            fell=bool(a_["fell"]), dur=a_["dur"]))
                        log(f"[probe-drv] fwd {lab:18s} vx_cmd {vx_c:+.2f} -> vx {a_['vx']:+.3f} fell {a_['fell']}")
                        if fell:
                            clock["falls"] += 1; spawn()
            rob.command(dict(nom)); rob.stop()
        # back to the nominal posture
        rob.command({c["name"]: c["nominal"] for c in spec["commands"] if c["role"] not in CS.VEL_ROLES})
        rob.stop()
        tr = rob.trace_arrays(geo, fz); rob._v_trace = None
        self._last_probe_tests = tests
        summ = PR.summarize(spec, tests)
        out = dict(ctrl_id=spec["ctrl_id"], base=spec["base"], spec_hash=CS.spec_hash(spec), lane=lane.get("id"), env=getattr(self, "env_id", None),
                   fz=fz, t_robot=round(clock["t"], 2), t_wall=round(time.time() - t_wall0, 1), n_falls=clock["falls"], tests=tests,
                   trace={k: v.tolist() for k, v in tr.items()}, **summ)
        if drv_ops:
            out["drv_ops"] = drv_ops
        log(f"[probe] {spec['ctrl_id']} done t_robot {out['t_robot']}s wall {out['t_wall']}s falls {clock['falls']} | "
            f"stand_top {summ['fp']['stand_top']:.3f} min_top_walk {summ['fp']['min_top_walk']:.3f} vx_hi {summ['fp']['vx_ach_hi']:.2f} "
            f"vy L/R {summ['caps']['vy_max_L']:.2f}/{summ['caps']['vy_max_R']:.2f} wz {summ['caps']['wz_ach']:.2f} "
            f"walk ops {len([o for o in summ['ops'] if o['moving'] and o['kind'] in ('posture', 'mode', 'speed')])}")
        return out

    # ------------------------------------------------------------------ one rollout of a driver program
    def rollout_v3(self, geo, dec, prog, act, vid=None):
        self.restore_state(dec["state"])
        self.rob._v_trace = []
        goal = lambda s, e, t, st, bs_: (self.goal_reached(bs_, geo, s, e) or ("stuck" if st else None))
        tmax = self.t_max(geo, dec["s_dec"])
        if prog["macro"] == "turn_side_turn":
            m = TurnSideCtl3(self, geo, dec, prog, act)
            stall0 = self.STALL_S; self.STALL_S = stall0 + 6.0
            try:
                res = self.drive2(geo, None, goal, tmax + 15.0, method=m, vid=vid)
            finally:
                self.STALL_S = stall0
        else:
            m = ProgramCtl3(self, geo, dec, prog, act)
            extra = 0.0
            if prog["macro"] == "detour":
                extra = 2.0 * abs(float(prog.get("lat_off", 0.0))) / max(self.v_nom, 0.1)
            if prog["macro"] == "two_phase":
                va = max(float(prog["phases"]["A"].get("pred", {}) and prog["phases"]["A"]["pred"].get("vx") or 0.1), 0.1)
                extra = max(0.0, float(geo["s_front"]) - float(dec["s_dec"])) / va * 0.5
            res = self.drive2(geo, None, goal, tmax + extra, method=m, vid=vid)
        self.rob.stop()
        out = self._finish(res, geo, dec)
        out["t_switch"] = None if m.t_switch is None else round(float(m.t_switch), 2)
        out["phase_end"] = m.phase
        tot = sum(getattr(m, "t_in", {}).values()) if hasattr(m, "t_in") else 0.0
        out["frac_a"] = round(m.t_in.get("A", 0.0) / tot, 3) if hasattr(m, "t_in") and tot > 0 else 0.0
        tr = self.rob.trace_arrays(geo, geo["fz"]); self.rob._v_trace = None
        out["tr"] = tr
        return out


class GenSimV3(V3Mixin, G2.V2Mixin, GV.GenSimV1):
    pass


def make_envsim_v3():
    import envsim as ES

    class EnvSimV3(V3Mixin, G2.V2Mixin, ES.EnvSim):
        pass
    return EnvSimV3
