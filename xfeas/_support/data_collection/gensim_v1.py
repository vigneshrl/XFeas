"""
gen_v1 harness extensions (2026-10-10, overnight collection). Import AFTER SimulationApp exists.

  * robot factories for Jackal / Iris with a .SPEC (GenSim needs FACTORY.SPEC): gensim_v1:make_jackal, gensim_v1:make_iris
  * per-environment HEADLIGHT profile + automatic pixel check of every night decision snapshot: the headlight is scaled
    by the env's base factor, then halved while > SAT_MAX of the pixels are saturated (any channel >= 250) or doubled
    while > BLACK_MAX are black (luminance < 8), at most 4 re-renders; a frame that still fails is rejected
    ('exposure'). Lighting never touches physics, so this is label-neutral. The final factor and the pixel stats are
    recorded per situation (dec['expo']).
  * save/restore for the Iris drone (no articulation): rigid-body pose/velocities through dynamic control + the
    velocity backend's reference/integrator state.
GenSimV1 = GenSim (school stage, real-asset generator families) + the above.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import copy, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import gensim as GS

# base headlight factor per environment (survey: the school-calibrated headlight saturates bright floors in office /
# hospital / GQ; terrain fields use 0.2, TERRAIN_ISAACLAB §5). The per-frame check below corrects what is left.
HL_SCALE = dict(
    school=1.0,
    office=0.45,
    warehouse=1.0,
    warehouse_forklifts=1.0,
    warehouse_shelves=1.0,
    full_warehouse=1.0,
    dt_warehouse=0.7,
    rivermark=1.0,
    simple_room=0.6,
    terrain=0.2,
    grid=0.4,
)
SAT_MAX, BLACK_MAX = 0.08, 0.55
REJ_SAT, REJ_BLACK = 0.15, 0.75


def pixel_stats(rgb):
    rgb = np.asarray(rgb)
    lum = rgb[..., :3].mean(-1)
    return dict(
        sat=round(float((rgb[..., :3].max(-1) >= 250).mean()), 4),
        black=round(float((lum < 8).mean()), 4),
        mean=round(float(lum.mean()), 1),
    )


def _robots_mod():
    import robots as RB

    return RB


def make_jackal(kind, prim_path, position, yaw=0.0):
    return _robots_mod().make_robot("jackal", prim_path, position, yaw)


def make_iris(kind, prim_path, position, yaw=0.0):
    """Iris + remove the Pegasus vehicle from the World's scene REGISTRY (USD prim kept): its articulation view is
    invalidated after the first Replicator frame and World.reset() -> post_reset() would crash on it; the vehicle is
    driven by its own physics callbacks and dynamic control, which survive resets (robots.DroneAdapter notes)."""
    rob = _robots_mod().make_robot("iris", prim_path, position, yaw)
    try:
        from omni.isaac.core.world import World

        W = World.instance()
        nm = rob.vehicle.name
        if W.scene.object_exists(nm):
            W.scene.remove_object(nm, registry_only=True)
            print(f"[gensim_v1] iris: removed '{nm}' from the scene registry", flush=True)
    except Exception as e:
        print(f"[gensim_v1] iris registry removal failed: {e}", flush=True)
    return rob


try:
    _RB = None
    import robots as _RB  # noqa (path set by gensim import)

    make_jackal.SPEC = _RB.SPECS["jackal"]
    make_iris.SPEC = _RB.SPECS["iris"]
except Exception:  # set lazily by GenSimV1 if the import order differs
    pass


class ExpoMixin:
    env_id = "school"

    def _hl_base(self):
        return float(HL_SCALE.get(self.env_id, 1.0))

    def set_lighting(self, sit):
        info = GS.GenSim.set_lighting(self, sit)
        self.expo = None
        if info.get("mode", "").startswith("night"):
            a = self.LIGHT.hl.GetIntensityAttr()
            a.Set(float(a.Get()) * self._hl_base())
            info = dict(info, hl_scale=self._hl_base())
        return info

    def snapshot(self, bs):
        out = GS.GenSim.snapshot(self, bs)
        if out is None:
            return out
        night = self.LIGHT.mode.startswith("night")
        a = self.LIGHT.hl.GetIntensityAttr()
        hist = []
        for it in range(5):
            st = pixel_stats(out[0])
            st["hl_I"] = round(float(a.Get()))
            hist.append(st)
            if not night or it == 4:
                break
            if st["sat"] > SAT_MAX:
                a.Set(float(a.Get()) * 0.5)
            elif st["black"] > BLACK_MAX and float(a.Get()) < 4e6:
                a.Set(float(a.Get()) * 2.0)
            else:
                break
            out = GS.GenSim.snapshot(self, bs)
            if out is None:
                return out
        fin = hist[-1]
        self.expo = dict(
            final=fin,
            n_adjust=len(hist) - 1,
            hist=hist,
            night=night,
            ok=bool((not night) or (fin["sat"] <= REJ_SAT and fin["black"] <= REJ_BLACK)),
        )
        return out

    def decision(self, sit, geo, light):
        dec = GS.GenSim.decision(self, sit, geo, light)
        dec["expo"] = getattr(self, "expo", None)
        if dec["expo"] is not None and not dec["expo"]["ok"]:
            raise GS.Reject(f"exposure {dec['expo']['final']}")
        return dec

    def check_view(self, geo, dist=2.5, h=1.0):
        if not hasattr(self.rob, "_art"):
            # drone: GenSim.decision parks the robot 30 m BELOW the floor for the audit view; the Pegasus Iris does not
            # recover from that (every hazard situation with an audit view ended in 'approach_fell'). Park it 400 m
            # away at floor height instead (it cannot occlude the hazard there either).
            P0 = geo["P0"]
            self.rob.teleport(float(P0[0]) + 400.0, float(P0[1]) + 400.0, 0.0, geo["fz"])
            for _ in range(5):
                self.world.step(render=False)
        return GS.GenSim.check_view(self, geo, dist, h)

    def rollout_sidestep(self, geo, dec, side, vy, e_gap, s_wall0, s_wall1, vid=None):
        """restore the decision state; nominal posture; scripted turn + sidestep + turn back (SidestepCtl) to the goal"""
        self.restore_state(dec["state"])
        nom = self.nominal()
        if self.NOM is not None and "body_height" in self.RANGES:
            self.rob.command(dict(body_height=nom["body_height"], pitch=0.0, roll=0.0, vx=0.0))
        m = SidestepCtl(self, geo, side, vy, e_gap, s_wall0, s_wall1)
        goal = lambda s, e, t, st, bs_: self.goal_reached(bs_, geo, s, e) or ("stuck" if st else None)
        stall0 = self.STALL_S
        self.STALL_S = stall0 + 6.0  # turning in place makes no lane progress for a few s
        try:
            res = self.drive(geo, None, goal, self.t_max(geo, dec["s_dec"]) + 15.0, method=m, vid=vid)
        finally:
            self.STALL_S = stall0
        self.rob.stop()
        out = self.outcome(res, geo, dec["s_dec"])
        out["t_goal"] = out["t"] if out["success"] else None
        out["posture_dev"] = 0.0
        out["side_phase_end"] = m.phase
        return out

    # ---------------------------------------------------------------- drone (no articulation) state
    def save_state(self):
        if hasattr(self.rob, "_art"):
            return GS.GenSim.save_state(self)
        from omni.isaac.dynamic_control import _dynamic_control as _dc

        dc = self.rob.vehicle.get_dc_interface()
        rb = dc.get_rigid_body(self.rob.prim_path + "/body")
        T = dc.get_rigid_body_pose(rb)
        be = self.rob.backend
        st = dict(
            drone=True,
            p=list(T.p),
            r=list(T.r),
            lin=list(dc.get_rigid_body_linear_velocity(rb)),
            ang=list(dc.get_rigid_body_angular_velocity(rb)),
            be={
                k: copy.deepcopy(getattr(be, k)) for k in ("p_ref", "yaw_ref", "int", "cmd", "rotor") if hasattr(be, k)
            },
            floor=self.rob._floor,
            pose=[np.array(list(T.p), float), np.array([T.r[3], T.r[0], T.r[1], T.r[2]], float)],
            q=np.zeros(1),
        )
        return st

    def restore_state(self, st):
        if not st.get("drone"):
            return GS.GenSim.restore_state(self, st)
        from omni.isaac.dynamic_control import _dynamic_control as _dc

        dc = self.rob.vehicle.get_dc_interface()
        rb = dc.get_rigid_body(self.rob.prim_path + "/body")
        dc.set_rigid_body_pose(rb, _dc.Transform(st["p"], st["r"]))
        dc.set_rigid_body_linear_velocity(rb, st["lin"])
        dc.set_rigid_body_angular_velocity(rb, st["ang"])
        for k, v in st["be"].items():
            setattr(self.rob.backend, k, copy.deepcopy(v))
        self.rob._floor = st["floor"]


class GenSimV1(ExpoMixin, GS.GenSim):
    """school stage (GenSim) + exposure check + drone state"""

    def __init__(self, app, factory_spec, kind, seed=0, side_cam=False, env_id="school"):
        self.env_id = env_id
        GS.GenSim.__init__(self, app, factory_spec, kind, seed=seed, side_cam=side_cam)


def bucket_of(sit):
    """the 4 user buckets (+ null) from a situation spec"""
    fam = sit["family"]
    if fam == "null":
        return "null"
    if fam in ("overhang", "compound", "gap", "step", "env_overhang", "env_gap", "env_climb", "env_compound"):
        return "setpoint"
    if fam == "terrain":
        return "slippery" if sit["params"].get("slippery") else "rough"
    if fam.startswith("env_"):
        return "scene"
    return "other"


def subfamily(sit):
    fam, P = sit["family"], sit["params"]
    if fam == "overhang":
        return "overhang_crouch"
    if fam == "compound":
        return "compound_oh_curb"
    if fam == "gap":
        return "narrow_gap"
    if fam == "step":
        return f"climb_{P.get('kind')}"
    if fam == "null":
        return f"null_{P.get('decoy', 'none')}"
    if fam == "terrain":
        return f"terrain_{P.get('type')}"
    return fam


class SidestepCtl:
    """scripted turn-90 + lateral walk (vy) + turn-back candidate for the oracle search (template:
    scripts/xembody/sidestep_methods.SidestepOracle). Lane frame from geo; gap centre e_gap.
    side=+1: face RIGHT of the lane (yaw = lane - 90 deg) and walk +vy; side=-1: face LEFT and walk -vy.
    G1/AMO: |vx| >= 0.1 is needed to step/turn, so vx alternates +-0.1 (sign centring on the gap)."""

    def __init__(self, sim, geo, side, vy, e_gap, s_wall0, s_wall1):
        self.G, self.geo, self.side, self.vy = sim, geo, float(side), float(vy)
        self.e_gap, self.s0, self.s1 = float(e_gap), float(s_wall0), float(s_wall1)
        self.g1 = sim.kind.startswith("g1")
        self.vxm = 0.1 if self.g1 else 0.0
        self.d_trig = 0.6 if self.g1 else 1.1
        self.past = 0.45 if self.g1 else 0.8
        self.phase = "approach"
        self.cap = float(sim.spec.max_cmd[4])
        self.log = []

    @staticmethod
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def act(self, obs):
        G, geo = self.G, self.geo
        bs = obs["bs"]
        yaw = float(bs["yaw"])
        h = geo["yaw"]
        s, e = G.lane_coords(bs, geo)
        cap = self.cap
        if self.phase == "approach":
            if s >= self.s0 - self.d_trig - 0.05:
                self.phase = "turn"
            else:
                err = self.wrap(h - yaw)
                return dict(
                    twist=np.array([G.v_nom, 0.0, 0.0, float(np.clip(2.0 * err - 1.5 * (e - self.e_gap), -1.0, 1.0))])
                )
        target = h - self.side * math.pi / 2
        if self.phase == "turn":
            err = self.wrap(target - yaw)
            if abs(err) < math.radians(5):
                self.phase = "side"
            else:
                return dict(twist=np.array([self.vxm, 0.0, 0.0, float(np.clip(1.5 * err, -cap, cap))]))
        if self.phase == "side":
            if s >= self.s1 + self.past:
                self.phase = "back"
            else:
                err = self.wrap(target - yaw)
                # body +x points to lane -side*e: centring on the gap
                want = -self.side * 1.5 * (self.e_gap - e)
                if self.vxm > 0:
                    vx = (
                        math.copysign(self.vxm, want)
                        if abs(self.e_gap - e) > 0.03
                        else (self.vxm if int(obs["t"]) % 2 == 0 else -self.vxm)
                    )
                else:
                    vx = float(np.clip(want, -0.2, 0.2))
                return dict(twist=np.array([vx, self.side * self.vy, 0.0, float(np.clip(1.5 * err, -cap, cap))]))
        if self.phase == "back":
            err = self.wrap(h - yaw)
            if abs(err) < math.radians(8):
                self.phase = "go"
            else:
                return dict(twist=np.array([self.vxm, 0.0, 0.0, float(np.clip(1.5 * err, -cap, cap))]))
        err = self.wrap(h - yaw)
        return dict(twist=np.array([G.v_nom, 0.0, 0.0, float(np.clip(2.0 * err - 1.5 * e, -1.0, 1.0))]))
