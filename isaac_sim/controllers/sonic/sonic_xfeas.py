#!/usr/bin/env python
"""
ZERO-SHOT adapter: the XFeas gen2-b decision layer (data/xe_train/gen2b, NOT retrained) on the released GEAR-SONIC G1
controller (g1_sonic.py). Additive 2026-10-17 (analysis/SONIC_GEN2B_ZEROSHOT.md). No learning, no tuning: every
constant below is copied from a flat-ground characterisation table (cited), never from a test-scene outcome.

    --robot g1s --robot-factory sonic_xfeas:make_g1s_xf          (scripts/sonic + scripts/gen_train on PYTHONPATH)
    --method sonic_xfeas:StraightXFeasGen2Sonic                  (episode_runner: D1g / HW1 / T0; run_terrain tiles)
    --method sonic_xfeas:StairsGen2Sonic                         (gym course GC0, stairs path follower)

1. What gen2-b is told (robot descriptor + setpoint-range tokens, sp_tokens.robot_box / gen_methods.gen_box)
   * body geometry = G1's EXISTING descriptor (the same physical robot): joint names / limits / vmax / tau_max (23
     joints), mass, bbox, family copied verbatim from the AMO-G1 (g1c) training descriptor (G1_DESC_SRC);
   * command / setpoint ranges = SONIC's executable range:
       vx  [v_min 0.2, 0.8] m/s  (slowWalk keyboard band 0.2-0.8, g1_sonic.SPEED_RANGE; nominal 0.3 = the harness's
                                  stock speed, XFEAS_GEN2_VNOM=0.3)
       vy  +-0.35 m/s            (lateral +0.36/-0.34 measured at cmd 0.4, G1_CONTROLLER_BRIDGES 2.2)
       wz  +-1.0 rad/s
       body_height [0.30, 0.75] m in G1 body-height units (nominal 0.75 = sp_tokens.NOMINAL_H['g1']). The lower bound
                                  is the G1 box bottom: SONIC's lowest mode used (handCrawl, top 0.51x) is LOWER than
                                  any G1 body height at pitch 0 can express (h 0.30 -> top 0.72x), so the whole
                                  downward range is executable (quantised to modes, rule 2). No upward range: SONIC
                                  has no 'stand taller' command, so hi = nominal.
       pitch, roll ABSENT        (not commandable on SONIC; mode-implied lean only). Consequence: gen2-b's go2
                                  macro (needs a pitch dim, select_gen2) is never offered.
2. Fixed geometric mode rule (commanded G1 body height -> SONIC planner mode)
   commanded robot top = top_G1(h): piecewise-linear in the AMO-G1 flat-ground probe (G1_OVERHANG.md table 1, pitch 0,
   standing top: h 0.30 / 0.45 / 0.60 / 0.75 -> 0.914 / 1.023 / 1.127 / 1.266 m); ratio r = top_G1(h) / 1.266.
   SONIC mode tops (Isaac plane, G1_CONTROLLER_BRIDGES 2.2): walk/idle 1.32 (1.00x), crouchWalk 1.07 (0.81x, mean of
   1.06-1.08), handCrawl 0.67 (0.51x). Rule = the highest SONIC mode whose top ratio is <= r (never taller than the
   commanded posture):
       r >= 1 - EPS       -> nominal walk (slowWalk at v)            EPS = 0.02 mode-switch dead-band (a-priori design
       0.81 <= r < 1-EPS  -> crouchWalk                               constant: gen2's own posture smoothing decays dh
       r < 0.81           -> handCrawl (planner height default 0.8)   geometrically back to 0 and never reaches it exactly)
   elbowCrawl (0.26x, 0.17 m/s, chaotic at low height) is outside the box (lowest box posture maps to handCrawl).
3. Speed: planner speed = gen2-b's v; in crouchWalk at least V_CROUCH_MIN = 0.5 m/s (flat characterisation:
   crouchWalk validated at 0.43 m/s achieved for cmd 0.5 and 0.62 default; cmd 0.3 unreliable); handCrawl keeps the
   adapter's own keyboard band 0.4-1.0. Lateral / turn twists (sidestep macro: turn 90 deg then vy; detour: heading)
   pass straight to SONIC's direction / facing interface (g1_sonic.command).
4. Logging: every gen2 decision row gets sonic_mode / sonic_speed / r_top / mode_rule; mode switches are counted.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "closed_loop"), os.path.join(ROOT, "closed_loop"),
           os.path.join(ROOT, "scenes"), os.path.join(ROOT, "scenes")):
    if _p not in sys.path:
        sys.path.append(_p)
import g1_sonic as GS

NOMINAL_H = 0.75                                   # G1 nominal body height (sp_tokens.NOMINAL_H['g1']); gen2 layer reads it
G1_DESC_SRC = os.path.join(ROOT, "data/gen_v1/dt_warehouse/g1c/ep/dt_nul_g1c_4000000_00.npz")   # AMO-G1 training descriptor
AMO_H = (0.30, 0.45, 0.60, 0.75)                   # G1_OVERHANG.md table 1 (pitch 0, standing top)
AMO_TOP = (0.914, 1.023, 1.127, 1.266)
AMO_TOP_NOM = 1.266
SONIC_TOP = dict(walk=1.32, crouchWalk=1.07, handCrawl=0.67)     # G1_CONTROLLER_BRIDGES 2.2 / 3 (Isaac plane)
R_CROUCH = SONIC_TOP["crouchWalk"] / SONIC_TOP["walk"]          # 0.811
R_CRAWL = SONIC_TOP["handCrawl"] / SONIC_TOP["walk"]            # 0.508
EPS = 0.02
V_CROUCH_MIN = 0.5
VX_MAX, VY_MAX, WZ_MAX, V_MIN = 0.8, 0.35, 1.0, 0.2
BOX_H = (0.30, 0.75)
RANGES_XF = dict(vx=(-VX_MAX, VX_MAX), vy=(-VY_MAX, VY_MAX), wz=(-WZ_MAX, WZ_MAX), body_height=BOX_H)


def top_g1(h):
    return float(np.interp(float(h), AMO_H, AMO_TOP))


def mode_rule(body_height):
    """commanded G1 body height (m) -> (sonic mode or None = nominal walk, r_top)"""
    r = top_g1(body_height) / AMO_TOP_NOM
    if r >= 1.0 - EPS:
        return None, r
    if r >= R_CROUCH:
        return "crouchWalk", r
    return "handCrawl", r


def _spec_xf():
    import robots as RB
    s = GS.make_g1s.SPEC
    ex = dict(s.extra); ex.update(setpoint_ranges=dict(RANGES_XF), controller="GEAR-SONIC (released) + zero-shot XFeas adapter",
                                  mode_rule=dict(eps=EPS, r_crouch=R_CROUCH, r_crawl=R_CRAWL, v_crouch_min=V_CROUCH_MIN))
    return RB.RobotSpec(s.kind, s.family, (VX_MAX, VX_MAX, VY_MAX, 0.0, WZ_MAX), s.spawn_dz, s.fall_tilt, s.phys_hz, usd=s.usd,
                        foot_expr=s.foot_expr, v_min=V_MIN, cam=s.cam, min_z=s.min_z, extra=ex)


class G1SonicXFeas(GS.G1SonicAdapter):
    """SONIC adapter + posture -> mode mapping, G1 descriptor, 23-joint G1 proprio (same names / order as AMO-G1)"""

    def __init__(self, kind, prim_path, position, yaw=0.0):
        super().__init__(kind, prim_path, position, yaw)
        self.spec = make_g1s_xf.SPEC
        z = np.load(G1_DESC_SRC, allow_pickle=False)
        self.g1_desc = json.loads(str(z["desc"]))
        self.posture_mode, self.r_top, self.bh_cmd = None, 1.0, NOMINAL_H
        self.n_switch, self.speed_floor_hits = 0, 0
        self._jsel = None

    # ---- command: posture keys -> mode; twist passes through
    def command(self, cmd=None, **kw):
        if isinstance(cmd, dict) and any(k in cmd for k in ("body_height", "pitch", "roll")) and "mode" not in cmd:
            c = dict(cmd)
            bh = c.pop("body_height", None); c.pop("pitch", None); c.pop("roll", None)
            if bh is not None:
                m, r = mode_rule(bh)
                if m != self.posture_mode:
                    self.n_switch += 1
                    print(f"[sonic_xf] mode {self.posture_mode or 'walk'} -> {m or 'walk'} (body_height cmd {bh:.3f}, "
                          f"r_top {r:.3f})", flush=True)
                self.posture_mode, self.r_top, self.bh_cmd = m, r, float(bh)
                c["mode"] = m                             # None = nominal 'walk at v' mapping of g1_sonic
                c["body_height"] = -1.0                   # planner height: the adapter default (0.8) for crawl modes
            return super().command(c, **kw)
        return super().command(cmd, **kw)

    def _resolve(self):
        c = super()._resolve()
        if c["mode"] == GS.MODES["crouch_walk"] and c["speed"] is not None and 0 < c["speed"] < V_CROUCH_MIN:
            c["speed"] = V_CROUCH_MIN; self.speed_floor_hits += 1
        return c

    def reset_setpoints(self):
        self.posture_mode, self.r_top, self.bh_cmd = None, 1.0, NOMINAL_H
        super().reset_setpoints()

    def mode_name(self):
        return self.posture_mode or "walk"

    # ---- what gen2-b sees
    def descriptor(self, stage=None):
        g = self.g1_desc
        d = {k: g[k] for k in ("family", "n_joints", "joint_names", "joint_lower", "joint_upper", "joint_vmax",
                               "joint_tau_max", "mass_kg", "bbox_xyz_m", "fall_tilt")}
        d.update(kind=self.kind, max_cmd=dict(vx_fwd=VX_MAX, vx_rev=VX_MAX, vy=VY_MAX, vz=0.0, wz=WZ_MAX), v_min=V_MIN,
                 variant=dict(self._variant), phys_hz=self.spec.phys_hz,
                 controller="GEAR-SONIC (released) via zero-shot XFeas adapter (G1 geometry from AMO-G1 descriptor)",
                 setpoint_ranges={k: list(v) for k, v in RANGES_XF.items()},
                 setpoints_supported=["vx", "vy", "wz", "body_height (-> mode)"],
                 setpoints_unsupported=["pitch", "roll", "torso_yaw", "gait"])
        return d

    def proprio(self, bs=None):
        out = super().proprio(bs)
        if out is None:
            return None
        b, j, e = out
        if self._jsel is None:
            names = list(self._art.dof_names)
            self._jsel = np.array([names.index(n) for n in self.g1_desc["joint_names"]])
        e = dict(e, sonic_mode=self.mode_name())
        return b, j[self._jsel], e


def make_g1s_xf(kind, prim_path, position, yaw=0.0):
    GS.register_local_nav(kind)
    return G1SonicXFeas(kind, prim_path, position, yaw)


try:
    make_g1s_xf.SPEC = _spec_xf()
except Exception as _e:                              # outside Isaac (no robots import): unit tests of the rule only
    make_g1s_xf.SPEC = None


# ------------------------------------------------------------------ methods (gen2-b layer + SONIC mode logging)
def _annotate(self, obs, r):
    rob = obs["rob"]
    if hasattr(rob, "mode_name"):
        ap = getattr(rob, "applied", None) or {}
        tag = f"SONIC {rob.mode_name()}" + (f" @ {ap['speed']:.2f}" if ap.get("speed") is not None else "")
        if self.log and "sonic_mode" not in self.log[-1]:
            self.log[-1].update(sonic_mode=rob.mode_name(), sonic_speed=ap.get("speed"), sonic_mode_id=ap.get("mode"),
                                r_top=round(float(rob.r_top), 3), sonic_vx=ap.get("vx"), sonic_vy=ap.get("vy"))
        r["info"] = f"{r.get('info', '')}  | {tag}"
    return r


def _load_gm2():
    import gen_methods2 as GM2
    return GM2


try:
    import gen_methods2 as _GM2
    import stairs_gen2 as _SG2

    class StraightXFeasGen2Sonic(_GM2.StraightXFeasGen2):
        name = "xfeasgen2_sonic"
        label = "+ XFeas gen2-b (zero-shot) -> SONIC modes (straight drive)"

        def setup(self, ctx):
            super().setup(ctx)
            self.label = os.environ.get("XFEAS_LABEL") or self.label

        def act(self, obs):
            r = super().act(obs)
            # the posture is pushed only at decisions; a macro / stock row re-sends nothing -> annotate the last row
            return _annotate(self, obs, r)

    class StairsGen2Sonic(_SG2.StairsGen2):
        name = "stairs_gen2_sonic"
        label = "+ XFeas gen2-b (zero-shot) -> SONIC modes (stairs path follower)"

        def setup(self, ctx):
            super().setup(ctx)
            self.label = os.environ.get("XFEAS_LABEL") or self.label

        def act(self, obs):
            return _annotate(self, obs, super().act(obs))
except Exception as _e:                              # outside Isaac
    print("[sonic_xf] methods not loaded:", repr(_e))


def selftest():
    rows = []
    for h in (0.75, 0.74, 0.73, 0.72, 0.70, 0.60, 0.50, 0.46, 0.45, 0.40, 0.30):
        m, r = mode_rule(h)
        rows.append((h, round(r, 3), m or "walk"))
    for row in rows:
        print("[sonic_xf] h %.2f  r_top %.3f  -> %s" % row)
    assert mode_rule(0.75)[0] is None and mode_rule(0.60)[0] == "crouchWalk" and mode_rule(0.30)[0] == "handCrawl"
    print(f"[sonic_xf] thresholds: walk if r >= {1 - EPS:.3f}; crouchWalk if r >= {R_CROUCH:.3f}; else handCrawl")
    return rows


if __name__ == "__main__":
    selftest()
