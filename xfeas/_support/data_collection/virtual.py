"""
Virtual controllers + the controller port (analysis/GEN3_SPEC.md sections 6-7).

VRob: a proxy around ONE harness robot adapter (AMO / GR00T / Isaac Lab / URMA / SONIC bridge). Everything not defined here
is forwarded to the adapter, so GenSim code keeps working. The proxy
  * accepts commands in the CURRENT controller's spec names (dict) or a twist list [vx, vy, vz, wz] (= roles lin_vel_x,
    lin_vel_y, yaw_rate of the current spec);
  * applies the current virtual wrapper's hidden transform (identity for a real controller) and sends the result to the
    adapter in its own command names (base spec binding);
  * keeps the 6 s / 10 Hz history h(t) (commanded by role, envelope intent, achieved base motion, joints) and, during a
    rollout, a trace with the same channels.
The wrapper's hidden parameters (wcfg) are never shown to the model; its spec (vspec) is.

sample_wrappers(base_spec, n, seed) -> [(vspec, wcfg)]   pure python (no Isaac)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import copy, math, os, sys
from collections import deque
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import ctrl_spec as CS

HIST_N, HIST_DT = 60, 0.1
# yaw, s, e are LANE-relative after pack_records(geo=...) (yaw - lane yaw, lane coordinates); foot_dz = |z_left - z_right| of
# the ankle_roll links (gait phase / step height signal)
ACH_KEYS = ("vx_b", "vy_b", "wz", "z_base", "top", "roll", "pitch", "yaw_rel", "tilt", "s", "e", "foot_dz")
RENAME = dict(
    lin_vel_x=["vx", "v_fwd", "fwd_speed", "x_vel", "forward"],
    lin_vel_y=["vy", "v_lat", "strafe", "y_vel", "side"],
    yaw_rate=["wz", "turn_rate", "yaw_vel", "omega", "steer"],
    pelvis_height=["body_height", "z_ref", "height", "h_cmd", "stance_h"],
    torso_pitch=["pitch", "lean", "torso_p", "bend"],
    torso_roll=["roll", "torso_r", "tilt_cmd"],
    torso_yaw=["torso_yaw", "waist_yaw", "twist"],
    mode=["mode", "gait", "style", "posture_mode"],
    speed=["speed", "target_vel"],
)


# ====================================================================== wrapper sampling (pure python)
def _primary_list(base):
    has_h = CS.has(base, "pelvis_height")
    has_y = CS.has(base, "lin_vel_y")
    out = ["quant" if has_h else "gears", "speed", "lag", "deadband", "mixed"]
    out.insert(1, "lateral" if has_y else "speed2")
    out.insert(4, "hclip" if has_h else "drift")
    return out  # 7 entries


def sample_wrapper(base, k, seed):
    """k-th virtual controller of base spec (k = 1..7); returns (vspec, wcfg)"""
    rng = np.random.default_rng([int(seed), k, sum(map(ord, base["ctrl_id"]))])
    prim = _primary_list(base)[(k - 1) % 7]
    has_h = CS.has(base, "pelvis_height")
    has_p = CS.has(base, "torso_pitch")
    has_y = CS.has(base, "lin_vel_y")
    vxr = CS.rng(base, "lin_vel_x")
    w = dict(
        primary=prim,
        sx=1.0,
        ox=0.0,
        sy=1.0,
        sw=1.0,
        ow=0.0,
        cap_f=None,
        lat="on",
        lat_k=1.0,
        h_clip=None,
        p_clip=None,
        latency=0.0,
        tau=0.0,
        db_v=0.0,
        db_h=0.0,
        quant=None,
        gears=None,
        rename=False,
        declared=[],
    )
    extras = rng.random(8)

    def want(name, p):
        return prim == name or (prim == "mixed" and rng.random() < 0.5) or rng.random() < p

    if want("speed", 0.25) or prim == "speed2":
        w["sx"] = round(float(rng.uniform(0.6, 1.4)), 3)
        if rng.random() < 0.5:
            w["ox"] = round(float(rng.uniform(-0.04, 0.04)), 3)
        if rng.random() < 0.6 or prim == "speed2":
            w["cap_f"] = round(float(rng.uniform(0.40, 0.80) * vxr[1]), 3)
    if want("drift", 0.2):
        w["sw"] = round(float(rng.uniform(0.7, 1.3)), 3)
        w["ow"] = round(float(rng.uniform(-0.08, 0.08)), 3)
    if has_y and want("lateral", 0.2):
        w["lat"] = str(rng.choice(["off_hidden", "off_declared", "limited"]))
        if w["lat"] == "limited":
            w["lat_k"] = round(float(rng.uniform(0.25, 0.6)), 3)
        w["sy"] = round(float(rng.uniform(0.8, 1.2)), 3)
    if want("lag", 0.3):
        w["latency"] = round(float(rng.uniform(0.05, 0.4)), 3)
        w["tau"] = round(float(rng.uniform(0.05, 0.5)), 3)
    if want("deadband", 0.25):
        w["db_v"] = round(float(rng.uniform(0.05, 0.15)), 3)
        if has_h:
            w["db_h"] = round(float(rng.uniform(0.02, 0.08)), 3)
    if has_h and prim == "quant" or (has_h and prim == "mixed" and rng.random() < 0.4):
        hlo, hhi = CS.rng(base, "pelvis_height")
        hnom = CS.cmd(base, "pelvis_height")["nominal"]
        plo, phi = CS.rng(base, "torso_pitch") if has_p else (0.0, 0.0)
        K = int(rng.choice([2, 3]))
        post = [(hnom, 0.0)]
        mid = (
            round(float(rng.uniform(hlo + 0.55 * (hnom - hlo), hlo + 0.85 * (hnom - hlo))), 3),
            round(float(rng.uniform(0.0, 0.8 * phi)), 3) if has_p else 0.0,
        )
        deep = (
            round(float(rng.uniform(hlo, hlo + 0.4 * (hnom - hlo))), 3),
            round(float(rng.uniform(0.4 * phi, phi)), 3) if has_p else 0.0,
        )
        post += [mid, deep][: K - 1] if K == 3 else [[mid, deep][int(rng.integers(2))]]
        names = [f"m{i}" for i in range(K)]
        perm = rng.permutation(K)
        w["quant"] = {names[perm[i]]: dict(h=post[i][0], p=post[i][1]) for i in range(K)}
        w["quant_nominal"] = names[perm[0]]
    elif not has_h and (prim == "gears" or (prim == "mixed" and rng.random() < 0.4)):
        mult = [1.0, round(float(rng.uniform(0.45, 0.75)), 3), round(float(rng.uniform(1.25, 1.6)), 3)]
        K = int(rng.choice([2, 3]))
        mult = mult[:K]
        names = [f"g{i}" for i in range(K)]
        perm = rng.permutation(K)
        w["gears"] = {names[perm[i]]: mult[i] for i in range(K)}
        w["gears_nominal"] = names[perm[0]]
    if has_h and w["quant"] is None and want("hclip", 0.2):
        hlo, hhi = CS.rng(base, "pelvis_height")
        hnom = CS.cmd(base, "pelvis_height")["nominal"]
        w["h_clip"] = [round(float(hlo + rng.uniform(0.3, 0.75) * (hnom - hlo)), 3), hhi]
        if has_p and rng.random() < 0.5:
            plo, phi = CS.rng(base, "torso_pitch")
            w["p_clip"] = [plo, round(float(rng.uniform(0.25, 0.7) * phi), 3)]
        if rng.random() < 0.5:
            w["declared"].append("h_clip")
    w["rename"] = bool(rng.random() < 0.5)
    vid = f"v_{base['ctrl_id']}_{k:02d}"
    return make_vspec(base, w, vid, rng), w


def make_vspec(base, w, vid, rng):
    s = CS.copy_spec(base)
    s.update(
        ctrl_id=vid,
        virtual=True,
        base=base["ctrl_id"],
        controller=f"virtual wrapper {vid} of {base['ctrl_id']}",
        doc_notes=[],
        sources=[],
    )
    cmds = []
    for c in s["commands"]:
        r = c["role"]
        if r == "lin_vel_y" and w["lat"] == "off_declared":
            continue
        if w["quant"] is not None and r in ("pelvis_height", "torso_pitch"):
            continue
        c = dict(c)
        if r == "pelvis_height" and w["h_clip"] is not None and "h_clip" in w["declared"]:
            c["range"] = list(w["h_clip"])
        if w["rename"]:
            c["name"] = str(rng.choice(RENAME[r]))
        cmds.append(c)
    seen = set()
    for c in cmds:  # unique names
        while c["name"] in seen:
            c["name"] = c["name"] + "_"
        seen.add(c["name"])
    modes = []
    if w["quant"] is not None or w["gears"] is not None:
        tab = w["quant"] if w["quant"] is not None else w["gears"]
        nm = sorted(tab)
        modes = [
            dict(
                name=n,
                id=i,
                group="posture" if w["quant"] is not None else "gait",
                moving=True,
                static=False,
                speed_range=None,
                uses_height=False,
                height_range=None,
                doc="",
            )
            for i, n in enumerate(nm)
        ]
        mname = str(rng.choice(RENAME["mode"])) if w["rename"] else "mode"
        while mname in seen:
            mname += "_"
        cmds.append(
            dict(
                name=mname,
                role="mode",
                type="discrete",
                values=nm,
                nominal=w.get("quant_nominal") if w["quant"] is not None else w.get("gears_nominal"),
            )
        )
    s["commands"] = cmds
    s["modes"] = modes
    s["binding"] = dict(adapter=base["binding"]["adapter"], call="virtual", base=base["ctrl_id"])
    CS.validate(s)
    return s


def sample_wrapper_modal(base, k, seed):
    """additive 2026-10-19: a MODAL virtual controller (k = 8, 9, ...): discrete posture modes (nominal + slow crouch walk
    [+ deep]) with a per-mode speed gear and per-mode lateral gain / deadband (lateral that only works in some modes or
    above a command threshold), on top of the usual speed / lag sampling. Bases with a height command only."""
    rng = np.random.default_rng([int(seed), 100 + k, sum(map(ord, base["ctrl_id"]))])
    vs, w = sample_wrapper(base, 1 if CS.has(base, "pelvis_height") else 7, seed + 7919 * k)  # quant (or mixed) primary
    has_h = CS.has(base, "pelvis_height")
    has_p = CS.has(base, "torso_pitch")
    if not has_h:
        return None
    hlo, hhi = CS.rng(base, "pelvis_height")
    hnom = CS.cmd(base, "pelvis_height")["nominal"]
    phi = CS.rng(base, "torso_pitch")[1] if has_p else 0.0
    K = int(rng.choice([2, 3]))
    lo_f, hi_f = (0.5, 0.8) if k < 10 else (0.65, 0.88)  # k >= 10: crouch band near the walking range (shallower)
    post = [
        (hnom, 0.0),
        (
            round(float(rng.uniform(hlo + lo_f * (hnom - hlo), hlo + hi_f * (hnom - hlo))), 3),
            round(float(rng.uniform(0.3 * phi if k >= 10 else 0, 0.8 * phi if k >= 10 else 0.6 * phi)), 3)
            if has_p
            else 0.0,
        ),
        (
            round(float(rng.uniform(hlo, hlo + 0.35 * (hnom - hlo))), 3),
            round(float(rng.uniform(0.3 * phi, phi)), 3) if has_p else 0.0,
        ),
    ][:K]
    names = [f"m{i}" for i in range(K)]
    perm = rng.permutation(K)
    w.update(
        primary="modal",
        quant={names[perm[i]]: dict(h=post[i][0], p=post[i][1]) for i in range(K)},
        quant_nominal=names[perm[0]],
        h_clip=None,
        p_clip=None,
        gears=None,
    )
    gear = [
        1.0,
        round(float(rng.uniform(0.35, 0.7) if k < 10 else rng.uniform(0.55, 0.9)), 3),
        round(float(rng.uniform(0.25, 0.5)), 3),
    ]
    w["quant_vx"] = {names[perm[i]]: gear[i] for i in range(K)}
    lat = []
    for i in range(K):
        if rng.random() < 0.3:
            lat.append(dict(k=0.0, db=0.0))  # no lateral in this mode
        else:
            lat.append(
                dict(
                    k=round(float(rng.uniform(0.3, 1.2)), 3),
                    db=round(float(rng.choice([0.0, rng.uniform(0.1, 0.35)])), 3),
                )
            )
    w["quant_lat"] = {names[perm[i]]: lat[i] for i in range(K)}
    if w["lat"] in ("off_hidden", "off_declared"):
        w["lat"] = "on"
    vid = f"v_{base['ctrl_id']}_{k:02d}"
    return make_vspec(base, w, vid, rng), w


def sample_wrappers(base, n=7, seed=0):
    return [sample_wrapper(base, k, seed) for k in range(1, n + 1)]


def identity_cfg():
    return dict(
        primary="identity",
        sx=1.0,
        ox=0.0,
        sy=1.0,
        sw=1.0,
        ow=0.0,
        cap_f=None,
        lat="on",
        lat_k=1.0,
        h_clip=None,
        p_clip=None,
        latency=0.0,
        tau=0.0,
        db_v=0.0,
        db_h=0.0,
        quant=None,
        gears=None,
        rename=False,
        declared=[],
    )


# ====================================================================== the proxy (Isaac side)
class VRob:
    _OWN = (
        "_v_inner",
        "_v_base",
        "_v_spec",
        "_v_w",
        "_v_state",
        "_v_q",
        "_v_applied",
        "_v_t",
        "_v_tnext",
        "_v_hnext",
        "_v_hist",
        "_v_trace",
        "_v_env",
        "_v_top_fn",
        "_v_last_sent",
        "_v_sub",
        "_v_jn",
        "_v_foot_fn",
    )

    def __init__(self, inner, base_spec, top_fn=None):
        object.__setattr__(self, "_v_inner", inner)
        object.__setattr__(self, "_v_base", base_spec)
        object.__setattr__(self, "_v_top_fn", top_fn)
        object.__setattr__(self, "_v_foot_fn", None)
        object.__setattr__(self, "_v_jn", None)
        self.set_controller(base_spec, identity_cfg())

    # ---- proxying
    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_v_inner"), k)

    def __setattr__(self, k, v):
        if k in VRob._OWN:
            object.__setattr__(self, k, v)
        else:
            setattr(self._v_inner, k, v)

    # ---- controller switching
    def set_controller(self, spec, wcfg):
        object.__setattr__(self, "_v_spec", spec)
        object.__setattr__(self, "_v_w", wcfg if wcfg is not None else identity_cfg())
        object.__setattr__(self, "_v_t", 0.0)
        object.__setattr__(self, "_v_tnext", 0.0)
        object.__setattr__(self, "_v_hnext", 0.0)
        object.__setattr__(self, "_v_hist", deque(maxlen=HIST_N))
        object.__setattr__(self, "_v_trace", None)
        object.__setattr__(self, "_v_env", [0.3, 9.9, 9.9, 0.0])
        object.__setattr__(self, "_v_state", CS.nominal_cmd(spec))
        object.__setattr__(self, "_v_q", deque())
        object.__setattr__(self, "_v_applied", None)
        object.__setattr__(self, "_v_last_sent", None)
        object.__setattr__(self, "_v_sub", 0)

    @property
    def vspec(self):
        return self._v_spec

    @property
    def wcfg(self):
        return self._v_w

    def is_identity(self):
        return self._v_w.get("primary") == "identity"

    def set_env_intent(self, v, top, wid, pitch):
        self._v_env = [float(v), float(top), float(wid), float(pitch)]

    # ---- role access
    def _role_val(self, role):
        c = CS.cmd(self._v_spec, role)
        return None if c is None else self._v_state.get(c["name"])

    def role_cmd(self):
        out = np.full(len(CS.HIST_ROLES), np.nan, np.float32)
        for i, r in enumerate(CS.HIST_ROLES):
            v = self._role_val(r)
            if v is None:
                continue
            if r == "mode":
                out[i] = CS.mode_index(self._v_spec, v)
            else:
                out[i] = float(v)
        return out

    # ---- command entry
    def command(self, cmd=None, **kw):
        s = self._v_spec
        if isinstance(cmd, dict):
            upd = dict(cmd)
        elif cmd is not None:
            t = np.zeros(4)
            t[: len(cmd)] = np.asarray(cmd, float)[:4]
            upd = {}
            for role, val in (("lin_vel_x", t[0]), ("lin_vel_y", t[1]), ("yaw_rate", t[3])):
                c = CS.cmd(s, role)
                if c is not None:
                    upd[c["name"]] = float(val)
        else:
            upd = {}
        upd.update(kw)
        names = {c["name"]: c for c in s["commands"]}
        for k, v in upd.items():
            if k not in names:
                raise KeyError(f"{s['ctrl_id']}: unknown command {k!r}; have {sorted(names)}")
            c = names[k]
            if c["type"] == "continuous":
                lo, hi = c["range"]
                v = (
                    float(np.clip(float(v), lo, hi))
                    if not (c["role"] in ("speed", "pelvis_height") and float(v) < 0)
                    else float(v)
                )
            else:
                if isinstance(v, (int, np.integer)) and v not in c["values"]:
                    v = c["values"][int(v)]
                assert v in c["values"], (k, v)
            self._v_state[k] = v
        base_cmd = self._transform()
        if self._v_w["latency"] <= 0 and self._v_w["tau"] <= 0:
            self._send(base_cmd)
        else:
            self._v_q.append((self._v_t + self._v_w["latency"], base_cmd))
        vx = self._role_val("lin_vel_x") or 0.0
        vy = self._role_val("lin_vel_y") or 0.0
        wz = self._role_val("yaw_rate") or 0.0
        return np.array([vx, vy, 0.0, wz])

    def _transform(self):
        """virtual state -> base command dict (base spec names)"""
        w, base = self._v_w, self._v_base
        g = lambda r, d=0.0: self._role_val(r) if self._role_val(r) is not None else d
        out = {}
        vx = float(g("lin_vel_x"))
        if abs(vx) < w["db_v"]:
            vx = 0.0
        if w.get("gears") is not None:
            vx *= float(w["gears"][self._role_val("mode")])
        if w.get("quant_vx") is not None:  # additive 2026-10-19 (modal wrappers): per-mode speed gear
            vx *= float(w["quant_vx"].get(self._role_val("mode"), 1.0))
        vx = w["sx"] * vx + (w["ox"] if abs(vx) > 1e-3 else 0.0)
        if w["cap_f"] is not None:
            vx = min(vx, w["cap_f"])
        vy = float(g("lin_vel_y"))
        if abs(vy) < w["db_v"]:
            vy = 0.0
        vy = (
            0.0
            if w["lat"] in ("off_hidden", "off_declared")
            else vy * w["sy"] * (w["lat_k"] if w["lat"] == "limited" else 1.0)
        )
        if w.get("quant_lat") is not None:  # additive 2026-10-19 (modal wrappers): per-mode lateral gain
            ml = w["quant_lat"].get(self._role_val("mode"), dict(k=1.0, db=0.0))  # + per-mode lateral deadband
            vy = 0.0 if abs(vy) < ml["db"] else vy * ml["k"]
        wz = float(g("yaw_rate"))
        wz = w["sw"] * wz + (w["ow"] if abs(wz) > 1e-3 or abs(vx) > 0.05 else 0.0)
        vals = dict(lin_vel_x=vx, lin_vel_y=vy, yaw_rate=wz)
        if CS.has(base, "pelvis_height"):
            hnom = CS.cmd(base, "pelvis_height")["nominal"]
            if w.get("quant") is not None:
                q = w["quant"][self._role_val("mode")]
                h, p = q["h"], q["p"]
            else:
                h = float(g("pelvis_height", hnom))
                p = float(g("torso_pitch", 0.0))
                if 0 <= h and abs(h - hnom) < w["db_h"]:
                    h = hnom
            if w["h_clip"] is not None and h >= 0:
                h = float(np.clip(h, *w["h_clip"]))
            if w["p_clip"] is not None:
                p = float(np.clip(p, *w["p_clip"]))
            vals["pelvis_height"] = h
            if CS.has(base, "torso_pitch"):
                vals["torso_pitch"] = p
        for r in ("torso_roll", "torso_yaw"):
            if CS.has(base, r) and self._role_val(r) is not None:
                vals[r] = float(self._role_val(r))
        if w.get("quant") is None and w.get("gears") is None:
            for r in ("speed", "mode"):
                if CS.has(base, r) and self._role_val(r) is not None:
                    vals[r] = self._role_val(r)
            # XFEAS_GEN3_SPEED_FOLLOW_V=1 (additive 2026-10-19, plumbing; default off = old behaviour): a 'speed' role at a
            # negative value means "the mode's default speed" (spec nominal -1). Sent explicitly it OVERRIDES the velocity
            # magnitude (SONIC bridge: target_vel = speed), so every lin_vel_x / lin_vel_y the probe and the driver command
            # ran at the mode default (~0.3 m/s slowWalk; probe office_s4_speedpin: cmd 0.28-0.80 -> 0.27-0.30 m/s steady).
            # With the flag a negative speed is sent as None (unset): the base derives the speed from |v| as documented.
            # XFEAS_GEN3_MODE_DEFAULT_SPEED=1 (additive 2026-10-19, generic spec rule; with SPEED_FOLLOW_V): every
            # discrete mode OTHER than the spec's nominal mode runs at that mode's own default speed (the spec's speed
            # nominal, -1 = "mode default"), in the probe AND in execution; only the nominal mode follows |v|. (A mode
            # commanded at an arbitrary |v| it was not designed for -- SONIC crouch at 0.3 m/s -- barely moves inside
            # the training-consistent probe window.)
            mode_default = False
            if os.environ.get("XFEAS_GEN3_MODE_DEFAULT_SPEED", "0") == "1" and CS.has(base, "mode"):
                mnom = CS.cmd(base, "mode").get("nominal")
                mcur = self._role_val("mode")
                mode_default = mcur is not None and mnom is not None and str(mcur) != str(mnom)
            if os.environ.get("XFEAS_GEN3_SPEED_FOLLOW_V", "0") == "1" and "speed" in vals and not mode_default:
                try:
                    if vals["speed"] is not None and float(vals["speed"]) < 0:
                        vals["speed"] = None
                except (TypeError, ValueError):
                    pass
        if CS.has(base, "mode") and "mode" in vals:
            m = CS.mode_info(base, vals["mode"])
            vals["mode"] = int(m["id"]) if m is not None else vals["mode"]
        for c in base["commands"]:
            if c["role"] in vals:
                out[base["binding"]["names"].get(c["name"], c["name"])] = vals[c["role"]]
        return out

    def _send(self, base_cmd):
        if base_cmd != self._v_last_sent:
            self._v_inner.command(dict(base_cmd))
            self._v_last_sent = dict(base_cmd)

    # ---- physics tick (registered via GenSim's "drive" callback -> rob.step)
    def step(self, dt):
        self._v_t += dt
        w = self._v_w
        if (w["latency"] > 0 or w["tau"] > 0) and self._v_t >= self._v_tnext:
            self._v_tnext = self._v_t + 0.02
            tgt = None
            while self._v_q and self._v_q[0][0] <= self._v_t:
                tgt = self._v_q.popleft()[1]
            if tgt is not None:
                object.__setattr__(self, "_v_sub", tgt)  # the delayed target now in force
            tgt = self._v_sub if isinstance(self._v_sub, dict) else None
            if tgt is not None:
                if self._v_applied is None or w["tau"] <= 0:
                    app = dict(tgt)
                else:
                    a = 1.0 - math.exp(-0.02 / w["tau"])
                    app = {}
                    for k, v in tgt.items():
                        p0 = self._v_applied.get(k, v)
                        app[k] = (p0 + a * (v - p0)) if isinstance(v, float) and isinstance(p0, float) else v
                self._v_applied = app
                self._send({k: (round(v, 4) if isinstance(v, float) else v) for k, v in app.items()})
        if self._v_t >= self._v_hnext:
            self._v_hnext = self._v_t + HIST_DT - 1e-9
            try:
                self._record()
            except Exception:
                pass
        self._v_inner.step(dt)

    def _record(self):
        inner = self._v_inner
        if not getattr(inner, "ready", False):
            return
        bs = inner.base_state()
        if bs is None:
            return
        q = np.asarray(inner._art.get_joint_positions(), np.float32).reshape(-1)
        dq = np.asarray(inner._art.get_joint_velocities(), np.float32).reshape(-1)
        top = float(self._v_top_fn()) if self._v_top_fn is not None else np.nan
        nc = float(self._v_foot_fn()) if self._v_foot_fn is not None else np.nan
        ach = np.array(
            [
                bs["v_b"][0],
                bs["v_b"][1],
                bs["w_b"][2],
                bs["pos"][2],
                top,
                bs["roll"],
                bs["pitch"],
                bs["yaw"],
                bs["tilt"],
                bs["pos"][0],
                bs["pos"][1],
                nc,
            ],
            np.float64,
        )
        rec = (self._v_t, self.role_cmd(), np.array(self._v_env, np.float32), ach, q, dq)
        self._v_hist.append(rec)
        if self._v_trace is not None:
            self._v_trace.append(rec)

    # ---- adapter calls that must also reset the wrapper
    def teleport(self, x, y, yaw, floor_z):
        self._v_inner.teleport(x, y, yaw, floor_z)
        self._v_q.clear()
        self._v_applied = None
        self._v_last_sent = None
        self._v_sub = 0
        self._v_hist.clear()
        self._v_state = CS.nominal_cmd(self._v_spec)
        self.command({})

    def reset_setpoints(self):
        self._v_state = CS.nominal_cmd(self._v_spec)
        self._v_inner.reset_setpoints()
        self._v_q.clear()
        self._v_applied = None
        self._v_last_sent = None
        self._v_sub = 0
        self.command({})

    def stop(self):
        upd = {}
        for r in ("lin_vel_x", "lin_vel_y", "yaw_rate"):
            c = CS.cmd(self._v_spec, r)
            if c is not None:
                upd[c["name"]] = 0.0
        self.command(upd)

    # ---- save / restore of the wrapper's own state
    def v_save(self):
        return dict(
            state=copy.deepcopy(self._v_state),
            q=copy.deepcopy(self._v_q),
            applied=copy.deepcopy(self._v_applied),
            t=self._v_t,
            tnext=self._v_tnext,
            hnext=self._v_hnext,
            hist=copy.deepcopy(self._v_hist),
            env=list(self._v_env),
            last=copy.deepcopy(self._v_last_sent),
            sub=copy.deepcopy(self._v_sub),
        )

    def v_restore(self, st):
        self._v_state = copy.deepcopy(st["state"])
        self._v_q = copy.deepcopy(st["q"])
        self._v_applied = copy.deepcopy(st["applied"])
        self._v_t = st["t"]
        self._v_tnext = st["tnext"]
        self._v_hnext = st["hnext"]
        self._v_hist = copy.deepcopy(st["hist"])
        self._v_env = list(st["env"])
        self._v_last_sent = None
        self._v_sub = copy.deepcopy(st["sub"])
        # re-send the command in force (the adapter state was restored separately)
        cur = self._v_sub if isinstance(self._v_sub, dict) else self._transform()
        if self._v_applied is not None:
            cur = self._v_applied
        self._send(dict(cur))

    def hist_arrays(self, geo=None, floor_z=0.0, t_ref=None):
        """-> dict of (60, .) arrays, oldest first, zero-padded (hist_valid 0) when fewer than 60 frames exist"""
        H = list(self._v_hist)
        return pack_records(H, geo, floor_z, t_ref if t_ref is not None else self._v_t, HIST_N)

    def trace_arrays(self, geo=None, floor_z=0.0):
        T = list(self._v_trace or [])
        return pack_records(T, geo, floor_z, None, None)


def pack_records(R, geo, floor_z, t_ref, n_fix):
    n = len(R) if n_fix is None else n_fix
    J = len(R[-1][4]) if R else 0
    t = np.zeros(n, np.float32)
    valid = np.zeros(n, np.uint8)
    cm = np.full((n, len(CS.HIST_ROLES)), np.nan, np.float32)
    ev = np.zeros((n, 4), np.float32)
    ach = np.zeros((n, len(ACH_KEYS)), np.float32)
    q = np.zeros((n, J), np.float32)
    dq = np.zeros((n, J), np.float32)
    off = n - len(R[-n:]) if R else n
    for i, rec in enumerate(R[-n:] if R else []):
        j = off + i
        tt, c, e, a, qq, dd = rec
        t[j] = tt - (t_ref if t_ref is not None else 0.0)
        valid[j] = 1
        cm[j] = c
        ev[j] = e
        a = np.array(a, np.float64)
        a[3] -= floor_z
        if np.isfinite(a[4]):
            a[4] -= floor_z
        if geo is not None:
            p = np.array([a[9], a[10]]) - geo["P0"]
            a[9], a[10] = float(p @ geo["d"]), float(p @ geo["n"])
            a[7] = (a[7] - geo["yaw"] + math.pi) % (2 * math.pi) - math.pi
        ach[j] = a
        if len(qq) == J:
            q[j] = qq
            dq[j] = dd
    return dict(t=t, valid=valid, cmd=cm, env=ev, ach=ach, q=q, dq=dq)


if __name__ == "__main__":
    import json

    for b in ("g1c", "g1g", "g1l", "g1u"):
        base = CS.load(b)
        for vs, w in sample_wrappers(base, 7, seed=3):
            print(
                vs["ctrl_id"],
                w["primary"],
                [(c["name"], c["role"]) for c in vs["commands"]],
                {k: w[k] for k in ("sx", "cap_f", "lat", "latency", "tau", "db_v", "quant", "gears", "h_clip")},
            )
