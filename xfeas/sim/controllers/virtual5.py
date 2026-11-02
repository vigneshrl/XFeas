"""
gen_v5 additions to the controller port (xfeas/_support/data_collection/virtual.py is NOT modified; this module patches the VRob class in
the importing process only). import virtual5 once, after virtual. Wrappers without an 'l5' list behave exactly as before.

Hidden wrapper effects (wcfg['l5'], list of dicts; written by make_controller_specs.py, never a model input):
  tall_crouch    {k_h, k_p}: h_eff = h_nom - k_h (h_nom - h); p_eff = k_p * p           (static, in _transform)
  lat_deadzone   {dz_f}:     vy_eff = sign(vy) max(0, |vy| - dz_f vy_max)              (static)
  mode_roll      {roll: {mode: rad}}: roll offset added in that style mode               (static)
  height_lag     {tau, delay}: height / pitch reach the base through a delay + first-order lag   (dynamic, 50 Hz)
  switch_glitch  {trigger vx_up | posture_change | mode_in, vx_hi_f, vx_prev_f, modes, T, period, amp}:
                 on the trigger a T-second burst of alternating max roll / lateral / yaw-rate commands (+ lowest height and
                 max forward lean / alternating pitch where the base has them) is sent instead
                 (dynamic). Triggers compare the NEW base command with the previous one:
                   vx_up           new vx >= vx_hi_f vmax, raised by >= 2 % of vmax, previous vx >= vx_prev_f vmax
                   posture_change  height / pitch / mode changed while the base vx >= vx_prev_f vmax
                   mode_in         the virtual mode changed INTO one of `modes` while the base vx >= vx_prev_f vmax
Base-binding plumbing (real controllers, gen5 specs):
  binding.mode_values == 'name'  the adapter takes the mode NAME (PHP direction classes), not the id
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
from collections import deque
import numpy as np
import ctrl_spec as CS
import virtual as VC

if "_v5" not in VC.VRob._OWN:
    VC.VRob._OWN = tuple(VC.VRob._OWN) + ("_v5",)
_ORIG_TRANSFORM = VC.VRob._transform
_ORIG_SEND = VC.VRob._send
_ORIG_STEP = VC.VRob.step
_ORIG_SETCTRL = VC.VRob.set_controller
_ORIG_TELEPORT = VC.VRob.teleport
_ORIG_RESET = VC.VRob.reset_setpoints


def _keys(base):
    """role -> adapter key of the base spec"""
    b = base["binding"].get("names", {})
    return {c["role"]: b.get(c["name"], c["name"]) for c in base["commands"]}


def _l5(self, kind):
    return [e for e in (self._v_w.get("l5") or []) if e.get("kind") == kind]


def _new_state():
    return dict(
        prev=None,
        glitch_t1=-1.0,
        glitch_t0=-1.0,
        lag_q=deque(),
        lag_app=None,
        desired=None,
        t_next=0.0,
        n_glitch=0,
        last_sent=None,
    )


def set_controller(self, spec, wcfg):
    _ORIG_SETCTRL(self, spec, wcfg)
    object.__setattr__(self, "_v5", _new_state())


def teleport(self, x, y, yaw, floor_z):
    object.__setattr__(self, "_v5", _new_state())
    return _ORIG_TELEPORT(self, x, y, yaw, floor_z)


def reset_setpoints(self):
    object.__setattr__(self, "_v5", _new_state())
    return _ORIG_RESET(self)


def _transform(self):
    out = _ORIG_TRANSFORM(self)
    base = self._v_base
    K = _keys(base)
    for e in _l5(self, "tall_crouch"):
        if "pelvis_height" in K and K["pelvis_height"] in out:
            hnom = CS.cmd(base, "pelvis_height")["nominal"]
            h = float(out[K["pelvis_height"]])
            if h >= 0:
                out[K["pelvis_height"]] = hnom - float(e["k_h"]) * (hnom - h)
        if "torso_pitch" in K and K["torso_pitch"] in out:
            out[K["torso_pitch"]] = float(e["k_p"]) * float(out[K["torso_pitch"]])
    for e in _l5(self, "lat_deadzone"):
        if "lin_vel_y" in K and K["lin_vel_y"] in out:
            vmax = max(abs(x) for x in CS.rng(base, "lin_vel_y"))
            dz = float(e["dz_f"]) * vmax
            v = float(out[K["lin_vel_y"]])
            out[K["lin_vel_y"]] = math.copysign(max(0.0, abs(v) - dz), v)
    for e in _l5(self, "mode_roll"):
        m = self._role_val("mode")
        if "torso_roll" in K and m in e["roll"]:
            lo, hi = CS.rng(base, "torso_roll")
            out[K["torso_roll"]] = float(np.clip(float(out.get(K["torso_roll"], 0.0)) + float(e["roll"][m]), lo, hi))
    if base["binding"].get("mode_values") == "name" and "mode" in K and K["mode"] in out:
        v = out[K["mode"]]
        if isinstance(v, (int, np.integer)):
            out[K["mode"]] = base["modes"][int(v)]["name"]
    return out


def _dynamic(self):
    return bool(_l5(self, "height_lag") or _l5(self, "switch_glitch"))


def _filtered(self, desired):
    """desired base command -> what the hidden controller actually sends (dynamic effects)"""
    st = self._v5
    base = self._v_base
    K = _keys(base)
    t = float(self._v_t)
    out = dict(desired)
    for e in _l5(self, "height_lag"):
        ks = [K[r] for r in ("pelvis_height", "torso_pitch") if r in K and K[r] in desired]
        if not ks:
            continue
        q = st["lag_q"]
        if not q or q[-1][1] != {k: desired[k] for k in ks}:
            q.append((t + float(e["delay"]), {k: desired[k] for k in ks}))
        tgt = None
        while q and q[0][0] <= t:
            tgt = q.popleft()[1]
        if tgt is not None:
            st["lag_tgt"] = tgt
        tgt = st.get("lag_tgt")
        if st["lag_app"] is None:  # starts from the base's nominal posture (the state after a teleport / reset)
            nomv = {
                K[r]: (float(CS.cmd(base, r)["nominal"]) if r == "pelvis_height" else 0.0)
                for r in ("pelvis_height", "torso_pitch")
                if r in K
            }
            st["lag_app"] = {k: nomv.get(k, float(desired[k])) for k in ks}
        if tgt is not None:
            a = 1.0 - math.exp(-0.02 / float(e["tau"]))
            for k in ks:
                st["lag_app"][k] += a * (float(tgt[k]) - st["lag_app"][k])
        for k in ks:
            out[k] = round(st["lag_app"][k], 4)
    if st["glitch_t1"] > t:
        e = _l5(self, "switch_glitch")[0]
        ph = int((t - st["glitch_t0"]) / float(e["period"])) % 2
        sg = 1.0 if ph == 0 else -1.0
        amp = float(e["amp"])
        for r in ("torso_roll", "lin_vel_y", "yaw_rate"):
            if r in K:
                lo, hi = CS.rng(base, r)
                out[K[r]] = round(sg * amp * min(abs(lo), abs(hi)), 4)
        # the burst also drops the body and throws the torso (deep forward lean on a humanoid, alternating pitch on a
        # quadruped): AMO ignored roll / lateral bursts in the first transition probe (max tilt 0.04)
        if "torso_pitch" in K:
            lo, hi = CS.rng(base, "torso_pitch")
            hum = str(base.get("robot", "")).startswith("unitree")
            out[K["torso_pitch"]] = round(amp * hi if hum else sg * amp * min(abs(lo), abs(hi)), 4)
        if "pelvis_height" in K:
            lo, hi = CS.rng(base, "pelvis_height")
            nom = float(CS.cmd(base, "pelvis_height")["nominal"])
            out[K["pelvis_height"]] = round(lo + (1.0 - amp) * (nom - lo), 4)
    return out


def _check_trigger(self, desired):
    st = self._v5
    base = self._v_base
    K = _keys(base)
    prev = st["prev"]
    st["prev"] = dict(desired)
    if prev is None or st["glitch_t1"] > float(self._v_t):
        return
    vxr = CS.rng(base, "lin_vel_x")
    if vxr is None:
        return
    vmax = float(vxr[1])
    kx = K.get("lin_vel_x")
    vx_new = float(desired.get(kx, 0.0))
    vx_old = float(prev.get(kx, 0.0))
    for e in _l5(self, "switch_glitch"):
        tr = e["trigger"]
        fire = False
        if tr == "vx_up":
            fire = (
                vx_new >= float(e["vx_hi_f"]) * vmax
                and vx_new >= vx_old + 0.02 * vmax
                and vx_old >= float(e["vx_prev_f"]) * vmax
            )
        elif tr == "posture_change":
            ks = [K[r] for r in ("pelvis_height", "torso_pitch", "mode") if r in K]
            fire = (
                any(desired.get(k) != prev.get(k) for k in ks) and min(vx_new, vx_old) >= float(e["vx_prev_f"]) * vmax
            )
        elif tr == "mode_in":
            m = self._role_val("mode")
            fire = m in e["modes"] and st.get("mode_prev") not in e["modes"] and vx_old >= float(e["vx_prev_f"]) * vmax
        if fire:
            st["glitch_t0"] = float(self._v_t)
            st["glitch_t1"] = float(self._v_t) + float(e["T"])
            st["n_glitch"] += 1
            break
    st["mode_prev"] = self._role_val("mode")


def _send(self, base_cmd):
    if not (self._v_w.get("l5") and _dynamic(self)):
        return _ORIG_SEND(self, base_cmd)
    st = self._v5
    st["desired"] = dict(base_cmd)
    _check_trigger(self, st["desired"])
    act = _filtered(self, st["desired"])
    if act != st["last_sent"]:
        self._v_inner.command(dict(act))
        st["last_sent"] = dict(act)
    self._v_last_sent = dict(base_cmd)  # the fall judge reads the COMMANDED posture (spec semantics)


def step(self, dt):
    if self._v_w.get("l5") and _dynamic(self):
        st = self._v5
        if st["desired"] is not None and float(self._v_t) >= st["t_next"]:
            st["t_next"] = float(self._v_t) + 0.02
            act = _filtered(self, st["desired"])
            if act != st["last_sent"]:
                self._v_inner.command(dict(act))
                st["last_sent"] = dict(act)
    return _ORIG_STEP(self, dt)


def glitch_count(rob):
    try:
        return int(rob._v5["n_glitch"])
    except Exception:
        return 0


VC.VRob.set_controller = set_controller
VC.VRob.teleport = teleport
VC.VRob.reset_setpoints = reset_setpoints
VC.VRob._transform = _transform
VC.VRob._send = _send
VC.VRob.step = step
