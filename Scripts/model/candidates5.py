#!/usr/bin/env python
"""
gen5 candidate enumerator (pure python, deterministic). analysis/GEN5_DESIGN.md 3.1, analysis/GEN5_DATA_SPEC.md 3.

enumerate_candidates(spec) -> list of candidate dicts, index = position in the list (stable for a spec):
  [0] KEEP     the current command unchanged (cmd None)
  [1] NOMINAL  the stock command: lin_vel_x = spec nominal.speed, lateral 0, height / pitch at their nominal, the nominal
               mode (spec nominal.mode if given -- PHP: 'forward' -- else the mode command's nominal)
  then, for a spec WITHOUT a discrete mode command:
      posture grid   P = height levels x pitch levels (or {nominal} if the spec has neither)
                       height: 4 levels linspace(range lo, nominal); pitch: humanoid {0, hi/2, hi}, quadruped {lo, 0, hi}
      forward        every posture in P x vx in {0, .25, .5, .75, 1} x vx_max (vy 0)
      lateral        if the spec has lin_vel_y: vy in {-vy_max, +vy_max} x vx levels x height levels (pitch 0 / nominal)
                     (vy_max = min(|lo|, |hi|) of the lateral range)
  for a spec WITH a mode command:
      every mode except steering classes (spec mode 'steer': true -- the path follower uses them):
        static modes (moving false)  one candidate (vx 0 if the spec has lin_vel_x)
        moving modes                 x 3 speeds {lo, mid, hi} of the mode's speed_range (if none: {0.25, 0.625, 1} x vx_max),
                                     clipped to the lin_vel_x range; no vx command -> one candidate
        + lateral (if lin_vel_y)     vy in {-vy_max, +vy_max} at the mid speed
Roll stays at its nominal, wz is NEVER a candidate (steering stays with the stock path follower), torso_yaw nominal.
A candidate's cmd is {role: value} (roles, not names; to_names() maps onto the spec's names).

transition_class(spec, cmd) -> the class used by the transition probe / descriptor:
  'nominal' | 'stop' (vx 0) | 'speed_max' (vx >= 0.9 vx_max) | 'lat_L' / 'lat_R' | 'post:h<h>_p<p>' | 'mode:<name>'
probe_classes(spec) -> the classes the transition probe measures (every class a candidate can have, except 'nominal').

    python3 Scripts/model/candidates5.py            # sizes per controller (data/gen_v5/specs)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import glob, json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import ctrl5 as K5
import ctrl_spec as CS

CMD_ROLES = ("lin_vel_x", "lin_vel_y", "pelvis_height", "torso_pitch", "mode")
VX_F = (0.0, 0.25, 0.5, 0.75, 1.0)
MODE_VX_F = (0.25, 0.625, 1.0)


def r4(x):
    return round(float(x), 4)


def is_humanoid(spec):
    return str(spec.get("robot", "")).startswith("unitree") or K5.robot_of(spec["ctrl_id"]) == "g1"


def vx_max(spec):
    r = CS.rng(spec, "lin_vel_x")
    return None if r is None else float(r[1])


def vy_max(spec):
    r = CS.rng(spec, "lin_vel_y")
    return None if r is None else float(min(abs(r[0]), abs(r[1])))


def height_levels(spec):
    c = CS.cmd(spec, "pelvis_height")
    if c is None or c["type"] != "continuous" or c.get("applies_to"):
        return None
    lo, hi = c["range"]; nom = float(c["nominal"])
    return [r4(x) for x in np.linspace(lo, nom, 4)]


def pitch_levels(spec):
    c = CS.cmd(spec, "torso_pitch")
    if c is None or c["type"] != "continuous":
        return None
    lo, hi = c["range"]
    return [0.0, r4(0.5 * hi), r4(hi)] if is_humanoid(spec) else [r4(lo), 0.0, r4(hi)]


def nominal_cmd(spec):
    out = {}
    if CS.has(spec, "lin_vel_x"):
        out["lin_vel_x"] = r4(spec.get("nominal", {}).get("speed", 0.3))
    if CS.has(spec, "lin_vel_y"):
        out["lin_vel_y"] = 0.0
    for r in ("pelvis_height", "torso_pitch"):
        c = CS.cmd(spec, r)
        if c is not None and c["type"] == "continuous" and not c.get("applies_to"):
            out[r] = r4(c["nominal"])
    if CS.has(spec, "mode"):
        out["mode"] = spec.get("nominal", {}).get("mode", CS.cmd(spec, "mode")["nominal"])
    return out


def enumerate_candidates(spec):
    C = [dict(kind="KEEP", cmd=None), dict(kind="NOMINAL", cmd=nominal_cmd(spec))]
    vm, ym = vx_max(spec), vy_max(spec)
    nom = nominal_cmd(spec)
    vxs = [r4(f * vm) for f in VX_F] if vm is not None else [None]
    if not CS.has(spec, "mode"):
        hs, ps = height_levels(spec), pitch_levels(spec)
        posts = []
        for h in (hs or [None]):
            for p in (ps or [None]):
                posts.append((h, p))
        p0 = 0.0 if ps is not None else None
        for (h, p) in posts:
            for v in vxs:
                c = {}
                if v is not None:
                    c["lin_vel_x"] = v
                if ym is not None:
                    c["lin_vel_y"] = 0.0
                if h is not None:
                    c["pelvis_height"] = h
                if p is not None:
                    c["torso_pitch"] = p
                C.append(dict(kind="grid", cmd=c))
        if ym is not None:
            for vy in (-ym, ym):
                for h in (hs or [None]):
                    for v in vxs:
                        c = {"lin_vel_y": r4(vy)}
                        if v is not None:
                            c["lin_vel_x"] = v
                        if h is not None:
                            c["pelvis_height"] = h
                        if p0 is not None:
                            c["torso_pitch"] = p0
                        C.append(dict(kind="lateral", cmd=c))
    else:
        mc = CS.cmd(spec, "mode")
        for m in mc["values"]:
            mi = CS.mode_info(spec, m) or {}
            if mi.get("steer"):
                continue
            if not mi.get("moving", True) or vm is None:
                c = {"mode": m}
                if vm is not None:
                    c["lin_vel_x"] = 0.0
                if ym is not None:
                    c["lin_vel_y"] = 0.0
                C.append(dict(kind="mode", cmd=c))
                continue
            sr = mi.get("speed_range")
            lo, hi = CS.rng(spec, "lin_vel_x")
            if sr:
                a, b = float(np.clip(sr[0], 0.0, hi)), float(np.clip(sr[1], 0.0, hi))
                lv = [a, 0.5 * (a + b), b]
            else:
                lv = [f * vm for f in MODE_VX_F]
            for v in lv:
                c = {"mode": m, "lin_vel_x": r4(v)}
                if ym is not None:
                    c["lin_vel_y"] = 0.0
                C.append(dict(kind="mode", cmd=c))
            if ym is not None:
                for vy in (-ym, ym):
                    C.append(dict(kind="mode_lateral", cmd={"mode": m, "lin_vel_x": r4(lv[1]), "lin_vel_y": r4(vy)}))
    for i, c in enumerate(C):
        c["idx"] = i
        c["cls"] = transition_class(spec, c["cmd"] if c["cmd"] is not None else nom)
    return C


def transition_class(spec, cmd):
    nom = nominal_cmd(spec)
    if cmd is None:
        return "keep"
    if "mode" in cmd and cmd["mode"] != nom.get("mode"):
        return f"mode:{cmd['mode']}"
    h, p = cmd.get("pelvis_height", nom.get("pelvis_height")), cmd.get("torso_pitch", nom.get("torso_pitch"))
    if (h is not None and abs(h - nom.get("pelvis_height", h)) > 1e-3) or (p is not None and abs(p - nom.get("torso_pitch", p)) > 1e-3):
        return f"post:h{h if h is not None else 0:.2f}_p{p if p is not None else 0:.2f}"
    vy = float(cmd.get("lin_vel_y", 0.0) or 0.0)
    if abs(vy) > 1e-3:
        return "lat_L" if vy > 0 else "lat_R"
    vm = vx_max(spec)
    if vm is not None and "lin_vel_x" in cmd:
        v = float(cmd["lin_vel_x"])
        if v <= 1e-3:
            return "stop"
        if v >= 0.9 * vm:
            return "speed_max"
    return "nominal"


def probe_classes(spec):
    """-> ordered list of (class, cmd) the transition probe measures (one representative command per class)"""
    seen, out = set(), []
    nom = nominal_cmd(spec)
    for c in enumerate_candidates(spec):
        if c["cmd"] is None or c["cls"] in ("nominal", "keep") or c["cls"] in seen:
            continue
        rep = dict(c["cmd"])
        if c["cls"].startswith(("post:", "mode:")):     # posture / mode classes are switched at the probe speed (vx follows)
            rep.pop("lin_vel_x", None); rep.pop("lin_vel_y", None)
        if c["cls"] in ("lat_L", "lat_R"):
            rep = {"lin_vel_y": c["cmd"]["lin_vel_y"]}
        if c["cls"] == "speed_max":
            rep = {"lin_vel_x": vx_max(spec)}
        if c["cls"] == "stop":
            rep = {"lin_vel_x": 0.0}
        seen.add(c["cls"]); out.append((c["cls"], rep))
    return out


def to_names(spec, cmd):
    """{role: value} -> {spec name: value}"""
    out = {}
    for r, v in (cmd or {}).items():
        c = CS.cmd(spec, r)
        if c is not None:
            out[c["name"]] = v
    return out


def cmd_of_state(spec, state):
    """VRob state {name: value} -> {role: value} over CMD_ROLES"""
    out = {}
    for r in CMD_ROLES:
        c = CS.cmd(spec, r)
        if c is not None and c["name"] in state:
            v = state[c["name"]]
            out[r] = v if r == "mode" else r4(v)
    return out


def same_cmd(a, b):
    if a is None or b is None:
        return False
    for k in set(a) | set(b):
        x, y = a.get(k), b.get(k)
        if isinstance(x, str) or isinstance(y, str):
            if x != y:
                return False
        elif x is None or y is None or abs(float(x) - float(y)) > 1e-3:
            return False
    return True


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(K5.SPEC_D5, "*.json"))):
        s = CS.load(f)
        C = enumerate_candidates(s)
        kinds = {}
        for c in C:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        rows.append((s["ctrl_id"], len(C), kinds, len(probe_classes(s))))
    for r in rows:
        print(f"{r[0]:12s} N={r[1]:4d} {r[2]} transition classes {r[3]}")


if __name__ == "__main__":
    main()
