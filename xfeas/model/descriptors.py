#!/usr/bin/env python
"""Describes each candidate command by what the controller's self-test MEASURED it does (body height and width
reached, speed, settle time, tilt, falls, how safe it is to switch to it at the current speed), plus the command
normalised by the spec range. This is what lets one model compare commands across different controllers.

Details:
gen5 candidate DESCRIPTOR (pure python).,.

Describer(spec, probe, trans).describe(cands, cur_cmd, cur_speed) -> (N, F) float32 matrix, F = len(DESC_KEYS).
Every candidate is described by WHAT THE SELF-TEST MEASURED it does on this controller, never by a name:

 effect (static per candidate; from the nearest probed op of probes_v4 'ops' + 'drv_ops'):
   e_top, e_width, e_depth, e_pitch       body top height / width / depth / torso pitch of the matched op (m, rad)
   e_vx, e_vy                             steady forward / lateral speed predicted for THIS command: e_vx = posture speed
                                          ratio (op vx / nominal-posture vx at the op's command) x nominal-posture steady
                                          speed interpolated at the candidate vx (speed ops + drv_ops forward rows); e_vy =
                                          drv_ops lateral row of that posture at the nearest vy command (0 if none)
   e_settle                               nominal -> this class transition settle time (transitions, mean of speeds), else
                                          the speed ops' lag63, else 0 (+ e_settle_missing)
   e_tilt, e_moving, e_fell, e_match_d    max tilt, op moving, op fell, normalised command distance to the matched op
   e_missing                              no op matched (no probe)
 raw command, normalised by the spec range (role absent -> 0 and has_* = 0):
   c_vx = vx / vx_max, c_vy = vy / vy_max, c_h = (h - h_nom) / (h_nom - h_lo) (<= 0 crouch), c_p = p / max|p range|,
   c_mode_static, c_mode_v (mode candidate's speed / vx_max), has_vx, has_vy, has_h, has_p, has_mode
 flags: is_keep, is_nominal, is_current (equal to the command in force), is_stop (vx 0 or a static mode)
 transition current -> candidate (dynamic; transitions.json; speed-blended between the slow and fast probe records by the
 measured |vx| now):
   t_settle, t_tilt, t_fell, t_stepping, t_same (same class: no switch), t_composed (X -> nominal -> Y), t_missing
Names never enter (mode names are only used to LOOK UP their measured effect).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import controller_registry as K5
import ctrl_spec as CS
import candidates as CA

DESC_KEYS = [
    "e_top",
    "e_width",
    "e_depth",
    "e_pitch",
    "e_vx",
    "e_vy",
    "e_settle",
    "e_settle_missing",
    "e_tilt",
    "e_moving",
    "e_fell",
    "e_match_d",
    "e_missing",
    "c_vx",
    "c_vy",
    "c_h",
    "c_p",
    "c_mode_static",
    "c_mode_v",
    "has_vx",
    "has_vy",
    "has_h",
    "has_p",
    "has_mode",
    "is_keep",
    "is_nominal",
    "is_current",
    "is_stop",
    "t_settle",
    "t_tilt",
    "t_fell",
    "t_stepping",
    "t_same",
    "t_composed",
    "t_missing",
]
F_DESC = len(DESC_KEYS)
KI = {k: i for i, k in enumerate(DESC_KEYS)}


def _nz(x, d=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return d
    return x if np.isfinite(x) else d


class Describer:
    def __init__(self, spec, probe=None, trans=None):
        self.spec, self.P, self.T = spec, probe, trans
        self.nom = CA.nominal_cmd(spec)
        self.vm, self.ym = CA.vx_max(spec), CA.vy_max(spec)
        hc = CS.cmd(spec, "pelvis_height")
        self.h = (
            (float(hc["range"][0]), float(hc["nominal"]))
            if hc is not None and hc["type"] == "continuous" and not hc.get("applies_to")
            else None
        )
        pc = CS.cmd(spec, "torso_pitch")
        self.pmax = max(abs(pc["range"][0]), abs(pc["range"][1])) if pc is not None else None
        self.names = {r: (CS.cmd(spec, r)["name"] if CS.has(spec, r) else None) for r in CA.CMD_ROLES}
        self._speed_curve()
        self._static = {}

    # ---------------------------------------------------------------- probe look-ups
    def _speed_curve(self):
        """nominal-posture steady vx as a function of the vx command (speed ops + drv_ops forward rows), monotone interp"""
        pts = [(0.0, 0.0)]
        if self.P is not None:
            for o in self.P.get("ops", []):
                if o["kind"] == "speed" and not o["fell"]:
                    pts.append((float(o["vx_cmd"]), float(o["vx_ach"])))
            for o in self.P.get("drv_ops", []) or []:
                if o.get("kind") == "speed" and o.get("post_label") == "nominal" and not o.get("fell"):
                    pts.append((float(o["vx_cmd"]), float(o["vx_ach"])))
        pts.sort()
        self.sc_x = np.array([p[0] for p in pts])
        self.sc_y = np.maximum.accumulate(np.array([p[1] for p in pts]))
        self.has_speed_ops = len(pts) > 1

    def vx_nom(self, v):
        if not self.has_speed_ops:
            return float(v)
        return float(np.interp(v, self.sc_x, self.sc_y))

    def _ops(self):
        return [] if self.P is None else self.P.get("ops", [])

    def _match_op(self, cmd):
        """nearest posture / mode op -> (op, distance) ; nominal posture -> the best speed op"""
        ops = self._ops()
        if not ops:
            return None, 9.0
        if "mode" in cmd and cmd["mode"] != self.nom.get("mode"):
            lab = f"mode:{cmd['mode']}"
            for o in ops:
                if o["label"] == lab:
                    return o, 0.0
            for o in ops:
                if o["kind"] == "stand" and not (CS.mode_info(self.spec, cmd["mode"]) or {}).get("moving", True):
                    return o, 0.5
            return None, 9.0
        hn, pn = self.names["pelvis_height"], self.names["torso_pitch"]
        h = cmd.get("pelvis_height")
        p = cmd.get("torso_pitch")
        post = (h is not None and abs(h - self.nom.get("pelvis_height", h)) > 1e-3) or (
            p is not None and abs(p - self.nom.get("torso_pitch", p)) > 1e-3
        )
        if not post:
            sp = [o for o in ops if o["kind"] == "speed" and not o["fell"]]
            if not sp:
                sp = [o for o in ops if o["kind"] == "stand"]
            v = float(cmd.get("lin_vel_x", 0.0) or 0.0)
            return (min(sp, key=lambda o: abs(o["vx_cmd"] - v)) if sp else None), 0.0
        best, bd = None, 9.0
        hspan = (self.h[1] - self.h[0]) if self.h else 1.0
        for o in ops:
            if o["kind"] != "posture":
                continue
            oh = o["cmd"].get(hn, self.nom.get("pelvis_height")) if hn else None
            op = o["cmd"].get(pn, 0.0) if pn else 0.0
            dd = 0.0
            if h is not None and oh is not None:
                dd += ((h - oh) / max(hspan, 1e-3)) ** 2
            if p is not None and self.pmax:
                dd += ((p - op) / self.pmax) ** 2
            dd = math.sqrt(dd)
            if dd < bd:
                best, bd = o, dd
        return best, bd

    def _lat(self, cmd, op_label):
        vy = float(cmd.get("lin_vel_y", 0.0) or 0.0)
        if abs(vy) < 1e-3 or self.P is None:
            return 0.0
        rows = [o for o in (self.P.get("drv_ops") or []) if o.get("kind") == "lateral" and not o.get("fell")]
        r1 = [o for o in rows if o.get("post_label") == op_label] or [
            o for o in rows if o.get("post_label") == "nominal"
        ]
        r1 = [o for o in r1 if np.sign(o["vy_cmd"]) == np.sign(vy)]
        if r1:
            o = min(r1, key=lambda o: abs(abs(o["vy_cmd"]) - abs(vy)))
            return float(o["vy_ach"])
        caps = self.P.get("caps", {})
        return float(caps.get("vy_max_L" if vy > 0 else "vy_max_R", 0.0))

    def _trans_rec(self, cls, direction, speed):
        """transition record of class cls ('to' = nominal -> cls, 'back' = cls -> nominal), blended by speed"""
        if self.T is None or cls not in self.T.get("classes", {}):
            return None
        R = self.T["classes"][cls][direction]
        sp = self.T.get("speeds", {})
        keys = [k for k in R if R[k] is not None]
        if not keys:
            return None
        if len(keys) == 1 or "slow" not in R or "fast" not in R:
            r = R[keys[0]]
            return dict(
                settle=_nz(r.get("settle_s")),
                tilt=_nz(r.get("max_tilt")),
                fell=float(bool(r.get("fell"))),
                step=float(bool(r.get("stepping"))),
            )
        vs, vf = _nz(sp.get("slow")), _nz(sp.get("fast"))
        rs, rf = R["slow"], R["fast"]
        # measured speeds of the two run-ups (vx_before) when known, else the commands
        ms = _nz(rs.get("vx_before"), vs)
        mf = _nz(rf.get("vx_before"), vf)
        w = float(np.clip((abs(speed) - ms) / max(mf - ms, 1e-3), 0.0, 1.0))
        g = lambda r, k: _nz(r.get(k))
        return dict(
            settle=(1 - w) * g(rs, "settle_s") + w * g(rf, "settle_s"),
            tilt=(1 - w) * g(rs, "max_tilt") + w * g(rf, "max_tilt"),
            fell=max(float(bool(rs.get("fell"))) * (1 - w), float(bool(rf.get("fell"))) * w)
            if (rs.get("fell") or rf.get("fell"))
            else 0.0,
            step=(1 - w) * float(bool(rs.get("stepping"))) + w * float(bool(rf.get("stepping"))),
        )

    # ---------------------------------------------------------------- per candidate
    def static_row(self, c):
        key = (
            c["kind"] + "|" + (json.dumps(c["cmd"], sort_keys=True) if c["cmd"] is not None else "KEEP")
        )  # kind: flags differ
        if key in self._static:
            return self._static[key].copy()
        x = np.zeros(F_DESC, np.float32)
        cmd = c["cmd"] if c["cmd"] is not None else self.nom
        op, dist = self._match_op(cmd)
        v = float(cmd.get("lin_vel_x", 0.0) or 0.0)
        if op is not None:
            x[KI["e_top"]] = _nz(op.get("top_min"))
            x[KI["e_width"]] = _nz(op.get("width"))
            x[KI["e_depth"]] = _nz(op.get("depth"))
            x[KI["e_pitch"]] = _nz(op.get("pitch_ach"))
            x[KI["e_tilt"]] = _nz(op.get("tilt_max"))
            x[KI["e_moving"]] = float(
                bool(op.get("moving")) and not (self.vm is not None and "lin_vel_x" in cmd and v <= 1e-3)
            )
            x[KI["e_fell"]] = float(bool(op.get("fell")))
            x[KI["e_match_d"]] = min(dist, 3.0)
            if op["kind"] in ("posture", "mode"):
                base_v = self.vx_nom(float(op["vx_cmd"]))
                ratio = _nz(op["vx_ach"]) / base_v if base_v > 0.02 else (1.0 if op.get("moving") else 0.0)
                if self.vm is None or "lin_vel_x" not in cmd:
                    x[KI["e_vx"]] = _nz(op["vx_ach"])
                elif op["kind"] == "mode" and self.vm is not None:
                    x[KI["e_vx"]] = float(np.clip(ratio, 0.0, 3.0)) * self.vx_nom(v) if v > 0 else 0.0
                else:
                    x[KI["e_vx"]] = float(np.clip(ratio, 0.0, 3.0)) * self.vx_nom(v)
            else:
                x[KI["e_vx"]] = self.vx_nom(v) if self.vm is not None else _nz(op["vx_ach"])
            x[KI["e_vy"]] = self._lat(cmd, op["label"] if op["kind"] in ("posture", "mode") else "nominal")
        else:
            x[KI["e_missing"]] = 1.0
            x[KI["e_match_d"]] = 3.0
        tr = self._trans_rec(c["cls"], "to", 0.0) if c["cls"] not in ("nominal", "keep") else None
        if tr is not None:
            tf = self._trans_rec(c["cls"], "to", 9.0)
            x[KI["e_settle"]] = 0.5 * (tr["settle"] + tf["settle"])
        else:
            lag = _nz((self.P or {}).get("caps", {}).get("lag_tau"), np.nan) if self.P else np.nan
            if np.isfinite(lag):
                x[KI["e_settle"]] = lag
            else:
                x[KI["e_settle_missing"]] = 1.0
        if self.vm is not None:
            x[KI["has_vx"]] = 1.0
            x[KI["c_vx"]] = v / self.vm
        if self.ym is not None:
            x[KI["has_vy"]] = 1.0
            x[KI["c_vy"]] = float(cmd.get("lin_vel_y", 0.0) or 0.0) / self.ym
        if self.h is not None:
            x[KI["has_h"]] = 1.0
            x[KI["c_h"]] = (float(cmd.get("pelvis_height", self.h[1])) - self.h[1]) / max(self.h[1] - self.h[0], 1e-3)
        if self.pmax:
            x[KI["has_p"]] = 1.0
            x[KI["c_p"]] = float(cmd.get("torso_pitch", 0.0)) / self.pmax
        if "mode" in cmd:
            x[KI["has_mode"]] = 1.0
            mi = CS.mode_info(self.spec, cmd["mode"]) or {}
            x[KI["c_mode_static"]] = float(not mi.get("moving", True))
            x[KI["c_mode_v"]] = (v / self.vm) if self.vm else float(bool(mi.get("moving", True)))
        x[KI["is_keep"]] = float(c["kind"] == "KEEP")
        x[KI["is_nominal"]] = float(c["kind"] == "NOMINAL")
        x[KI["is_stop"]] = float((self.vm is not None and v <= 1e-3) or x[KI["c_mode_static"]] > 0.5)
        self._static[key] = x
        return x.copy()

    def describe(self, cands, cur_cmd, cur_speed):
        """cands: candidates list; cur_cmd: {role: value} in force; cur_speed: measured forward speed (m/s)"""
        cur_cls = CA.transition_class(self.spec, cur_cmd) if cur_cmd else "nominal"
        X = np.zeros((len(cands), F_DESC), np.float32)
        for i, c in enumerate(cands):
            if c["kind"] == "KEEP":
                x = self.static_row(dict(c, cmd=cur_cmd, kind="KEEP", cls=cur_cls))
            else:
                x = self.static_row(c)
            x[KI["is_current"]] = float(c["kind"] == "KEEP" or CA.same_cmd(c["cmd"], cur_cmd))
            new_cls = cur_cls if c["kind"] == "KEEP" else c["cls"]
            if new_cls == cur_cls:
                x[KI["t_same"]] = 1.0
            else:
                if cur_cls == "nominal":
                    r = self._trans_rec(new_cls, "to", cur_speed)
                    comp = False
                elif new_cls == "nominal":
                    r = self._trans_rec(cur_cls, "back", cur_speed)
                    comp = False
                else:
                    r1 = self._trans_rec(cur_cls, "back", cur_speed)
                    r2 = self._trans_rec(new_cls, "to", cur_speed)
                    comp = True
                    r = (
                        None
                        if (r1 is None or r2 is None)
                        else dict(
                            settle=r1["settle"] + r2["settle"],
                            tilt=max(r1["tilt"], r2["tilt"]),
                            fell=max(r1["fell"], r2["fell"]),
                            step=max(r1["step"], r2["step"]),
                        )
                    )
                if r is None:
                    x[KI["t_missing"]] = 1.0
                else:
                    x[KI["t_settle"]] = r["settle"]
                    x[KI["t_tilt"]] = r["tilt"]
                    x[KI["t_fell"]] = r["fell"]
                    x[KI["t_stepping"]] = r["step"]
                    x[KI["t_composed"]] = float(comp)
            X[i] = x
        return X


def load_probe_pair(ctrl_id, root=None):
    import controller_registry as K5

    d = os.path.join(root or K5.PROBES_V5, ctrl_id)
    P = (
        json.load(open(os.path.join(d, "office_v4.json")))
        if os.path.exists(os.path.join(d, "office_v4.json"))
        else None
    )
    T = (
        json.load(open(os.path.join(d, "transitions.json")))
        if os.path.exists(os.path.join(d, "transitions.json"))
        else None
    )
    return P, T


if __name__ == "__main__":
    import controller_registry as K5

    for cid in sys.argv[1:] or ["g1c", "spotp", "v_g1c_08"]:
        s = CS.load(K5.spec_path(cid))
        P, T = load_probe_pair(cid)
        if P is None:
            P = (
                json.load(open(os.path.join(K5.PROBES_V4, cid, "office_v4.json")))
                if os.path.exists(os.path.join(K5.PROBES_V4, cid, "office_v4.json"))
                else None
            )
        D = Describer(s, P, T)
        C = CA.enumerate_candidates(s)
        X = D.describe(C, CA.nominal_cmd(s), 0.2)
        print(cid, X.shape, "trans" if T else "no-trans", "probe" if P else "no-probe")
        for i in [0, 1, 2, 7, len(C) // 2, len(C) - 1]:
            print(
                "  ",
                i,
                C[i]["kind"],
                C[i]["cls"],
                C[i]["cmd"],
                {k: round(float(X[i, j]), 3) for j, k in enumerate(DESC_KEYS) if abs(X[i, j]) > 1e-6},
            )
