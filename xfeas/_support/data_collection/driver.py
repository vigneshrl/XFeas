"""
DRIVER (not learned): envelope action + macro -> the controller's own commands, from the probe's measured operating points.
analysis/GEN3_SPEC.md sections 2, 3, 5. Pure python.

    D = Driver(spec, probe)            # probe = summarize() output (ops, caps, fp)
    prog = D.plan(act)                 # act = envelope action dict (see ACT_KEYS), returns an executable program dict
    v, m = act_vec(act)                # (16,) env_act + mask
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import ctrl_spec as CS

MACROS = dict(constant=0, two_phase=1, turn_side_turn=2, detour=3)
UNC_TOP, UNC_WID = 9.9, 9.9
TOL = 0.01
WZ_WORKS = (math.pi / 2) / 15.0  # rad/s: a 90 deg turn within 15 s (3 x the 5 s stall watchdog)


def stock_act(spec):
    v = float(spec.get("nominal", {}).get("speed", 0.3))
    return dict(macro="constant", v_b=v, top_b=UNC_TOP, wid_b=UNC_WID, pitch_b=0.0, stock=True)


def act_vec(a):
    """envelope action dict -> (16,) f32 + (16,) u8 mask (GEN3_SPEC section 2)"""
    v = np.zeros(16, np.float32)
    m = np.zeros(16, np.uint8)
    mac = a["macro"]
    g = lambda k, d=0.0: float(a.get(k, d))
    v[0], v[1], v[2], v[3] = g("v_b"), g("top_b", UNC_TOP), g("wid_b", UNC_WID), g("pitch_b")
    v[4], v[5], v[6], v[7] = (
        g("v_a", g("v_b")),
        g("top_a", g("top_b", UNC_TOP)),
        g("wid_a", g("wid_b", UNC_WID)),
        g("pitch_a", g("pitch_b")),
    )
    m[0:4] = 1
    if mac == "two_phase":
        v[8], v[9] = g("d_commit"), g("d_level")
        m[0:10] = 1
        if a.get("yaw_off"):
            v[11], v[15] = g("yaw_off"), g("d_yaw", 0.8)
            m[[11, 15]] = 1
    elif mac == "turn_side_turn":
        v[8] = g("d_commit", 0.6)
        v[10] = g("e_gap")
        v[11] = -g("side") * math.pi / 2
        v[12] = g("side") * abs(g("vy"))
        m[[0, 1, 2, 8, 10, 11, 12]] = 1
    elif mac == "detour":
        v[10] = g("lat_off")
        v[11] = math.atan2(g("lat_off"), max(g("L_ramp", 1.0), 1e-3))
        v[13] = g("L_ramp")
        v[14] = g("s_back")
        m[[10, 11, 13, 14]] = 1
    return v, m


class Driver:
    def __init__(self, spec, probe):
        self.spec, self.P = spec, probe
        self.ops = [o for o in probe["ops"] if not o["fell"]]
        self.walk = [o for o in self.ops if o["kind"] in ("speed", "posture", "mode") and o["moving"]]
        self.lat = [o for o in self.ops if o["kind"] == "lateral" and o["moving"]]
        self.caps = probe["caps"]
        fp = probe["fp"]
        self.stand_top = float(fp["stand_top"])
        self.stand_w = float(fp["stand_width"])
        self.stand_d = float(fp["stand_depth"])
        self.v_nom = float(spec.get("nominal", {}).get("speed", 0.3))
        self.vel_names = {
            r: (CS.cmd(spec, r)["name"] if CS.has(spec, r) else None)
            for r in ("lin_vel_x", "lin_vel_y", "yaw_rate", "speed")
        }
        self.nominal_posture = {c["name"]: c["nominal"] for c in spec["commands"] if c["role"] not in CS.VEL_ROLES}
        wr = CS.rng(spec, "yaw_rate")
        self.wz_max = float(wr[1]) if wr else 1.0

    # ------------------------------------------------------------------ op selection
    def _posture_of(self, op):
        p = dict(self.nominal_posture)
        for k, v in op["cmd"].items():
            c = next(c for c in self.spec["commands"] if c["name"] == k)
            if c["role"] not in CS.VEL_ROLES:
                p[k] = v
        return p

    def _mode_speed_range(self, posture):
        mc = CS.cmd(self.spec, "mode")
        if mc is None:
            return None
        m = CS.mode_info(self.spec, posture.get(mc["name"]))
        return None if m is None else m.get("speed_range")

    def choose(self, top, wid, pitch, v, travel="fwd"):
        """-> (op, info). An unconstrained phase is the nominal op (stock posture)."""
        cands = self.walk if travel == "fwd" else self.lat
        ext = (lambda o: o["width"]) if travel == "fwd" else (lambda o: o["depth"])
        if top >= self.stand_top - 0.005 and wid >= UNC_WID - 1e-6 and abs(pitch) < 1e-6:
            nomop = next((o for o in self.ops if o["kind"] == ("stand" if travel == "fwd" else "lateral")), None)
            if travel == "fwd":
                nomop = dict(
                    op_id=-1,
                    kind="nominal",
                    cmd={},
                    top_min=self.stand_top,
                    width=self.stand_w,
                    depth=self.stand_d,
                    pitch_ach=0.0,
                    ratio=self._nominal_ratio(),
                    vx_ach=None,
                    tilt_max=0.0,
                    label="nominal",
                )
            return nomop, dict(feasible=True, viol=0.0, unconstrained=True)
        # 2026-10-22 quick patch (XFEAS_DRV_CROUCH_MODES, default on): a lowered-top request (a crouch) may only be served by
        # posture-type ops (body height / pitch commands) or low-posture MODES (crouch walk, squat, kneel, crawls, lying);
        # never by a style mode that merely measured a slightly lower top (run 1.218 m, punches, dances...). Before this,
        # SONIC "crouch top 1.208" was executed as RUN mode at ~2 m/s. Hand-written rule, disclosed.
        if travel == "fwd" and os.environ.get("XFEAS_DRV_CROUCH_MODES", "1") == "1" and top < self.stand_top - 0.03:

            def _low_ok(o):
                if o["kind"] != "mode":
                    return True
                nm = str((o.get("cmd") or {}).get("mode", o.get("label", ""))).lower().replace("_", "")
                return any(w in nm for w in ("crouch", "squat", "kneel", "crawl", "lying"))

            cands = [o for o in cands if _low_ok(o)]
        if not cands:
            return None, dict(feasible=False, viol=9.9, reason="no_ops")
        feas, scored = [], []
        for o in cands:
            viol = max(0.0, o["top_min"] - top - TOL) + max(0.0, ext(o) - wid - TOL)
            if travel == "fwd":  # fastest speed this op can reach inside the spec / doc band (linear in ratio)
                sr = self._mode_speed_range(self._posture_of(o))
                vhi = CS.rng(self.spec, "lin_vel_x")[1]
                vcap = (o.get("ratio") or 0.0) * (min(vhi, sr[1]) if sr else vhi)
                vcap = max(vcap, o["vx_ach"])
            else:
                vcap = abs(o["vy_ach"])
            sc = (
                max(0.0, top - o["top_min"]) * (0.0 if top >= UNC_TOP - 1e-6 else 1.0)
                + 0.5 * abs(pitch - o["pitch_ach"])
                + 0.3 * max(0.0, v - vcap) / max(v, 0.05)
                + 0.5 * o["tilt_max"]
            )
            scored.append((viol, sc, o))
            if viol <= 0:
                feas.append((sc, o))
        if feas:
            sc, o = min(feas, key=lambda x: x[0])
            return o, dict(feasible=True, viol=0.0, score=round(sc, 4))
        viol, sc, o = min(scored, key=lambda x: (x[0], x[1]))
        return o, dict(feasible=False, viol=round(viol, 4), score=round(sc, 4))

    def lat_match(self, side, vy_req, top=UNC_TOP, wid=UNC_WID):
        """probe drv_ops lateral entry on this side minimising |measured |vy| - requested vy| (ties: the faster one), among
        entries that did not fall and fit the envelope (top_min <= top + TOL, depth <= wid + TOL)"""
        T = [
            o
            for o in (self.P.get("drv_ops") or [])
            if o.get("kind") == "lateral"
            and not o.get("fell")
            and (o["vy_cmd"] > 0) == (side > 0)
            and (o["vy_ach"] > 0) == (side > 0)
        ]
        T = [o for o in T if (not np.isfinite(o.get("top_min", np.nan)) or o["top_min"] <= top + TOL)]
        if not T:
            return None
        # primary: speed match (3 cm/s resolution); secondary: the smaller measured depth (the depth measured while
        # stepping sideways is a p90 incl. leg / arm swing, so it is a tie-break, not a hard filter)
        return min(
            T,
            key=lambda o: (
                int(round(abs(abs(o["vy_ach"]) - vy_req) / 0.03)),
                float(o.get("depth") or 9.0),
                -abs(o["vy_ach"]),
            ),
        )

    def fwd_cmd_match(self, v):
        """nominal-posture forward command whose probe-measured steady-state speed best matches v (drv_ops), or None"""
        T = [
            o
            for o in (self.P.get("drv_ops") or [])
            if o.get("kind") == "speed" and o.get("post_label") == "nominal" and not o.get("fell")
        ]
        if not T:
            return None
        o = min(T, key=lambda o: (abs(o["vx_ach"] - v), -o["vx_ach"]))
        return float(o["vx_cmd"]), float(o["vx_ach"])

    def _nominal_ratio(self):
        sp = [o for o in self.ops if o["kind"] == "speed" and o.get("ratio")]
        if not sp:
            return 1.0
        o = min(sp, key=lambda o: abs(o["vx_cmd"] - self.v_nom))
        return float(o["ratio"])

    def v_cmd_for(self, op, v):
        """speed command for target v (m/s) on op: v / measured ratio, clipped to the spec range and the mode's doc band"""
        r = op.get("ratio") if op is not None else None
        if r is None or not np.isfinite(r) or r < 0.2:
            vc = float(op["vx_cmd"]) if op is not None and op.get("vx_cmd") else v
        else:
            vc = v / r
        lo, hi = CS.rng(self.spec, "lin_vel_x")
        vc = float(np.clip(vc, max(lo, 0.0), hi))
        sr = self._mode_speed_range(self._posture_of(op)) if op is not None else None
        if sr:
            vc = float(np.clip(vc, sr[0], sr[1]))
        return round(vc, 3)

    # ------------------------------------------------------------------ programs
    def phase(self, top, wid, pitch, v, travel="fwd"):
        op, info = self.choose(top, wid, pitch, v, travel)
        if op is None:
            return dict(op_id=None, posture=dict(self.nominal_posture), v_cmd=self.v_nom, pred=None, **info)
        post = self._posture_of(op) if op.get("kind") != "nominal" else dict(self.nominal_posture)
        vc = self.v_cmd_for(op, v) if travel == "fwd" else None
        if travel == "fwd" and op.get("kind") == "nominal" and os.environ.get("XFEAS_GEN3_DRV_LAT", "0") == "1":
            fm = self.fwd_cmd_match(v)  # additive 2026-10-19: steady-state forward table of the probe
            if fm is not None:
                vc = round(fm[0], 3)
        pred = dict(
            top=op["top_min"],
            width=op["width"],
            depth=op["depth"],
            pitch=op["pitch_ach"],
            vx=None if vc is None or op.get("ratio") is None else round(vc * op["ratio"], 3),
        )
        return dict(op_id=op["op_id"], op_label=op.get("label"), posture=post, v_cmd=vc, pred=pred, **info)

    def plan(self, a):
        mac = a["macro"]
        if a.get("stock"):
            ph = dict(
                op_id=-1,
                op_label="nominal",
                posture=dict(self.nominal_posture),
                v_cmd=float(a["v_b"]),
                pred=None,
                feasible=True,
                viol=0.0,
            )
            return dict(
                macro="constant",
                phases=dict(B=ph),
                driver_feasible=True,
                executable=True,
                stock=True,
                wz_max=self.wz_max,
                names=self.vel_names,
            )
        prog = dict(
            macro=mac,
            phases={},
            wz_max=self.wz_max,
            names=self.vel_names,
            stock=False,
            turn_vx=float(self.caps.get("turn_vx", 0.1)) if self.caps.get("turn_needs_vx") else 0.0,
            lat_vx=float(self.caps.get("lat_vx", 0.1)) if self.caps.get("lateral_needs_vx") else 0.0,
        )
        if (
            mac == "turn_side_turn"
        ):  # top_b / wid_b describe the SIDE phase; after the turn back: nominal posture at v_b
            B = self.phase(UNC_TOP, UNC_WID, 0.0, a["v_b"])
        else:
            B = self.phase(a.get("top_b", UNC_TOP), a.get("wid_b", UNC_WID), a.get("pitch_b", 0.0), a["v_b"])
        prog["phases"]["B"] = B
        if mac == "two_phase":
            prog["phases"]["A"] = self.phase(
                a.get("top_a", UNC_TOP), a.get("wid_a", UNC_WID), a.get("pitch_a", 0.0), a.get("v_a", a["v_b"])
            )
            prog.update(
                d_commit=float(a["d_commit"]),
                d_level=float(a.get("d_level", 0.0)),
                yaw_off=float(a.get("yaw_off", 0.0)),
                d_yaw=float(a.get("d_yaw", 0.8)),
            )
        if mac == "detour":
            prog.update(lat_off=float(a["lat_off"]), L_ramp=float(a["L_ramp"]), s_back=float(a["s_back"]))
        if mac == "turn_side_turn":
            S = self.phase(a.get("top_b", UNC_TOP), a.get("wid_b", UNC_WID), 0.0, abs(a["vy"]), travel="side")
            side = float(a["side"])
            ok = self.lat_side_ok(side) and self.caps.get("turn_ok", True) and S["op_id"] is not None
            # lateral speed command from the measured lateral ratio on that side
            vy_t = abs(float(a["vy"]))
            lo, hi = CS.rng(self.spec, "lin_vel_y") if CS.has(self.spec, "lin_vel_y") else (0.0, 0.0)
            ach = abs(self.caps["vy_max_L"] if side > 0 else self.caps["vy_max_R"])
            lat_ops = [o for o in self.lat if (o["vy_cmd"] > 0) == (side > 0)]
            ratio = (
                (ach / abs(lat_ops[0]["vy_cmd"]))
                if lat_ops and abs(lat_ops[0]["vy_cmd"]) > 1e-3 and ach > 0.02
                else None
            )
            vy_c = float(np.clip(vy_t / ratio, 0.0, hi)) if ratio else float(min(vy_t, hi))
            S.update(vy_cmd=round(vy_c, 3))
            if os.environ.get("XFEAS_GEN3_DRV_LAT", "0") == "1":
                # additive 2026-10-19: pick the (posture, vy command) whose PROBE-MEASURED steady-state lateral speed on this
                # side best matches the requested vy, among postures that fit the envelope (top / depth); generic table
                # lookup over the probe's drv_ops addendum (gensim3 XFEAS_GEN3_PROBE_DRVSWEEP), no decision logic
                best = self.lat_match(side, vy_t, a.get("top_b", UNC_TOP), a.get("wid_b", UNC_WID))
                if best is not None:
                    S.update(
                        posture=dict(best["posture"]),
                        vy_cmd=round(abs(float(best["vy_cmd"])), 3),
                        op_label=f"drv:{best['post_label']}",
                        lat_match=dict(
                            vy_req=round(vy_t, 3), vy_ach=best["vy_ach"], vy_cmd=best["vy_cmd"], post=best["post_label"]
                        ),
                    )
                    vy_c = abs(float(best["vy_cmd"]))
            prog["phases"]["S"] = S
            prog.update(side=side, e_gap=float(a.get("e_gap", 0.0)), d_commit=float(a.get("d_commit", 0.6)))
            # XFEAS_GEN3_DRV_PROBE=1 (additive 2026-10-19): honour the probe -- if turning was measured NOT to work
            # (caps turn_ok false, or |measured yaw rate| < WZ_WORKS, i.e. a 90 deg turn would take > 15 s), the sidestep
            # macro is executed as PURE lateral motion (no turns: vy to the gap line at the measured lateral speed, then on)
            if os.environ.get("XFEAS_GEN3_DRV_PROBE", "0") == "1":
                wz = abs(float(self.caps.get("wz_ach", 0.0) or 0.0))
                turn_works = bool(self.caps.get("turn_ok", True)) and wz >= WZ_WORKS
                prog.update(
                    pure_lateral=not turn_works,
                    turn_measured=dict(turn_ok=bool(self.caps.get("turn_ok", True)), wz_ach=round(wz, 3)),
                )
                if not turn_works:
                    ok = self.lat_side_ok(side) and S["op_id"] is not None
            prog["executable"] = bool(ok and CS.has(self.spec, "lin_vel_y") and vy_c > 0.02)
        else:
            prog["executable"] = True
        prog["driver_feasible"] = bool(
            all(p.get("feasible", True) for p in prog["phases"].values()) and prog["executable"]
        )
        return prog

    def lat_side_ok(self, side):
        """lateral capability of ONE side from the probe (XFEAS_GEN3_LAT_PER_SIDE=1, default on since 2026-10-21): the
        side's measured steady lateral speed passes the probe's own lateral_ok test (|vy_ach| > max(0.06, 0.3 |vy_cmd|)).
        Before: caps lateral_ok required BOTH sides, so AMO (L 0.149, R 0.052 m/s) could not sidestep at all."""
        if os.environ.get("XFEAS_GEN3_LAT_PER_SIDE", "1") != "1":
            return bool(self.caps.get("lateral_ok", False))
        if self.caps.get("lateral_ok", False):
            return True
        return lat_side_ok(self.caps, side, self.lat)

    def program_key(self, prog):
        """identity of the EXECUTED commands (two envelopes that map to the same commands share one simulation)"""
        P = {
            k: dict(posture=v["posture"], v_cmd=v.get("v_cmd"), vy_cmd=v.get("vy_cmd"))
            for k, v in prog["phases"].items()
        }
        keep = {
            k: prog.get(k)
            for k in (
                "macro",
                "d_commit",
                "d_level",
                "yaw_off",
                "d_yaw",
                "lat_off",
                "L_ramp",
                "s_back",
                "side",
                "e_gap",
            )
        }
        if keep["macro"] == "two_phase" and P.get("A") == P.get("B"):
            keep = dict(macro="constant")
            P = dict(B=P["B"])
        if keep["macro"] == "two_phase":
            keep["d_level"] = keep["d_level"] if P["B"] != {} else 0
        return json.dumps(dict(P=P, **keep), sort_keys=True, default=float)

    def op_top(self, prog, ph="B"):
        p = prog["phases"].get(ph)
        if p is None or p.get("pred") is None:
            return self.stand_top
        return float(p["pred"]["top"])


def env_dev(driver, prog, frac_a=0.0):
    """time-weighted (1 - top_op / stand_top) + 0.2 |pitch_op|; frac_a = share of time in phase A (two_phase)"""

    def dev(ph):
        p = prog["phases"].get(ph)
        if p is None or p.get("pred") is None:
            return 0.0
        return max(0.0, 1.0 - float(p["pred"]["top"]) / max(driver.stand_top, 0.3)) + 0.2 * abs(
            float(p["pred"]["pitch"])
        )

    if prog["macro"] == "two_phase":
        return round(frac_a * dev("A") + (1 - frac_a) * dev("B"), 4)
    if prog["macro"] == "turn_side_turn":
        return round(dev("S"), 4)
    return round(dev("B"), 4)


def lat_side_ok(caps, side, lat_ops=None):
    """probe lateral_ok test applied to one side (side > 0 = left / +y)"""
    ach = abs(float(caps.get("vy_max_L" if side > 0 else "vy_max_R", 0.0) or 0.0))
    cmd = [
        abs(float(o["vy_cmd"]))
        for o in (lat_ops or [])
        if (float(o["vy_cmd"]) > 0) == (side > 0) and abs(float(o["vy_cmd"])) > 1e-3
    ]
    return bool(ach > max(0.06, 0.3 * (cmd[0] if cmd else 0.0)))
