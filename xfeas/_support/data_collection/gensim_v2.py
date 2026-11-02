"""
gen_v2 harness extensions (2026-10-14; analysis/GEN_V2_SPEC.md). Import AFTER SimulationApp exists.

V2Mixin (put FIRST in the MRO, before gensim_v1.ExpoMixin / envsim.EnvSim):
  * decision():  as gensim.GenSim.decision + ExpoMixin, but the decision is 1-3 m before the hazard and the approach
                 captures K=4 head-camera frames at t = -1.5, -1.0, -0.5, 0 s (memory) with camera matrices and base
                 poses; the clearance map (clearmap.py) is computed from them. Exposure adjustment + near-black relight
                 run on the FIRST history frame (label-neutral) and are re-checked at the decision.
  * rollout_v2(geo, dec, act):  restore the decision state and execute one action of the gen_v2 vocabulary
        go       constant setpoint (= gen_v1 rollout)
        go2      two-phase setpoint: A until the base is d_commit before the front edge, then B; nominal posture
                 d_level past the edge (0 = hold B)
        detour   lane following with a shifted lateral reference e_ref(s) (route offset) at a constant setpoint
        sidestep turn 90 deg + lateral walk + turn back (gensim_v1.SidestepCtl)
    + traces (commanded twist/posture and lane-frame base state at 10 Hz) and the stance-foot SLIP label.
GenSimV2 = school; EnvSimV2 = Isaac environments / terrain.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import gensim as GS
import gensim_v1 as GV
import clearmap as CM

T_HIST = (-1.5, -1.0, -0.5, 0.0)
N_HIST = 128
SLIP_STANCE_H, SLIP_STANCE_VZ = 0.03, 0.15
SLIP_MEAN_THR, SLIP_P90_THR, SLIP_FRAC_THR = 0.12, 0.35, 0.15
SKILL_ID = dict(go=0, go2=3, detour=7, sidestep=4)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def ramp(x):
    return float(np.clip(x, 0.0, 1.0))


class ProgramCtl:
    """go / go2 / detour executor (method plugin for drive2). Lane frame from geo; posture via rob.command(dict)."""

    def __init__(self, sim, geo, dec, act):
        self.G, self.geo, self.A = sim, geo, act
        self.sk = act["skill"]
        self.s_front, self.s_end = float(geo["s_front"]), float(geo["s_end"])
        self.s_dec = float(dec["s_dec"])
        self.phase = "A" if self.sk == "go2" else "B"
        self.t_switch = None
        self.posture = None
        self.v = float(act["v_a"] if self.sk == "go2" else act["v"])
        L = float(act.get("L_ramp", 1.0))
        self.L = max(L, 0.3)
        self.s_out0 = self.s_dec + 0.1
        self.s_back0 = self.s_end + float(act.get("s_back", 0.3))
        self.lat = float(act.get("lat_off", 0.0))

    def pose_cmd(self, which):
        a, G = self.A, self.G
        if which == "nominal":
            sp = dict(body_height=G.nominal()["body_height"], pitch=0.0, roll=0.0)
        elif self.sk == "go2" and which == "A":
            sp = dict(body_height=a["h_a"], pitch=a["p_a"], roll=a["r_a"])
        else:
            sp = dict(body_height=a["h"], pitch=a["p"], roll=a["r"])
        if G.NOM is not None and "body_height" in G.RANGES:
            G.rob.command(dict(sp))
        self.posture = sp

    def e_ref(self, s):
        if self.sk != "detour":
            return 0.0, 0.0
        if s < self.s_back0:
            x = (s - self.s_out0) / self.L
            return self.lat * ramp(x), (self.lat / self.L if 0.0 <= x <= 1.0 else 0.0)
        x = (s - self.s_back0) / self.L
        return self.lat * (1.0 - ramp(x)), (-self.lat / self.L if 0.0 <= x <= 1.0 else 0.0)

    def act(self, obs):
        G, geo = self.G, self.geo
        bs = obs["bs"]
        s, e = G.lane_coords(bs, geo)
        a = self.A
        if self.posture is None:
            self.pose_cmd("A" if self.phase == "A" else "B")
        if self.sk == "go2":
            if self.phase == "A" and s >= self.s_front - float(a["d_commit"]):
                self.phase = "B"
                self.t_switch = obs["t"]
                self.v = float(a["v"])
                self.pose_cmd("B")
            if self.phase == "B" and float(a.get("d_level", 0.0)) > 0 and s >= self.s_front + float(a["d_level"]):
                self.phase = "L"
                self.pose_cmd("nominal")
        er, slope = self.e_ref(s)
        href = geo["yaw"] + math.atan(slope)
        kl = 1.5
        yo = float(a.get("yaw_off", 0.0))
        if yo and self.phase == "B" and s < self.s_front + float(a.get("d_yaw", 0.8)):
            # go2 crossing at an angle: from the commit point until d_yaw past the front edge, hold the lane yaw + yaw_off
            # (no lateral correction, the body drifts sideways); then normal lane following resumes
            href += yo
            kl = 0.0
        err = wrap(href - float(bs["yaw"]))
        wz = float(np.clip(2.0 * err - kl * (e - er), -1.0, 1.0))
        return dict(twist=np.array([self.v, 0.0, 0.0, wz]))


class V2Mixin:
    v2 = True

    # ------------------------------------------------------------------ feet (slip label)
    def _feet(self):
        """(physics view, foot body indices) or (None, None) for robots without legs / articulation"""
        try:
            import re

            av = self.rob._art._articulation_view
            pv = av._physics_view
            names = list(av.body_names)
            idx = [i for i, n in enumerate(names) if re.search(r"(_foot$|ankle_roll_link$)", n)]
            return (pv, idx, len(names)) if idx else (None, None, None)
        except Exception:
            return None, None, None

    # ------------------------------------------------------------------ rendered-floor check (2026-10-14)
    def floor_render_check(self, M, dep, step=12):
        """PhysX rays through a pixel grid of the LOWER HALF of the image (any collider except the robot): the rendered
        depth must agree where a collider is hit within 10 m. gen_v1 had 104 decision frames whose physics floor was
        not rendered (lanes outside an env's visual geometry: a backdrop dome / white void below the robot)."""
        eye = M[3, :3]
        n_hit, n_ok, n_all = 0, 0, 0
        for v in range(112 + step // 2, 224, step):
            for u in range(step // 2, 224, step):
                dc = np.array([(u + 0.5 - 112) / GS.FX, -(v + 0.5 - 112) / GS.FX, -1.0, 0.0])
                dw = (dc @ M)[:3]
                nrm = np.linalg.norm(dw)
                dw = dw / nrm
                o = eye.copy()
                acc = 0.0
                dist = None
                for _ in range(4):  # skip the robot's own colliders
                    dd, who, _ = self.ray(o + dw * acc, dw, 10.0 - acc)
                    if dd is None:
                        break
                    if who.startswith(self.root):
                        acc += dd + 0.01
                        continue
                    dist = acc + dd
                    break
                n_all += 1
                if dist is None:
                    continue
                n_hit += 1
                z_ray = dist / nrm
                n_ok += int(abs(float(dep[v, u]) - z_ray) < 0.05 + 0.03 * z_ray)
        return dict(n_all=n_all, n_hit=n_hit, n_ok=n_ok, ok=bool(n_hit >= 0.5 * n_all and n_ok >= 0.8 * n_hit))

    # ------------------------------------------------------------------ approach with memory
    def spawn_s_v2(self, s_dec_t):
        if self.kind.startswith("g1"):
            return max(0.0, s_dec_t - 1.0)
        return max(0.2, s_dec_t - 2.6)

    def _snap_raw(self, bs):
        return GS.GenSim.snapshot(self, bs)

    def _relight_if_black(self, snap, bs, light):
        relit = None
        if snap is not None and float(snap[0].mean()) < 10.0:
            src = "dark" if light.get("dark") else light["mode"]
            if not light["mode"].startswith("night"):
                light = dict(self.LIGHT.set("day", self.rng), relit_from=src)
                snap = self.snapshot(bs)
            if snap is not None and float(snap[0].mean()) < 10.0:
                m_ = light["mode"] if light["mode"].startswith("night") else "night_white"
                light = dict(self.LIGHT.set(m_, self.rng), dark=False, relit_from=src)
                snap = self.snapshot(bs)
            relit = light.get("relit_from")
        return snap, light, relit

    def decision(self, sit, geo, light):
        import cv2

        P0, d, n, fz = geo["P0"], geo["d"], geo["n"], geo["fz"]
        s_dec_t = geo["s_front"] - float(sit["d_dec"])
        run_min = 0.45 if self.kind.startswith("g1") else 0.9
        s_min = (0.0 if self.kind.startswith("g1") else 0.2) + run_min
        if s_dec_t < s_min:  # the BUILT front edge is earlier than the spec's (yawed / deep slab): decide closer
            s_dec_t = s_min
            if geo["s_front"] - s_dec_t < 0.8:
                raise GS.Reject(f"no_runup s_front {geo['s_front']:.2f}")
        s_sp = self.spawn_s_v2(s_dec_t)
        q0 = P0 + d * s_sp + n * float(sit["lat0"])
        vis_check = None
        if geo.get("aim") is not None:
            self.rob.teleport(q0[0], q0[1], geo["yaw"], fz - 30.0)
            self.world.step(render=False)
            low = geo.get("family") in ("overhang", "compound") or (geo.get("family") == "null" and geo["aim"][2] > 0.6)
            vis_check = self.check_view(geo, 1.6, h=0.25) if low else self.check_view(geo, 2.5)
        self.settle(q0[0], q0[1], geo["yaw"] + float(sit["yaw0"]), fz, 1.0 if not self.kind.startswith("g1") else 2.0)
        bs = self.rob.base_state()
        if bs is None or self.rob.fallen(bs):
            raise GS.Reject("bad_spawn")
        self.stand_z = float(bs["pos"][2]) - fz
        # history captures: distance targets from the nominal speed (actual times are recorded)
        v_est = max(self.v_nom * (0.85 if self.kind.startswith("g1") else 0.95), 0.05)
        frames = []
        t_acc = 0.0
        relit = None
        for i, th in enumerate(T_HIST[:-1]):
            tgt = s_dec_t + v_est * th
            s_now, _ = self.lane_coords(bs, geo)
            if tgt > s_now + 0.02:
                stop = lambda s, e, t, st, bs_, tg=tgt: "h" if s >= tg else ("stall" if st else None)
                a = self.drive(geo, self.v_nom, stop, 30.0, track=False)
                if a["reason"] != "h":
                    raise GS.Reject(f"approach_{a['reason']}")
                bs = a["bs"]
                t_acc += a["t"]
            if i == 0:
                snap = self.snapshot(bs)  # ExpoMixin: exposure adjustment on the first frame
                snap, light, relit = self._relight_if_black(snap, bs, light)
                expo0 = getattr(self, "expo", None)
            else:
                snap = self._snap_raw(bs)
            if snap is None:
                raise GS.Reject("render_failed_hist")
            frames.append(
                (
                    t_acc,
                    snap,
                    dict(
                        pos=np.array(bs["pos"], float),
                        yaw=float(bs["yaw"]),
                        pitch=float(bs["pitch"]),
                        roll=float(bs["roll"]),
                    ),
                )
            )
        stop = lambda s, e, t, st, bs_: "dec" if (s >= s_dec_t and self.buf.full()) else ("stall" if st else None)
        a = self.drive(geo, self.v_nom, stop, 30.0, track=False)
        if a["reason"] != "dec":
            raise GS.Reject(f"approach_{a['reason']}")
        bs = a["bs"]
        t_acc += a["t"]
        snap = self.snapshot(bs)
        if snap is None:
            raise GS.Reject("render_failed")
        if float(snap[0].mean()) < 10.0:
            raise GS.Reject("near_black_snapshot")
        frames.append(
            (
                t_acc,
                snap,
                dict(
                    pos=np.array(bs["pos"], float),
                    yaw=float(bs["yaw"]),
                    pitch=float(bs["pitch"]),
                    roll=float(bs["roll"]),
                ),
            )
        )
        vis = self.visible(snap[2], snap[1])
        frc = self.floor_render_check(snap[2], snap[1])
        if not frc["ok"]:
            raise GS.Reject(f"floor_not_rendered {frc}")
        pb, pj = self.buf.arrays()
        s_dec, e_dec = self.lane_coords(bs, geo)
        st = self.save_state()
        # ---------------- memory arrays
        t_dec = frames[-1][0]
        x0, y0, yaw0 = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
        c0, s0_ = math.cos(yaw0), math.sin(yaw0)
        rgb_h = np.stack([cv2.resize(f[1][0], (N_HIST, N_HIST), interpolation=cv2.INTER_AREA) for f in frames]).astype(
            np.uint8
        )
        dep_h = np.stack([CM.min_pool_depth(np.clip(f[1][1], 0, 20.0), N_HIST) for f in frames]).astype(np.float32)
        cam_h = np.stack([f[1][2] for f in frames]).astype(np.float64)
        pose_h = []
        for f in frames:
            p = f[2]
            dx, dy = p["pos"][0] - x0, p["pos"][1] - y0
            pose_h.append(
                [
                    c0 * dx + s0_ * dy,
                    -s0_ * dx + c0 * dy,
                    p["pos"][2] - fz,
                    wrap(p["yaw"] - yaw0),
                    p["pitch"],
                    p["roll"],
                    f[0] - t_dec,
                ]
            )
        pose_h = np.array(pose_h, np.float32)
        cmap = CM.clear_map(dep_h.astype(np.float32), cam_h, (x0, y0, yaw0), fz)
        goal_xy = P0 + d * geo["s_goal"]
        gx, gy = goal_xy[0] - x0, goal_xy[1] - y0
        mem = dict(
            rgb_hist=rgb_h,
            depth_hist=dep_h.astype(np.float16),
            t_hist=pose_h[:, 6].copy(),
            cam_hist=cam_h,
            pose_hist=pose_h,
            K128=CM.intrinsics(N_HIST),
            K224=CM.intrinsics(224),
            clear_map=cmap,
            clear_meta=CM.meta(),
            base_z_floor=np.float32(float(bs["pos"][2]) - fz),
            lane_yaw_rel=np.float32(wrap(geo["yaw"] - yaw0)),
            goal_rel=np.array([c0 * gx + s0_ * gy, -s0_ * gx + c0 * gy], np.float32),
            hist_light=[f[1][0].mean() for f in frames],
            base_pose_dec=np.array([x0, y0, float(bs["pos"][2]), yaw0], np.float64),
            floor_z=np.float64(fz),
        )
        dec = dict(
            rgb=snap[0],
            depth=snap[1],
            cam=snap[2],
            pb=pb,
            pj=pj,
            s_dec=s_dec,
            e_dec=e_dec,
            state=st,
            light=light,
            vis_check=vis_check,
            vis=vis,
            relit=relit,
            mean_rgb=round(float(snap[0].mean()), 1),
            t_app=t_acc,
            mem=mem,
            expo=getattr(self, "expo", None),
            expo_hist0=expo0,
            floor_render=frc,
        )
        if dec["expo"] is not None and not dec["expo"]["ok"]:
            raise GS.Reject(f"exposure {dec['expo']['final']}")
        return dec

    # ------------------------------------------------------------------ drive with traces + slip
    def drive2(self, geo, v, stop_fn, t_max, method=None, vid=None):
        HZ, EVERY, rob = self.HZ, self.EVERY, self.rob
        best, t_best, maxtilt, energy, min_clear, top_max = -1e9, 0.0, 0.0, 0.0, 9.9, None
        goal_xy = geo["P0"] + geo["d"] * geo["s_goal"]
        idx = getattr(rob, "idx", None)
        pv, feet, nb = self._feet()
        slips = []
        ctr, btr = [], []
        last_tw = np.zeros(4)
        s = e = 0.0
        t = 0.0
        bs = None
        k = 0
        res = None
        for step in range(int(t_max * HZ)):
            t = step / HZ
            self.world.step(render=False)
            self.buf.tick(rob)
            if step % EVERY:
                continue
            k += 1
            bs = rob.base_state()
            if bs is None or rob.fallen(bs):
                res = dict(
                    reason="fell",
                    t=t,
                    s=s,
                    e=e,
                    max_tilt=maxtilt,
                    energy=energy,
                    min_clear=min_clear,
                    top=top_max,
                    bs=bs,
                )
                break
            maxtilt = max(maxtilt, float(bs["tilt"]))
            s, e = self.lane_coords(bs, geo)
            tau = getattr(rob, "last_tau", None)
            if tau is not None:
                dq = np.asarray(rob._art.get_joint_velocities(), float).reshape(-1)
                if idx is not None and len(tau) == len(idx):
                    dq = dq[idx]
                if len(dq) == len(tau):
                    energy += float(np.abs(np.asarray(tau) * dq).sum()) * EVERY / HZ
            if pv is not None:
                try:
                    T_ = np.asarray(pv.get_link_transforms()).reshape(-1, nb, 7)[0]
                    V_ = np.asarray(pv.get_link_velocities()).reshape(-1, nb, 6)[0]
                    fz_ = T_[feet, 2]
                    vz = V_[feet, 2]
                    vh = np.hypot(V_[feet, 0], V_[feet, 1])
                    stc = (fz_ - fz_.min() < SLIP_STANCE_H) & (np.abs(vz) < SLIP_STANCE_VZ)
                    slips.extend(vh[stc].tolist())
                except Exception:
                    pv = None
            if method is None:
                err = wrap(geo["yaw"] - bs["yaw"])
                last_tw = np.array([v, 0.0, 0.0, float(np.clip(2.0 * err - 1.5 * e, -1.0, 1.0))])
                rob.command(list(last_tw))
            else:
                x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
                dd = goal_xy - np.array([x, y])
                c, s_ = math.cos(yaw), math.sin(yaw)
                obs = dict(
                    t=t,
                    dt=EVERY / HZ,
                    bs=bs,
                    rob=rob,
                    goal_rel=np.array([c * dd[0] + s_ * dd[1], -s_ * dd[0] + c * dd[1]]),
                    goal_xy=goal_xy.copy(),
                    goal_dist=float(np.hypot(*dd)),
                )
                self.LIGHT.follow(bs, self.spec.cam)
                r = method.act(obs)
                last_tw = np.asarray(r["twist"], float)
                rob.command(last_tw)
            if k % 5 == 1:  # 10 Hz traces
                po = getattr(method, "posture", None) or getattr(self, "_cur_posture", None) or {}
                nom = self.nominal()
                ctr.append(
                    [
                        last_tw[0],
                        last_tw[1],
                        0.0,
                        last_tw[3],
                        (po.get("body_height", nom["body_height"]) - nom["body_height"])
                        if self.NOM is not None
                        else 0.0,
                        po.get("pitch", 0.0),
                        po.get("roll", 0.0),
                    ]
                )
                btr.append(
                    [s, e, wrap(float(bs["yaw"]) - geo["yaw"]), float(bs["pos"][2]) - geo["fz"], float(bs["tilt"])]
                )
            if s > best + 0.10:
                best, t_best = s, t
            if geo.get("u_track") and k % 5 == 0 and geo["s_front"] - 0.7 <= s <= geo["s_end"] + 0.7:
                top = self.robot_top() - geo["fz"]
                u = self.u_of(geo, s)
                top_max = top if top_max is None else max(top_max, top)
                if u is not None:
                    min_clear = min(min_clear, u - top)
            if vid is not None and k % 6 == 0:
                self.side_frame(vid, bs, geo)
            r_ = stop_fn(s, e, t, (t - t_best) > self.STALL_S, bs)
            if r_:
                res = dict(
                    reason=r_, t=t, s=s, e=e, max_tilt=maxtilt, energy=energy, min_clear=min_clear, top=top_max, bs=bs
                )
                break
        if res is None:
            res = dict(
                reason="timeout",
                t=t_max,
                s=s,
                e=e,
                max_tilt=maxtilt,
                energy=energy,
                min_clear=min_clear,
                top=top_max,
                bs=bs,
            )
        sl = np.array(slips, float)
        if len(sl) >= 10:
            res["slip"] = dict(
                slip_mean=round(float(sl.mean()), 4),
                slip_p90=round(float(np.percentile(sl, 90)), 4),
                slip_frac=round(float((sl > SLIP_FRAC_THR).mean()), 4),
                n_stance=int(len(sl)),
            )
        else:
            res["slip"] = dict(slip_mean=None, slip_p90=None, slip_frac=None, n_stance=int(len(sl)))
        res["cmd_trace"] = np.array(ctr, np.float32).reshape(-1, 7)
        res["base_trace"] = np.array(btr, np.float32).reshape(-1, 5)
        return res

    def _finish(self, res, geo, dec):
        out = self.outcome(res, geo, dec["s_dec"])
        out["t_goal"] = out["t"] if out["success"] else None
        sl = res["slip"]
        out.update(sl)
        out["slipped"] = bool(
            sl["slip_mean"] is not None and (sl["slip_mean"] > SLIP_MEAN_THR or sl["slip_p90"] > SLIP_P90_THR)
        )
        out["cmd_trace"] = res["cmd_trace"]
        out["base_trace"] = res["base_trace"]
        return out

    def _posture_dev_trace(self, ctr):
        if len(ctr) == 0 or self.NOM is None and all(self.BOX[k][1] - self.BOX[k][0] < 1e-6 for k in ("pitch", "roll")):
            return 0.0
        nom = self.nominal()
        devs = [self.posture_dev(dict(body_height=nom["body_height"] + r[4], pitch=r[5], roll=r[6])) for r in ctr]
        return float(np.mean(devs))

    # ------------------------------------------------------------------ one branch of the v2 vocabulary
    def rollout_v2(self, geo, dec, act, vid=None):
        """act: dict(skill=go|go2|detour, v, h, p, r [, v_a, h_a, p_a, r_a, d_commit, d_level] [, lat_off, L_ramp, s_back])
        or dict(skill=sidestep, side, vy, e_gap)"""
        self.restore_state(dec["state"])
        goal = lambda s, e, t, st, bs_: self.goal_reached(bs_, geo, s, e) or ("stuck" if st else None)
        if act["skill"] == "sidestep":
            nom = self.nominal()
            if self.NOM is not None and "body_height" in self.RANGES:
                self.rob.command(dict(body_height=nom["body_height"], pitch=0.0, roll=0.0, vx=0.0))
            m = GV.SidestepCtl(self, geo, act["side"], act["vy"], act["e_gap"], geo["s_front"], geo["s_end"])
            stall0 = self.STALL_S
            self.STALL_S = stall0 + 6.0
            try:
                res = self.drive2(geo, None, goal, self.t_max(geo, dec["s_dec"]) + 15.0, method=m, vid=vid)
            finally:
                self.STALL_S = stall0
            self.rob.stop()
            out = self._finish(res, geo, dec)
            out["posture_dev"] = 0.0
            out["side_phase_end"] = m.phase
            out["t_switch"] = None
            return out
        m = ProgramCtl(self, geo, dec, act)
        extra = 0.0
        if act["skill"] == "detour":
            extra = 2.0 * abs(float(act.get("lat_off", 0.0))) / max(self.v_nom, 0.1)
        if act["skill"] == "go2":
            extra = max(0.0, float(geo["s_front"]) - float(dec["s_dec"])) / max(float(act["v_a"]), 0.1) * 0.5
        res = self.drive2(geo, None, goal, self.t_max(geo, dec["s_dec"]) + extra, method=m, vid=vid)
        self.rob.stop()
        out = self._finish(res, geo, dec)
        out["posture_dev"] = round(self._posture_dev_trace(res["cmd_trace"]), 3)
        out["t_switch"] = None if m.t_switch is None else round(float(m.t_switch), 2)
        out["phase_end"] = m.phase
        return out


class GenSimV2(V2Mixin, GV.GenSimV1):
    pass


def make_envsim_v2():
    import envsim as ES

    class EnvSimV2(V2Mixin, ES.EnvSim):
        pass

    return EnvSimV2
