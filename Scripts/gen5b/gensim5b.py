"""
gen5b harness: SIDE-BY-SIDE BRANCHES (import AFTER SimulationApp exists). analysis/GEN5B_DATA_PLAN.md.

V5bMixin (MRO: V5bMixin, gensim5.V5Mixin, gensim4.V4Mixin, ...) adds:
  b_save / b_restore   the gen4 branching state (gensim3.save_state: articulation pose / velocities / joints, adapter and
                       controller python state, VRob wrapper state) + the gen5 look-alike state (_v5: glitch / height-lag
                       queues), the fall judge, the proprio buffer, and the run-loop clock (episode time, tick phase, stall
                       clock, command in force, time of the last switch, frame memory). Scene objects are static colliders
                       (rigid bodies disabled by the builders), so a moment is fully described by the robot state.
  run5b                one RUN = from the spawn (stock warm-up) or from a saved moment, ONE command applied at the start and
                       then HELD to the end (goal strict 0.5 m / fall (mode-aware) / stall / cap). Steering = the stock path
                       follower (steer5), as gen5. master runs render the head camera every tick (the 4-frame memory of
                       their moments) and save MOMENTS at given path arclengths (state + the gen5 model inputs at that
                       tick: frames at -1.5/-1/-0.5/0 s, feas v0.4, h(t), 2 s proprio, descriptors given the command in
                       force); light runs render only 2 thumbnails.
gen4's branching (run_gen_v4 / gensim3.rollout_v3) restored a saved decision state for every option; gen5b does the same
at several moments. Restore fidelity is checked (driver): from a moment, the prefix command is re-run for 1 s and the pose
is compared with the master's own continuation.
"""
import copy, math, os, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
for _p in (THIS, os.path.join(ROOT, "scripts", "gen5")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gensim as GS
import gensim5 as G5
import gensim4 as G4
import gensim3 as G3
import gensim_v2 as G2
import gensim_v1 as GV
import ctrl_spec as CS
import candidates5 as CA
import steer5 as ST

TICK_S, WARM_S, HIST_TICKS = G5.TICK_S, G5.WARM_S, G5.HIST_TICKS
STALL_B_G1, STALL_B_SPOT = 6.0, 5.0          # gen5b: no 0.1 m progress for this long ends a run (gen5 10 / 8 s, gen4 8 / 5 s)
K_PROP = 100


class V5bMixin:
    # ------------------------------------------------------------------ branching state
    def b_save(self):
        st = self.save_state()
        rob = self.rob
        st["v5"] = copy.deepcopy(getattr(rob, "_v5", None))
        j = self.judge
        st["judge"] = dict(reason=j.reason, legacy_fired=j.legacy_fired, n_low_tilt=j.n_low_tilt, t_low_end=j.t_low_end)
        b = self.buf
        st["buf"] = (copy.deepcopy(b.base), copy.deepcopy(b.joints), copy.deepcopy(b.extras), b.k)
        return st

    def b_restore(self, st):
        self.restore_state(st)
        if st.get("v5") is not None:
            object.__setattr__(self.rob, "_v5", copy.deepcopy(st["v5"]))
        for k, v in st["judge"].items():
            setattr(self.judge, k, v)
        b = self.buf
        b.base, b.joints, b.extras = copy.deepcopy(st["buf"][0]), copy.deepcopy(st["buf"][1]), copy.deepcopy(st["buf"][2])
        b.k = st["buf"][3]

    # ------------------------------------------------------------------ spawn (= episode5 spawn)
    def spawn5b(self, sit, geo):
        rob = self.rob
        pf = ST.PathFollower(geo["path_world"])
        fz = float(geo["fz"])
        arc_sp = float(sit["s_spawn"])
        q = pf.point(arc_sp); _, _, seg = pf.project(q)
        hd = pf.heading(seg)
        nvec = np.array([-math.sin(hd), math.cos(hd)])
        q = q + nvec * float(sit.get("lat0", 0.0))
        vis_check = None
        if geo.get("aim") is not None:
            rob.teleport(q[0], q[1], hd, fz - 30.0)
            self.world.step(render=False)
            low = geo.get("family") in ("overhang", "compound") or (geo.get("family") == "null" and geo["aim"][2] > 0.6)
            vis_check = self.check_view(geo, 1.6, h=0.25) if low else self.check_view(geo, 2.5)
        g1 = self.kind.startswith("g1")
        self.settle(q[0], q[1], hd + float(sit.get("yaw0", 0.0)), fz, 2.0 if g1 else 1.0)
        bs = rob.base_state()
        if bs is None or rob.fallen(bs):
            raise GS.Reject("bad_spawn")
        self.stand_z = float(bs["pos"][2]) - fz
        self.judge.reset()
        nom = CA.nominal_cmd(self.ctrl_spec)
        rob.set_env_intent(float(nom.get("lin_vel_x", 0.3) or 0.3), 9.9, 9.9, 0.0)
        rob.command(CA.to_names(self.ctrl_spec, nom))
        t_max = float(min(self.T_CAP, max(12.0, (geo["arc_goal"] - arc_sp) / self.V_FLOOR + 6.0)))
        return dict(t=0.0, t_next=0.0, k=0, best=-1e9, t_best=0.0, cur=dict(nom), t_sw=None, frames=[], t_clock0=float(rob._v_t),
                    t_max=t_max, vis_check=vis_check, first=True)

    # ------------------------------------------------------------------ one run
    def run5b(self, sit, geo, ctl, start, cmd, master=False, pts=(), t_cap=25.0, thumbs=(4, 12), vid=None, side_at=(8,)):
        """start: the spawn loop dict (spawn5b) or a moment dict (its 'state' + 'loop'); cmd: role dict applied at the start
        (None = keep). pts: [(label, arc)] moments to save (master only). -> run dict"""
        spec, C, D, P = ctl["spec"], ctl["cands"], ctl["desc"], ctl["P"]
        rob = self.rob
        pf = ST.PathFollower(geo["path_world"])
        fz = float(geo["fz"])
        HZ, EVERY = self.HZ, self.EVERY
        if "state" in start:
            self.b_restore(start["state"])
            L = dict(start["loop"]); L["frames"] = list(L["frames"]); L["first"] = False
        else:
            L = start
        g1 = self.kind.startswith("g1")
        stall_s = STALL_B_G1 if g1 else STALL_B_SPOT
        self.contacts.begin()
        rob._v_trace = []
        t0 = float(L["t"]); tc0 = float(rob._v_t)
        moving = lambda c: bool((CS.mode_info(spec, c["mode"]) or {}).get("moving", True)) if "mode" in c else True
        resend = bool(spec.get("binding", {}).get("resend_every_step")) or bool(self.base_spec.get("binding", {}).get("resend_every_step"))
        cur = dict(L["cur"]); t_sw = L["t_sw"]
        if cmd is not None:
            if not CA.same_cmd(cmd, cur):
                t_sw = t0
            cur = dict(cmd)
            rob.command(CA.to_names(spec, cur))
            bs = rob.base_state()
            if bs is not None:
                sc = ST.follower_cmd(spec, None, pf.errors(bs["pos"][:2], float(bs["yaw"])), moving=moving(cur))
                if sc:
                    rob.command(sc)
        t_end_cap = min(float(L["t_max"]) if master and "state" not in start else 1e9, t0 + float(t_cap))
        n_steps = int(round((t_end_cap - t0) * HZ)) + 1
        frames = L["frames"]
        best, t_best, t_next, k = L["best"], L["t_best"], L["t_next"], L["k"]
        haz = geo.get("hazards", [])
        pend = sorted([p for p in pts], key=lambda p: p[1])
        ticks, moments, thumbs_out = [], [], []
        end = None; t = t0; k_rel = 0
        last_ref = self.buf.base[-1] if self.buf.base else None
        prof = dict(phys=0.0, render=0.0, feas=0.0)
        _tp = time.time()
        floor_render = None; vis_dec = None
        for step in range(n_steps):
            t = t0 + (step + 1) / HZ
            self.world.step(render=False); self.buf.tick(rob)
            is_tick = t + 1e-9 >= t_next
            if (step + 1) % EVERY and not is_tick:
                continue
            bs = rob.base_state()
            if bs is None or rob.fallen(bs):
                end = dict(reason="fell", t=t, fall_reason=self.judge.reason or ("no_state" if bs is None else "legacy")); break
            er = pf.errors(bs["pos"][:2], float(bs["yaw"]))
            sc = ST.follower_cmd(spec, None, er, moving=moving(cur))
            if sc:
                rob.command(sc)
            if resend:
                rob._v_inner.command(dict(rob._v_last_sent or {}))
            if er["s"] > best + 0.10:
                best, t_best = er["s"], t
            if er["s"] >= geo["arc_goal"] and abs(er["e"]) < 0.5:
                end = dict(reason="crossed_low" if (geo.get("z_top") is not None and float(bs["pos"][2]) - geo["z_top"] < 0.25) else "success", t=t)
                break
            if t - t_best > stall_s:
                end = dict(reason="stall", t=t); break
            if t >= t_end_cap - 1e-9:
                end = dict(reason="timeout", t=t); break
            if not is_tick:
                continue
            # ------------------------------------------------ TICK (0.25 s)
            t_next += TICK_S
            _t1 = time.time(); prof["phys"] += _t1 - _tp
            try:
                top = float(self.link_extents(bs)["top_abs"]) - fz
            except Exception:
                top = float("nan")
            dist = min([h["arc0"] - er["s"] for h in haz if er["s"] <= h["arc1"] + 0.5] or [99.0])
            ticks.append((t, er["s"], er["e"], float(bs["pos"][0]), float(bs["pos"][1]), float(bs["pos"][2]), float(bs["yaw"]),
                          float(bs["tilt"]), top, float(bs["v_b"][0]), float(bs["v_b"][1]), float(dist)))
            if master:
                if L.get("first"):
                    snap = self.snapshot(bs); L["expo"] = getattr(self, "expo", None)
                    light = L.get("light")
                    snap, light, relit = self._relight_if_black(snap, bs, light)
                    L["light"] = light; L["first"] = False
                else:
                    snap = self._snap_raw(bs)
                if snap is None:
                    raise GS.Reject(f"render_failed t {t:.2f}")
                rgb, dep, M = snap
                info = dict(getattr(self, "snap_info", None) or {})
                dep = np.nan_to_num(np.asarray(dep, np.float32), posinf=60.0)
                import clearmap as CMg
                d128 = CMg.min_pool_depth(np.clip(dep, 0, 20.0), 128).astype(np.float32)
                frames.append(dict(t=t, rgb=np.asarray(rgb, np.uint8), dep=dep.astype(np.float16), d128=d128.astype(np.float16), M=np.asarray(M, np.float64),
                                   mask=float(info.get("mask_frac", np.nan)), cam_h=float(info.get("cam_h") if info.get("cam_h") is not None else np.nan),
                                   bw=(float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"]))))
                if len(frames) > 40:                      # memory needs the last 7 ticks; keep 10 s for sheets
                    frames[:] = frames[-40:]
                if floor_render is None and t >= WARM_S - 1e-6 and "state" not in start:
                    floor_render = self.floor_render_check(M, dep)
                    if not floor_render["ok"]:
                        raise GS.Reject(f"floor_not_rendered {floor_render}")
                    try:
                        vis_dec = self.visible(M, dep)
                    except Exception:
                        vis_dec = None
                _t2 = time.time(); prof["render"] += _t2 - _t1
                while pend and er["s"] >= pend[0][1] - 1e-6:
                    lab, arc_p = pend.pop(0)
                    if t < WARM_S - 1e-6 or len(frames) < 7 or er["s"] > arc_p + 0.25:
                        moments.append(dict(label=lab, arc_target=arc_p, skipped=f"t {t:.2f} arc {er['s']:.2f} frames {len(frames)}"))
                        continue
                    idx = [len(frames) - 1 + o for o in HIST_TICKS]
                    fr = [(frames[i]["d128"].astype(np.float32), frames[i]["M"], frames[i]["bw"]) for i in idx]
                    try:
                        fv, fd = G5.feas_v4(fr, (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])), fz, P)
                    except Exception as ex:
                        fv = np.full(16, np.nan, np.float32)
                    X = D.describe(C, cur, float(bs["v_b"][0]))
                    st = self.b_save()
                    loop = dict(t=t, t_next=t_next, k=k + 1, best=best, t_best=t_best, cur=dict(cur), t_sw=t_sw, frames=list(frames[-8:]),
                                t_clock0=L["t_clock0"], t_max=L["t_max"])
                    obs = dict(fr=[frames[i] for i in idx], feas=np.asarray(fv, np.float32), X=X.astype(np.float16),
                               rec=list(rob._v_hist), prop_b=np.stack(self.buf.base[-K_PROP:]).astype(np.float32) if self.buf.base else np.zeros((0, 13), np.float32),
                               prop_j=np.stack(self.buf.joints[-K_PROP:]).astype(np.float32) if self.buf.joints else np.zeros((0, 0, 7), np.float32))
                    moments.append(dict(label=lab, arc_target=arc_p, t=t, arc=er["s"], e=er["e"], pos=[float(x) for x in bs["pos"]], yaw=float(bs["yaw"]),
                                        tilt=float(bs["tilt"]), top=top, vx=float(bs["v_b"][0]), d_haz=float(dist), cmd_prev=dict(cur),
                                        t_since=(t - t_sw) if t_sw is not None else t, any_switch=t_sw is not None,
                                        state=st, loop=loop, obs=obs, clock=float(rob._v_t)))
                    if vid is not None:
                        self.side_frame(vid, bs, geo)
                prof["feas"] += time.time() - _t2
            elif k_rel in thumbs:
                snap = self._snap_raw(bs)
                if snap is not None:
                    thumbs_out.append((t, np.asarray(snap[0], np.uint8)[::2, ::2].copy()))
            if vid is not None and k_rel in side_at:
                self.side_frame(vid, bs, geo)
            k += 1; k_rel += 1
            _tp = time.time()
        if end is None:
            end = dict(reason="timeout", t=t)
        ct = self.contacts.end()
        nb = int((t - t0) / G4.CONTACT_BIN) + 2
        bins = self.contacts.bins(nb)
        trace = list(rob._v_trace or []); rob._v_trace = None
        return dict(t0=t0, clock0=tc0, cmd=dict(cur), end=end, contact=ct, bins=bins, ticks=np.array(ticks, np.float64).reshape(-1, 12),
                    trace=trace, moments=moments, thumbs=thumbs_out, prof={k_: round(v_, 1) for k_, v_ in prof.items()},
                    floor_render=floor_render, vis_dec=vis_dec, light=L.get("light"), expo=L.get("expo"), fall_legacy=bool(self.judge.legacy_fired))


class GenSimV5b(V5bMixin, G5.V5Mixin, G4.V4Mixin, G3.V3Mixin, G2.V2Mixin, GV.GenSimV1):
    pass


def make_envsim_v5b():
    import envsim as ES

    class EnvSimV5b(V5bMixin, G5.V5Mixin, G4.V4Mixin, G3.V3Mixin, G2.V2Mixin, ES.EnvSim):
        pass
    return EnvSimV5b
