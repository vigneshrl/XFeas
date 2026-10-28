#!/usr/bin/env python
"""
gen5 self-test = probe_v4 protocol + the TRANSITION probe (GEN5_DESIGN 5.2; analysis/GEN5_DATA_SPEC.md 4).
One Isaac process = one BASE controller (+ its virtual wrappers), the probe_v4 office training lane, hazards parked.

Per controller:
  1. probe_v4: data/probes_v4/<cid>/office_v4.json is COPIED when it exists (same protocol, no re-run); otherwise the
     probe_v4 protocol is run here (gensim3.run_probe with the probe_v4 flags, scripts/probe_v4/run_probes_v4.py) ->
     data/probes_v5/<cid>/office_v4.json
  2. transitions (new): for every transition class of the controller's candidate set (candidates5.probe_classes: stop,
     speed_max, lat_L / lat_R, every posture grid point, every non-steering mode) and 2 speeds (slow = 0.35, fast = 0.95 x
     min(vx_max, 1); one speed if there is no forward command), walking at the speed in the NOMINAL posture:
        switch nominal -> X, hold 1.5 s (record 'to'), switch X -> nominal, hold 1.5 s (record 'back')
     a fresh spawn (+ 1.5 s nominal run-up) whenever the lane runs out or the robot fell. Each record: settle_s (first time
     after which top height, vx, vy stay within 2 cm / max(0.05, 15 %) / 0.05 m/s of their last-0.4 s values; = hold if
     never), max_tilt, fell (+ reason), stepping (foot height difference std > 1 cm or |v| > 0.05), top / vx / vy before and
     after, n_glitch (hidden wrapper bursts; audit only).
     -> data/probes_v5/<cid>/transitions.json
  Steering during the probe = the stock follower (steer5) along the lane axis.
SONIC (g1s) and the Isaac Spot flat policy (spotf) are HELD OUT: only with --heldout (probe only, never data).

  scripts/gen5/run_isaac5.sh 0 scripts/gen5/probe_transitions5.py --base g1c --ctrls g1c,lk_g1c_01,...
"""
import argparse, json, math, os, shutil, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--ctrls", default="")
ap.add_argument("--suite", default=os.path.join(ROOT, "data", "gen_v3", "suites", "office_g1g_p1.json"))
ap.add_argument("--out", default=os.path.join(ROOT, "data", "probes_v5"))
ap.add_argument("--heldout", action="store_true")
ap.add_argument("--hold", type=float, default=1.5)
ap.add_argument("--redo", action="store_true", help="re-run transitions even if transitions.json exists")
a = ap.parse_args()
sys.path.insert(0, THIS)
import ctrl5 as K5
assert a.base in K5.ALL_BASES, a.base
if a.base in K5.HELD_OUT:
    assert a.heldout, f"{a.base} is held out: --heldout (probe only)"
for k, v in K5.env_for(a.base).items():
    os.environ.setdefault(k, v)
os.environ["XFEAS_FOOT_MATERIAL"] = "1"
PROTO = dict(XFEAS_GEN3_PROBE_STEADY="1", XFEAS_GEN3_SPEED_FOLLOW_V="1", XFEAS_GEN3_MODE_DEFAULT_SPEED="1", XFEAS_GEN3_PROBE_DRVSWEEP="1",
             XFEAS_GEN3_DRV_LAT_VX="1", XFEAS_PROBE_BUDGET="1e9")
os.environ.update(PROTO)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 224, "height": 224, "renderer": "RaytracedLighting"})
print(f"[p5] app up {time.strftime('%H:%M:%S')} gpu {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
from isaacsim.core.prims import RigidPrim as _RP
_orig_opr = _RP._on_physics_ready


def _safe_opr(self, event):
    try:
        return _orig_opr(self, event)
    except Exception as _e:                      # noqa: BLE001
        if event is not None:
            raise
        print(f"[p5] RigidPrim eager physics init skipped ({str(_e)[:80]}); initialised later", flush=True)


_safe_opr.__name__ = _orig_opr.__name__; _safe_opr.__qualname__ = _orig_opr.__qualname__
_RP._on_physics_ready = _safe_opr
import gensim3 as G3
import ctrl_spec as CS
import virtual as VC
import virtual5 as V5
import candidates5 as CA
import steer5 as ST

S = json.load(open(a.suite))
LANE = next(s["lane"] for s in S if s["lane"].get("split", "train") == "train" and s["family"] != "env_curb")
G = G3.make_envsim_v3()(app, K5.FACT[a.base], a.base, "office", seed=1, side_cam=False, tiles=None)
BASE = CS.load(K5.spec_path(a.base))
G.setup_v3(BASE)
G.park_scene()


def _f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def transitions(spec, lane, hold, log):
    rob = G.rob
    P0, yaw0, d, n, fz0 = G.lane_frame(lane)
    L = float(lane.get("len", 8.0))
    fz = G.floor_at(P0 + d * 1.5, fz0)
    geo = dict(P0=P0, d=d, n=n, yaw=yaw0, fz=fz)
    pf = ST.PathFollower([P0, P0 + d * L])
    HZ, EVERY = G.HZ, G.EVERY
    nom = CA.nominal_cmd(spec)
    vm = CA.vx_max(spec)
    hi = min(vm, 1.0) if vm is not None else None
    speeds = [("slow", round(0.35 * hi, 3)), ("fast", round(0.95 * hi, 3))] if hi is not None else [("own", None)]
    classes = CA.probe_classes(spec)
    clock = dict(t=0.0, falls=0, spawns=0)
    t_wall0 = time.time()

    def full(cmd_roles, v):
        c = dict(nom); c.update(cmd_roles)
        if v is not None and "lin_vel_x" not in cmd_roles and "lin_vel_x" in c:
            c["lin_vel_x"] = v
        return c

    def moving_of(c):
        m = c.get("mode")
        if m is None:
            return True
        mi = CS.mode_info(spec, m) or {}
        return bool(mi.get("moving", True))

    def run(cmd, dur, measure=True):
        g0 = V5.glitch_count(rob)
        rob.command(CA.to_names(spec, cmd))
        samp, fell, reason = [], False, None
        mv = moving_of(cmd)
        for k in range(int(round(dur * HZ))):
            G.world.step(render=False); G.buf.tick(rob)
            if (k + 1) % EVERY:
                continue
            bs = rob.base_state()
            if bs is None or rob.fallen(bs):
                fell = True; reason = "no_state" if bs is None else f"tilt {float(bs['tilt']):.2f} z {float(bs['pos'][2]) - fz:.2f}"
                break
            er = pf.errors(bs["pos"][:2], float(bs["yaw"]))
            sc = ST.follower_cmd(spec, None, er, moving=mv)
            if sc:
                rob.command(sc)
            if spec["binding"].get("resend_every_step"):
                rob._v_inner.command(dict(rob._v_last_sent or {}))
            if measure and (k + 1) % (5 * EVERY) == 0:
                r = dict(t=round((k + 1) / HZ, 3), vx=float(bs["v_b"][0]), vy=float(bs["v_b"][1]), tilt=float(bs["tilt"]), s=er["s"])
                try:
                    r["top"] = G.link_extents(bs)["top_abs"] - fz
                except Exception:
                    r["top"] = np.nan
                r["fdz"] = float(G._foot_dz()) if hasattr(G, "_foot_dz") else np.nan
                samp.append(r)
        clock["t"] += (k + 1) / HZ
        return samp, fell, reason, V5.glitch_count(rob) - g0

    def summary(samp, fell, reason, dur, ng):
        if not samp:
            return dict(settle_s=dur, max_tilt=None, fell=bool(fell), fall_reason=reason, stepping=False, n=0, n_glitch=ng)
        T = np.array([r["t"] for r in samp]); t_end = T[-1]
        W = [r for r in samp if r["t"] >= t_end - 0.4] or samp[-1:]
        top_f = np.nanmean([r["top"] for r in W]); vx_f = np.mean([r["vx"] for r in W]); vy_f = np.mean([r["vy"] for r in W])
        ok = [abs(r["top"] - top_f) < 0.02 if np.isfinite(r["top"]) and np.isfinite(top_f) else True for r in samp]
        ok = [o and abs(r["vx"] - vx_f) < max(0.05, 0.15 * abs(vx_f)) and abs(r["vy"] - vy_f) < 0.05 for o, r in zip(ok, samp)]
        settle = dur
        for i in range(len(samp)):
            if all(ok[i:]):
                settle = float(T[i]) if i > 0 else 0.0
                break
        if fell:
            settle = dur
        fdz = np.array([r["fdz"] for r in samp], float)
        stepping = bool((np.isfinite(fdz).sum() > 3 and np.nanstd(fdz) > 0.01) or np.mean(np.hypot([r["vx"] for r in samp], [r["vy"] for r in samp])) > 0.05)
        return dict(settle_s=round(settle, 3), max_tilt=round(max(r["tilt"] for r in samp), 4), fell=bool(fell), fall_reason=reason,
                    stepping=stepping, top_before=_f(samp[0]["top"]), top_after=_f(top_f), vx_before=_f(samp[0]["vx"]), vx_after=_f(vx_f),
                    vy_after=_f(vy_f), n=len(samp), settled=bool(settle < dur - 1e-6), n_glitch=int(ng),
                    trace=[[r["t"], round(r["vx"], 3), round(r["vy"], 3), None if not np.isfinite(r["top"]) else round(r["top"], 3), round(r["tilt"], 3)]
                           for r in samp])

    def spawn(v):
        q0 = P0 + d * 0.6
        G.settle(q0[0], q0[1], yaw0, fz, 1.0)
        clock["t"] += 1.0; clock["spawns"] += 1
        run(full({}, v), 1.5, measure=False)

    out = dict(ctrl_id=spec["ctrl_id"], base=spec["base"], spec_hash=CS.spec_hash(spec), protocol="transitions_v5", hold_s=hold,
               speeds={k: v for k, v in speeds}, classes={}, lane=lane.get("id"), env="office")
    for sk, v in speeds:
        rob.set_env_intent(v or 0.0, 9.9, 9.9, 0.0)
        spawn(v)
        for cls, rep in classes:
            bs = rob.base_state(); s_now = pf.project(bs["pos"][:2])[0] if bs is not None else 0.0
            need = (v or 0.9) * 2 * hold + 0.6
            if s_now + need > L - 0.4:
                spawn(v)
            cx = full(rep, v)
            samp, fell, reason, ng = run(cx, hold)
            rto = summary(samp, fell, reason, hold, ng)
            if fell:
                clock["falls"] += 1
                rb = dict(settle_s=hold, max_tilt=None, fell=None, skipped="fell_in_to", n=0)
                spawn(v)
            else:
                samp, fell2, reason2, ng2 = run(full({}, v), hold)
                rb = summary(samp, fell2, reason2, hold, ng2)
                if fell2:
                    clock["falls"] += 1; spawn(v)
            out["classes"].setdefault(cls, dict(cmd=rep, to={}, back={}))
            out["classes"][cls]["to"][sk] = rto; out["classes"][cls]["back"][sk] = rb
            log(f"[trans] {spec['ctrl_id']} {sk:4s} v={v} {cls:18s} to: settle {rto['settle_s']:.2f} tilt {rto['max_tilt']} fell {rto['fell']} "
                f"step {rto['stepping']} gl {rto.get('n_glitch')} | back: settle {rb['settle_s']:.2f} tilt {rb.get('max_tilt')} fell {rb.get('fell')}")
    rob.command(CA.to_names(spec, dict(nom, lin_vel_x=0.0) if "lin_vel_x" in nom else nom)); rob.stop()
    out.update(t_robot=round(clock["t"], 2), t_wall=round(time.time() - t_wall0, 1), n_falls=clock["falls"], n_spawns=clock["spawns"])
    return out


for cid in (a.ctrls or a.base).split(","):
    if K5.is_held_out(cid) and not a.heldout:
        raise SystemExit(f"{cid} is held out")
    s, w = K5.load_ctrl(cid)
    assert s["base"] == a.base, (cid, s["base"])
    od = os.path.join(a.out, cid); os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "office_v4.json"); ft = os.path.join(od, "transitions.json")
    G.use_controller(s, w if w is not None else VC.identity_cfg())
    t0 = time.time()
    src4 = os.path.join(K5.PROBES_V4, cid, "office_v4.json")
    if not os.path.exists(fp):
        if os.path.exists(src4):
            shutil.copy(src4, fp); print(f"[p5] {cid} probe_v4 copied from {src4}", flush=True)
        else:
            G.park_scene()
            P = G.run_probe(LANE, budget=float(os.environ["XFEAS_PROBE_BUDGET"]), log=lambda m: print(f"[{cid}] {m}", flush=True))
            P.update(session="v5", protocol="probe_v4", protocol_flags=PROTO, lane_used=LANE, env="office",
                     spec_file=os.path.relpath(K5.spec_path(cid), ROOT), held_out=bool(K5.is_held_out(cid)), wall_s=round(time.time() - t0, 1),
                     note="probe_v4 protocol run by scripts/gen5/probe_transitions5.py (no probe_v4 file existed)")
            json.dump(P, open(fp + ".tmp", "w"), default=float); os.replace(fp + ".tmp", fp)
            print(f"[p5] {cid} probe_v4 protocol run -> {fp} ({time.time() - t0:.0f}s, robot {P['t_robot']} s, falls {P['n_falls']})", flush=True)
    if os.path.exists(ft) and not a.redo:
        print(f"[p5] {cid} transitions exist, skip", flush=True); continue
    t1 = time.time()
    G.use_controller(s, w if w is not None else VC.identity_cfg())
    T = transitions(s, LANE, a.hold, lambda m: print(m, flush=True))
    T.update(held_out=bool(K5.is_held_out(cid)), spec_file=os.path.relpath(K5.spec_path(cid), ROOT), wall_s=round(time.time() - t1, 1))
    json.dump(T, open(ft + ".tmp", "w"), default=float); os.replace(ft + ".tmp", ft)
    print(f"[p5] {cid} transitions -> {ft} ({T['t_wall']:.0f}s wall, robot {T['t_robot']} s, {len(T['classes'])} classes, falls {T['n_falls']})", flush=True)
print("[p5] DONE", flush=True)
app.close()
