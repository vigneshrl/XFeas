#!/usr/bin/env python
"""
gen_v5 SWITCHING collector (2026-10-22; analysis/GEN5_DATA_SPEC.md). One process = one BASE controller (g1c / g1g / g1l /
g1u / g1p / spotp; never g1s / spotf) + its virtual wrappers (same robot asset). Per situation x controller: ONE episode
(gensim5.V5Mixin.episode5): spawn 3-5 m (+ warm-up) before the hazard, decision tick every 0.25 s, behaviour policy
(policy5), stock path follower, to the end of the lane / fall / stall. Probes: data/probes_v5 only (probe_v4 + transitions).

  scripts/gen5/run_isaac5.sh 0 scripts/gen5/run_gen_v5.py --env school --base g1c --suite data/gen_v5/suites/school_g1c_p1.json \
      --chunk 0:6 --part school_g1c_p1_0000 [--smoke]
Output: <root>/<env>/<base>/{ep/<sid>__<cid>.npz, sits_<part>.jsonl, rejects_<part>.jsonl, videos/<sid>__<cid>/}
  root = data/gen_v5 (production) | data/gen_v5_smoke (--smoke; never read by training or the production audit)
Resumable (sid__cid already in sits_ / rejects_ of this part are skipped).
"""
import argparse, json, math, os, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--env", default="school")
ap.add_argument("--suite", required=True)
ap.add_argument("--base", default="g1c")
ap.add_argument("--chunk", default="")
ap.add_argument("--part", default="")
ap.add_argument("--video-every", type=int, default=4)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
sys.path.insert(0, THIS)
import ctrl5 as K5
assert a.base in K5.BASES, f"gen_v5 collected bases: {K5.BASES} (g1s / spotf are held out and never collected)"
assert os.environ.get("XFEAS_FOOT_MATERIAL") == "1" and os.environ.get("XFEAS_PERC_MINPOOL_FIX") == "1"
assert os.environ.get("XFEAS_LEGACY_PRIMITIVES", "0") != "1", "real assets only"
assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1"), "GPU 0 or 1 only"
HELD_ENV = ("gq", "hospital", "interior")
assert not any(h in a.env.lower() for h in HELD_ENV), "held-out environment"
PROD = not a.smoke
assert not (PROD and any(k.startswith("GEN5_DEBUG") for k in os.environ)), "GEN5_DEBUG_* flags are for debugging only (never data)"
assert os.path.exists(os.path.join(K5.PROBES_V5, "MANIFEST.json")) or not PROD, "production needs data/probes_v5/MANIFEST.json"
for k, v in K5.env_for(a.base).items():
    os.environ.setdefault(k, v)
RUN_ID = "gen_v5" if PROD else "gen_v5_smoke"
OUT_ROOT = os.path.join(ROOT, "data", "gen_v5" if PROD else "gen_v5_smoke")
OUT = os.path.join(OUT_ROOT, a.env, a.base)
for sub in ("ep", "videos"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
part = a.part or (a.chunk.replace(":", "-") if a.chunk else "all")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 224, "height": 224, "renderer": "RaytracedLighting"})
print(f"[v5] app up {time.strftime('%H:%M:%S')} gpu {os.environ.get('CUDA_VISIBLE_DEVICES')} prod {PROD}", flush=True)
from isaacsim.core.prims import RigidPrim as _RP
_orig_opr = _RP._on_physics_ready


def _safe_opr(self, event):
    try:
        return _orig_opr(self, event)
    except Exception as _e:                      # noqa: BLE001
        if event is not None:
            raise
        print(f"[v5] RigidPrim eager physics init skipped ({str(_e)[:80]}); initialised later", flush=True)


_safe_opr.__name__ = _orig_opr.__name__; _safe_opr.__qualname__ = _orig_opr.__qualname__
_RP._on_physics_ready = _safe_opr
import gensim as GS
import gensim5 as G5
import ctrl_spec as CS
import virtual as VC
import candidates5 as CA
import describe5 as DS
import feas as FE

S_ALL = json.load(open(a.suite))
if a.chunk:
    _i0, _i1 = map(int, a.chunk.split(":")); S_ALL = S_ALL[_i0:_i1]
TILES = None
if a.env == "terrain":
    TILES = []
    for s_ in S_ALL:
        k_ = int(round(s_["lane"]["start"][1] / 14.0))
        TILES.append(dict(s_["params"]["tile"], id=f"{s_['params']['type']}_{k_:02d}", origin=[0.0, 14.0 * k_, 0.0]))
seed = sum(map(ord, part)) % 997
FACT = K5.FACT[a.base]
if a.env == "school":
    G = G5.GenSimV5(app, FACT, a.base, seed=seed, side_cam=a.video_every > 0, env_id="school")
else:
    G = G5.make_envsim_v5()(app, FACT, a.base, a.env, seed=seed, side_cam=a.video_every > 0, tiles=TILES)
BASE = CS.load(K5.spec_path(a.base))
G.setup_v5(BASE)
DESC_PUB = json.dumps({k: v for k, v in G.desc.items() if k not in ("controller", "setpoint_ranges", "setpoints_supported",
                                                                     "setpoints_unsupported", "layer_box", "extra")}, default=float)
ROBOT = K5.ROBOT_OF[a.base]
JN = json.dumps(list(G.rob._v_inner._art.dof_names))
print(f"[v5] env {a.env} base {a.base} robot {ROBOT} cam {G.cam4.src}:{G.cam4.mount}", flush=True)
CTRL = {}


def load_ctrl(cid):
    assert not K5.is_held_out(cid), f"{cid} is held out"
    s, w = K5.load_ctrl(cid)
    assert s["base"] == a.base, (cid, s["base"])
    d = os.path.join(K5.PROBES_V5, cid)
    fp, ft = os.path.join(d, "office_v4.json"), os.path.join(d, "transitions.json")
    assert os.path.exists(fp) and os.path.exists(ft), f"no gen5 probe for {cid} (data/probes_v5/{cid})"
    P, T = json.load(open(fp)), json.load(open(ft))
    assert P["ctrl_id"] == cid and T["ctrl_id"] == cid
    C = CA.enumerate_candidates(s)
    return dict(cid=cid, spec=s, w=w if w is not None else VC.identity_cfg(), P=P, T=T, cands=C, desc=DS.Describer(s, P, T),
                pfile=os.path.relpath(fp, ROOT), tfile=os.path.relpath(ft, ROOT))


def use(ctl):
    G.rob.set_controller(ctl["spec"], ctl["w"])
    G.ctrl_spec = ctl["spec"]; G.v_nom = float(ctl["spec"].get("nominal", {}).get("speed", 0.3)); G.driver = None


ROLE_V = ("lin_vel_x", "lin_vel_y", "pelvis_height", "torso_pitch")


def cmd_vec(spec, c):
    v = [float(c.get(r, np.nan)) if c.get(r) is not None else np.nan for r in ROLE_V]
    m = c.get("mode")
    v.append(float(CS.mode_index(spec, m)) if m is not None else np.nan)
    return v


def save_episode(fn, sit, geo, ctl, ep, wall):
    T, Fr = ep["ticks"], ep["frames"]
    spec = ctl["spec"]
    n = len(T); N = len(ctl["cands"]); F = DS.F_DESC
    X = np.zeros((n, N, F), np.float16); feas = np.full((n, 16), np.nan, np.float32)
    chosen = np.full(n, -1, np.int16)
    for i, r in enumerate(T):
        if r["dec"]:
            X[i] = r["X"]; feas[i] = r["feas"]; chosen[i] = r["chosen"]
    g = lambda k, dt=np.float32: np.array([r.get(k, np.nan) for r in T], dt)
    R = ep["records"]
    J = len(R[-1][4]) if R else 0
    rec_t = np.array([x[0] for x in R], np.float64)
    rec_cmd = np.array([x[1] for x in R], np.float32).reshape(-1, len(CS.HIST_ROLES))
    rec_env = np.array([x[2] for x in R], np.float32).reshape(-1, 4)
    rec_ach = np.array([x[3] for x in R], np.float64).reshape(-1, len(VC.ACH_KEYS))
    rec_q = np.array([x[4] for x in R if len(x[4]) == J], np.float32).reshape(-1, J)
    rec_dq = np.array([x[5] for x in R if len(x[5]) == J], np.float32).reshape(-1, J)
    e = ep["end"]
    np.savez_compressed(
        fn, fmt="gen_v5", run_id=RUN_ID, sid=f"{sit['sid']}__{ctl['cid']}", geo_sit=sit["sid"], ctrl_id=ctl["cid"], base=spec["base"],
        robot=ROBOT, virtual=bool(spec.get("virtual")), spec_json=json.dumps(spec), probe_file=ctl["pfile"], trans_file=ctl["tfile"],
        probe_src="probes_v5", fingerprint=np.array(ctl["P"]["fingerprint"], np.float32), fp_keys=json.dumps(ctl["P"]["fp_keys"]),
        desc=DESC_PUB, env=a.env, family=sit["family"], family5=sit["family5"], aggr=bool(sit.get("aggr")), zone_split=sit["lane"].get("split", "train"),
        lighting=json.dumps(ep["light"], default=float), light_mode=str(ep["light"]["mode"]),
        cands_json=json.dumps([dict(kind=c["kind"], cmd=c["cmd"], cls=c["cls"]) for c in ctl["cands"]]), desc_keys=json.dumps(DS.DESC_KEYS),
        cand_roles=json.dumps(list(ROLE_V) + ["mode_index"]), aggr_set=np.array(ep["aggr_set"], np.int16), posture_set=np.array(ep["posture_set"], np.int16),
        style=ep["style"], tick_s=np.float32(G5.TICK_S), warm_s=np.float32(G5.WARM_S), hist_ticks=np.array(G5.HIST_TICKS, np.int8),
        # ---- per tick
        t=g("t"), dec=np.array([r["dec"] for r in T], bool), arc=g("arc"), e=g("e"), herr=g("herr"), pos=np.array([r["pos"] for r in T], np.float32).reshape(-1, 3),
        yaw=g("yaw"), pitch=g("pitch"), roll=g("roll"), v_b=np.array([r["v_b"] for r in T], np.float32).reshape(-1, 3),
        w_b=np.array([r["w_b"] for r in T], np.float32).reshape(-1, 3), tilt=g("tilt"), top=g("top"), d_haz=g("d_haz"), near=np.array([r["near"] for r in T], bool),
        hist_n=g("hist_n", np.int32), prop_n=g("prop_n", np.int32), glitch=g("glitch", np.int16), mean_rgb=g("mean_rgb"),
        chosen=chosen, pol_mode=np.array([r.get("pol_mode", "warm") for r in T]), prop=g("prop"), switched=np.array([bool(r.get("switched", False)) for r in T]),
        cls_prev=np.array([r.get("cls_prev", "") for r in T]), cls_new=np.array([r.get("cls_new", "") for r in T]),
        cmd=np.array([cmd_vec(spec, r["cmd"]) for r in T], np.float32).reshape(-1, 5), cmd_json=np.array([json.dumps(r["cmd"]) for r in T]),
        state_json=np.array([json.dumps(r["state"]) for r in T]), sent_json=np.array([json.dumps(r["sent"]) for r in T]),
        feas=feas, feas_keys=json.dumps(FE.FEAS_KEYS), X=X,
        # ---- per tick frames (shared head camera)
        rgb=np.stack([f["rgb"] for f in Fr]).astype(np.uint8), depth=np.stack([f["dep"] for f in Fr]).astype(np.float16),
        depth128=np.stack([f["d128"] for f in Fr]).astype(np.float16), cam=np.stack([f["M"] for f in Fr]).astype(np.float64),
        selfmask=np.array([f["mask"] for f in Fr], np.float32), cam_h=np.array([f["cam_h"] for f in Fr], np.float32),
        base_world=np.array([f["bw"] for f in Fr], np.float64), cam_source=G.cam4.src, cam_mount=G.cam4.mount,
        # ---- history h(t) records (VRob, 10 Hz; ach RAW world: pos z / top not floor-relative) + proprio stream (50 Hz)
        rec_t=rec_t, rec_cmd=rec_cmd, rec_env=rec_env, rec_ach=rec_ach, rec_q=rec_q, rec_dq=rec_dq, n_prefix=np.int32(ep["n_prefix"]),
        t_clock0=np.float64(ep["t_clock0"]), prop_base=ep["prop_base"].astype(np.float32), prop_joints=ep["prop_joints"].astype(np.float32),
        joint_names=JN, hist_roles=json.dumps(list(CS.HIST_ROLES)), ach_keys=json.dumps(list(VC.ACH_KEYS)),
        # ---- geometry / path
        P0=np.asarray(geo["P0"], np.float64), d=np.asarray(geo["d"], np.float64), n=np.asarray(geo["n"], np.float64), lane_yaw=np.float64(geo["yaw"]),
        floor_z=np.float64(geo["fz"]), z_top=np.float64(geo["z_top"] if geo.get("z_top") is not None else np.nan),
        path_world=np.array(geo["path_world"], np.float64), path_lane=np.array(geo["path_lane"], np.float64), hazards_json=json.dumps(geo["hazards"]),
        arc_goal=np.float32(geo["arc_goal"]), s_spawn=np.float32(sit["s_spawn"]), d_start=np.float32(sit["d_start"]), value=np.float32(geo.get("value") or 0.0),
        # ---- outcome
        end_reason=e["reason"], t_end=np.float32(e["t"]), fall_reason=str(e.get("fall_reason") or ""), fall_rule="mode_aware_v4",
        fell_legacy=bool(ep["fall_legacy"]), success=e["reason"] == "success", fell=e["reason"] == "fell",
        contact_json=json.dumps(ep["contact"], default=float), bin_s=np.float32(0.02), bins_haz_body=ep["bins"]["hb"], bins_haz_foot=ep["bins"]["hf"],
        bins_scene_body=ep["bins"]["sb"], t_max=np.float32(ep["t_max"]), wall_s=np.float32(wall),
        pol_commit=json.dumps(ep.get("pol_commit") or {}))


def main():
    fs = os.path.join(OUT, f"sits_{part}.jsonl"); rej = os.path.join(OUT, f"rejects_{part}.jsonl")
    done = set()
    for f in (fs, rej):
        if os.path.exists(f):
            R = [json.loads(l) for l in open(f) if l.strip()]
            # floor_not_rendered = a render failure, not a property of the situation: retried on resume (queue floor-burst retry)
            done |= {r["sid"] for r in R if not str(r.get("reason", "")).startswith("floor_not_rendered")}
    for sit in S_ALL:
        for cid in sit["ctrls"]:
            if cid not in CTRL:
                CTRL[cid] = load_ctrl(cid)
    t_all = time.time(); n_ep = 0
    for j, sit in enumerate(S_ALL):
        todo = [c for c in sit["ctrls"] if f"{sit['sid']}__{c}" not in done]
        if not todo:
            continue
        cdir = os.path.join(OUT, "crash"); os.makedirs(cdir, exist_ok=True)
        cfn = os.path.join(cdir, f"{sit['sid']}.cnt")
        ncr = int(open(cfn).read().strip() or 0) if os.path.exists(cfn) else 0
        if ncr >= 2:
            for cid in todo:
                open(rej, "a").write(json.dumps(dict(sid=f"{sit['sid']}__{cid}", ctrl_id=cid, family5=sit["family5"], reason="crash_repeat")) + "\n")
            continue
        open(cfn, "w").write(str(ncr + 1))
        t0 = time.time()
        try:
            bsit = dict(sit, family=sit.get("family_build") or sit["family"])
            geo = G.build(bsit)
        except GS.Reject as e:
            for cid in todo:
                open(rej, "a").write(json.dumps(dict(sid=f"{sit['sid']}__{cid}", ctrl_id=cid, family5=sit["family5"], reason=str(e)[:400], stage="build")) + "\n")
            print(f"[v5] REJECT {sit['sid']}: {str(e)[:200]}", flush=True)
            os.path.exists(cfn) and os.remove(cfn)
            continue
        for cid in todo:
            run_one(sit, geo, CTRL[cid], j, fs, rej)
            n_ep += 1
        os.path.exists(cfn) and os.remove(cfn)
        print(f"[v5] situation {sit['sid']} done ({len(todo)} ctrl) {time.time() - t0:.0f}s | {n_ep} episodes {(time.time() - t_all) / 60:.1f} min", flush=True)


def run_one(sit, geo, ctl, j, fs, rej):
    t0 = time.time()
    cid = ctl["cid"]; sid = f"{sit['sid']}__{cid}"
    use(ctl)
    rng = np.random.default_rng([int(sit["seed"]), sum(map(ord, cid)), 5])
    vid = None
    if a.video_every > 0 and j % a.video_every == 0:
        vd = os.path.join(OUT, "videos", sid); os.makedirs(vd, exist_ok=True); vid = dict(dir=vd, k=0)
    try:
        light = G.set_lighting(sit)
        ep = G.episode5(sit, geo, ctl, ctl["P"], light, rng, vid=vid)
    except GS.Reject as e:
        open(rej, "a").write(json.dumps(dict(sid=sid, ctrl_id=cid, family5=sit["family5"], reason=str(e)[:400], stage="episode",
                                             wall_s=round(time.time() - t0, 1))) + "\n")
        print(f"[v5] REJECT {sid}: {str(e)[:200]}", flush=True); return
    wall = time.time() - t0
    if not any(r["dec"] for r in ep["ticks"]):           # ended (fell / stalled) inside the 1.5 s warm-up: no decision -> reject
        open(rej, "a").write(json.dumps(dict(sid=sid, ctrl_id=cid, family5=sit["family5"], reason=f"no_decision_tick end {ep['end']['reason']} "
                                             f"t {ep['end']['t']:.2f} {ep['end'].get('fall_reason') or ''}", stage="warmup", wall_s=round(wall, 1))) + "\n")
        print(f"[v5] REJECT {sid}: ended in warm-up ({ep['end']['reason']} t {ep['end']['t']:.2f})", flush=True); return
    fn = os.path.join(OUT, "ep", f"{sid}.npz")
    save_episode(fn, sit, geo, ctl, ep, wall)
    T = ep["ticks"]; D_ = [r for r in T if r["dec"]]
    e = ep["end"]; ct = ep["contact"]
    modes = {}
    for r in D_:
        modes[r["pol_mode"]] = modes.get(r["pol_mode"], 0) + 1
    arc_max = max([r["arc"] for r in T] or [0.0])
    passed = [h["kind"] for h in geo["hazards"] if arc_max > h["arc1"] + 0.1]
    rec = dict(sid=sid, geo_sit=sit["sid"], robot=ROBOT, ctrl_id=cid, base=ctl["spec"]["base"], virtual=bool(ctl["spec"].get("virtual")), env=a.env,
               family=sit["family"], family5=sit["family5"], aggr=bool(sit.get("aggr")), zone_split=sit["lane"].get("split", "train"),
               lane=sit["lane"]["id"], value=geo.get("value"), d_start=sit["d_start"], s_spawn=sit["s_spawn"], arc_goal=geo["arc_goal"],
               hazards=geo["hazards"], light=ep["light"], night=str(ep["light"]["mode"]).startswith("night"), expo=(ep["expo"] or {}).get("final"),
               floor_render=ep["floor_render"], vis_check=ep["vis_check"], vis_dec=ep["vis_dec"], checks=geo.get("checks"),
               end=e["reason"], t_end=round(e["t"], 2), fall_reason=e.get("fall_reason"), fell_legacy=ep["fall_legacy"], success=e["reason"] == "success",
               arc_max=round(arc_max, 3), hazards_passed=passed, n_ticks=len(T), n_dec=len(D_), pol_modes=modes, style=ep["style"],
               n_switch=int(sum(r.get("switched", False) for r in D_)), n_glitch=int(T[-1]["glitch"]) if T else 0, contact=ct,
               clean=bool(e["reason"] == "success" and ct["haz_body_s"] < 0.05 and ct["haz_body_Ns"] < 1.0),
               cls_hist={}, selfmask_mean=float(np.nanmean([f["mask"] for f in ep["frames"]])) if ep["frames"] else None,
               ep_file=os.path.relpath(fn, OUT_ROOT), probe_file=ctl["pfile"], trans_file=ctl["tfile"], probe_src="probes_v5", wall_s=round(wall, 1),
               prof=ep.get("prof"), run_id=RUN_ID, pol_commit=(ep.get("pol_commit") or {}).get("p", 0.0),
               n_commit=len((ep.get("pol_commit") or {}).get("log", [])), video=None if vid is None else os.path.relpath(vid["dir"], OUT_ROOT))
    for r in D_:
        rec["cls_hist"][r["cls_new"]] = rec["cls_hist"].get(r["cls_new"], 0) + 1
    open(fs, "a").write(json.dumps(rec, default=float) + "\n")
    print(f"[v5] {sid} {sit['family5']:12s} {ep['light']['mode'][:9]} {e['reason']:11s} t {e['t']:5.1f}s ticks {len(T)} dec {len(D_)} "
          f"sw {rec['n_switch']} {modes} passed {passed} haz_body {ct['haz_body_s']:.2f}s style {ep['style']} gl {rec['n_glitch']} "
          f"wall {wall:.0f}s ({wall / max(e['t'], 1e-3):.1f}x) prof {ep.get('prof')}", flush=True)


main()
print("[v5] DONE", flush=True)
app.close()
