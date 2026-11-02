#!/usr/bin/env python
"""Collects training data in Isaac Sim, one controller per process. For each scene it walks the robot from the
start, saves the simulator state at several moments before the obstacle, and replays each moment with
different commands, so every command's real outcome from the same state is known.

Details:
gen5b BRANCH collector. One process = one BASE controller (+ its v_/lk_/mm_
variants, same robot asset); never g1s / spotf. Per situation x controller (a PAIR):
  A  master run from the spawn with the stock command (warm-up 1.5 s, then held to the end); moments at d before the hazard
     start (branch_schedule.A_D); from every moment one run per command class (branch_schedule.pick_classes) held to the end
  B  (low-obstacle families) the low1 / low2 runs from the d = 1.5 moment are masters too: moments at the B points; from
     every B moment: stock (stand up) / the other posture / slow-in-posture, each held to the end (keep = the master)
  C  (null) 2 moments; stock (master) / slow / low1 / fast
Each run is started from the saved moment (branch_sim.b_restore, gen4's branching); restore fidelity: from every moment (smoke)
or the d1.5 / B-mid moments (production) the prefix command is re-run for 1 s and compared with the master (flag > 2 cm);
--replay-check additionally re-simulates the stock prefix from the spawn and compares the pose at every A moment.
Output: <root>/<env>/<base>/{br/<sid>__<cid>.npz, sits_<part>.jsonl, rejects_<part>.jsonl, side/<sid>__<cid>/}
  root = data/gen_v5b (production) | data/gen_v5b_smoke (--smoke). Resumable (pairs in sits_ / rejects_ are skipped).

  xfeas/data_collection/run_isaac.sh 0 xfeas/data_collection/branch_collector.py --env school --base g1c --suite data/gen_v5b/suites/school_g1c_b1.json       --chunk 0:4 --part school_g1c_b1_0000 [--smoke] [--replay-check]
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, json, math, os, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
G5D = os.path.join(ROOT, "xfeas", "_support", "closed_loop")
ap = argparse.ArgumentParser()
ap.add_argument("--env", default="school")
ap.add_argument("--suite", required=True)
ap.add_argument("--base", default="g1c")
ap.add_argument("--chunk", default="")
ap.add_argument("--part", default="")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--replay-check", action="store_true")
ap.add_argument("--fid-all", action="store_true")
ap.add_argument("--side", type=int, default=1, help="side-camera frames for contact sheets (every k-th pair)")
a = ap.parse_args()
import controller_registry as K5

assert a.base in K5.BASES and a.base not in ("jackal", "iris"), f"gen5b bases (tonight): G1 / Spot; {a.base}"
assert not K5.is_held_out(a.base)
assert os.environ.get("XFEAS_FOOT_MATERIAL") == "1" and os.environ.get("XFEAS_PERC_MINPOOL_FIX") == "1"
assert os.environ.get("XFEAS_LEGACY_PRIMITIVES", "0") != "1", "real assets only"
assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1"), "GPU 0 or 1 only"
HELD_ENV = ("gq", "hospital", "interior")
assert not any(h in a.env.lower() for h in HELD_ENV), "held-out environment"
PROD = not a.smoke
assert not (PROD and any(k.startswith("GEN5_DEBUG") for k in os.environ)), "debug flags never in data"
for k, v in K5.env_for(a.base).items():
    os.environ.setdefault(k, v)
RUN_ID = "gen_v5b" if PROD else "gen_v5b_smoke"
OUT_ROOT = os.path.join(ROOT, "data", "gen_v5b" if PROD else "gen_v5b_smoke")
OUT = os.path.join(OUT_ROOT, a.env, a.base)
for sub in ("br", "side"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)
part = a.part or (a.chunk.replace(":", "-") if a.chunk else "all")
FID_ALL = a.fid_all or a.smoke

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 224, "height": 224, "renderer": "RaytracedLighting"})
print(f"[v5b] app up {time.strftime('%H:%M:%S')} gpu {os.environ.get('CUDA_VISIBLE_DEVICES')} prod {PROD}", flush=True)
from isaacsim.core.prims import RigidPrim as _RP

_orig_opr = _RP._on_physics_ready


def _safe_opr(self, event):
    try:
        return _orig_opr(self, event)
    except Exception as _e:  # noqa: BLE001
        if event is not None:
            raise
        print(f"[v5b] RigidPrim eager physics init skipped ({str(_e)[:80]})", flush=True)


_safe_opr.__name__ = _orig_opr.__name__
_safe_opr.__qualname__ = _orig_opr.__qualname__
_RP._on_physics_ready = _safe_opr
import gensim as GS
import gensim5 as G5
import branch_sim as GB
import ctrl_spec as CS
import virtual as VC
import candidates as CA
import descriptors as DS
import branch_schedule as SB
import branch_labels as LB
import feas as FE

S_ALL = json.load(open(a.suite))
if a.chunk:
    _i0, _i1 = map(int, a.chunk.split(":"))
    S_ALL = S_ALL[_i0:_i1]
seed = sum(map(ord, part)) % 997
FACT = K5.FACT[a.base]
if a.env == "school":
    G = GB.GenSimV5b(app, FACT, a.base, seed=seed, side_cam=a.side > 0, env_id="school")
else:
    G = GB.make_envsim_v5b()(app, FACT, a.base, a.env, seed=seed, side_cam=a.side > 0, tiles=None)
BASE = CS.load(K5.spec_path(a.base))
G.setup_v5(BASE)
DESC_PUB = json.dumps(
    {
        k: v
        for k, v in G.desc.items()
        if k
        not in ("controller", "setpoint_ranges", "setpoints_supported", "setpoints_unsupported", "layer_box", "extra")
    },
    default=float,
)
ROBOT = K5.ROBOT_OF[a.base]
JN = json.dumps(list(G.rob._v_inner._art.dof_names))
print(f"[v5b] env {a.env} base {a.base} robot {ROBOT} cam {G.cam4.src}:{G.cam4.mount}", flush=True)
CTRL = {}
T_CAP = dict(g1=25.0, spot=20.0)[ROBOT]  # a light branch run: held at most this long after the branch
T_CAP_B = dict(g1=35.0, spot=20.0)[ROBOT]  # a B master (posture held through the hazard)
FID_S, FID_TOL = 1.0, 0.02


def load_ctrl(cid):
    assert not K5.is_held_out(cid), f"{cid} is held out"
    s, w = K5.load_ctrl(cid)
    assert s["base"] == a.base, (cid, s["base"])
    d = os.path.join(K5.PROBES_V5, cid)
    fp, ft = os.path.join(d, "office_v4.json"), os.path.join(d, "transitions.json")
    P, T = json.load(open(fp)), json.load(open(ft))
    assert P["ctrl_id"] == cid and T["ctrl_id"] == cid
    C = CA.enumerate_candidates(s)
    return dict(
        cid=cid,
        spec=s,
        w=w if w is not None else VC.identity_cfg(),
        P=P,
        T=T,
        cands=C,
        desc=DS.Describer(s, P, T),
        pfile=os.path.relpath(fp, ROOT),
        tfile=os.path.relpath(ft, ROOT),
    )


def use(ctl):
    G.rob.set_controller(ctl["spec"], ctl["w"])
    G.ctrl_spec = ctl["spec"]
    G.v_nom = float(ctl["spec"].get("nominal", {}).get("speed", 0.3))
    G.driver = None


ROLE_V = ("lin_vel_x", "lin_vel_y", "pelvis_height", "torso_pitch")


def cmd_vec(spec, c):
    v = [float(c.get(r)) if c.get(r) is not None else np.nan for r in ROLE_V]
    m = c.get("mode")
    v.append(float(CS.mode_index(spec, m)) if m is not None else np.nan)
    return v


def tick_at(ticks, t):
    if len(ticks) == 0:
        return None
    i = int(np.argmin(np.abs(ticks[:, 0] - t)))
    return ticks[i] if abs(ticks[i, 0] - t) < 1e-3 else None


def fidelity(sit, geo, ctl, m, R_master):
    """restore the moment, re-run the master's command 1 s, pose difference vs the master at the same time (m)"""
    R = G.run_training(sit, geo, ctl, m, None, master=False, t_cap=FID_S + 0.3, thumbs=())
    t1 = m["t"] + FID_S
    a_, b_ = tick_at(R["ticks"], t1), tick_at(R_master["ticks"], t1)
    if a_ is None or b_ is None:
        return float("nan")
    return float(np.linalg.norm(a_[3:6] - b_[3:6]))


def pair(sit, geo, ctl, j, fs, rej):
    t_w0 = time.time()
    cid = ctl["cid"]
    sid = f"{sit['sid']}__{cid}"
    use(ctl)
    spec, C = ctl["spec"], ctl["cands"]
    nom = CA.nominal_cmd(spec)
    X0 = ctl["desc"].describe(C, nom, float(nom.get("lin_vel_x", 0.3) or 0.3))
    pl = SB.plan(sit["family5"], C, X0, ROBOT, geo["hazards"], geo["arc_goal"], front_reach=G5.COMMIT_REACH[ROBOT][0])
    cl = pl["classes"]
    vid = None
    if a.side > 0 and j % a.side == 0:
        vd = os.path.join(OUT, "side", sid)
        os.makedirs(vd, exist_ok=True)
        vid = dict(dir=vd, k=0)
    runs, moments, branches = [], [], []
    try:
        light = G.set_lighting(sit)
        L = G.spawn5b(sit, geo)
        L["light"] = light
        R0 = G.run_training(sit, geo, ctl, L, None, master=True, pts=pl["A"], t_cap=1e9, vid=vid)
    except GS.Reject as e:
        open(rej, "a").write(
            json.dumps(dict(sid=sid, ctrl_id=cid, family5=sit["family5"], reason=str(e)[:400], stage="master")) + "\n"
        )
        print(f"[v5b] REJECT {sid}: {str(e)[:200]}", flush=True)
        return
    light = R0["light"] or light
    runs.append(dict(R0, kind="A0", start=-1, cls="stock"))

    def add_moments(R, ri, group):
        out = []
        for m in R["moments"]:
            if m.get("skipped"):
                moments.append(dict(m, group=group, run=ri, skipped=m["skipped"]))
                continue
            m.update(group=group, run=ri)
            moments.append(m)
            out.append(len(moments) - 1)
        return out

    def label(R, t_b, arc_b):
        return LB.branch_labels(
            R["ticks"],
            R["bins"]["hb"],
            R["bins"]["hf"],
            R["bins"]["sb"],
            R["t0"],
            R["end"]["reason"],
            R["end"]["t"],
            t_b,
            arc_b,
            ROBOT,
            geo["hazards"],
        )

    def branch(mi, cls, ci, ri, keep):
        m = moments[mi]
        R = runs[ri]
        branches.append(
            dict(
                moment=mi,
                cls=cls,
                cand=int(ci),
                run=ri,
                keep=bool(keep),
                cmd=C[ci]["cmd"],
                tcls=C[ci]["cls"],
                lab=label(R, m["t"], m["arc"]),
            )
        )

    A_m = add_moments(R0, 0, "A")
    fid = {}
    rep = {}
    if a.replay_check and A_m:
        Lr = G.spawn5b(sit, geo)
        Lr["light"] = light
        Rr = G.run_training(
            sit, geo, ctl, Lr, None, master=False, t_cap=max(moments[i]["t"] for i in A_m) + 0.3, thumbs=()
        )
        for i in A_m:
            q = tick_at(Rr["ticks"], moments[i]["t"])
            rep[i] = float(np.linalg.norm(q[3:6] - np.asarray(moments[i]["pos"]))) if q is not None else float("nan")
    B_masters = {}
    for mi in A_m:
        m = moments[mi]
        if FID_ALL or m["label"] in ("d1.5", "null_a"):
            fid[mi] = fidelity(sit, geo, ctl, m, R0)
        branch(mi, "stock", cl["stock"], 0, True)
        for k in pl["opts_A"] if not pl["null"] else pl["opts_A"]:
            ci = cl[k]
            is_bm = k in pl["B"] and m["label"] == f"d{SB.B_FROM_D:.1f}"
            try:
                R = G.run_training(
                    sit,
                    geo,
                    ctl,
                    m,
                    C[ci]["cmd"],
                    master=is_bm,
                    pts=pl["B"].get(k, []) if is_bm else (),
                    t_cap=T_CAP_B if is_bm else T_CAP,
                    vid=vid if is_bm else None,
                )
            except GS.Reject as e:
                print(f"[v5b] run reject {sid} {m['label']} {k}: {str(e)[:120]}", flush=True)
                continue
            runs.append(dict(R, kind="B0" if is_bm else "A", start=mi, cls=k))
            ri = len(runs) - 1
            print(
                f"[v5b.run] {sid} {m['label']} {k} {R['end']['reason']} {R['end']['t'] - R['t0']:.1f}s sim {time.time() - t_w0:.0f}s",
                flush=True,
            )
            branch(mi, k, ci, ri, False)
            if is_bm:
                B_masters[k] = ri
    for P, ri in B_masters.items():
        Bm = add_moments(runs[ri], ri, f"B:{P}")
        for mi in Bm:
            m = moments[mi]
            if FID_ALL or m["label"] == "mid":
                fid[mi] = fidelity(sit, geo, ctl, m, runs[ri])
            branch(mi, "keep", cl[P], ri, True)
            for k, ci in pl["opts_B"][P].items():
                try:
                    R = G.run_training(sit, geo, ctl, m, C[ci]["cmd"], master=False, t_cap=T_CAP)
                except GS.Reject as e:
                    print(f"[v5b] run reject {sid} {m['label']} {k}: {str(e)[:120]}", flush=True)
                    continue
                runs.append(dict(R, kind="B", start=mi, cls=k))
                branch(mi, k, ci, len(runs) - 1, False)
                print(
                    f"[v5b.run] {sid} {P}:{m['label']} {k} {R['end']['reason']} {R['end']['t'] - R['t0']:.1f}s sim {time.time() - t_w0:.0f}s",
                    flush=True,
                )
    for mi, v in fid.items():
        moments[mi]["fid_dpos"] = v
    for mi, v in rep.items():
        moments[mi]["rep_dpos"] = v
    G.rob.stop()
    wall = time.time() - t_w0
    if not branches:
        open(rej, "a").write(
            json.dumps(
                dict(
                    sid=sid,
                    ctrl_id=cid,
                    family5=sit["family5"],
                    reason="no_moment " + ";".join(str(m.get("skipped")) for m in moments)[:300],
                    stage="moments",
                    wall_s=round(wall, 1),
                )
            )
            + "\n"
        )
        print(f"[v5b] REJECT {sid}: no moment", flush=True)
        return
    fn = os.path.join(OUT, "br", f"{sid}.npz")
    save_pair(fn, sit, geo, ctl, pl, runs, moments, branches, light, R0, wall)
    sim_s = sum(r["end"]["t"] - r["t0"] for r in runs)
    mv = [m for m in moments if not m.get("skipped")]
    fd = [m.get("fid_dpos") for m in mv if m.get("fid_dpos") is not None and np.isfinite(m.get("fid_dpos"))]
    succ = {}
    for b in branches:
        key = f"{moments[b['moment']]['group']}|{moments[b['moment']]['label']}|{b['cls']}"
        succ[key] = int(b["lab"]["success"])
    rec = dict(
        sid=sid,
        geo_sit=sit["sid"],
        robot=ROBOT,
        ctrl_id=cid,
        base=a.base,
        virtual=bool(spec.get("virtual")),
        env=a.env,
        family=sit["family"],
        family5=sit["family5"],
        zone_split=sit["lane"].get("split", "train"),
        lane=sit["lane"]["id"],
        hazards=geo["hazards"],
        arc_goal=geo["arc_goal"],
        light=light,
        night=str(light["mode"]).startswith("night"),
        floor_render=R0["floor_render"],
        vis_check=None,
        vis_dec=R0["vis_dec"],
        checks=geo.get("checks"),
        crawl_v2=sit["params"].get("crawl_v2"),
        classes={k: [int(v), C[v]["cls"]] for k, v in cl.items()},
        n_runs=len(runs),
        n_moments=len(mv),
        n_skipped=len(moments) - len(mv),
        n_branches=len(branches),
        skipped=[m["label"] + ":" + m["skipped"] for m in moments if m.get("skipped")],
        fid_dpos=fd,
        fid_flag=int(sum(x > FID_TOL for x in fd)),
        rep_dpos=[moments[i].get("rep_dpos") for i in rep],
        stock_end=R0["end"]["reason"],
        succ=succ,
        sim_s=round(sim_s, 1),
        wall_s=round(wall, 1),
        s_per_branch=round(wall / len(branches), 1),
        br_file=os.path.relpath(fn, OUT_ROOT),
        probe_file=ctl["pfile"],
        trans_file=ctl["tfile"],
        run_id=RUN_ID,
        video=None if vid is None else os.path.relpath(vid["dir"], OUT_ROOT),
    )
    open(fs, "a").write(json.dumps(rec, default=float) + "\n")
    print(
        f"[v5b] {sid} {sit['family5']:12s} {light['mode'][:9]} stock {R0['end']['reason']:8s} runs {len(runs)} moments {len(mv)} "
        f"branches {len(branches)} succ {sum(succ.values())} fid {['%.3f' % x for x in fd]} rep {['%.3f' % x for x in rec['rep_dpos'] if x is not None]} "
        f"sim {sim_s:.0f}s wall {wall:.0f}s ({wall / max(sim_s, 1):.1f}x)",
        flush=True,
    )


def save_pair(fn, sit, geo, ctl, pl, runs, moments, branches, light, R0, wall):
    spec, C = ctl["spec"], ctl["cands"]
    M = [m for m in moments if not m.get("skipped")]
    mid = {id(m): i for i, m in enumerate(M)}
    remap = {i: mid[id(m)] for i, m in enumerate(moments) if not m.get("skipped")}
    nM = len(M)
    J = 0
    o = {}
    # ---- moments: the gen5 model input at the moment (frames -1.5 / -1 / -0.5 / 0 s, h(t), proprio, feas, descriptors)
    o["m_group"] = np.array([m["group"] for m in M])
    o["m_label"] = np.array([m["label"] for m in M])
    for k in ("t", "arc", "e", "yaw", "tilt", "top", "vx", "d_haz", "t_since", "arc_target", "clock"):
        o[f"m_{k}"] = np.array([m[k] for m in M], np.float64)
    o["m_any_switch"] = np.array([m["any_switch"] for m in M], bool)
    o["m_pos"] = np.array([m["pos"] for m in M], np.float64).reshape(-1, 3)
    o["m_run"] = np.array([m["run"] for m in M], np.int16)
    o["m_cmd_prev"] = np.array([cmd_vec(spec, m["cmd_prev"]) for m in M], np.float32).reshape(-1, 5)
    o["m_cmd_prev_json"] = np.array([json.dumps(m["cmd_prev"]) for m in M])
    o["m_cls_prev"] = np.array([CA.transition_class(spec, m["cmd_prev"]) for m in M])
    o["m_fid_dpos"] = np.array([m.get("fid_dpos", np.nan) for m in M], np.float32)
    o["m_rep_dpos"] = np.array([m.get("rep_dpos", np.nan) for m in M], np.float32)
    o["rgb"] = np.stack([np.stack([f["rgb"] for f in m["obs"]["fr"]]) for m in M]).astype(np.uint8)
    o["depth"] = np.stack([m["obs"]["fr"][-1]["dep"] for m in M]).astype(np.float16)
    o["depth128"] = np.stack([np.stack([f["d128"] for f in m["obs"]["fr"]]) for m in M]).astype(np.float16)
    o["cam"] = np.stack([np.stack([f["M"] for f in m["obs"]["fr"]]) for m in M]).astype(np.float64)
    o["base_world"] = np.array([[f["bw"] for f in m["obs"]["fr"]] for m in M], np.float64)
    o["cam_h"] = np.array([[f["cam_h"] for f in m["obs"]["fr"]] for m in M], np.float32)
    o["selfmask"] = np.array([[f["mask"] for f in m["obs"]["fr"]] for m in M], np.float32)
    o["t_hist"] = np.array([[f["t"] - m["t"] for f in m["obs"]["fr"]] for m in M], np.float32)
    o["feas"] = np.stack([m["obs"]["feas"] for m in M]).astype(np.float32)
    o["X"] = np.stack([m["obs"]["X"] for m in M]).astype(np.float16)
    nP = max(len(m["obs"]["prop_b"]) for m in M)
    pb = np.zeros((nM, nP, 13), np.float32)
    pjs = [m["obs"]["prop_j"] for m in M]
    J = max(p.shape[1] for p in pjs) if pjs and pjs[0].ndim == 3 else 0
    pj = np.zeros((nM, nP, J, 7), np.float32)
    pn = np.zeros(nM, np.int16)
    for i, m in enumerate(M):
        n = len(m["obs"]["prop_b"])
        pn[i] = n
        pb[i, nP - n :] = m["obs"]["prop_b"]
        pj[i, nP - n :] = m["obs"]["prop_j"]
    o.update(prop_base=pb, prop_joints=pj, prop_n=pn)
    nR = max(len(m["obs"]["rec"]) for m in M)
    Rq = [r for m in M for r in m["obs"]["rec"]]
    Jq = len(Rq[-1][4]) if Rq else 0
    rt = np.zeros((nM, nR), np.float64)
    rc = np.full((nM, nR, len(CS.HIST_ROLES)), np.nan, np.float32)
    re_ = np.zeros((nM, nR, 4), np.float32)
    ra = np.zeros((nM, nR, len(VC.ACH_KEYS)), np.float64)
    rq = np.zeros((nM, nR, Jq), np.float32)
    rdq = np.zeros((nM, nR, Jq), np.float32)
    rn = np.zeros(nM, np.int16)
    for i, m in enumerate(M):
        R = m["obs"]["rec"]
        rn[i] = len(R)
        for k_, x in enumerate(R):
            rt[i, k_] = x[0]
            rc[i, k_] = np.asarray(x[1], np.float32)
            re_[i, k_] = np.asarray(x[2], np.float32)
            ra[i, k_] = np.asarray(x[3], np.float64)
            if len(x[4]) == Jq:
                rq[i, k_] = x[4]
                rdq[i, k_] = x[5]
    o.update(rec_t=rt, rec_cmd=rc, rec_env=re_, rec_ach=ra, rec_q=rq, rec_dq=rdq, rec_n=rn)
    # ---- branches (one row per moment x command; keep rows point at the master run)
    o["b_moment"] = np.array([remap[b["moment"]] for b in branches], np.int16)
    o["b_cls"] = np.array([b["cls"] for b in branches])
    o["b_cand"] = np.array([b["cand"] for b in branches], np.int16)
    o["b_tcls"] = np.array([b["tcls"] for b in branches])
    o["b_run"] = np.array([b["run"] for b in branches], np.int16)
    o["b_keep"] = np.array([b["keep"] for b in branches], bool)
    o["b_cmd"] = np.array([cmd_vec(spec, b["cmd"]) for b in branches], np.float32).reshape(-1, 5)
    for k in LB.LAB_KEYS:
        o[f"y_{k}"] = np.array([b["lab"][k] for b in branches], np.float32)
    # ---- runs: ticks (4 Hz), 10 Hz trace (t, cmd, ach), contact bins, thumbnails
    o["r_kind"] = np.array([r["kind"] for r in runs])
    o["r_cls"] = np.array([r["cls"] for r in runs])
    o["r_start"] = np.array([remap.get(r["start"], -1) for r in runs], np.int16)
    o["r_t0"] = np.array([r["t0"] for r in runs], np.float64)
    o["r_clock0"] = np.array([r["clock0"] for r in runs], np.float64)
    o["r_end"] = np.array([r["end"]["reason"] for r in runs])
    o["r_t_end"] = np.array([r["end"]["t"] for r in runs], np.float64)
    o["r_fall_reason"] = np.array([str(r["end"].get("fall_reason") or "") for r in runs])
    o["r_cmd"] = np.array([cmd_vec(spec, r["cmd"]) for r in runs], np.float32).reshape(-1, 5)
    o["r_contact_json"] = np.array([json.dumps(r["contact"], default=float) for r in runs])
    cat = lambda xs: (np.concatenate(xs) if xs else np.zeros(0), np.cumsum([0] + [len(x) for x in xs]).astype(np.int32))
    o["tk"], o["tk_off"] = cat([r["ticks"] for r in runs])
    o["tk"] = o["tk"].reshape(-1, 12).astype(np.float32)
    o["tk_t"] = np.concatenate([r["ticks"][:, 0] for r in runs]) if runs else np.zeros(0)
    for kk in ("hb", "hf", "sb"):
        o[f"bins_{kk}"], o[f"bins_{kk}_off"] = cat([np.asarray(r["bins"][kk], np.float32) for r in runs])
    tr = [
        np.array(
            [[x[0]] + list(np.asarray(x[1], float)) + list(np.asarray(x[3], float)) for x in r["trace"]], np.float64
        ).reshape(-1, 1 + len(CS.HIST_ROLES) + len(VC.ACH_KEYS))
        for r in runs
    ]
    o["tr"], o["tr_off"] = cat(tr)
    o["tr"] = o["tr"].reshape(-1, 1 + len(CS.HIST_ROLES) + len(VC.ACH_KEYS))
    th = [(ri, t, im) for ri, r in enumerate(runs) for t, im in r["thumbs"]]
    o["th_run"] = np.array([x[0] for x in th], np.int16)
    o["th_t"] = np.array([x[1] for x in th], np.float32)
    o["th_rgb"] = np.stack([x[2] for x in th]).astype(np.uint8) if th else np.zeros((0, 112, 112, 3), np.uint8)
    np.savez_compressed(
        fn,
        fmt="gen_v5b",
        run_id=RUN_ID,
        sid=f"{sit['sid']}__{ctl['cid']}",
        geo_sit=sit["sid"],
        ctrl_id=ctl["cid"],
        base=spec["base"],
        robot=ROBOT,
        virtual=bool(spec.get("virtual")),
        spec_json=json.dumps(spec),
        probe_file=ctl["pfile"],
        trans_file=ctl["tfile"],
        probe_src="probes_v5",
        fingerprint=np.array(ctl["P"]["fingerprint"], np.float32),
        fp_keys=json.dumps(ctl["P"]["fp_keys"]),
        desc=DESC_PUB,
        env=a.env,
        family=sit["family"],
        family5=sit["family5"],
        zone_split=sit["lane"].get("split", "train"),
        lighting=json.dumps(light, default=float),
        light_mode=str(light["mode"]),
        cands_json=json.dumps([dict(kind=c["kind"], cmd=c["cmd"], cls=c["cls"]) for c in C]),
        desc_keys=json.dumps(DS.DESC_KEYS),
        cand_roles=json.dumps(list(ROLE_V) + ["mode_index"]),
        classes_json=json.dumps(pl["classes"]),
        plan_json=json.dumps(dict(A=pl["A"], B=pl["B"], opts_A=pl["opts_A"], opts_B=pl["opts_B"]), default=float),
        tick_s=np.float32(G5.TICK_S),
        hist_ticks=np.array(G5.HIST_TICKS, np.int8),
        cam_source=G.cam4.src,
        cam_mount=G.cam4.mount,
        joint_names=JN,
        hist_roles=json.dumps(list(CS.HIST_ROLES)),
        ach_keys=json.dumps(list(VC.ACH_KEYS)),
        t_clock0=np.float64(R0["clock0"]),
        tk_cols=json.dumps(["t", "arc", "e", "x", "y", "z", "yaw", "tilt", "top", "vx", "vy", "d_haz"]),
        P0=np.asarray(geo["P0"], np.float64),
        d=np.asarray(geo["d"], np.float64),
        n=np.asarray(geo["n"], np.float64),
        lane_yaw=np.float64(geo["yaw"]),
        floor_z=np.float64(geo["fz"]),
        z_top=np.float64(geo["z_top"] if geo.get("z_top") is not None else np.nan),
        path_world=np.array(geo["path_world"], np.float64),
        hazards_json=json.dumps(geo["hazards"]),
        arc_goal=np.float32(geo["arc_goal"]),
        s_spawn=np.float32(sit["s_spawn"]),
        bin_s=np.float32(0.02),
        fall_rule="mode_aware_v4",
        tilt_fall=np.float32(LB.TILT_FALL),
        wall_s=np.float32(wall),
        crawl_v2=json.dumps(sit["params"].get("crawl_v2")),
        **o,
    )


def main():
    fs = os.path.join(OUT, f"sits_{part}.jsonl")
    rej = os.path.join(OUT, f"rejects_{part}.jsonl")
    done = set()
    for f in (fs, rej):
        if os.path.exists(f):
            R = [json.loads(l) for l in open(f) if l.strip()]
            done |= {r["sid"] for r in R if not str(r.get("reason", "")).startswith("floor_not_rendered")}
    for sit in S_ALL:
        for cid in sit["ctrls"]:
            if cid not in CTRL:
                CTRL[cid] = load_ctrl(cid)
    t_all = time.time()
    n = 0
    for j, sit in enumerate(S_ALL):
        todo = [c for c in sit["ctrls"] if f"{sit['sid']}__{c}" not in done]
        if not todo:
            continue
        cdir = os.path.join(OUT, "crash")
        os.makedirs(cdir, exist_ok=True)
        cfn = os.path.join(cdir, f"{sit['sid']}.cnt")
        ncr = int(open(cfn).read().strip() or 0) if os.path.exists(cfn) else 0
        if ncr >= 2:
            for cid in todo:
                open(rej, "a").write(
                    json.dumps(
                        dict(sid=f"{sit['sid']}__{cid}", ctrl_id=cid, family5=sit["family5"], reason="crash_repeat")
                    )
                    + "\n"
                )
            continue
        open(cfn, "w").write(str(ncr + 1))
        t0 = time.time()
        try:
            geo = G.build(dict(sit, family=sit.get("family_build") or sit["family"]))
        except GS.Reject as e:
            for cid in todo:
                open(rej, "a").write(
                    json.dumps(
                        dict(
                            sid=f"{sit['sid']}__{cid}",
                            ctrl_id=cid,
                            family5=sit["family5"],
                            reason=str(e)[:400],
                            stage="build",
                        )
                    )
                    + "\n"
                )
            print(f"[v5b] REJECT {sit['sid']}: {str(e)[:200]}", flush=True)
            os.path.exists(cfn) and os.remove(cfn)
            continue
        for cid in todo:
            pair(sit, geo, CTRL[cid], j, fs, rej)
            n += 1
        os.path.exists(cfn) and os.remove(cfn)
        print(
            f"[v5b] situation {sit['sid']} done ({len(todo)} ctrl) {time.time() - t0:.0f}s | {n} pairs {(time.time() - t_all) / 60:.1f} min",
            flush=True,
        )


main()
print("[v5b] DONE", flush=True)
app.close()
