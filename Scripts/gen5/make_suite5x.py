#!/usr/bin/env python
"""
make_suite5x.py = make_suite5.py (copied 2026-10-25 23:59, kept unchanged in its G1 / Spot logic) for the WHEELED / AERIAL
bases jackal / iris (ctrl5.XBODY_BASES). make_suite5.py itself is not modified (G1 / Spot production runs from it). Differences:
  * robot key = the base (jackal / iris; the gen_v1 / gen_v2 generators already take those keys); own seed offsets;
  * nominal speed 0.8 m/s (warm-up 1.2 m);
  * family mixes / bands per body (XB_FAMS, XB_BANDS below):
      jackal (wheeled, no posture, cannot climb): curbs it cannot / can just mount (stepover h U(0.04, 0.30), env_step,
        env_curb outdoors), narrow passages (gap w U(0.36, 0.90), env_gap, seq funnel), low overhangs it may not fit
        under (u U(0.28, 0.60); top 0.43 m), clutter, ramps / humps / decks (seq), bends (jog), terrain + slopes;
      iris (aerial; altitude 0.35-1.5 m, stock 1.5 m): overhangs / low ceilings to fly under or over (u 50 % U(0.55, 1.25), 50 % U(1.30, 1.45)),
        overhang + curb behind it (compound, overhang band as above), doorbeams / beams (seq), narrow
        passages (gap / funnel), low bars to fly over, clutter (env), bends; no terrain / rivermark (nothing for a drone).
Original docstring follows.

gen_v5 suites (plain python, seeded). analysis/GEN5_DATA_SPEC.md section 5. Copies / extends make_suite_v4.py (not modified).
  data/gen_v5/suites/<env>_<base>_<tag>.json (+ HASHES.txt)        (--smoke: data/gen_v5_smoke/suites/)

Episode placement (new in gen5): the robot spawns at path arclength s_spawn >= 0.2, walks a 1.5 s NOMINAL warm-up (frame
history), and the first decision tick is D = U(3, 5) m before the hazard front (s_front = s_spawn + warm + D); the episode
runs to the end of the lane / sequence, a fall or a stall. Short lanes: D is reduced to >= 3.0 m, else the lane is skipped.

Families (family5 = data category; family = gen_v4 builder family; all real assets, nothing floating, test scenes >= 3 m):
  school
    crawl      overhang builder; v2 (default since 10-26): underside = assigned controller's lowest MOVING probe top
               + U(-0.12, 0.18), door / plank / bench / partition only, policy5 COMMIT (sit pol_commit) [aggressive]
               (--crawl-legacy: p1 band G1 U(0.50, 1.00) / Spot U(0.45, 0.80), all classes)
    overhang   overhang builder, gen_v4 band (u_draw)
    slot       gap builder, width G1 U(0.25, 0.50) (sidestep) / Spot U(0.40, 0.65)                  [aggressive]
    gap        gap builder, gen_v4 band
    stepover   step 'curb' (one plank-stack bar, step up + down), h U(0.08, 0.40)                    [aggressive]
    climb      step 'ledge' (3 m plank-stack platform, goal ON TOP), h U(0.08, 0.40)                 [aggressive]
    climb2     step 'steps' n=2 (double stack: two levels, tread 0.5-0.8 m), riser U(0.08, 0.20)    [aggressive]
    stairs     step 'steps' n U{2..6} + landing (top level 2 m deep), riser U(0.08, 0.20), tread U(0.25, 0.35)  [aggressive]
    compound   overhang + curb (gen_v4)
    seq        gen_v4 course-like fenced lanes (beam / doorbeam / funnel / hump / deck), FOCUS = the first element
               (aggressive when the focus is beam / doorbeam / funnel / deck)
    jog        NEW fenced lane with a 45 or 90 deg jog (A 0 deg -> J +-theta, length Lj -> C 0 deg), real school slat
               posts + rails (the gym-course fence recipe), 45: Lj U(1.2, 1.8); 90: Lj U(1.6, 2.2)
    null
  office / warehouse_shelves / full_warehouse (env assets): env_overhang (50 % crawl band), env_gap (50 % slot band),
    env_climb (h U(0.08, 0.40), 60 % platform), env_step, env_lowclutter, env_compound, env_clutter, null
  rivermark: env_curb, null
  terrain (Isaac Lab photoreal field): terrain (rough / slippery tiles incl. stairs_pyramid, slope_pyramid) and
    terrain_jog = a tile crossed along a path with a 45 / 90 deg bend ON the tile (60 % slope_pyramid = tilted section)
Lighting >= 60 % night (sample_lighting(0.70), topped up to 0.62). One controller per situation (+ twin p 0.2), sampled
from the base's pool (base, v_*, lk_*, mm_*; never g1s / spotf). Base family weights: g1l x1.5 and g1p x2 on the
climb / stepover / climb2 / stairs families (users' 10-22 notes).
    python3 scripts/gen5/make_suite5.py --env school --base g1c --n 40 --tag p1 [--smoke] [--families climb,stairs --weights 1,1]
"""
import argparse, glob, hashlib, json, math, os, sys
from collections import Counter
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
for p in (THIS, os.path.join(ROOT, "scripts", "gen"), os.path.join(ROOT, "scripts", "gen_train"), os.path.join(ROOT, "scripts", "gen3"),
          os.path.join(ROOT, "scripts", "gen4"), os.path.join(ROOT, "scripts", "terrain")):
    sys.path.insert(0, p)
import ctrl5 as K5
import envfam as EF
import envfam_v2 as EV
import families as FA

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="school")
ap.add_argument("--base", default="g1c")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--twin", type=float, default=0.2)
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--seed-off", type=int, default=0)
ap.add_argument("--tag", default="")
ap.add_argument("--families", default="")
ap.add_argument("--weights", default="")
ap.add_argument("--ctrls", default="", help="restrict the controller pool (comma list)")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--crawl-legacy", action="store_true", help="p1 crawl generator (fixed band, all overhang classes); reproduces old suites")
ap.add_argument("--crawl-pool", action="store_true", help="restrict the controller pool to crawl-capable controllers "
                "(probe lowest MOVING top >= 0.15 m below the standing top)")
ap.add_argument("--commit", type=float, default=0.7, help="crawl situations: policy5 COMMIT probability (sit['pol_commit'])")
a = ap.parse_args()
assert a.base in K5.XBODY_BASES, f"make_suite5x is for {K5.XBODY_BASES}; G1 / Spot: make_suite5.py"
ROBOT = K5.ROBOT_OF[a.base]
RK = a.base
SEED_BASE = dict(school=1_000_000, office=2_000_000, dt_warehouse=4_000_000, rivermark=5_000_000, simple_room=6_000_000,
                 terrain=7_000_000, full_warehouse=3_500_000, warehouse_shelves=3_200_000)
BOFF = dict(g1c=0, g1g=250_000, g1l=500_000, g1u=750_000, g1p=850_000, spotp=900_000, jackal=950_000, iris=980_000)
SB = SEED_BASE[a.env] + 50_000_000 + BOFF[a.base] + a.seed_off + (5_000_000 if a.smoke else 0)     # never gen_v1..v4 seeds
D5 = os.path.join(ROOT, "data", "gen_v5_smoke" if a.smoke else "gen_v5")
OUTD = os.path.join(D5, "suites"); os.makedirs(OUTD, exist_ok=True)
V_NOM = dict(jackal=0.8, iris=0.8)[a.base]
V_MEAS = dict(jackal=0.8, iris=0.8)[a.base]          # probe: both track the commanded speed (jackal 0.95 -> 0.95, iris 0.95 -> 0.94)      # probe_v4 steady speed at nominal (approx.)
WARM = round(1.5 * V_MEAS, 3)
IND = (["env_overhang", "env_gap", "env_climb", "env_step", "env_lowclutter", "env_compound", "env_clutter", "null"],
       [0.22, 0.16, 0.20, 0.06, 0.10, 0.08, 0.04, 0.08])
FAMS = dict(office=IND, warehouse_shelves=IND, full_warehouse=IND, rivermark=(["env_curb", "null"], [0.85, 0.15]),
            simple_room=(["null"], [1.0]), dt_warehouse=(["null"], [1.0]),
            school=(["crawl", "overhang", "slot", "gap", "stepover", "climb", "climb2", "stairs", "compound", "seq", "jog", "null"],
                    [0.12, 0.05, 0.10, 0.04, 0.08, 0.10, 0.06, 0.10, 0.04, 0.15, 0.10, 0.06]),
            terrain=(["terrain", "terrain_jog"], [0.70, 0.30]))
XB_IND = dict(jackal=(["env_gap", "env_step", "env_lowclutter", "env_clutter", "env_overhang", "env_compound", "null"],
                      [0.25, 0.20, 0.20, 0.10, 0.10, 0.05, 0.10]),
              iris=(["env_overhang", "env_gap", "env_lowclutter", "env_clutter", "env_compound", "env_step", "null"],
                    [0.30, 0.15, 0.15, 0.15, 0.10, 0.05, 0.10]))
# school 'compound' needs 3.2 m after the front: with the 1.2 m warm-up no school lane is long enough (smoke: always skipped) -> not used
XB_SCHOOL = dict(jackal=(["stepover", "gap", "overhang", "seq", "jog", "null"], [0.24, 0.14, 0.10, 0.20, 0.20, 0.12]),
                 iris=(["overhang", "gap", "stepover", "seq", "jog", "null"], [0.36, 0.12, 0.08, 0.20, 0.14, 0.10]))
FAMS = dict(FAMS, office=XB_IND[a.base], warehouse_shelves=XB_IND[a.base], full_warehouse=XB_IND[a.base], school=XB_SCHOOL[a.base])
XB_BANDS = dict(jackal=dict(u=(0.28, 0.60), w=(0.36, 0.90), h=(0.04, 0.30)), iris=dict(u=(0.55, 1.25), w=(0.50, 1.00), h=(0.08, 0.40)))   # + 50 % u U(1.30, 1.45) (u_draw4; more support rejects)
XB = XB_BANDS[a.base]
CLIMBY = ("stepover", "climb", "climb2", "stairs", "env_climb")
UPW = dict(g1l=1.5, g1p=2.0)
AGGR = dict(jackal=("stepover", "gap", "env_step", "env_gap", "env_curb"),
            iris=("overhang", "compound", "gap", "env_overhang", "env_compound"))[a.base]      # policy5 AGGRESSIVE set near these
BUILD = dict(crawl="overhang", overhang="overhang", slot="gap", gap="gap", stepover="step", climb="step", climb2="step", stairs="step",
             compound="compound", null="null")
B4 = None                                              # gen_v4 G1 / Spot bands: not used for jackal / iris


def r3(x):
    return round(float(x), 3)


def u_draw4(rng):
    if a.base == "iris" and rng.random() < 0.5:     # undersides 1.30-1.45: the object's top blocks the stock 1.5 m altitude
        return r3(rng.uniform(1.30, 1.45))          # (fly under needed); below 1.25 the drone can also pass OVER the object
    return r3(rng.uniform(*XB["u"]))


def _u_draw4_g1spot(rng):
    r = rng.random()
    lo, hi = (1.05, 1.25) if ROBOT == "g1" else (B4["min_top"] + 0.03, B4["stand_top"] - 0.02)
    if r < 0.75:
        return r3(rng.uniform(lo, hi))
    return r3(rng.uniform(hi, hi + 0.20)) if r < 0.85 else r3(rng.uniform(lo - 0.20, lo))


def u_crawl(rng):
    return u_draw4(rng)


def _u_crawl_g1spot(rng):
    return r3(rng.uniform(0.50, 1.00) if ROBOT == "g1" else rng.uniform(0.45, 0.80))


# ---------------------------------------------------------------- crawl v2 (2026-10-26)
# p1 drew the underside from a fixed band (G1 U(0.50, 1.00), Spot U(0.45, 0.80)), but the probe-measured lowest top while
# MOVING is G1 AMO 0.63 (0.08 m/s), GR00T 1.04, Isaac Lab 1.12, URMA 1.19, PHP 1.28, Spot/ReLIC 0.71 (arm included):
# 24 / 27 production crawl episodes were infeasible by construction. v2 draws the underside RELATIVE to the assigned
# controller's own lowest moving top t_lo: u = t_lo + U(-0.12, +0.18) (~50 % passable with that posture held, the rest
# not), clipped below the standing top. Object classes / poses that the support builder rejected most (grating 6/6,
# window 2/2, shelf 4/6: too deep / too narrow for the support columns; yaw / roll: no column contact) are not used.
CRAWL_CLS = dict(door=0.40, plank=0.25, bench=0.20, partition=0.15)
CRAWL_REL = (-0.12, 0.18)


def crawl_obj(o, rng):
    o = dict(o)
    o["cls"] = str(rng.choice(list(CRAWL_CLS), p=np.array(list(CRAWL_CLS.values())) / sum(CRAWL_CLS.values())))
    o.update(across=True, roll=0.0, slant=0.0, yaw_rel=r3(rng.uniform(-0.25, 0.25)) if rng.random() < 0.5 else 0.0)
    o.setdefault("copy_idx", int(rng.integers(1 << 20))); o.setdefault("scale", r3(rng.uniform(0.85, 1.15)))
    return o


_LOWTOP = {}


def ctrl_low_top(cid):
    """(t_lo, stand_top) from the gen5 probe: lowest top_p50 over non-fallen tests that MOVE (achieved vx >= 0.03)."""
    if cid not in _LOWTOP:
        P = json.load(open(os.path.join(K5.PROBES_V5, cid, "office_v4.json")))
        stand, lo = None, 9.0
        for t in P["tests"]:
            ach = t.get("ach") or {}
            if "top_p50" not in ach or ach.get("fell"):
                continue
            if t.get("label") == "stand":
                stand = float(ach["top_p50"])
            if float(ach.get("vx", 0.0)) >= 0.03:
                lo = min(lo, float(ach["top_p50"]))
        _LOWTOP[cid] = (lo, stand if stand is not None else lo)
    return _LOWTOP[cid]


def crawl_capable(cid):
    try:
        lo, st = ctrl_low_top(cid)
    except Exception:
        return False
    return st - lo >= 0.15


def w_draw4(rng):
    return r3(rng.uniform(*XB["w"]))


def _w_draw4_g1spot(rng):
    if ROBOT == "g1":
        return r3(rng.uniform(B4["depth"], B4["stand_w"]) if rng.random() < 0.75 else rng.uniform(B4["stand_w"], 0.85))
    w = B4["stand_w"]
    return r3(rng.uniform(w - 0.05, w + 0.20) if rng.random() < 0.75 else rng.uniform(w + 0.20, 1.0))


def w_slot(rng):
    return r3(rng.uniform(*XB["w"]))


def _w_slot_g1spot(rng):
    return r3(rng.uniform(0.25, 0.50) if ROBOT == "g1" else rng.uniform(0.40, 0.65))


# ---------------------------------------------------------------- school lanes (gen_v4 test exclusion, copied)
def school_lanes():
    import test_exclusion as TE
    tp = TE.TestProximity()
    GYM = ((-24.445, -6.946), (-2.62, 7.849), 0.009)
    L = []
    for fn, pre in (("gen_v2/lanes_school.json", "a"), ("gen_v2/lanes_school_L7.4.json", "b"), ("gen_v2/lanes_school_L6.8.json", "c"),
                    ("gen/lanes.json", "d")):
        fp = os.path.join(ROOT, "data", fn)
        if os.path.exists(fp):
            for l in json.load(open(fp))["lanes"]:
                L.append(dict(l, id=f"{pre}{l['id']}", len=float(l.get("len", 5.6))))
    Bx = json.load(open(os.path.join(ROOT, "data", "real_scenes", "bake_report.json")))["boxes"]
    TEST_PTS = [(-22.84, 4.6, 0.022), (-24.8, -6.18, 5.101), (-25.0, -6.17, 5.101), (-23.65, -6.23, 5.101)]
    TEST_SEGS = [((-22.95, 0.05), (-22.48, 6.23), 0.022), ((-21.649, -6.327), (-27.343, -6.061), 5.101)]

    def seg_pt(p0, p1, q):
        p0, p1, q = map(np.asarray, (p0, p1, q)); d = p1 - p0
        t = np.clip(((q - p0) @ d) / max(d @ d, 1e-9), 0, 1); return float(np.linalg.norm(p0 + t * d - q))

    def seg_box(p0, p1, lo, hi):
        return min(float(np.linalg.norm(np.maximum(0, np.maximum(np.asarray(lo) - q, q - np.asarray(hi)))))
                   for q in [np.asarray(p0) + t * (np.asarray(p1) - np.asarray(p0)) for t in np.linspace(0, 1, 81)])
    out, why = [], {}
    for l in L:
        p0 = np.array(l["start"], float); y = float(l["yaw"])
        p1 = p0 + float(l.get("len", 8.0)) * np.array([math.cos(y), math.sin(y)])
        fz = float(l["floor"])
        # the jog family shifts the lane up to 2.2 m sideways: the exclusion distance is taken with 3 m + 2.2 m margin
        dmin = min([seg_pt(p0, p1, q[:2]) for q in TEST_PTS if abs(q[2] - fz) < 2.0] +
                   [min(seg_pt(p0, p1, q) for q in np.linspace(g[0], g[1], 30)) for g in TEST_SEGS if abs(g[2] - fz) < 2.0] +
                   [seg_box(p0, p1, b[0][:2], b[1][:2]) for b in Bx.values() if b[0][2] - 2.0 < fz < b[1][2] + 1.0] + [99.0])
        if dmin < 3.0:
            why["test_pts/baked"] = why.get("test_pts/baked", 0) + 1; continue
        if abs(fz - GYM[2]) < 1.5 and seg_box(p0, p1, (GYM[0][0], GYM[1][0]), (GYM[0][1], GYM[1][1])) < 3.0:
            why["gym"] = why.get("gym", 0) + 1; continue
        t = tp.near_poly(np.array([p0, p1]), fz)
        if t is not None:
            why[t] = why.get(t, 0) + 1; continue
        out.append(dict(l, split="val" if l["split"] == "heldout" else "train", d_test=round(dmin, 2), clr_haz=9.0))
    print(f"[suite5] school lanes kept {len(out)} dropped {why}")
    return out


# ---------------------------------------------------------------- placement
NEED_AFTER = dict(crawl=2.5, overhang=2.5, slot=2.1, gap=2.1, stepover=1.5, climb=1.5, compound=3.2, null=1.6, env_overhang=1.8,
                  env_gap=2.1, env_climb=2.2, env_step=1.6, env_lowclutter=2.4, env_compound=3.2, env_clutter=2.4, null_wide=1.6)


def place(rng, L, need_after, s_front_max=None):
    """-> (s_spawn, D, s_front) or None"""
    D = float(rng.uniform(3.0, 5.0))
    s_sp = float(rng.uniform(0.2, 0.5))
    s_f = s_sp + WARM + D
    lim = L - need_after if s_front_max is None else min(L - need_after, s_front_max)
    if s_f > lim:
        D = lim - s_sp - WARM
        if D < 3.0:
            s_sp = 0.2; D = lim - s_sp - WARM
            if D < 3.0:
                return None
        s_f = s_sp + WARM + D
    return r3(s_sp), r3(D), r3(s_f)


# ---------------------------------------------------------------- seq (gen_v4 elements, focus = first)
EL_LEN = dict(beam=lambda e: e["n"] * 0.427, doorbeam=lambda e: e["n"] * 0.733, funnel=lambda e: 0.80, hump=lambda e: 1.45,
              deck=lambda e: 0.72 + e["n"] * 0.717)


def seq_element(kind, rng):
    if kind == "beam":
        u = float(rng.uniform(1.0, 1.25))
        return dict(kind="beam", u=r3(np.clip(u, 0.97, 1.29)), n=int(rng.integers(2, 5)))
    if kind == "doorbeam":
        u = float(rng.uniform(0.66, 0.87))
        return dict(kind="doorbeam", desk_scale=r3(np.clip(u / 0.757, 0.851, 1.149)), n=int(rng.integers(1, 3)))
    if kind == "funnel":
        return dict(kind="funnel", g=r3(np.clip(w_slot(rng) if rng.random() < 0.6 else w_draw4(rng), 0.28, 1.0)))
    if kind == "hump":
        return dict(kind="hump")
    return dict(kind="deck", layers=int(rng.integers(2, 6)), n=int(rng.integers(2, 4)))


def make_seq5(lane, rng):
    L = float(lane["len"])
    kinds = ["beam", "funnel", "hump", "deck", "doorbeam"]
    w = dict(jackal=[0.0, 0.45, 0.30, 0.25, 0.0], iris=[0.30, 0.30, 0.05, 0.05, 0.30])[a.base]      # beam funnel hump deck doorbeam
    for _ in range(60):
        ne = 1 if rng.random() < 0.45 else 2
        els = [seq_element(str(rng.choice(kinds, p=np.array(w) / sum(w))), rng) for _ in range(ne)]
        if sum(e["kind"] in ("beam", "doorbeam") for e in els) > 1:
            continue
        lens = [EL_LEN[e["kind"]](e) for e in els]
        gaps = [float(rng.uniform(0.9, 1.4)) for _ in range(ne - 1)]
        tot = sum(lens) + sum(gaps)
        pl = place(rng, L, tot + 0.7)
        if pl is None:
            continue
        s_sp, D, s_f = pl
        for e, g in zip(els, gaps):
            e["gap_after"] = r3(g)
        pos, s = [], s_f
        for i, ln in enumerate(lens):
            pos.append((s, s + ln)); s = s + ln + (gaps[i] if i < len(gaps) else 0.0)
        return dict(elements=els, focus=0, focus_kind=els[0]["kind"], s_first=r3(s_f), est=[[r3(x) for x in p] for p in pos]), pl
    return None


def retag(x, fam5):
    x = dict(x)
    x["sid"] = f"{a.env[:2]}5_{fam5}_{a.base}_{x['seed']}"
    x["robot_key"] = RK; x["robot"] = a.base; x["gen"] = "v5"; x["family5"] = fam5
    return x


def main():
    rng = np.random.default_rng(SB + 11)
    S = []
    fams, w = FAMS[a.env]
    if a.families:
        fams = a.families.split(","); w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(fams)
    w = np.array([wi * (UPW.get(a.base, 1.0) if f in CLIMBY else 1.0) for f, wi in zip(fams, w)], float); w /= w.sum()
    cnt = np.floor(w * a.n).astype(int)
    while cnt.sum() < a.n:
        cnt[np.argmax(w * a.n - cnt)] += 1
    seqf = sum([[f] * c for f, c in zip(fams, cnt)], [])
    seqf = [seqf[i] for i in np.random.default_rng(SB + 5).permutation(len(seqf))]
    if a.env == "terrain":
        lanes = None
    elif a.env == "school":
        lanes = school_lanes()
    else:
        lanes = json.load(open(os.path.join(ROOT, "data", "gen_v2", f"lanes_{a.env}.json")))["lanes"]
        rf = os.path.join(ROOT, "data", "gen_v2", f"lanes_{a.env}_render.json")
        if os.path.exists(rf):
            RO = json.load(open(rf)); lanes = [l for l in lanes if RO.get(l["id"], {}).get("ok", False)]
        if a.env == "rivermark":
            up = json.load(open(os.path.join(ROOT, "data", "gen_v2", "lanes_rivermark_up.json")))["lanes"]
            lanes = [l for l in lanes if l.get("kind") != "curb"] + up
    sd = SB; skipped = Counter()
    for i, f in enumerate(seqf):
        sd += 1
        lrng = np.random.default_rng([sd, 4242])
        if a.env == "terrain":
            import catalog as CT
            nt = 14; k = i % nt; field = 600 + (a.seed_off // 1000) + BOFF[a.base] // 1000 + i // nt + (500 if a.smoke else 0)
            x = EV.make_terrain_v2(sd, RK, k, field)
            if f == "terrain_jog":
                trng = np.random.default_rng([sd, 77, 5])
                t = "slope_pyramid" if trng.random() < 0.6 else str(trng.choice(CT.ROUGH))
                tile = CT.sample_tile(trng, t, float(trng.uniform(0.3, 1.0)))
                for Dd in tile["dressing"]:
                    if Dd.get("kind") == "rocks":
                        Dd["dynamic"] = False
                x["params"] = dict(type=t, slippery=tile["bucket"] == "slippery", difficulty=tile["difficulty"], tile=tile)
                th = float(trng.choice([45.0, 90.0])); Lj = float(trng.uniform(1.0, 2.0)); side = float(trng.choice([-1.0, 1.0]))
                b = float(trng.uniform(0.8, 2.5))
                x["jog"] = dict(theta_deg=th, Lj=r3(Lj), side=side, s_bend=r3(5.0 + b), fence=False, tilted=t == "slope_pyramid")
            pl = place(np.random.default_rng([sd, 9]), 11.5, 0.0, s_front_max=5.0)
            s_sp, D, _ = pl
            s_sp = r3(5.0 - WARM - D)
            x = retag(x, f if f == "terrain" else "terrain_jog")
            x.update(s_spawn=s_sp, d_start=D, s_front=5.0 if f == "terrain" else x["jog"]["s_bend"] - (0.0 if f == "terrain" else 0.0))
            if f == "terrain_jog":                    # decision zone measured to the tile front; the bend lies on the tile
                x["s_front"] = 5.0
            x["aggr"] = x["params"]["type"] in ("stairs_pyramid", "slope_pyramid", "rubble_rocky")
            S.append(x); continue
        ok = lanes
        if f == "env_curb":
            ok = [l for l in lanes if l.get("kind") == "curb" and l.get("rise", 0) > 0 and float(l["s_riser"]) >= 0.2 + WARM + 3.0]
        elif a.env != "school":
            ok = [l for l in lanes if EV.lane_ok_v2(f if f != "null" else "null", l, a.env)]
        TR = [l for l in ok if l["split"] == "train"]; VA = [l for l in ok if l["split"] == "val"]
        pool = VA if (VA and rng.random() < a.val_frac) else (TR or VA)
        if not pool:
            skipped["no_lane"] += 1; continue
        # prefer lanes long enough for the family (start >= 3 m + warm-up before the hazard, room after it)
        MIN_AFTER = dict(stairs=1.4, climb2=1.9, jog=2.45, seq=1.6)
        need_L = 0.2 + WARM + 3.0 + MIN_AFTER.get(f, NEED_AFTER.get(f, 2.0))
        pool = [l for l in pool if float(l["len"]) >= need_L] or pool
        lane = pool[int(rng.integers(len(pool)))]
        L = float(lane["len"])
        lane_d = {k: lane[k] for k in ("id", "level", "start", "yaw", "floor", "zone", "split", "kind", "clr_haz", "rise", "s_riser") if k in lane} | dict(len=L)
        if f == "seq":
            r = None
            for _ in range(8):
                r = make_seq5(lane, rng)
                if r is not None:
                    break
                lane = pool[int(rng.integers(len(pool)))]; L = float(lane["len"])
                lane_d = {k: lane[k] for k in ("id", "level", "start", "yaw", "floor", "zone", "split") if k in lane} | dict(len=L)
            if r is None:
                skipped["seq_no_layout"] += 1; continue
            Q, (s_sp, D, s_f) = r
            x = dict(seed=int(sd), family="seq", robot=a.base, robot_key=RK, split="school", env="school", params=dict(e=0.0), seq=Q,
                     family_oracle={"beam": "overhang", "doorbeam": "overhang", "funnel": "gap", "hump": "step", "deck": "step"}[Q["focus_kind"]],
                     s_front=s_f, d_dec=D, lane=lane_d, lighting=FA.sample_lighting(lrng, 0.70), lat0=r3(lrng.uniform(-0.04, 0.04)),
                     yaw0=r3(lrng.normal(0, 0.02)))
            x = retag(x, "seq"); x.update(s_spawn=s_sp, d_start=D, aggr=Q["focus_kind"] in ("beam", "doorbeam", "funnel", "deck"))
            S.append(x); continue
        if f == "jog":
            jr = np.random.default_rng([sd, 31])
            # test-scene exclusion holds for the SHIFTED lane too: lateral reach (offset + fence 0.75 m) <= d_test - 3.0
            reach = float(lane.get("d_test", 99.0)) - 3.75
            th = float(jr.choice([45.0, 90.0]))
            if th == 90.0 and reach < 1.6:
                th = 45.0
            if th == 45.0:
                hi = min(1.8, reach / math.sin(math.radians(45.0)))
                if hi < 1.2:
                    skipped["jog_near_test"] += 1; continue
                Lj = float(jr.uniform(1.2, hi))
            else:
                Lj = float(jr.uniform(1.6, min(2.2, reach)))
            need = Lj * math.cos(math.radians(th)) + 1.6
            pl = place(jr, L, need)
            if pl is None:
                skipped["jog_short"] += 1; continue
            s_sp, D, s_f = pl
            x = dict(seed=int(sd), family="jog", robot=a.base, robot_key=RK, split="school", env="school", params=dict(e=0.0),
                     jog=dict(theta_deg=th, Lj=r3(Lj), side=float(jr.choice([-1.0, 1.0])), s_bend=s_f, fence=True, tilted=False),
                     family_oracle="null", s_front=s_f, d_dec=D, lane=lane_d, lighting=FA.sample_lighting(lrng, 0.70),
                     lat0=r3(lrng.uniform(-0.04, 0.04)), yaw0=r3(lrng.normal(0, 0.02)))
            x = retag(x, "jog"); x.update(s_spawn=s_sp, d_start=D, aggr=False)
            S.append(x); continue
        bf = BUILD.get(f, f)
        if a.env == "school":
            x = FA.make_situation(int(sd), bf, lane, RK, "school")
        else:
            x = EV.make_situation_v2(sd, f, lane, RK, a.env)
        P = x["params"]; prng = np.random.default_rng([sd, 13])
        if f == "compound" and isinstance(P.get("obj"), dict):
            P["obj"] = dict(P["obj"], u=u_draw4(prng))
        if f == "crawl":
            P["obj"] = dict(P["obj"], u=u_crawl(prng))
            if not a.crawl_legacy:                      # u is re-drawn per controller after the controller assignment
                P["obj"] = crawl_obj(P["obj"], np.random.default_rng([sd, 13, 2]))
        elif f == "overhang":
            P["obj"] = dict(P["obj"], u=u_draw4(prng))
        elif f == "slot":
            P["w"] = w_slot(prng)
        elif f == "gap":
            P["w"] = w_draw4(prng)
        elif f == "stepover":
            P.update(kind="curb", h=r3(prng.uniform(*XB["h"])), width=r3(prng.uniform(1.4, 2.4)), lc=r3(prng.uniform(-0.1, 0.1)), rot=0.0)
        elif f == "climb":
            P.update(kind="ledge", h=r3(prng.uniform(0.08, 0.40)), width=r3(prng.uniform(1.4, 2.4)), lc=r3(prng.uniform(-0.1, 0.1)), rot=0.0)
        elif f == "climb2":
            P.update(kind="steps", h=r3(prng.uniform(0.08, 0.20)), n=2, tread=r3(prng.uniform(0.5, 0.8)), rot=0.0)
        elif f == "stairs":
            tr_ = r3(prng.uniform(0.25, 0.35)); n_ = int(prng.integers(2, 7))
            n_max = int((L - (0.2 + WARM + 3.0) - 0.9) / tr_)
            P.update(kind="steps", h=r3(prng.uniform(0.08, 0.20)), n=int(max(2, min(n_, n_max))), tread=tr_, rot=0.0)
        elif f == "env_overhang":
            P["u"] = u_crawl(prng)
        elif f == "env_gap":
            P["w"] = w_slot(prng)
        elif f == "env_climb":
            P["h"] = r3(prng.uniform(0.08, 0.40))
        need = NEED_AFTER.get(f)
        if f in ("climb2", "stairs"):
            need = P["n"] * P["tread"] + 0.9
        if f == "env_curb":
            need = L - float(lane["s_riser"])
            pl = place(prng, L, need, s_front_max=float(lane["s_riser"]))
            if pl is not None and abs(pl[2] - float(lane["s_riser"])) > 1e-3:
                s_sp = r3(float(lane["s_riser"]) - WARM - pl[1]); pl = (s_sp, pl[1], r3(float(lane["s_riser"]))) if s_sp >= 0.2 else None
        else:
            pl = place(prng, L, need or 2.0)
        if pl is None:
            skipped[f"{f}_short_lane"] += 1; continue
        s_sp, D, s_f = pl
        x["s_front"] = s_f; x["d_dec"] = D
        x["lighting"] = FA.sample_lighting(lrng, 0.70)
        x["family"] = bf if a.env == "school" else f
        if f == "env_lowclutter":
            x["family_build"] = "env_clutter"
        x = retag(x, f)
        x.update(s_spawn=s_sp, d_start=D, aggr=(f in AGGR) or bool(P.get("crawl")) or bool(P.get("slot")))
        S.append(x)
    nf = lambda: np.mean([x["lighting"]["mode"].startswith("night") for x in S]) if S else 1.0
    trng = np.random.default_rng(SB + 77)
    order = [i for i in trng.permutation(len(S)) if not S[i]["lighting"]["mode"].startswith("night")]
    while nf() < 0.62 and order:
        i = order.pop()
        S[i]["lighting"] = dict(mode="night_white" if trng.random() < 0.5 else "night_yellow", dark=bool(trng.random() < 0.3), topped_up=True)
    # controller pool of the base (only controllers with a gen5 probe in production; smoke: spec present)
    pool = pool_of(a.base)
    if a.ctrls:
        pool = [c for c in a.ctrls.split(",")]
    have = lambda c: os.path.exists(os.path.join(K5.PROBES_V5, c, "transitions.json")) and os.path.exists(os.path.join(K5.PROBES_V5, c, "office_v4.json"))
    if not a.smoke:
        miss = [c for c in pool if not have(c)]
        pool = [c for c in pool if have(c)]
        if miss:
            print(f"[suite5] controllers without a gen5 probe (not assigned): {miss}")
        assert a.base in pool, f"base {a.base} has no gen5 probe"
    if a.crawl_pool:
        drop = [c for c in pool if not crawl_capable(c)]
        pool = [c for c in pool if crawl_capable(c)]
        print(f"[suite5] crawl pool: {len(pool)} capable, dropped {drop}")
        assert pool, "no crawl-capable controller in the pool"
    wts = np.array([pool_weight(c) for c in pool], float); wts /= wts.sum()
    crng = np.random.default_rng(SB + 99)
    for x in S:
        c = [str(crng.choice(pool, p=wts))]
        if crng.random() < a.twin:
            c2 = str(crng.choice(pool, p=wts))
            if c2 != c[0]:
                c.append(c2)
        assert not any(K5.is_held_out(z) for z in c)
        x["ctrls"] = c
        if x["family5"] == "crawl" and not a.crawl_legacy:
            lo, st = ctrl_low_top(c[0])
            cr = np.random.default_rng([int(x["seed"]), 13, 3])
            rel = float(cr.uniform(*CRAWL_REL))
            u = float(np.clip(lo + rel, 0.40, st - 0.06))
            x["params"]["obj"]["u"] = r3(u)
            x["params"]["crawl_v2"] = dict(ctrl=c[0], t_lo=r3(lo), stand=r3(st), rel=r3(u - lo), passable_est=bool(u - lo >= 0.03))
            x["pol_commit"] = float(a.commit)
    name = f"{a.env}_{a.base}" + (f"_{a.tag}" if a.tag else "")
    pth = os.path.join(OUTD, f"{name}.json")
    s = json.dumps(S, sort_keys=True); json.dump(S, open(pth, "w"), indent=0)
    h = hashlib.sha1(s.encode()).hexdigest()
    open(os.path.join(OUTD, "HASHES.txt"), "a").write(f"{name}.json {h} n={len(S)}\n")
    fam_c = Counter(x["family5"] for x in S)
    print(f"[suite5] {pth}: {len(S)} {dict(fam_c)} ctrl-runs {sum(len(x['ctrls']) for x in S)} night={nf():.2f} "
          f"lanes={len(set(x['lane']['id'] for x in S))} D={np.mean([x['d_start'] for x in S]):.2f} skipped {dict(skipped)}")


def pool_of(base):
    if base == "g1p":
        return ["g1p"]
    out = [base] + [os.path.basename(f)[:-5] for f in sorted(glob.glob(os.path.join(K5.SPEC_D5, f"v_{base}_*.json")))]
    out += [os.path.basename(f)[:-5] for f in sorted(glob.glob(os.path.join(K5.SPEC_D5, f"lk_{base}_*.json")))]
    out += [os.path.basename(f)[:-5] for f in sorted(glob.glob(os.path.join(K5.SPEC_D5, f"mm_{base}_*.json")))]
    return out


def pool_weight(c):
    """shares: base 30 %, look-alikes 25 %, many-mode 20 % (g1c / g1g), v_* the rest (spread evenly inside each group)"""
    grp = "base" if not c.startswith(("v_", "lk_", "mm_")) else c[:2]
    P = pool_of(a.base)
    n = Counter(("base" if not z.startswith(("v_", "lk_", "mm_")) else z[:2]) for z in P)
    share = dict(base=0.30, lk=0.25, mm=0.20, v_=0.25)
    tot = sum(share[g] for g in n)
    return share[grp] / tot / n[grp]


main()
