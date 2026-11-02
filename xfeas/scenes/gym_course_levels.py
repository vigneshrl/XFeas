#!/usr/bin/env python
"""
GYM OBSTACLE COURSE -- 5 LEVELS (additive 2026-10-19; v1 gym_course.py and v3 gym_course_v3.py are NOT changed).
Everything is built IN MEMORY in assets/school.usd (the file is never written). Mobility hurdles only.

  L1  plain       = the v1 course exactly (gym_course.make_spec(seed) + GymCourse): ramp, deck, humps / valleys,
                    stairs, platform, fenced straight lane.
  L2  + overhang  = the v3 course exactly (gym_course_v3.make_spec_v3(seed) + GymCourseV3): L1 + the wooden BEAM on
                    deck pillars (underside 1.24 m) + the narrow sidestep FUNNEL (gap 0.36 m).
  L3  + rough     = L2's hurdles on one straight lane + ROUGH-TERRAIN insets in the deck: the top `recess_layers` desk
                    tops of whole deck slots are left out and a photoreal Isaac Lab rough tile fills the hole, its
                    edges flush with the deck (relief from isaaclab.terrains height-field generators, NVIDIA library
                    PBR textures, NVIDIA rocks / leaf litter, bark-textured roots; PhysX material of the catalogue).
  L4  + turns     = L3's hurdles on a longer serpentine lane over the whole gym floor: 3 long straights joined by two
                    U-turns (4 x 90 deg corners: left, left, right, right = an S); the fence follows every corner.
                    4 rough insets (gravel, rubble, roots, dirt), humps / valleys split over two straights.
  L5  + slippery  = L4 + irregular OIL and ICE patches (scripts/terrain catalogue looks + PhysX materials) on flat
                    deck and on corner landings.

L3-L5 use LevelCourse (this file): the same real assets and the same element geometry as v1 / v3 (desk tops
BP_school_desk_26/SM_school_desk_2_2, apron boards .../SM_school_desk_2_3, planks SM_literature_props_1), built segment
by segment in a lane frame per segment. Corners: the incoming segment's deck runs through the square corner landing,
the outgoing segment's deck starts beside it; the outer fence closes the corner, the inner fence stops at it.

  spec = make_level_spec(level, seed)                 # pure python, JSON-able
  plan = plan_layout(spec, dims)                      # pure python: segments, frames, modules, walls, terrace
  python xfeas/scenes/gym_course_levels.py --level 4 --seed 0      # print spec + plan (no Isaac)
  # in a SimulationApp started with xfeas/sim/isaaclab_python.sh (Isaac Lab terrains are needed for L3-L5):
  B, spec = level_builder(stage, level, seed); info = B.build(spec); world.reset(); ver = B.verify(pxq, carb)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, copy, json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import gym_course as GC
import gym_course_v3 as GV

OUT_ROOT = xfeas.paths.COURSES + "/gym_obstacle_course/levels"
NOMINAL = dict(board=(1.2, 0.65, 0.025721), slat=(1.1292, 0.1874, 0.0259), plank=(3.0, 0.5, 0.040))
GYM = GC.GYM  # inner wall faces (x -24.445 .. -6.946, y -2.62 .. 7.849)
# free floor of the gym for the course (survey data/gym_course/survey/gym_survey.json + top render):
#   -y side: wall bars / benches / bleacher steps up to y ~ -1.5, the gym A-frame (SM_gym_props_10) up to y -0.75 at
#   x -20.25 .. -15.5; +y side: wall pilasters / pole (SM_gym_props_9) from y 7.27.
FREE = dict(x=(-24.445 + 0.05, -6.946 - 0.05), y=(-0.60, 7.20))
EXTRA_OFF = {4: ["SM_gym_props_11"], 5: ["SM_gym_props_11"]}  # pommel horse in the middle of the north lane
ROOT_PRIM = "/World/GymCourseLevels"
GAP = GC.GAP
HEAD_SEAT = GC.HEAD_SEAT
RL = 6  # deck = one riser = 6 desk tops


# ------------------------------------------------------------------ spec (pure python)
def _base(seed):
    sp = GC.make_spec(0)  # element geometry = v1 / v3 canonical (seed jitters layout only)
    v3 = GV.make_spec_v3(0)
    return dict(
        board_scale=sp["board_scale"],
        riser_layers=RL,
        hump_scale=sp["hump_scale"],
        valley_scale=sp["valley_scale"],
        valley_inset=sp["valley_inset"],
        cross_gap=sp["cross_gap"],
        support_scale=0.85,
        n_steps=sp["n_steps"],
        tread=sp["tread"],
        platform_len=sp["platform_len"],
        fence=sp["fence"],
        handrail=sp["handrail"],
        crouch=v3["crouch"],
        funnel=v3["funnel"],
    )


ROUGH_LOOK = {  # catalogue type -> (difficulty band, dressing): rock scales in NVIDIA rock units (~0.2 m across at 1.0)
    "gravel_loose": dict(
        diff=(0.45, 0.6), rocks=dict(per_m2=(55, 75), scale=(0.10, 0.22), embed=0.40, zflat=(0.6, 1.0))
    ),
    "rubble_rocky": dict(
        diff=(0.45, 0.6), rocks=dict(per_m2=(9, 13), scale=(0.30, 0.65), embed=0.45, zflat=(0.5, 0.9))
    ),
    "roots_leaves": dict(
        diff=(0.5, 0.7),
        roots=dict(n=(4, 6), dia=(0.05, 0.08)),
        debris=dict(n=(6, 9), size=(0.25, 0.45)),
        rocks=dict(per_m2=(2, 4), scale=(0.25, 0.45), embed=0.45, zflat=(0.5, 0.9)),
    ),
    "dirt_bumpy": dict(diff=(0.5, 0.65), rocks=dict(per_m2=(6, 10), scale=(0.20, 0.45), embed=0.40, zflat=(0.5, 0.9))),
}


def make_level_spec(level=1, seed=0):
    level = int(level)
    if level == 1:
        return dict(level=1, seed=int(seed), base="v1 gym_course.make_spec", course=GC.make_spec(seed))
    if level == 2:
        return dict(
            level=2, seed=int(seed), base="v3 gym_course_v3.make_spec_v3", course=GV.make_spec_v3(seed, gap=0.36)
        )
    rng = np.random.default_rng(1000 * level + int(seed))
    sp = dict(level=level, seed=int(seed), version="levels", **_base(seed))
    sp["rough"] = dict(recess_layers=3, hscale=0.025, vscale=0.002, margin_x=0.14, margin_y=0.10, zero_mean=True)
    sp["start_back"] = 0.5  # robot start on the gym floor, this far before the ramp foot

    def rough(t, slots):
        lk = ROUGH_LOOK[t]
        return dict(
            kind="rough",
            type=t,
            slots=slots,
            difficulty=round(float(rng.uniform(*lk["diff"])), 3),
            seed=int(rng.integers(1 << 30)),
        )

    cross = lambda kinds: dict(kind="crossing", types=list(kinds))
    if level == 3:
        sp["segments"] = [
            dict(
                name="A",
                heading=0.0,
                P0=[-23.60, 2.60],
                last=True,
                x_end_max=-7.05,
                modules=[
                    rough("gravel_loose", 1),
                    dict(kind="beam"),
                    cross(["hump", "valley", "hump", "valley"]),
                    rough("rubble_rocky", 1),
                    dict(kind="funnel"),
                ],
            )
        ]
    else:
        y0 = 0.45
        sp["segments"] = [
            dict(
                name="A",
                heading=0.0,
                P0=[-23.35, y0],
                n_slots=21,
                turn=+1,
                modules=[rough("gravel_loose", 2), dict(kind="beam"), cross(["hump", "valley"])],
            ),
            dict(name="K1", heading=90.0, n_slots=4, turn=+1, modules=[]),
            dict(
                name="B",
                heading=180.0,
                n_slots=21,
                turn=-1,
                modules=[rough("rubble_rocky", 2), cross(["hump", "valley"]), rough("roots_leaves", 2)],
            ),
            dict(name="K2", heading=90.0, n_slots=4, turn=-1, modules=[]),
            dict(
                name="C", heading=0.0, last=True, x_end_max=-7.15, modules=[dict(kind="funnel"), rough("dirt_bumpy", 2)]
            ),
        ]
    if level == 5:
        # slippery patches: (segment, where, index) -> resolved in plan_layout; where = flat k of the segment / corner
        def slick(t, seg, where, k, size):
            return dict(type=t, seg=seg, where=where, k=k, size=list(size), seed=int(rng.integers(1 << 30)))

        sp["slippery"] = [
            slick("oil", "A", "flat", 0, (0.95, 0.80)),
            slick("ice", "A", "corner", 0, (1.00, 0.95)),
            slick("oil", "B", "flat", 1, (0.95, 0.85)),
            slick("ice", "B", "corner", 0, (1.00, 0.95)),
            slick("ice", "C", "flat", 1, (1.00, 0.85)),
            slick("oil", "C", "flat", 2, (1.10, 0.85)),
        ]
    return sp


# ------------------------------------------------------------------ layout (pure python, dims = nominal or measured)
def element_dims(dims, spec):
    bs = spec["board_scale"]
    BL, BD, t = (x * bs for x in dims["board"])
    SL, SH, ST = dims["slat"]
    c = spec["crouch"]
    Pl, Pw, Pt = dims["plank"]
    D = c["n_planks"] * Pw * c["plank_scale"] + (c["n_planks"] - 1) * c["plank_gap"]
    f = spec["funnel"]
    BL1, BD1, BT1 = dims["board"]
    Wb, tb = BD1 * f["board_scale"], BT1 * f["board_scale"]
    l_rail_in = 0.5 * BL + 0.012
    l_lim = l_rail_in - f["rail_clear"]
    th = math.asin((l_lim - 0.5 * f["gap"]) / Wb)
    for _ in range(50):
        th = math.asin((l_lim - 0.5 * f["gap"] - tb * math.cos(th)) / Wb)
    Ls = Wb * math.cos(th) + tb * math.sin(th)
    head = (RL - 1) * t
    th_r = math.asin(head / BD)
    return dict(
        BL=BL,
        BD=BD,
        t=t,
        P=BD + GAP,
        deck_top=RL * t,
        head=head,
        ramp_deg=math.degrees(th_r),
        s_d=BD * math.cos(th_r),
        SL=SL,
        SH=SH,
        ST=ST,
        beam_D=D,
        funnel_L=Ls,
        funnel_th=th,
        l_rail_in=l_rail_in,
    )


def cross_el(kind, s, dims, spec):
    """one crossing element starting at s (lane frame) -- the v1 layout() code for a single hump / valley"""
    SLs, SHs, STs = dims["slat"]
    ss_ = spec["support_scale"]
    Hs, Ts = SHs * ss_, STs * ss_
    if kind == "hump":
        PL, PD, PT = (x * spec["hump_scale"] for x in dims["board"])
        apex = Hs + 0.003
        for _ in range(30):
            th = math.asin(apex / PD)
            apex = Hs + 0.5 * Ts * math.tan(th) + 0.0005
        th = math.asin(apex / PD)
        Hh = PD * math.cos(th) + PT * math.sin(th)
        c = s + Hh
        return dict(
            type="hump",
            c=c,
            extent=(s, c + Hh),
            deg=math.degrees(th),
            apex=apex,
            panel=(PL, PD, PT),
            support=(SLs * ss_, Hs, Ts),
            lat=0.5 * PL,
        )
    PL, PD, PT = (x * spec["valley_scale"] for x in dims["board"])
    ins = spec["valley_inset"]
    lo_, hi_ = math.radians(5), math.radians(30)
    for _ in range(60):
        th = 0.5 * (lo_ + hi_)
        if PD * math.sin(th) - ins * math.tan(th) > Hs:
            hi_ = th
        else:
            lo_ = th
    l0 = 0.002 + PT * math.sin(th)
    l_out = l0 + PD * math.cos(th)
    return dict(
        type="valley",
        c=s + 0.5 * PL,
        extent=(s, s + PL),
        deg=math.degrees(th),
        panel=(PL, PD, PT),
        support=(SLs * ss_, Hs, Ts),
        l0=l0,
        l_out=l_out,
        l_sup=l_out - ins,
        lat=l_out + PT * math.sin(th),
    )


def module_len(m, E, dims, spec):
    if m["kind"] == "rough":
        return m["slots"] * E["P"] - GAP
    if m["kind"] == "beam":
        return E["beam_D"]
    if m["kind"] == "funnel":
        return E["funnel_L"]
    s, out = 0.0, []
    for k, t in enumerate(m["types"]):
        el = cross_el(t, s, dims, spec)
        out.append(el)
        s = el["extent"][1] + (spec["cross_gap"] if k + 1 < len(m["types"]) else 0.0)
    return s


def _terrace(spec, E, s0, s_st):
    """v1 terrace logic: step starts, board counts (from the top step down), ends; s0 = start of the deck (step 0)"""
    starts = [s0, s_st] + [s_st + k * spec["tread"] for k in range(1, spec["n_steps"] + 1)]
    s_p0 = starts[-1]
    req = s_p0 + spec["platform_len"]
    cnt = [0] * len(starts)
    for k in range(len(starts) - 1, -1, -1):
        cnt[k] = max(1, int(math.ceil((req - starts[k]) / E["P"] - 1e-6)))
        req = starts[k] + cnt[k] * E["P"] - GAP
    ends = [starts[k] + cnt[k] * E["P"] - GAP for k in range(len(starts))]
    return dict(
        starts=starts,
        cnt=cnt,
        ends=ends,
        s_p0=s_p0,
        s_end_req=s_p0 + spec["platform_len"],
        tops=[(k + 1) * RL * E["t"] for k in range(len(starts))],
    )


def _place_modules(mods, s_lo, s_hi, s0, E, dims, spec, min_flat=0.15):
    """equal flats between modules (rough insets snapped up to the deck-slot grid s0 + i P). -> list, flat"""
    L = [module_len(m, E, dims, spec) for m in mods]

    def place(flat):
        cur, out = s_lo, []
        for m, ln in zip(mods, L):
            a = cur + flat
            if m["kind"] == "rough":
                i = int(math.ceil((a - s0) / E["P"] - 1e-9))
                a = s0 + i * E["P"]
            out.append(dict(m, s=[a, a + ln], slot0=(int(round((a - s0) / E["P"])) if m["kind"] == "rough" else None)))
            cur = a + ln
        return out, s_hi - cur

    lo, hi = 0.0, max(0.0, (s_hi - s_lo - sum(L)) / (len(mods) + 1))
    if not mods:
        return [], s_hi - s_lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        out, rest = place(mid)
        if rest >= mid:
            lo = mid
        else:
            hi = mid
    out, rest = place(lo)
    if lo < min_flat - 1e-6:
        raise ValueError(
            f"modules do not fit: flat {lo:.3f} < {min_flat} (range {s_lo:.2f}..{s_hi:.2f}, need {sum(L):.2f})"
        )
    return out, lo


def _rot(h):
    return np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])


def plan_layout(spec, dims=None):
    """segments with world frames, deck slots, module positions (lane s), corners, fence walls, terrace, path"""
    dims = dims or NOMINAL
    E = element_dims(dims, spec)
    segs = []
    prev = None
    nseg = len(spec["segments"])
    for k, sg in enumerate(spec["segments"]):
        h = math.radians(sg["heading"])
        d, n = _rot(h)
        if k == 0:
            P0 = np.array(sg["P0"], float)
            s0 = E["s_d"]
        else:
            P0 = prev["C"] + d * (0.5 * E["BL"] + GAP)
            s0 = 0.0
        S = dict(
            name=sg["name"],
            k=k,
            heading=sg["heading"],
            yaw=h,
            d=d,
            n=n,
            P0=P0,
            s0=s0,
            first=(k == 0),
            last=bool(sg.get("last")),
            turn=sg.get("turn"),
        )
        s_lo = s0 if k == 0 else 0.0
        if S["last"]:
            # largest stairs start whose terrace end stays inside x_end_max (world x along d = +x only)
            lim = sg["x_end_max"]
            s_st = None
            for s_try in np.arange(30.0, 1.0, -0.005):
                T = _terrace(spec, E, s0, float(s_try))
                xe = (P0 + d * T["ends"][0])[0] if abs(d[0]) > 0.5 else None
                if xe is not None and (xe - lim) * np.sign(d[0]) <= 0:
                    s_st = float(s_try)
                    break
            T = _terrace(spec, E, s0, s_st)
            S.update(s_st=s_st, terrace=T, n_slots=T["cnt"][0], s_end=T["ends"][0])
            s_hi = s_st
        else:
            S["n_slots"] = sg["n_slots"]
            S["s_end"] = s0 + sg["n_slots"] * E["P"] - GAP
            S["s_c"] = S["s_end"] - 0.5 * E["BL"]
            S["C"] = P0 + d * S["s_c"]
            s_hi = S["s_end"] - E["BL"]
        mods, flat = _place_modules(sg["modules"], s_lo, s_hi, s0, E, dims, spec)
        S.update(modules=mods, flat=flat, s_lo=s_lo, s_hi=s_hi)
        # the flats between modules (for slippery patches): [a, b] in s
        fl, cur = [], s_lo
        for m in mods:
            fl.append([cur, m["s"][0]])
            cur = m["s"][1]
        fl.append([cur, s_hi])
        S["flats"] = fl
        segs.append(S)
        prev = S
    # fence walls (two passes: corner offsets use the neighbouring walls' rail thickness)
    walls = _walls(segs, E, spec)
    walls = _walls(segs, E, spec, prev=walls)
    # robot path along the centre line + start / goal
    first, last = segs[0], segs[-1]
    path = [dict(p=first["P0"] + first["d"] * (-spec["start_back"]), z=0.0, what="start")]
    path.append(dict(p=first["P0"].copy(), z=0.0, what="ramp foot"))
    path.append(dict(p=first["P0"] + first["d"] * E["s_d"], z=E["deck_top"], what="deck"))
    for S in segs[:-1]:
        path.append(dict(p=S["C"].copy(), z=E["deck_top"], what=f"corner {S['name']}"))
    T = last["terrace"]
    path.append(dict(p=last["P0"] + last["d"] * last["s_st"], z=E["deck_top"], what="stairs foot"))
    for j in range(2, len(T["starts"])):
        path.append(dict(p=last["P0"] + last["d"] * T["starts"][j], z=T["tops"][j - 1], what=f"tread {j - 1}"))
    sgoal = T["s_p0"] + 0.5 * spec["platform_len"]
    path.append(dict(p=last["P0"] + last["d"] * sgoal, z=T["tops"][-1], what="goal"))
    plen = 0.0
    for a, b in zip(path[:-1], path[1:]):
        plen += float(np.linalg.norm(b["p"] - a["p"]))
    out = dict(E=E, segments=segs, walls=walls, path=path, path_len_xy=plen)
    out["slippery"] = _slick_places(spec, segs, E) if spec.get("slippery") else []
    out["checks"] = _plan_checks(out, spec)
    return out


def _walls(segs, E, spec, prev=None):
    """per segment two walls ('L' = +n side, 'R' = -n side): rails span [a, b] (lane s), end-post extension beyond the
    rail end (ext; SH/2 = post centred on the rail end), rail scale / thickness"""
    SL, SH, ST = E["SL"], E["SH"], E["ST"]
    li = E["l_rail_in"]
    tw = {}
    if prev:
        for w in prev:
            tw[(w["seg"], w["side"])] = w["rail_t"]
    T = lambda k, side: tw.get((k, side), ST)
    out = []
    for k, S in enumerate(segs):
        for side in ("L", "R"):
            a, ext_a = 0.0, 0.5 * SH  # first segment: from the ramp foot, post centred (v1)
            if k > 0:
                tn = segs[k - 1]["turn"]
                outer = (side == "R") if tn > 0 else (side == "L")
                sC = -(0.5 * E["BL"] + GAP)  # the corner point in this segment's frame
                if outer:
                    oth = "R" if tn > 0 else "L"
                    a, ext_a = sC - (li + T(k - 1, oth)), 0.0
                else:
                    oth = "L" if tn > 0 else "R"
                    a, ext_a = sC + li + T(k - 1, oth) + ST, 0.0
            if S["last"]:
                b, ext_b = S["s_st"], 0.5 * SH  # stairs foot, post centred (v1)
            else:
                tn = S["turn"]
                outer = (side == "R") if tn > 0 else (side == "L")
                if outer:
                    oth = "R" if tn > 0 else "L"
                    b, ext_b = S["s_c"] + li, T(k + 1, oth) + ST
                else:
                    b, ext_b = S["s_c"] - li, 0.0
            Lw = b - a
            nb = max(1, int(round(Lw / SL)))
            while Lw / nb > 1.15 * SL:
                nb += 1
            if Lw / nb < 0.85 * SL:
                nb = max(1, nb - 1)
            bay = Lw / nb
            shrink = 0.0
            if bay > 1.15 * SL + 1e-9 or bay < 0.85 * SL - 1e-9:  # no feasible split: shrink, centred
                bay = min(max(bay, 0.85 * SL), 1.15 * SL)
                shrink = Lw - nb * bay
                a2, b2 = a + 0.5 * shrink, b - 0.5 * shrink
                ext_a, ext_b = 0.0, 0.0
            else:
                a2, b2 = a, b
            rsc = bay / SL
            rail_t = ST * rsc
            posts = [a2 - ext_a + 0.5 * SH] + [a2 + i * bay for i in range(1, nb)] + [b2 + ext_b - 0.5 * SH]
            out.append(
                dict(
                    seg=k,
                    side=side,
                    a=a2,
                    b=b2,
                    nb=nb,
                    bay=bay,
                    rsc=rsc,
                    rail_t=rail_t,
                    posts=posts,
                    shrunk=round(shrink, 4),
                    l_post=li + rail_t + 0.5 * ST,
                    l_rail=li + 0.5 * rail_t,
                )
            )
    return out


def _slick_places(spec, segs, E):
    by = {S["name"]: S for S in segs}
    out = []
    for p in spec["slippery"]:
        S = by[p["seg"]]
        if p["where"] == "corner":
            s_c, room = S["s_c"], (0.5 * E["BL"] - 0.08)
            out.append(dict(p, s=s_c, l=0.0, room_s=(s_c - room, s_c + room), room_l=0.5 * E["BL"] - 0.08))
        else:
            a, b = S["flats"][p["k"]]
            out.append(dict(p, s=0.5 * (a + b), l=0.0, room_s=(a + 0.10, b - 0.10), room_l=0.5 * E["BL"] - 0.08))
    return out


def _plan_checks(P, spec):
    """footprints inside the free gym floor; lanes do not overlap; slippery patches fit their room"""
    E = P["E"]
    res = dict(ok=True, notes=[])
    ext = E["l_rail_in"] + 0.03 + 0.026  # fence outer face
    pts = []
    for S in P["segments"]:
        for s in np.linspace(-(spec["start_back"] if S["first"] else 0.0), S["s_end"], 40):
            for l in (-ext, ext):
                pts.append(S["P0"] + S["d"] * s + S["n"] * l)
    pts = np.array(pts)
    res["footprint_x"] = [round(float(pts[:, 0].min()), 3), round(float(pts[:, 0].max()), 3)]
    res["footprint_y"] = [round(float(pts[:, 1].min()), 3), round(float(pts[:, 1].max()), 3)]
    if (
        pts[:, 0].min() < FREE["x"][0]
        or pts[:, 0].max() > FREE["x"][1]
        or pts[:, 1].min() < FREE["y"][0]
        or pts[:, 1].max() > FREE["y"][1]
    ):
        res["ok"] = False
        res["notes"].append("footprint outside the free gym floor")
    for sl in P["slippery"]:
        if sl["room_s"][1] - sl["room_s"][0] < 0.6:
            res["ok"] = False
            res["notes"].append(f"slippery {sl['type']} {sl['seg']} room too small")
    return res


def jsonable(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 4).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def plan_summary(P):
    out = dict(
        path_len_xy=round(P["path_len_xy"], 3),
        turns=sum(1 for S in P["segments"] if S.get("turn")),
        checks=P["checks"],
        segments=[],
    )
    for S in P["segments"]:
        out["segments"].append(
            dict(
                name=S["name"],
                heading=S["heading"],
                P0=np.round(S["P0"], 3).tolist(),
                n_slots=S["n_slots"],
                s_end=round(S["s_end"], 3),
                flat=round(S["flat"], 3),
                turn=S.get("turn"),
                s_st=round(S["s_st"], 3) if S.get("s_st") else None,
                modules=[
                    dict(
                        kind=m["kind"],
                        type=m.get("type") or m.get("types"),
                        s=[round(m["s"][0], 3), round(m["s"][1], 3)],
                    )
                    for m in S["modules"]
                ],
            )
        )
    out["walls"] = [
        dict(
            seg=w["seg"],
            side=w["side"],
            a=round(w["a"], 3),
            b=round(w["b"], 3),
            nb=w["nb"],
            rsc=round(w["rsc"], 4),
            shrunk=w["shrunk"],
        )
        for w in P["walls"]
    ]
    out["slippery"] = [
        dict(
            type=s["type"],
            seg=s["seg"],
            where=s["where"],
            s=round(s["s"], 3),
            room=[round(s["room_s"][0], 3), round(s["room_s"][1], 3)],
        )
        for s in P["slippery"]
    ]
    return out


# ------------------------------------------------------------------ builder (needs a stage)
def prepare_level(stage, level):
    """v1 gym preparation (in memory) + the level's extra deactivations (L4/L5: the pommel horse in the north lane)"""
    off = GC.prepare_gym(stage)
    for nme in EXTRA_OFF.get(int(level), []):
        pr = stage.GetPrimAtPath(GC.G + nme)
        if pr.IsValid() and pr.IsActive():
            pr.SetActive(False)
            off.append(GC.G + nme)
    return off


def level_builder(stage, level, seed=0):
    """-> (builder, spec): L1 = v1 GymCourse, L2 = v3 GymCourseV3, L3-L5 = LevelCourse"""
    sp = make_level_spec(level, seed)
    if int(level) == 1:
        return GC.GymCourse(stage), sp["course"]
    if int(level) == 2:
        return GV.GymCourseV3(stage), sp["course"]
    return LevelCourse(stage), sp


def _terrain_mods():
    import catalog as CAT, build as TB

    return CAT, TB


def _hf_isaaclab(layers, Lt, Wt, hs, vs, seed):
    """sum of Isaac Lab height-field layers (raw, no taper) on an (nx, ny) grid -> metres"""
    from isaaclab.terrains.height_field import hf_terrains as H, hf_terrains_cfg as C

    nx, ny = int(round(Lt / hs)) + 1, int(round(Wt / hs)) + 1
    Z = np.zeros((nx, ny))
    for k, ly in enumerate(layers):
        np.random.seed((seed + 7919 * k) % (2**31))
        base = dict(size=(Lt, Wt), horizontal_scale=hs, vertical_scale=vs, border_width=0.0)
        if ly["fn"] == "random_uniform":
            cfg = C.HfRandomUniformTerrainCfg(
                noise_range=tuple(ly["noise"]),
                noise_step=ly["step"],
                downsampled_scale=(None if ly["down"] is None else max(hs, min(ly["down"], Lt / 5.0, Wt / 5.0))),
                **base,
            )
            a = H.random_uniform_terrain.__wrapped__(1.0, cfg)
        elif ly["fn"] == "wave":
            cfg = C.HfWaveTerrainCfg(amplitude_range=(ly["amp"], ly["amp"]), num_waves=ly["waves"], **base)
            a = H.wave_terrain.__wrapped__(1.0, cfg)
        else:
            raise ValueError(ly["fn"])
        a = np.asarray(a, float)[:nx, :ny]
        a = np.pad(a, ((0, nx - a.shape[0]), (0, ny - a.shape[1])), mode="edge")
        Z += a
    return Z * vs


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class LevelCourse(GC.GymCourse):
    """L3-L5: segments (lane frame per segment), v1/v3 elements, rough insets, slippery patches"""

    def __init__(self, stage, root=ROOT_PRIM):
        super().__init__(stage, root=root)
        self.dims["plank"] = self.lib.flat_dims("plank")
        self.troot = "/World/GymCourseTerrain"
        self.beams, self.funnels, self.crossings, self.tiles, self.slicks = [], [], [], [], []

    # ---- frames
    def frame(self, S):
        self.S = S
        self.P0 = np.asarray(S["P0"], float)
        self.d = np.asarray(S["d"], float)
        self.n = np.asarray(S["n"], float)
        self.yaw = float(S["yaw"])

    def put(self, role, *a, **k):
        p = GC.GymCourse.put(self, role, *a, **k)
        p.update(seg=self.S["name"], d=self.d.copy(), n=self.n.copy(), P0=self.P0.copy())
        return p

    def _yaw_al(self):
        sy = self.lib.size("slat")
        return self.yaw + (0.0 if sy[0] >= sy[1] else math.pi / 2)

    def _byaw(self):
        return self.yaw + math.pi / 2 + self.lib.flat_yaw("board")

    def Wl(self, S, s, l, z=0.0):
        xy = np.asarray(S["P0"]) + np.asarray(S["d"]) * s + np.asarray(S["n"]) * l
        return np.array([xy[0], xy[1], GC.FLOOR_Z + z])

    # ------------------------------------------------------------ build
    def build(self, spec):
        from pxr import UsdGeom

        self.spec = spec
        P = plan_layout(spec, self.dims)
        self.plan = P
        E = P["E"]
        self.E = E
        UsdGeom.Xform.Define(self.stage, self.troot)
        bs = spec["board_scale"]
        t = E["t"]
        BD = E["BD"]
        Pp = E["P"]
        kr = spec["rough"]["recess_layers"]
        info = dict(level=spec["level"], seed=spec["seed"], plan=plan_summary(P))
        # 1. deck of every segment (desk tops across the lane, RL layers; rough slots leave out the top kr layers)
        for S in P["segments"]:
            self.frame(S)
            byaw = self._byaw()
            recess = set()
            for m in S["modules"]:
                if m["kind"] == "rough":
                    recess |= set(range(m["slot0"], m["slot0"] + m["slots"]))
            S["recess"] = sorted(recess)
            for i in range(S["n_slots"]):
                for jl in range(RL):
                    if i in recess and jl >= RL - kr:
                        continue
                    s0k = S["s0"] + (HEAD_SEAT if (S["first"] and jl == RL - 1) else 0.0)
                    w = self.W(s0k + i * Pp + BD / 2, 0.0, jl * t)
                    self.put(
                        f"{S['name']}_deck_L{jl}_{i:02d}",
                        "board",
                        bs,
                        byaw,
                        center=w[:2],
                        z_bottom=w[2],
                        orient="flat",
                        contacts=[("down", "flat")],
                    )
        # 2. entry ramp (first segment)
        S = P["segments"][0]
        self.frame(S)
        byaw = self._byaw()
        th = math.radians(E["ramp_deg"])
        for sg in (1, -1):
            Pc, signs, size = self._local("board", "flat", byaw, bs, sg * th, tuple(self.n))
            Bf = self.bottom_face(Pc, signs, size)
            hi = Bf[np.argsort(Bf[:, 2])[2:]]
            if (hi.mean(0)[:2] - Bf.mean(0)[:2]) @ self.d > 0:
                pitch_r = sg * th
                break

        def head_anchor(Pc, signs, size):
            Bf = self.bottom_face(Pc, signs, size)
            return Bf[np.argsort(Bf[:, 2])[2:]].mean(0)

        self.put(
            "ramp",
            "board",
            bs,
            byaw,
            pitch=pitch_r,
            axis=tuple(self.n),
            anchor=head_anchor,
            target=self.W(E["s_d"] + 0.5 * HEAD_SEAT, 0.0, E["head"]),
            orient="flat",
            contacts=[("down", "edges")],
        )
        info["ramp"] = dict(deg=round(E["ramp_deg"], 3), head_z=round(E["head"], 4), s_d=round(E["s_d"], 4))
        # 3. modules
        nh = dict(hump=0, valley=0)
        for S in P["segments"]:
            self.frame(S)
            for m in S["modules"]:
                if m["kind"] == "crossing":
                    s = m["s"][0]
                    for k, typ in enumerate(m["types"]):
                        el = cross_el(typ, s, self.dims, spec)
                        nh[typ] += 1
                        self._cross(el, f"{S['name']}_{typ}{nh[typ]}")
                        self.crossings.append(dict(el, seg=S["name"], name=f"{typ} {nh[typ]}"))
                        s = el["extent"][1] + spec["cross_gap"]
                elif m["kind"] == "beam":
                    self.beams.append(self._beam(m["s"][0], f"{S['name']}_beam"))
                elif m["kind"] == "funnel":
                    self.funnels.append(self._funnel(m["s"][0], f"{S['name']}_funnel"))
        # 4. terrace (stairs + platform) + handrails on the last segment
        info["terrace"] = self._terrace(P["segments"][-1])
        # 5. fence
        info["fence"] = self._fence()
        # 6. rough insets
        CAT, TB = _terrain_mods()
        self.mats = TB.Materials(self.stage)
        for S in P["segments"]:
            for m in S["modules"]:
                if m["kind"] == "rough":
                    self.tiles.append(self._rough(m, S, CAT, TB))
        # 7. slippery patches
        for k, sl in enumerate(P["slippery"]):
            self.slicks.append(self._slick(sl, k, CAT, TB))
        info["crossing"] = [
            dict(
                name=c["name"], seg=c["seg"], extent_s=[round(x, 4) for x in c["extent"]], panel_deg=round(c["deg"], 3)
            )
            for c in self.crossings
        ]
        info["beam"] = [{k: v for k, v in b.items() if k not in ("paths", "d", "n", "P0")} for b in self.beams]
        info["funnel"] = [{k: v for k, v in f.items() if k not in ("d", "n", "P0")} for f in self.funnels]
        info["rough"] = [
            {k: v for k, v in r.items() if k not in ("Z", "frame", "rocks_bb", "hfun")} for r in self.tiles
        ]
        info["slippery"] = self.slicks
        info["n_parts"] = len(self.parts)
        # robot task
        path = [dict(xyz=np.round(np.r_[p["p"], GC.FLOOR_Z + p["z"]], 3).tolist(), what=p["what"]) for p in P["path"]]
        first = P["segments"][0]
        info["robot"] = dict(
            start=[*np.round(path[0]["xyz"][:2], 3).tolist(), round(first["yaw"], 4)],
            goal=path[-1]["xyz"],
            path=path,
            path_len_xy=round(P["path_len_xy"], 3),
            turns=self._turns(),
        )
        self.info = info
        return info

    def _turns(self):
        out = []
        segs = self.plan["segments"]
        for a, b in zip(segs[:-1], segs[1:]):
            dh = (b["heading"] - a["heading"] + 180) % 360 - 180
            out.append(
                dict(
                    at=a["name"] + "->" + b["name"],
                    corner_xy=np.round(a["C"], 3).tolist(),
                    deg=dh,
                    dir="left" if dh > 0 else "right",
                )
            )
        return out

    # ---- elements (ported from gym_course.py / gym_course_v3.py, in the current segment frame)
    def _cross(self, el, tag):
        s_ = self.spec
        dk = self.E["deck_top"]
        byaw = self._byaw()
        yaw_al = self._yaw_al()
        PL, PD, PT = el["panel"]
        SL_, SH_, ST_ = el["support"]
        th = math.radians(el["deg"])
        ss_ = s_["support_scale"]

        def upper_anchor(P, signs, size):
            B = self.bottom_face(P, signs, size)
            return B[np.argsort(B[:, 2])[2:]].mean(0)

        if el["type"] == "hump":
            C = self.W(el["c"], 0.0, dk)
            self.put(
                f"{tag}_ridge",
                "slat",
                ss_,
                yaw_al + math.pi / 2,
                center=C[:2],
                z_bottom=C[2],
                contacts=[("down", "flat")],
            )
            u_sup = (0.25 * ST_) / (PD * math.cos(th))
            v_sup = min(0.3, 0.4 * SL_) / PL
            for side in (-1, 1):
                axis = tuple(self.n)
                for sg in (1, -1):
                    P, signs, size = self._local("board", "flat", byaw, s_["hump_scale"], sg * th, axis)
                    Bf = self.bottom_face(P, signs, size)
                    hi = Bf[np.argsort(Bf[:, 2])[2:]]
                    if ((hi.mean(0)[:2] - Bf.mean(0)[:2]) @ self.d) * (-side) > 0:
                        break
                self.put(
                    f"{tag}_panel{'ab'[side > 0]}",
                    "board",
                    s_["hump_scale"],
                    byaw,
                    pitch=sg * th,
                    axis=axis,
                    anchor=upper_anchor,
                    target=C + np.r_[0, 0, el["apex"]],
                    orient="flat",
                    contacts=[("down", ("ridge", u_sup, v_sup))],
                )
        else:
            vyaw = self.yaw + self.lib.flat_yaw("board")
            for side in (-1, 1):
                lc = side * (el["l_sup"] + 0.5 * ST_)
                w = self.W(el["c"], lc, dk)
                self.put(
                    f"{tag}_slat{'RL'[side > 0]}",
                    "slat",
                    ss_,
                    yaw_al,
                    center=w[:2],
                    z_bottom=w[2],
                    contacts=[("down", "flat")],
                )
                axis = tuple(self.d)
                for sg in (1, -1):
                    P, signs, size = self._local("board", "flat", vyaw, s_["valley_scale"], sg * th, axis)
                    Bf = self.bottom_face(P, signs, size)
                    hi = Bf[np.argsort(Bf[:, 2])[2:]]
                    if ((hi.mean(0)[:2] - Bf.mean(0)[:2]) @ self.n) * side > 0:
                        break
                tgt = self.W(el["c"], side * el["l_out"], dk + PD * math.sin(th))
                u_sup = (s_["valley_inset"] - 0.003) / (PD * math.cos(th))
                v_sup = min(0.35, 0.4 * SL_) / PL
                self.put(
                    f"{tag}_panel{'RL'[side > 0]}",
                    "board",
                    s_["valley_scale"],
                    vyaw,
                    pitch=sg * th,
                    axis=axis,
                    anchor=upper_anchor,
                    target=tgt,
                    orient="flat",
                    contacts=[("down", ("ridge", u_sup, v_sup))],
                )

    def _beam(self, s_door0, tag):
        c = self.spec["crouch"]
        E = self.E
        dk = E["deck_top"]
        SLs, SHs, STs = self.dims["slat"]
        yaw_al = self._yaw_al()
        Pl, Pw_, Pt = self.lib.flat_dims("plank")
        bsc_ = c["plank_scale"]
        pw_b = Pw_ * bsc_
        npk = c["n_planks"]
        D = npk * pw_b + (npk - 1) * c["plank_gap"]
        pyaw_b = self.yaw + math.pi / 2 + self.lib.flat_yaw("plank")
        psc = c["u"] / SLs
        assert 0.85 <= psc <= 1.15, psc
        l_pil = 0.5 * E["BL"] - 0.5 * STs * psc - c["pillar_inset"]
        posts, planks, paths = [], [], []
        for k in range(npk):
            s_k = s_door0 + k * (pw_b + c["plank_gap"]) + 0.5 * pw_b
            tops = []
            for side in (-1, 1):
                wp = self.W(s_k, side * l_pil, dk)
                p = self.put(
                    f"{tag}_pillar{'RL'[side > 0]}{k}",
                    "slat",
                    psc,
                    yaw_al,
                    pitch=math.pi / 2,
                    axis=tuple(self.n),
                    center=wp[:2],
                    z_bottom=wp[2],
                    contacts=[("down", "flat")],
                )
                top = float(p["corners"][:, 2].max()) - GC.FLOOR_Z
                tops.append(top)
                posts.append(
                    dict(
                        role=p["role"],
                        s=round(s_k, 4),
                        l=round(side * l_pil, 4),
                        scale=round(psc, 4),
                        top_above_floor=round(top, 4),
                    )
                )
            w = self.W(s_k, 0.0, max(tops))
            pk = self.put(
                f"{tag}_plank{k}", "plank", bsc_, pyaw_b, center=w[:2], z_bottom=w[2], orient="flat", contacts=[]
            )
            paths.append(pk["path"])
            planks.append(dict(role=pk["role"], s=round(s_k, 4), path=pk["path"]))
        return dict(
            seg=self.S["name"],
            s=[round(s_door0, 4), round(s_door0 + D, 4)],
            depth=round(D, 4),
            width=round(Pl * bsc_, 4),
            thick=round(Pt * bsc_, 4),
            u_target=c["u"],
            plank_scale=bsc_,
            n_planks=npk,
            pillars=posts,
            planks=planks,
            paths=paths,
            d=self.d.copy(),
            n=self.n.copy(),
            P0=self.P0.copy(),
        )

    def _funnel(self, s_o, tag):
        RA = self.RA
        f = self.spec["funnel"]
        E = self.E
        dk = E["deck_top"]
        SLs, SHs, STs = self.dims["slat"]
        yaw_al = self._yaw_al()
        lp = lambda P: (P[:, :2] - self.P0) @ self.n
        sp = lambda P: (P[:, :2] - self.P0) @ self.d
        bsc = f["board_scale"]
        BL1, BD1, BT1 = self.dims["board"]
        Hb, Wb, tb = BL1 * bsc, BD1 * bsc, BT1 * bsc
        g = f["gap"]
        l_lim = E["l_rail_in"] - f["rail_clear"]
        th = math.asin((l_lim - 0.5 * g) / Wb)
        for _ in range(50):
            th = math.asin((l_lim - 0.5 * g - tb * math.cos(th)) / Wb)
        Ls = Wb * math.cos(th) + tb * math.sin(th)
        s_t = s_o + Wb * math.cos(th)
        wl = lambda s, l: np.r_[self.d * s + self.n * l]
        wings, fposts = [], []
        for side in (-1, 1):
            v = np.array([-math.cos(th), side * math.sin(th)])
            nb = np.array([math.sin(th), side * math.cos(th)])
            Ft = np.array([s_t, side * 0.5 * g])
            Cc = Ft + v * (0.5 * Wb) + nb * (0.5 * tb)
            a_w, v_w = wl(*nb), wl(*v)
            byaw = math.atan2(a_w[1], a_w[0]) + self.lib.flat_yaw("board")
            wc = self.W(Cc[0], Cc[1], dk)
            part = self.put(
                f"{tag}_wing{'RL'[side > 0]}",
                "board",
                bsc,
                byaw,
                pitch=math.pi / 2,
                axis=(float(v_w[0]), float(v_w[1])),
                center=wc[:2],
                z_bottom=wc[2],
                orient="flat",
                contacts=[("down", "flat")],
            )
            Pw = RA.world_points(self.stage.GetPrimAtPath(part["path"]))
            delta = float((side * lp(Pw)).min()) - 0.5 * g
            rec = self.lib.placed[-1]
            dxy = -side * delta * self.n
            self.lib.shift([rec], dxy)
            part["corners"] = part["corners"] + np.r_[dxy, 0.0]
            Ft = Ft + np.array([0.0, -side * delta])
            P = part["corners"]
            assert abs(np.ptp(P[:, 2]) - Hb) < 0.01, (np.ptp(P[:, 2]), Hb)
            wings.append(
                dict(
                    role=part["role"],
                    mesh_edge_correction=round(-delta, 4),
                    s=[round(float(sp(P).min()), 4), round(float(sp(P).max()), 4)],
                    l=[round(float(lp(P).min()), 4), round(float(lp(P).max()), 4)],
                    height=round(float(np.ptp(P[:, 2])), 4),
                    angle_to_lane_deg=round(math.degrees(th), 3),
                )
            )
            ps = f["post_scale"]
            pt = STs * ps
            pyaw = math.atan2(v_w[1], v_w[0]) + (yaw_al - self.yaw)
            for k, fr in enumerate(f["post_at"]):
                Q = Ft + v * (Wb * (1.0 - fr)) + nb * (tb + 0.5 * pt + 0.0005)
                wq = self.W(Q[0], Q[1], dk)
                pp = self.put(
                    f"{tag}_post{'RL'[side > 0]}{k}",
                    "slat",
                    ps,
                    pyaw,
                    pitch=math.pi / 2,
                    axis=(float(a_w[0]), float(a_w[1])),
                    center=wq[:2],
                    z_bottom=wq[2],
                    contacts=[("down", "flat")],
                )
                PP = pp["corners"]
                assert float(np.abs(lp(PP)).max()) <= 0.5 * E["BL"] - 0.003, float(np.abs(lp(PP)).max())
                fposts.append(
                    dict(
                        role=pp["role"],
                        s=round(float(Q[0]), 4),
                        l=round(float(Q[1]), 4),
                        height=round(SLs * ps, 4),
                        back_face_dir=nb.round(4).tolist(),
                        wing=part["role"],
                    )
                )
        return dict(
            seg=self.S["name"],
            gap_design=g,
            angle_to_lane_deg=round(math.degrees(th), 3),
            wing_h=round(Hb, 4),
            wing_w=round(Wb, 4),
            s=[round(s_o, 4), round(s_o + Ls, 4)],
            s_throat=round(s_t, 4),
            wings=wings,
            posts=fposts,
            mouth_width_design=round(2 * (0.5 * g + Wb * math.sin(th)), 4),
            d=self.d.copy(),
            n=self.n.copy(),
            P0=self.P0.copy(),
        )

    def _terrace(self, S):
        """stairs + platform boards on top of the last segment's deck + v1 handrails / platform enclosure"""
        self.frame(S)
        s_ = self.spec
        E = self.E
        T = S["terrace"]
        bs = s_["board_scale"]
        t = E["t"]
        BD = E["BD"]
        Pp = E["P"]
        byaw = self._byaw()
        starts, cnt, ends = T["starts"], T["cnt"], T["ends"]
        nsteps = len(starts)
        layer = RL
        for k in range(1, nsteps):
            for jl in range(RL):
                for ia in range(cnt[k]):
                    w = self.W(starts[k] + ia * Pp + BD / 2, 0.0, layer * t)
                    role = "platform" if k == nsteps - 1 else f"step{k}"
                    self.put(
                        f"{S['name']}_{role}_L{layer}_{ia}",
                        "board",
                        bs,
                        byaw,
                        center=w[:2],
                        z_bottom=w[2],
                        orient="flat",
                        contacts=[("down", "flat")],
                    )
                layer += 1
        tops = [(k + 1) * RL * t for k in range(nsteps)]
        # handrails (v1 code)
        SLs, SHs, STs = self.dims["slat"]
        yaw_al = self._yaw_al()
        hr = s_["handrail"]
        hp = hr["post_scale"]
        Hh = (SLs * hp, SHs * hp, STs * hp)
        l_hp = 0.5 * E["BL"] - 0.5 * Hh[2] - 0.004
        s_t1 = 0.5 * (starts[1] + starts[2]) if nsteps > 2 else starts[1] + 0.15
        s_pf = T["s_p0"] + 0.5 * Hh[1] + 0.02
        z_t1, z_pf = tops[1], tops[-1]
        pe = ends[-1]
        n_side = max(1, int(round((pe - 0.5 * Hh[1] - 0.02 - s_pf) / 1.0)))
        side_posts = list(np.linspace(s_pf, pe - 0.5 * Hh[1] - 0.02, n_side + 1))
        for side in (-1, 1):
            tag = 'RL'[side > 0]
            self.put(
                f"hand{tag}_post_t1",
                "slat",
                hp,
                yaw_al,
                pitch=math.pi / 2,
                axis=tuple(self.n),
                center=self.W(s_t1, side * l_hp)[:2],
                z_bottom=GC.FLOOR_Z + z_t1,
                contacts=[("down", "flat")],
            )
            for j_, sp_ in enumerate(side_posts):
                self.put(
                    f"plat{tag}_post{j_}",
                    "slat",
                    hp,
                    yaw_al,
                    pitch=math.pi / 2,
                    axis=tuple(self.n),
                    center=self.W(sp_, side * l_hp)[:2],
                    z_bottom=GC.FLOOR_Z + z_pf,
                    contacts=[("down", "flat")],
                )
            run = s_pf - s_t1
            rise = z_pf - z_t1
            phi = math.atan2(rise, run)
            Lr = math.hypot(run, rise)
            rs_ = min(max(Lr / SLs, 0.85), 1.15)
            l_r = side * (l_hp - 0.5 * Hh[2] - 0.5 * STs * rs_)
            for k, hz in enumerate(hr["rails_above"]):
                mid = self.W(0.5 * (s_t1 + s_pf), l_r, 0.5 * (z_t1 + z_pf) + hz)
                for sg in (1, -1):
                    P, signs, size = self._local("slat", "as_is", yaw_al, rs_, sg * phi, tuple(self.n))
                    i_long = int(np.argmax(size))
                    hiend = P[signs[:, i_long] == 1].mean(0)
                    loend = P[signs[:, i_long] == -1].mean(0)
                    if (hiend[2] - loend[2]) * ((hiend[:2] - loend[:2]) @ self.d) > 0:
                        break
                self.put(
                    f"hand{tag}_rail{k}",
                    "slat",
                    rs_,
                    yaw_al,
                    pitch=sg * phi,
                    axis=tuple(self.n),
                    anchor=lambda P, signs, size: P.mean(0),
                    target=mid,
                    contacts=[("lat", side)],
                )
            for j_ in range(len(side_posts) - 1):
                a_, b_ = side_posts[j_], side_posts[j_ + 1]
                rsc2 = (b_ - a_) / SLs
                for k, hz in enumerate(hr["rails_above"]):
                    zc = z_pf + hz
                    w = self.W(0.5 * (a_ + b_), side * (l_hp - 0.5 * Hh[2] - 0.5 * STs * rsc2), zc)
                    self.put(
                        f"plat{tag}_rail{j_}_{k}",
                        "slat",
                        rsc2,
                        yaw_al,
                        center=w[:2],
                        z_bottom=GC.FLOOR_Z + zc - 0.5 * SHs * rsc2,
                        contacts=[("lat", side)],
                    )
        span = 2 * (l_hp - 0.5 * Hh[2])
        rs_e = span / SLs
        for k, hz in enumerate(hr["rails_above"]):
            zc = z_pf + hz
            w = self.W(side_posts[-1] + 0.5 * Hh[1] - 0.5 * STs * rs_e - 0.003, 0.0, zc)
            self.put(
                f"plat_end_rail{k}",
                "slat",
                rs_e,
                yaw_al + math.pi / 2,
                center=w[:2],
                z_bottom=GC.FLOOR_Z + zc - 0.5 * SHs * rs_e,
                contacts=[("ends", None)],
            )
        self.terrace = dict(
            seg=S["name"],
            steps=nsteps,
            layers=layer,
            boards_per_step=cnt,
            starts=[round(x, 4) for x in starts],
            ends=[round(x, 4) for x in ends],
            tops=[round(x, 4) for x in tops],
            s_st=round(S["s_st"], 4),
            s_p0=round(T["s_p0"], 4),
            stair_rail_deg=round(math.degrees(phi), 3),
            platform_posts_s=[round(x, 4) for x in side_posts],
            end_x=round(float(self.W(ends[0], 0)[0]), 4),
        )
        return self.terrace

    def _fence(self):
        E = self.E
        fs = self.spec["fence"]
        SLs, SHs, STs = self.dims["slat"]
        ps = fs["post_scale"]
        out = []
        for w in self.plan["walls"]:
            S = self.plan["segments"][w["seg"]]
            self.frame(S)
            yaw_al = self._yaw_al()
            side = 1 if w["side"] == "L" else -1
            l_post = E["l_rail_in"] + w["rail_t"] + 0.5 * STs * ps
            for i, sp_ in enumerate(w["posts"]):
                pw = self.W(sp_, side * l_post, 0.0)
                self.put(
                    f"fence{S['name']}{w['side']}_post{i}",
                    "slat",
                    ps,
                    yaw_al,
                    pitch=math.pi / 2,
                    axis=tuple(self.n),
                    center=pw[:2],
                    z_bottom=pw[2],
                    contacts=[("down", "flat")],
                )
            for i in range(w["nb"]):
                for k, hz in enumerate(fs["rails_above_deck"]):
                    zc = E["deck_top"] + hz
                    rw = self.W(w["a"] + (i + 0.5) * w["bay"], side * (E["l_rail_in"] + 0.5 * w["rail_t"]), zc)
                    self.put(
                        f"fence{S['name']}{w['side']}_rail{i}_{k}",
                        "slat",
                        w["rsc"],
                        yaw_al,
                        center=rw[:2],
                        z_bottom=zc - 0.5 * SHs * w["rsc"] + GC.FLOOR_Z,
                        contacts=[("lat", side)],
                    )
            out.append(
                dict(
                    seg=S["name"],
                    side=w["side"],
                    s=[round(w["a"], 4), round(w["b"], 4)],
                    bays=w["nb"],
                    bay=round(w["bay"], 4),
                    rail_scale=round(w["rsc"], 4),
                    posts=len(w["posts"]),
                    shrunk_at_corners=w["shrunk"],
                )
            )
        return dict(
            walls=out,
            rail_inner_l=round(E["l_rail_in"], 4),
            lane_width_design=round(2 * E["l_rail_in"], 4),
            rails_above_deck=fs["rails_above_deck"],
        )

    # ---- rough terrain inset
    def _rough(self, m, S, CAT, TB):
        from pxr import UsdGeom, UsdPhysics, Usd, Gf, Vt

        rs = self.spec["rough"]
        E = self.E
        dk = E["deck_top"]
        t = E["t"]
        depth = rs["recess_layers"] * t
        T = CAT.TYPES[m["type"]]
        d_ = m["difficulty"]
        layers = []
        for L_ in T["hf"]:
            if L_["fn"] == "random_uniform":
                hi = CAT.lerp((L_["noise"][0][1], L_["noise"][1][1]), d_)
                layers.append(dict(fn="random_uniform", noise=[0.0, CAT.r3(hi)], step=L_["step"], down=L_["down"]))
            elif L_["fn"] == "wave":
                layers.append(dict(fn="wave", amp=CAT.r3(CAT.lerp(L_["amp"], d_)), waves=L_["waves"]))
        s_a = m["s"][0] + (HEAD_SEAT if S["first"] else 0.0)
        s_b = m["s"][1]
        Lt, Wt = s_b - s_a, E["BL"]
        hs, vs = rs["hscale"], rs["vscale"]
        Zr = _hf_isaaclab(layers, Lt, Wt, hs, vs, m["seed"])
        nx, ny = Zr.shape
        xs = np.linspace(0.0, Lt, nx)
        ys = np.linspace(-Wt / 2, Wt / 2, ny)
        if rs.get("zero_mean", True):
            Zr = Zr - Zr.mean()
        w = (
            _smooth(np.minimum(xs, Lt - xs) / rs["margin_x"])[:, None]
            * _smooth((Wt / 2 - np.abs(ys)) / rs["margin_y"])[None, :]
        )
        Z = np.maximum(Zr * w, -(depth - 0.012))

        def hfun(x, y):  # tile-local bilinear height (relative to the deck top)
            fx = np.clip(np.asarray(x, float) / Lt * (nx - 1), 0, nx - 1.001)
            fy = np.clip((np.asarray(y, float) + Wt / 2) / Wt * (ny - 1), 0, ny - 1.001)
            i, j = fx.astype(int), fy.astype(int)
            u, v = fx - i, fy - j
            return (
                Z[i, j] * (1 - u) * (1 - v)
                + Z[i + 1, j] * u * (1 - v)
                + Z[i, j + 1] * (1 - u) * v
                + Z[i + 1, j + 1] * u * v
            )

        d, n, P0 = S["d"], S["n"], S["P0"]

        def world(x, y, z):  # tile-local -> world (z relative to the deck top)
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            X = P0[0] + d[0] * (s_a + x) + n[0] * y
            Y = P0[1] + d[1] * (s_a + x) + n[1] * y
            return np.stack([X, Y, GC.FLOOR_Z + dk + np.asarray(z, float)], -1)

        # closed slab: top grid + 4 walls + bottom
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        Vt_ = world(XX.ravel(), YY.ravel(), Z.ravel())
        idx = np.arange(nx * ny).reshape(nx, ny)
        F = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                a, b, c_, e = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                F += [[a, b, c_], [a, c_, e]]
        V = [Vt_]
        F = [np.array(F)]
        base = len(Vt_)
        zb = -depth

        def wall(px, py, pz):  # vertical strip from the top edge (px, py, pz) down to zb
            nonlocal base
            m_ = len(px)
            top = world(px, py, pz)
            bot = world(px, py, np.full(m_, zb))
            V.append(np.vstack([top, bot]))
            ff = []
            for k_ in range(m_ - 1):
                a, b = base + k_, base + k_ + 1
                c_, e = base + m_ + k_ + 1, base + m_ + k_
                ff += [[a, e, c_], [a, c_, b]]
            F.append(np.array(ff))
            base += 2 * m_

        wall(xs, np.full(nx, Wt / 2), Z[:, -1])  # +n side (outward +n)
        wall(xs[::-1], np.full(nx, -Wt / 2), Z[::-1, 0])  # -n side
        wall(np.zeros(ny), ys[::-1], Z[0, ::-1])  # start (s_a) side
        wall(np.full(ny, Lt), ys, Z[-1, :])  # end side
        Bv = world(np.array([0, Lt, Lt, 0.0]), np.array([-Wt / 2, -Wt / 2, Wt / 2, Wt / 2]), np.full(4, zb))
        V.append(Bv)
        F.append(np.array([[base, base + 2, base + 1], [base, base + 3, base + 2]]))
        base += 4
        V = np.vstack(V)
        F = np.vstack(F)
        k = len(self.tiles)
        root = f"{self.troot}/rough{k}_{m['type']}"
        UsdGeom.Xform.Define(self.stage, root)
        vis = T["vis"]
        phys = T["phys"]
        mesh = TB.make_mesh(self.stage, root + "/Ground", V, F, CAT.VIS[vis]["uv_m"])
        TB.bind(mesh.GetPrim(), self.mats.visual(vis), self.mats.physics(phys))
        rng = np.random.default_rng(m["seed"])
        look = ROUGH_LOOK[m["type"]]
        rec = dict(
            seg=S["name"],
            type=m["type"],
            visual=vis,
            physics=phys,
            phys_params=CAT.PHYS[phys],
            difficulty=d_,
            seed=m["seed"],
            slots=m["slots"],
            s=[round(s_a, 4), round(s_b, 4)],
            size=[round(Lt, 4), round(Wt, 4)],
            recess_depth=round(depth, 4),
            recess_layers=rs["recess_layers"],
            hf_layers=layers,
            path=root,
            relief_design=dict(
                min=round(float(Z.min()), 4), max=round(float(Z.max()), 4), std=round(float(Z.std()), 4)
            ),
            centre_xy=np.round(world(Lt / 2, 0.0, 0.0)[:2], 3).tolist(),
        )
        rec["Z"], rec["hfun"], rec["frame"] = Z, hfun, (d, n, P0, s_a)
        # dressing
        rocks_bb = []
        self._cur_tile_frame = (d, n, P0, s_a)
        if "rocks" in look:
            rocks_bb = self._rocks(root + "/rocks", rng, look["rocks"], Lt, Wt, hfun, world, self.mats.physics(phys))
        nroots = (
            self._roots(root + "/roots", rng, look["roots"], Lt, Wt, hfun, world, CAT, TB) if "roots" in look else 0
        )
        ndeb = self._debris(root + "/debris", rng, look["debris"], Lt, Wt, hfun, world, TB) if "debris" in look else 0
        rec.update(n_rocks=len(rocks_bb), n_roots=nroots, n_debris=ndeb)
        rec["rocks_bb"] = rocks_bb
        return rec

    def _rocks(self, path, rng, D, Lt, Wt, hfun, world, phys_mat):
        from pxr import UsdGeom, UsdPhysics, Usd, Gf

        CAT, TB = _terrain_mods()
        UsdGeom.Xform.Define(self.stage, path)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        n = int(round(rng.uniform(*D["per_m2"]) * Lt * Wt))
        out = []
        for i in range(n):
            x = rng.uniform(0.08, Lt - 0.08)
            y = rng.uniform(-Wt / 2 + 0.08, Wt / 2 - 0.08)
            s = rng.uniform(*D["scale"])
            zf = rng.uniform(*D["zflat"])
            p = f"{path}/r{i:03d}"
            xf = UsdGeom.Xform.Define(self.stage, p)
            xf.GetPrim().GetReferences().AddReference(TB.ROCKS[int(rng.integers(len(TB.ROCKS)))])
            w = world(x, y, 0.0)
            tr = xf.AddTranslateOp()
            tr.Set(Gf.Vec3d(*map(float, w)))
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rng.uniform(-20, 20, 2)), float(rng.uniform(0, 360))))
            sz = 0.01 * s
            xf.AddScaleOp().Set(Gf.Vec3f(sz, sz, sz * zf))
            for q in Usd.PrimRange(xf.GetPrim()):
                if q.IsInstanceable():
                    q.SetInstanceable(False)
            for q in Usd.PrimRange(xf.GetPrim()):
                if q.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(q)
                    UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr().Set("convexHull")
            TB.bind(xf.GetPrim(), None, phys_mat)
            cache.Clear()
            bb = cache.ComputeWorldBound(xf.GetPrim()).ComputeAlignedRange()
            mn, mx = np.array(bb.GetMin()), np.array(bb.GetMax())
            hgt = mx[2] - mn[2]
            # surface under the rock footprint (tile-local samples) -> sink the rock's lowest point embed x height below it
            d, n_, P0, s_a = self._cur_tile_frame
            xs_ = np.linspace(mn[0], mx[0], 5)
            ys_ = np.linspace(mn[1], mx[1], 5)
            G = np.array([(a_, b_) for a_ in xs_ for b_ in ys_])
            lx = (G - P0) @ d - s_a
            ly = (G - P0) @ n_
            hmin = (
                float(np.min(hfun(np.clip(lx, 0, Lt), np.clip(ly, -Wt / 2, Wt / 2)))) + GC.FLOOR_Z + self.E["deck_top"]
            )
            dz = (hmin - D["embed"] * hgt) - mn[2]
            tr.Set(Gf.Vec3d(float(w[0]), float(w[1]), float(w[2] + dz)))
            hc = float(hfun(x, y)) + GC.FLOOR_Z + self.E["deck_top"]
            out.append(
                dict(
                    path=p,
                    centre=np.round(w[:2], 4).tolist(),
                    bottom=round(mn[2] + dz, 4),
                    top=round(mx[2] + dz, 4),
                    surface_at_centre=round(hc, 4),
                    surface_min_under=round(hmin, 4),
                    height=round(hgt, 4),
                )
            )
        return out

    def _roots(self, path, rng, D, Lt, Wt, hfun, world, CAT, TB):
        from pxr import UsdGeom, Vt

        UsdGeom.Xform.Define(self.stage, path)
        vis, phys = self.mats.visual("bark"), self.mats.physics("firm")
        n = int(rng.integers(D["n"][0], D["n"][1] + 1))
        made = 0
        for i in range(n):
            x0 = rng.uniform(0.15, Lt - 0.15)
            side = -1 if rng.random() < 0.5 else 1
            y0 = side * (Wt / 2 - 0.04)
            ang = math.atan2(-side, 0.0) + rng.uniform(-0.55, 0.55)  # across the lane, from one edge
            r0 = rng.uniform(*D["dia"]) / 2
            P = [np.array([x0, y0])]
            a_ = ang
            while True:
                a_ += rng.normal(0, 0.10)
                q = P[-1] + 0.03 * np.array([math.cos(a_), math.sin(a_)])
                if not (0.04 <= q[0] <= Lt - 0.04 and -Wt / 2 + 0.03 <= q[1] <= Wt / 2 - 0.03) or len(P) > 80:
                    break
                P.append(q)
            if len(P) < 12:
                continue
            P = np.array(P)
            m_ = len(P)
            R = r0 * np.linspace(1.0, 0.35, m_) * (1 + 0.12 * np.sin(np.linspace(0, rng.uniform(6, 14), m_)))
            u = np.linspace(0, 1, m_)
            dive = np.clip(np.minimum(u, 1 - u) / 0.12, 0, 1)  # both ends dive under the surface
            zrel = hfun(P[:, 0], P[:, 1]) + R * (0.35 * dive - 1.2 * (1 - dive))
            P3 = world(P[:, 0], P[:, 1], zrel)
            V, F, UV = TB._tube(P3, R)
            mesh = TB.make_mesh(self.stage, f"{path}/root{i:02d}", V, F, 1.0)
            from pxr import UsdGeom as UG

            UG.PrimvarsAPI(mesh).GetPrimvar("st").Set(
                Vt.Vec2fArray.FromNumpy((UV / CAT.VIS["bark"]["uv_m"]).astype(np.float32))
            )
            TB.bind(mesh.GetPrim(), vis, phys)
            made += 1
        return made

    def _debris(self, path, rng, D, Lt, Wt, hfun, world, TB):
        from pxr import UsdGeom, Gf, Usd

        UsdGeom.Xform.Define(self.stage, path)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        n = int(rng.integers(D["n"][0], D["n"][1] + 1))
        for i in range(n):
            f = TB.DEBRIS[int(rng.integers(len(TB.DEBRIS)))]
            xf = UsdGeom.Xform.Define(self.stage, f"{path}/d{i:02d}")
            inner = UsdGeom.Xform.Define(self.stage, f"{path}/d{i:02d}/o")
            ref = UsdGeom.Xform.Define(self.stage, f"{path}/d{i:02d}/o/a")
            ref.GetPrim().GetReferences().AddReference(f)
            bb = cache.ComputeRelativeBound(ref.GetPrim(), inner.GetPrim()).ComputeAlignedRange()
            ext = bb.GetSize()
            foot = max(ext[0], ext[1], 1e-6)
            sz = rng.uniform(*D["size"])
            s = sz / foot
            c = bb.GetMidpoint()
            x, y = rng.uniform(0.5 * sz, Lt - 0.5 * sz), rng.uniform(-Wt / 2 + 0.5 * sz, Wt / 2 - 0.5 * sz)
            w = world(x, y, float(hfun(x, y)) - 0.003)
            xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, w)))
            xf.AddRotateZOp().Set(float(rng.uniform(0, 360)))
            xf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
            inner.AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -bb.GetMin()[2]))
        return n

    # ---- slippery patch (L5)
    def _slick(self, sl, k, CAT, TB):
        from pxr import UsdGeom

        S = next(s for s in self.plan["segments"] if s["name"] == sl["seg"])
        rng = np.random.default_rng(sl["seed"])
        E = self.E
        dk = E["deck_top"]
        a, b = 0.5 * sl["size"][0], 0.5 * sl["size"][1]
        lo, hi = sl["room_s"][0] - sl["s"], sl["room_s"][1] - sl["s"]
        rl = sl["room_l"]
        blobs = [(0.0, 0.0, a, b, 0.35)]
        for j in range(int(rng.integers(3, 7))):  # satellite drips / splashes
            ang = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(0.85, 1.2)
            blobs.append((rr * a * math.cos(ang), rr * b * math.sin(ang), *(rng.uniform(0.04, 0.12, 2)), 0.5))
        vis = sl["type"]
        phys = sl["type"]
        root = f"{self.troot}/slick{k}_{sl['type']}"
        UsdGeom.Xform.Define(self.stage, root)
        d, n, P0 = S["d"], S["n"], S["P0"]
        parts, area = [], 0.0
        for j, (bx, by, ba, bb, ro) in enumerate(blobs):
            P = TB._blob(rng, ba, bb, n=160, rough=ro)
            for f_ in (7, 11, 17):
                t_ = np.arctan2(P[:, 1], P[:, 0])
                P *= (1 + 0.05 * np.sin(f_ * t_ + rng.uniform(0, 6.3)))[:, None]
            Q = P + np.array([bx, by])
            sc = 1.0  # keep inside the room (deck flat / landing)
            if j == 0:
                sc = min(
                    1.0,
                    0.98 * hi / max(1e-6, Q[:, 0].max()),
                    0.98 * lo / min(-1e-6, Q[:, 0].min()),
                    0.98 * rl / np.abs(Q[:, 1]).max(),
                )
                P = P * sc
                Q = P + np.array([bx, by])
            elif Q[:, 0].max() > hi or Q[:, 0].min() < lo or np.abs(Q[:, 1]).max() > rl:
                continue
            V2, F = TB._fan(P, 0)
            ss = sl["s"] + bx + V2[:, 0]
            ll = sl["l"] + by + V2[:, 1]
            V = np.column_stack(
                [
                    P0[0] + d[0] * ss + n[0] * ll,
                    P0[1] + d[1] * ss + n[1] * ll,
                    np.full(len(V2), GC.FLOOR_Z + dk + 0.0015 + 0.0003 * j),
                ]
            )
            mesh = TB.make_mesh(self.stage, f"{root}/s{j:02d}", V, F, CAT.VIS[vis]["uv_m"], collider=True)
            TB.bind(mesh.GetPrim(), self.mats.visual(vis), self.mats.physics(phys))
            xy = P
            area += 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))
            parts.append(
                dict(
                    path=f"{root}/s{j:02d}",
                    s=[round(float(ss.min()), 3), round(float(ss.max()), 3)],
                    l=[round(float(ll.min()), 3), round(float(ll.max()), 3)],
                )
            )
        main = parts[0]
        return dict(
            type=sl["type"],
            seg=S["name"],
            where=sl["where"],
            k=k,
            path=root,
            n_blobs=len(parts),
            area_m2=round(area, 3),
            s_extent=[min(p["s"][0] for p in parts), max(p["s"][1] for p in parts)],
            l_extent=[min(p["l"][0] for p in parts), max(p["l"][1] for p in parts)],
            main=main,
            parts=parts,
            centre_xy=np.round(P0 + d * sl["s"] + n * sl["l"], 3).tolist(),
            physics=CAT.PHYS[phys],
            visual=vis,
            height_above_deck=0.0015,
        )

    # ------------------------------------------------------------ verification (PhysX rays)
    def floor_clear(self, pxq, carb, step=0.1):
        """before build: down rays over the plan footprint (+0.1 m) must hit the gym floor (nothing in the way)"""
        P = self.plan if hasattr(self, "plan") else plan_layout(self.spec, self.dims)
        E = P["E"]
        bad = {}
        n = 0
        ext = E["l_rail_in"] + 0.06 + 0.1
        for S in P["segments"]:
            s_from = -(self.spec["start_back"] + 0.1) if S["first"] else -0.8
            for s in np.arange(s_from, S["s_end"] + 0.1, step):
                for l in np.arange(-ext, ext + 1e-9, step):
                    xy = S["P0"] + S["d"] * s + S["n"] * l
                    h = pxq.raycast_closest(carb.Float3(float(xy[0]), float(xy[1]), 2.6), carb.Float3(0, 0, -1), 3.0)
                    n += 1
                    if not (h and h.get("hit")):
                        bad.setdefault("NO HIT", 0)
                        bad["NO HIT"] += 1
                        continue
                    c = str(h.get("collision", ""))
                    z = float(h["position"][2])
                    if "SM_floor_element" not in c or abs(z - GC.FLOOR_Z) > 0.01:
                        bad[c] = bad.get(c, 0) + 1
        return dict(rays=n, ok=not bad, offenders=bad)

    def _ray(self, pxq, carb, o, v, Lr, skip=(), keep=None):
        o = np.asarray(o, float)
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        done = 0.0
        if isinstance(skip, str):
            skip = (skip,)
        for _ in range(40):
            h = pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, v)), float(Lr - done))
            if not (h and h.get("hit")):
                return None
            c = str(h.get("collision", ""))
            dist = float(h["distance"])
            if any(c.startswith(s_) for s_ in skip if s_):
                o = o + v * (dist + 1e-4)
                done += dist + 1e-4
                continue
            return dict(d=done + dist, c=c, p=np.asarray(h["position"], float))
        return None

    def verify(self, pxq, carb):
        ray = lambda o, v, Lr, skip=(): self._ray(pxq, carb, o, v, Lr, skip)
        E = self.E
        dk = E["deck_top"]
        beam_skip = tuple(p for b in self.beams for p in b["paths"])
        res = dict(parts=len(self.parts), checks=0, fail=[], max_gap=0.0)
        # (1) contacts of every structure part (v1 recipe, lateral contacts in the part's own segment frame)
        for p in self.parts:
            P, sg, size = p["corners"], p["signs"], p["size"]
            for kind_, mode in p["contacts"]:
                pts, dirs = [], []
                if kind_ == "down":
                    B = self.bottom_face(P, sg, size)
                    cB = B.mean(0)
                    if mode == "flat":
                        pts = [cB + 0.7 * (b - cB) for b in B] + [cB]
                    elif isinstance(mode, tuple):
                        o = np.argsort(B[:, 2])
                        lo2, hi2 = B[o[:2]], B[o[2:]]
                        u, v = mode[1], mode[2]
                        lm, hm_ = lo2.mean(0), hi2.mean(0)
                        ev = hi2[1] - hi2[0]
                        pts = [lo2[0] + 0.25 * (lo2[1] - lo2[0]), lo2[1] + 0.25 * (lo2[0] - lo2[1])]
                        for sv in (-1, 1):
                            pts.append(hm_ + u * (lm - hm_) + sv * v * ev)
                    else:
                        o = np.argsort(B[:, 2])
                        lo2, hi2 = B[o[:2]], B[o[2:]]
                        k_ = 0.25
                        cc = (cB - hi2.mean(0)) / max(1e-9, np.linalg.norm(cB - hi2.mean(0)))
                        pts = [
                            lo2[0] + k_ * (lo2[1] - lo2[0]),
                            lo2[1] + k_ * (lo2[0] - lo2[1]),
                            hi2[0] + k_ * (hi2[1] - hi2[0]) + 0.002 * cc,
                            hi2[1] + k_ * (hi2[0] - hi2[1]) + 0.002 * cc,
                        ]
                    dirs = [np.array([0, 0, -1.0])] * len(pts)
                elif kind_ == "lat":
                    i_long = int(np.argmax(size))
                    n_out = np.r_[p["n"] * mode, 0.0]
                    for e in (-1, 1):
                        F = P[sg[:, i_long] == e]
                        ctr = P.mean(0) + 0.85 * (F.mean(0) - P.mean(0))
                        half_t = 0.5 * float(np.min(size)) * p["scale"]
                        pts.append(ctr + n_out * (half_t - 0.001))
                        dirs.append(n_out)
                elif kind_ == "ends":
                    i_long = int(np.argmax(size))
                    for e in (-1, 1):
                        F = P[sg[:, i_long] == e]
                        c0 = P.mean(0)
                        v = F.mean(0) - c0
                        v = v / np.linalg.norm(v)
                        pts.append(F.mean(0) - v * 0.001)
                        dirs.append(v)
                for q, v in zip(pts, dirs):
                    res["checks"] += 1
                    h = ray(q - v * 0.02, v, 0.08, skip=(p["path"],))
                    gap = None if h is None else h["d"] - 0.02
                    if gap is None or gap > 0.005 or gap < -0.01:
                        res["fail"].append(
                            dict(
                                part=p["role"],
                                mode=f"{kind_}:{mode}",
                                at=np.round(q, 3).tolist(),
                                gap=None if gap is None else round(gap, 4),
                                hit=None if h is None else h["c"][-60:],
                            )
                        )
                    else:
                        res["max_gap"] = max(res["max_gap"], round(gap, 4))
        # (2) part overlaps among axis-aligned fence / deck / terrace parts (corner joints)
        res["overlaps"] = self._overlaps()
        meas = {}
        # (3) walking-surface profiles along every segment (lane centre, +-0.35 m), beam planks skipped
        prof = {}
        for S in self.plan["segments"]:
            s_from = -(self.spec["start_back"]) if S["first"] else -(0.5 * E["BL"] + GAP)
            s_to = S["s_end"] + (0.3 if S["last"] else 0.0)
            for l in (-0.35, 0.0, 0.35):
                rows = []
                for s in np.arange(s_from, s_to, 0.02):
                    h = ray(self.Wl(S, s, l, 3.0), (0, 0, -1), 4.0, skip=beam_skip)
                    rows.append((round(float(s), 3), None if h is None else round(float(h["p"][2] - GC.FLOOR_Z), 4)))
                prof[f"{S['name']}:{l}"] = rows
        res["profile"] = prof
        segs = self.plan["segments"]
        S0, SL_ = segs[0], segs[-1]
        c = np.array([(s, z) for s, z in prof[f"{S0['name']}:0.0"] if z is not None])
        m = (c[:, 0] > 0.12) & (c[:, 0] < E["s_d"] - 0.12)
        meas["ramp_deg"] = round(math.degrees(math.atan(np.polyfit(c[m, 0], c[m, 1], 1)[0])), 2)
        # deck top on the flats of every segment (between modules / corners)
        dz = []
        for S in segs:
            cc = np.array([(s, z) for s, z in prof[f"{S['name']}:0.0"] if z is not None])
            for a, b in S["flats"]:
                mm = (cc[:, 0] > a + 0.1) & (cc[:, 0] < b - 0.1)
                if S["first"] and a < E["s_d"] + 0.05:
                    mm &= cc[:, 0] > E["s_d"] + 0.1
                for sl in self.slicks:  # oil / ice sheets are 1.5 mm above the deck
                    if sl["seg"] == S["name"]:
                        mm &= ~((cc[:, 0] > sl["s_extent"][0] - 0.02) & (cc[:, 0] < sl["s_extent"][1] + 0.02))
                dz += list(cc[mm, 1])
        meas["deck_top_z"] = dict(
            median=round(float(np.median(dz)), 4),
            min=round(float(np.min(dz)), 4),
            max=round(float(np.max(dz)), 4),
            n=len(dz),
        )
        st = self.terrace["starts"]
        tops = []
        cL = np.array([(s, z) for s, z in prof[f"{SL_['name']}:0.0"] if z is not None])
        for k in range(1, len(st)):
            a = st[k] + 0.05
            b = (st[k + 1] - 0.05) if k + 1 < len(st) else st[k] + 0.6
            mm = (cL[:, 0] > a) & (cL[:, 0] < b)
            tops.append(round(float(np.median(cL[mm, 1])), 4))
        meas["tread_tops_z"] = tops
        meas["risers"] = [round(b - a, 4) for a, b in zip([meas["deck_top_z"]["median"]] + tops[:-1], tops)]
        meas["platform_z"] = tops[-1]
        # crossing panels: plane from 3 rays inside each panel -> slope / pitch / roll in the segment frame
        hm = []
        for p in self.parts:
            if "_panel" not in p["role"]:
                continue
            P, sg, size = p["corners"], p["signs"], p["size"]
            B = self.bottom_face(P, sg, size)
            cB = B.mean(0)
            pts = []
            for k_ in range(3):
                q = cB + 0.5 * (B[k_] - cB)
                h = ray(np.r_[q[:2], q[2] + 1.0], (0, 0, -1), 1.5)
                pts.append(h["p"])
            A = np.array(pts)
            nrm = np.cross(A[1] - A[0], A[2] - A[0])
            nrm = nrm / np.linalg.norm(nrm)
            if nrm[2] < 0:
                nrm = -nrm
            g = -nrm[:2] / nrm[2]
            hm.append(
                dict(
                    part=p["role"],
                    slope_deg=round(math.degrees(math.acos(nrm[2])), 2),
                    pitch_deg=round(math.degrees(math.atan(abs(g @ p["d"]))), 2),
                    roll_deg=round(math.degrees(math.atan(abs(g @ p["n"]))), 2),
                )
            )
        meas["crossing_panels"] = hm
        xh = []
        for el in self.crossings:
            S = next(s for s in segs if s["name"] == el["seg"])
            if el["type"] == "hump":
                h = ray(self.Wl(S, el["c"], 0.0, 3.0), (0, 0, -1), 4.0)
                xh.append(
                    dict(name=el["name"], seg=el["seg"], top_above_deck=round(float(h["p"][2] - GC.FLOOR_Z) - dk, 4))
                )
            else:
                hc = ray(self.Wl(S, el["c"], 0.0, 3.0), (0, 0, -1), 4.0)
                ho = ray(self.Wl(S, el["c"], 0.55, 3.0), (0, 0, -1), 4.0)
                xh.append(
                    dict(
                        name=el["name"],
                        seg=el["seg"],
                        centre_above_deck=round(float(hc["p"][2] - GC.FLOOR_Z) - dk, 4),
                        at_l055_above_deck=round(float(ho["p"][2] - GC.FLOOR_Z) - dk, 4),
                    )
                )
        meas["crossing_heights"] = xh
        # (4) beam: pillar tops touch the planks; underside above the walking surface
        bm = []
        for b in self.beams:
            S = next(s for s in segs if s["name"] == b["seg"])
            self.frame(S)
            pre = b["paths"]
            gaps = {}
            for pl in b["pillars"]:
                p = next(q for q in self.parts if q["role"] == pl["role"])
                P = p["corners"]
                top = P[:, 2].max()
                cxy = P[:, :2].mean(0)
                cP = P[P[:, 2] > top - 1e-6][:, :2]
                half_al = 0.5 * float(np.ptp(cP @ self.d))
                half_t = 0.5 * float(np.ptp(cP @ self.n))
                gp = []
                for dl in np.linspace(-0.9, 0.9, 13):
                    for dt in (-0.6, 0.0, 0.6):
                        q = cxy + dl * half_al * self.d + dt * half_t * self.n
                        h = ray(np.r_[q, top - 0.02], (0, 0, 1), 0.1, skip=(p["path"],))
                        if h is not None and any(h["c"].startswith(x) for x in pre):
                            gp.append(h["d"] - 0.02)
                res["checks"] += 1
                ok = len(gp) == 39 and -0.001 <= min(gp) and max(gp) <= 0.003
                if not ok:
                    res["fail"].append(
                        dict(
                            part=pl["role"], mode="pillar->plank", n=len(gp), min_gap=round(min(gp), 4) if gp else None
                        )
                    )
                gaps[pl["role"]] = dict(
                    n=len(gp), min=round(min(gp), 4) if gp else None, max=round(max(gp), 4) if gp else None
                )
            s0, s1 = b["s"]
            U = []
            for s in np.arange(s0 - 0.05, s1 + 0.05, 0.01):
                for l in np.arange(-0.65, 0.651, 0.05):
                    dn = ray(self.W(s, l, dk + 0.3), (0, 0, -1), 0.6)
                    if dn is None:
                        continue
                    zw = float(dn["p"][2])
                    up = ray(np.r_[dn["p"][:2], zw + 0.05], (0, 0, 1), 2.0)
                    if up is not None and any(up["c"].startswith(x) for x in pre):
                        U.append(float(up["p"][2]) - zw)
            U = np.array(U)
            bm.append(
                dict(
                    seg=b["seg"],
                    n_rays=len(U),
                    underside_above_walk_min=round(float(U.min()), 4),
                    median=round(float(np.median(U)), 4),
                    max=round(float(U.max()), 4),
                    pillar_contact=gaps,
                )
            )
        meas["beam"] = bm
        # (5) funnel: narrowest gap (lateral rays every 5 mm along the lane, 0.03-1.38 m above the deck); wings on posts
        fu = []
        for f in self.funnels:
            S = next(s for s in segs if s["name"] == f["seg"])
            self.frame(S)
            a_, b_ = f["s"]
            best = None
            for s in np.unique(
                np.r_[
                    np.arange(a_ - 0.1, b_ + 0.1, 0.005), np.arange(f["s_throat"] - 0.01, f["s_throat"] + 0.01, 0.0005)
                ]
            ):
                for hz in np.arange(0.03, 1.40, 0.05):
                    o = self.W(s, 0.0, dk + hz)
                    hp = ray(o, np.r_[self.n, 0], 1.5)
                    hm_ = ray(o, np.r_[-self.n, 0], 1.5)
                    if hp and hm_:
                        wdt = hp["d"] + hm_["d"]
                        if best is None or wdt < best[0]:
                            best = (wdt, float(s), hz)
            wg = []
            for fp in f["posts"]:
                p = next(q for q in self.parts if q["role"] == fp["role"])
                nb = np.array(fp["back_face_dir"])
                dirw = -np.r_[self.d * nb[0] + self.n * nb[1], 0.0]
                wing = next(q for q in self.parts if q["role"] == fp["wing"])
                for hz in (0.25, 0.65, 1.05):
                    q = self.W(fp["s"], fp["l"], dk + hz)
                    h = ray(q, dirw, 0.1, skip=(p["path"],))
                    off = 0.5 * self.dims["slat"][2] * self.spec["funnel"]["post_scale"]
                    gap = None if h is None else h["d"] - off
                    res["checks"] += 1
                    if h is None or not h["c"].startswith(wing["path"]) or not (-0.01 <= gap <= 0.005):
                        res["fail"].append(
                            dict(part=fp["role"], mode="post->wing", gap=None if gap is None else round(gap, 4))
                        )
                    else:
                        wg.append(gap)
            fu.append(
                dict(
                    seg=f["seg"],
                    gap_min=round(best[0], 4),
                    gap_s=round(best[1], 4),
                    gap_height_above_deck=round(best[2], 3),
                    wing_post_max_gap=round(max(wg), 4) if wg else None,
                )
            )
        meas["funnel"] = fu
        # (6) lane width between the rails (mid-rail height) at every bay centre of every segment, hurdles excluded
        widths, wc = [], []
        busy = lambda S, s: any(
            m["s"][0] - 0.1 <= s <= m["s"][1] + 0.1 for m in S["modules"] if m["kind"] in ("funnel", "beam")
        )
        for w in self.plan["walls"]:
            if w["side"] != "L":
                continue
            S = segs[w["seg"]]
            for i in range(w["nb"]):
                s = w["a"] + (i + 0.5) * w["bay"]
                if busy(S, s) or s < 0.0 or s > S["s_end"]:
                    continue
                o = self.Wl(S, s, 0.0, dk + 0.58)
                hp = ray(o, np.r_[S["n"], 0], 1.5)
                hm_ = ray(o, np.r_[-S["n"], 0], 1.5)
                if hp and hm_:
                    widths.append(round(hp["d"] + hm_["d"], 4))
        meas["lane_width"] = dict(min=min(widths), max=max(widths), n=len(widths), note="beam / funnel bays skipped")
        # corners: from the corner point along the incoming direction and outwards -> rails at l_rail_in
        cw = []
        for a_s, b_s in zip(segs[:-1], segs[1:]):
            C = a_s["C"]
            o = np.r_[C, GC.FLOOR_Z + dk + 0.58]
            outer_in = -a_s["n"] * np.sign(a_s["turn"])  # outer side of the incoming lane
            hd = ray(o, np.r_[a_s["d"], 0], 2.0)
            ho = ray(o, np.r_[outer_in, 0], 2.0)
            cw.append(
                dict(
                    corner=a_s["name"] + "->" + b_s["name"],
                    to_far_rail=round(hd["d"], 4) if hd else None,
                    to_outer_rail=round(ho["d"], 4) if ho else None,
                )
            )
        meas["corners"] = cw
        # fence top rail above the walking surface
        fh = []
        for p in self.parts:
            if not (p["role"].startswith("fence") and "_rail" in p["role"] and p["role"].endswith("_2")):
                continue
            cP = p["corners"].mean(0)
            h = ray(np.r_[cP[:2], cP[2] + 1.0], (0, 0, -1), 1.2, skip=beam_skip)
            if h is None or not h["c"].startswith(p["path"]):
                continue
            S = next(s for s in segs if s["name"] == p["seg"])
            s_ = float((cP[:2] - S["P0"]) @ S["d"])
            g = ray(self.Wl(S, s_, 0.0, 3.0), (0, 0, -1), 4.0, skip=beam_skip)
            fh.append(round(float(h["p"][2] - g["p"][2]), 4))
        meas["fence_top_rail_above_walk"] = dict(min=min(fh), max=max(fh), n=len(fh)) if fh else None
        # (7) rough insets: flush edges, relief, support, rocks embedded
        rr = []
        for T in self.tiles:
            d, n, P0, s_a = T["frame"]
            Lt, Wt = T["size"]
            wl = lambda x, y, z: np.r_[P0 + d * (s_a + x) + n * y, GC.FLOOR_Z + z]
            fl = []
            for x_in, y_in, x_out, y_out in [(0.01, 0.0, -0.03, 0.0), (Lt - 0.01, 0.0, Lt + 0.03, 0.0)]:
                hi_ = ray(wl(x_in, y_in, 3.0), (0, 0, -1), 4.0)
                ho_ = ray(wl(x_out, y_out, 3.0), (0, 0, -1), 4.0)
                fl.append(round(float(hi_["p"][2] - ho_["p"][2]), 4))
            zs, hits = [], {}
            for x in np.arange(0.02, Lt - 0.01, 0.04):
                for y in np.arange(-Wt / 2 + 0.03, Wt / 2 - 0.02, 0.04):
                    h = ray(wl(x, y, 3.0), (0, 0, -1), 4.0)
                    if h is not None:
                        zs.append(float(h["p"][2] - GC.FLOOR_Z) - dk)
                        key = (
                            "tile"
                            if h["c"].startswith(T["path"] + "/Ground")
                            else ("rock" if "/rocks/" in h["c"] else ("root" if "/roots/" in h["c"] else h["c"][-40:]))
                        )
                        hits[key] = hits.get(key, 0) + 1
            zs = np.array(zs)
            sup = []
            for x in np.linspace(0.05, Lt - 0.05, 5):
                for y in np.linspace(-Wt / 2 + 0.05, Wt / 2 - 0.05, 4):
                    h = ray(wl(x, y, dk - T["recess_depth"] + 0.02), (0, 0, -1), 0.08, skip=(T["path"],))
                    res["checks"] += 1
                    gap = None if h is None else h["d"] - 0.02
                    if gap is None or not (-0.003 <= gap <= 0.003):
                        res["fail"].append(
                            dict(
                                part=T["path"], mode="tile bottom on boards", gap=None if gap is None else round(gap, 4)
                            )
                        )
                    else:
                        sup.append(gap)
            emb = [r["bottom"] - r["surface_min_under"] for r in T["rocks_bb"]]
            prot = [r["top"] - r["surface_at_centre"] for r in T["rocks_bb"]]
            for r in T["rocks_bb"]:
                res["checks"] += 1
                if r["bottom"] > r["surface_min_under"] - 0.002:
                    res["fail"].append(dict(part=r["path"], mode="rock not embedded"))
            rr.append(
                dict(
                    type=T["type"],
                    seg=T["seg"],
                    s=T["s"],
                    size=T["size"],
                    edge_step_vs_deck=fl,
                    walk_relief_above_deck=dict(
                        min=round(float(zs.min()), 4),
                        max=round(float(zs.max()), 4),
                        std=round(float(zs.std()), 4),
                        p05=round(float(np.percentile(zs, 5)), 4),
                        p95=round(float(np.percentile(zs, 95)), 4),
                        n=len(zs),
                    ),
                    hit_counts=hits,
                    bottom_support_max_gap=round(max(sup), 4) if sup else None,
                    rocks=dict(
                        n=len(emb),
                        sink_below_lowest_surface_min=round(-max(emb), 4) if emb else None,
                        protrusion_max=round(max(prot), 4) if prot else None,
                    ),
                )
            )
        meas["rough"] = rr
        # (8) slippery patches: the sheet is hit first, 1.5 mm above the deck, and the deck right under it
        sk = []
        for sl in self.slicks:
            S = next(s for s in segs if s["name"] == sl["seg"])
            main = sl["main"]
            ok_n, gaps = 0, []
            for s in np.linspace(main["s"][0] + 0.08, main["s"][1] - 0.08, 6):
                for l in np.linspace(main["l"][0] + 0.06, main["l"][1] - 0.06, 5):
                    h = ray(self.Wl(S, s, l, 3.0), (0, 0, -1), 4.0)
                    if h is None or not h["c"].startswith(sl["path"]):
                        continue
                    h2 = ray(h["p"] + np.r_[0, 0, 0.01], (0, 0, -1), 0.1, skip=(sl["path"],))
                    res["checks"] += 1
                    gap = None if h2 is None else h2["d"] - 0.01
                    if gap is None or not (0.0 <= gap <= 0.004):
                        res["fail"].append(
                            dict(part=sl["path"], mode="sheet on deck", gap=None if gap is None else round(gap, 4))
                        )
                    else:
                        gaps.append(gap)
                        ok_n += 1
            from pxr import UsdShade

            pm = UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(main["path"])).ComputeBoundMaterial("physics")[0]
            sk.append(
                dict(
                    type=sl["type"],
                    seg=sl["seg"],
                    where=sl["where"],
                    n_rays_on_sheet=ok_n,
                    sheet_above_deck=dict(min=round(min(gaps), 4), max=round(max(gaps), 4)) if gaps else None,
                    physics_material=str(pm.GetPath()) if pm else None,
                    area_m2=sl["area_m2"],
                )
            )
        meas["slippery"] = sk
        res["measured"] = meas
        res["ok"] = not res["fail"] and not res["overlaps"]
        return res

    def _overlaps(self, tol=0.003):
        """axis-aligned structure parts (deck / terrace / fence / handrail posts) must not interpenetrate (> tol on
        all three axes). Tilted parts (ramp, hump / valley panels, sloped handrails, funnel wings) are skipped."""
        boxes = []
        for p in self.parts:
            P = p["corners"]
            ext = np.ptp(P, axis=0)
            sz = np.sort(np.asarray(p["size"]) * p["scale"])
            if abs(np.sort(ext)[0] - sz[0]) > 0.003 or abs(np.sort(ext)[2] - sz[2]) > 0.003:
                continue  # not axis-aligned
            boxes.append((p["role"], P.min(0), P.max(0)))
        out = []
        for i in range(len(boxes)):
            ri, ai, bi = boxes[i]
            for j in range(i + 1, len(boxes)):
                rj, aj, bj = boxes[j]
                ov = np.minimum(bi, bj) - np.maximum(ai, aj)
                if (ov > tol).all():
                    out.append(dict(a=ri, b=rj, overlap=np.round(ov, 4).tolist()))
        return out[:50]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sp = make_level_spec(a.level, a.seed)
    if a.level <= 2:
        print(json.dumps(sp, indent=1, default=jsonable))
    else:
        print(json.dumps(plan_summary(plan_layout(sp)), indent=1, default=jsonable))
