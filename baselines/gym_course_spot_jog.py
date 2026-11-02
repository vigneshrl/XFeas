#!/usr/bin/env python
"""
SPOT GYM COURSE L1 WITH A 45 DEG JOG (experiments/4_obstacle_course/spot/L1, additive 2026-10-22).
gym_course_spot.py / gym_course_levels.py are NOT modified: SpotJogCourse subclasses SpotLevelCourse and reuses its
element code (_cross for humps, the valley recipe, _terrace, the ramp recipe, put / frame / contacts); only the lane
layout (3 lane frames A 0 deg -> J 45 deg -> C 0 deg), the deck, the valley halves and the fence are new.

Course (old L1 = new L0 + a jog in the middle of the crossing section):
  start on the gym floor -> 6.8 deg ramp -> deck -> hump 1 -> VALLEY 1 = a bent cross-slope trough:
      A-valley (0 deg) | B1: turn LEFT 45 | J-valley (45 deg, centre line B1->B2 = 1.5 m) | B2: turn RIGHT 45 | C-valley (0 deg)
  -> hump 2 -> valley 2 -> stairs (4 risers) -> platform (goal). After the jog the lane, stairs, platform and goal are
  shifted +y by 1.5 sin45 = 1.061 m.
  Both turns are INSIDE the trough: the bend points B1, B2 lie on the lane centre line between valley panels of the two
  lane frames, mitred (the panels of each frame stop at the bisector -- mitre -- line of the bend, 2D footprints
  disjoint). Rectangular boards cannot fill a mitre exactly, so each bend leaves deck-level pockets on the deck: on the
  inner side a sliver along the centre line, on the outer side a 45 deg wedge (both on the deck, 0 m above it).
  Deck under the jog: deck slots keep the 0 deg orientation (desk tops across x, the global slot grid of the straight
  lane); slots the jog crosses are a PAD of two boards end to end in y (y 1.86 .. 4.50), so the deck under the bends has
  no seams other than the 2 mm board gaps. Fence posts on the pad stand on the deck (post scale 0.90 -> same top height
  as the floor posts at 1.00).
  Fence: straight runs as v1 (bays 0.85-1.15 slat length, three rails on the inner faces of posts); the diagonal runs
  are 1.5 m between the mitre corners: one rail bay at the max scale (1.15 -> 1.299 m) + an in-line filler post (a
  standing slat) at the concave corner. At concave corners rails meet on the inner-face mitre point; at reflex corners
  the diagonal rail is set back just enough not to intersect the straight rail. Lane width >= 1.344 m (design) everywhere,
  1.344 / cos 22.5 = 1.455 m along the mitre lines.

  python baselines/gym_course_spot_jog.py           # print the plan (no Isaac)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, copy, json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(THIS)
import gym_course_spot as GS  # sets G.RL = 4 (Spot riser) and the Spot element dims
import gym_course_levels as G
import gym_course as GC

JOG = dict(L=1.5, turn_deg=45.0, mitre_margin=None, pad_ylo=1.86, stairs_x=-12.696, cross_gap=0.15)
CLR = 0.002  # clearance between fence parts that meet (rail ends, filler, posts)
POST_DECK_SCALE = 0.90  # posts standing on the deck: 0.113 + 0.9 x 1.129 = floor post top (1.129)


def _rot(h):
    return np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])


def frame(name, P0, heading_deg):
    h = math.radians(heading_deg)
    d, n = _rot(h)
    return dict(name=name, P0=np.asarray(P0, float), d=d, n=n, yaw=h, heading=heading_deg)


def W2(F, s, l):
    return F["P0"] + F["d"] * s + F["n"] * l


def to_sl(F, xy):
    v = np.asarray(xy, float) - F["P0"]
    return float(v @ F["d"]), float(v @ F["n"])


def rect(F, s0, s1, l0, l1):
    return np.array([W2(F, s0, l0), W2(F, s1, l0), W2(F, s1, l1), W2(F, s0, l1)])


def sat_sep(A, B):
    """signed separation of two convex polygons (>0 apart, <0 penetration depth)"""
    best = -1e9
    for P, Q in ((A, B), (B, A)):
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]
            nrm = np.array([e[1], -e[0]])
            nrm /= np.linalg.norm(nrm)
            if (P.mean(0) - P[i]) @ nrm > 0:
                nrm = -nrm
            best = max(best, float(np.min((Q - P[i]) @ nrm)))
    return best


def hull2d(P):
    """convex hull (Andrew monotone chain) of 2D points -> CCW polygon"""
    P = sorted(map(tuple, np.round(np.asarray(P, float), 9)))
    if len(P) < 3:
        return np.array(P)
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def line_x(F1, l1, F2, l2):
    """intersection of the offset line l1 of frame F1 and offset line l2 of frame F2 -> xy"""
    p1, d1 = W2(F1, 0.0, l1), F1["d"]
    p2, d2 = W2(F2, 0.0, l2), F2["d"]
    A = np.c_[d1, -d2]
    t = np.linalg.solve(A, p2 - p1)
    return p1 + d1 * t[0]


# ------------------------------------------------------------------ plan (pure python)
def plan_jog(spec, dims=None, jog=None):
    jog = dict(JOG, **(jog or {}))
    dims = dims or G.NOMINAL
    E = G.element_dims(dims, spec)
    BL, BD, Pp, t = E["BL"], E["BD"], E["P"], E["t"]
    SL, SH, ST = E["SL"], E["SH"], E["ST"]
    li = E["l_rail_in"]
    seg0 = spec["segments"][0]
    A = frame("A", seg0["P0"], 0.0)
    s0 = E["s_d"]
    hum = G.cross_el("hump", 0.0, dims, spec)
    val = G.cross_el("valley", 0.0, dims, spec)
    Hlen, Vlen = hum["extent"][1], val["extent"][1]
    lat_v = val["lat"]  # valley panel footprint half-width (incl. thickness)
    tq = math.tan(math.radians(jog["turn_deg"] / 2))
    pull = lat_v * tq  # mitre pull-back of a full-width panel half
    L = jog["L"]
    # mitre margin: the slack of the diagonal valley split evenly over its 4 panel ends
    need = Vlen + pull + 2 * CLR
    if L < need:
        raise ValueError(f"diagonal {L} < valley need {need:.3f}")
    m = jog["mitre_margin"] if jog["mitre_margin"] is not None else 0.5 * (L - Vlen - pull)
    gap = jog["cross_gap"]
    c45 = math.cos(math.radians(jog["turn_deg"]))
    # x budget: flats on both sides of the crossing equal; stairs foot fixed (old L1 x)
    x_st = jog["stairs_x"]
    s_st_A = x_st - A["P0"][0]
    span = Hlen + gap + (Vlen + pull + m) + L * c45 + (Vlen + pull + m) + gap + Hlen + gap + Vlen
    flat = 0.5 * ((s_st_A - s0) - span)
    if flat < 0.15:
        raise ValueError(f"flat {flat:.3f} < 0.15")
    s_h1 = s0 + flat
    sB1 = s_h1 + Hlen + gap + Vlen + pull + m
    B1 = W2(A, sB1, 0.0)
    J = frame("J", B1, jog["turn_deg"])
    B2 = W2(J, L, 0.0)
    C = frame("C", B2, 0.0)
    # valley halves: (frame, side, s_start) ; side +1 = left (+n), -1 = right
    halves = []
    #   A-valley ends at B1 (left turn: inner = left -> pulled back by `pull`)
    halves += [dict(seg="A", side=+1, s=sB1 - pull - m - Vlen), dict(seg="A", side=-1, s=sB1 - m - Vlen)]
    #   J-valley: starts after B1 (inner left pulled), ends before B2 (right turn: inner = right pulled)
    halves += [dict(seg="J", side=+1, s=pull + m), dict(seg="J", side=-1, s=m)]
    #   C-valley starts after B2 (inner = right pulled)
    halves += [dict(seg="C", side=+1, s=m), dict(seg="C", side=-1, s=pull + m)]
    sC_end = pull + m + Vlen
    s_h2 = sC_end + gap
    s_v2 = s_h2 + Hlen + gap
    s_st_C = x_st - C["P0"][0]
    # terrace in the C frame (deck = slot grid of the straight lane, s0 in C coordinates)
    s0_C = (A["P0"][0] + s0) - C["P0"][0]
    T = G._terrace(spec, E, s0_C, s_st_C)
    n_slots = T["cnt"][0]
    # ---- fence lines: corners (inner rail faces), runs per side
    Fr = dict(A=A, J=J, C=C)
    cor = {}
    for side in (+1, -1):
        cor[("B1", side)] = line_x(A, side * li, J, side * li)
        cor[("B2", side)] = line_x(J, side * li, C, side * li)
    # reflex corner = the inner side of a turn: B1 left (+1), B2 right (-1)
    reflex = {("B1", +1): True, ("B1", -1): False, ("B2", +1): False, ("B2", -1): True}
    runs = []
    for side in (+1, -1):
        a_A = 0.0
        b_A = to_sl(A, cor[("B1", side)])[0]
        a_J = to_sl(J, cor[("B1", side)])[0]
        b_J = to_sl(J, cor[("B2", side)])[0]
        a_C = to_sl(C, cor[("B2", side)])[0]
        b_C = s_st_C
        runs.append(
            dict(
                seg="A", side=side, a=a_A, b=b_A, a_kind="start", b_kind="reflex" if reflex[("B1", side)] else "concave"
            )
        )
        runs.append(
            dict(
                seg="J",
                side=side,
                a=a_J,
                b=b_J,
                a_kind="reflex" if reflex[("B1", side)] else "concave",
                b_kind="reflex" if reflex[("B2", side)] else "concave",
            )
        )
        runs.append(
            dict(
                seg="C",
                side=side,
                a=a_C,
                b=b_C,
                a_kind="reflex" if reflex[("B2", side)] else "concave",
                b_kind="stairs",
            )
        )
    # ---- deck slots: type A / pad / C (lane interior between the rail faces must be covered)
    ylo = jog["pad_ylo"]
    yA, yC = A["P0"][1], C["P0"][1]
    pad_b = [(ylo + 0.5 * BL), (ylo + BL + 0.002 + 0.5 * BL)]
    lane_polys = []
    for F, a, b in ((A, 0.0, sB1 + 1.0), (J, -1.0, L + 1.0), (C, -1.0, s_st_C)):
        lane_polys.append((F, a, b))

    def lane_y_range(x0, x1):
        """y range of the lane interior (between rail inner faces) over x in [x0, x1] (sampled)"""
        ys = []
        for x in np.linspace(x0, x1, 41):
            # A part valid for x <= corner of each side; build the two fence polylines as functions of x
            for side in (+1, -1):
                ys.append(fence_y(x, side))
        return min(ys), max(ys)

    def fence_y(x, side):
        c1, c2 = cor[("B1", side)], cor[("B2", side)]
        if x <= c1[0]:
            return yA + side * li
        if x >= c2[0]:
            return yC + side * li
        return c1[1] + (x - c1[0]) * (c2[1] - c1[1]) / (c2[0] - c1[0])

    slots = []
    for i in range(n_slots):
        x0 = A["P0"][0] + s0 + i * Pp
        x1 = x0 + BD
        ylo_, yhi_ = lane_y_range(x0, x1)
        if yhi_ <= yA + li + 1e-6 and ylo_ >= yA - li - 1e-6:
            typ, ys = "A", [yA]
        elif ylo_ >= yC - li - 1e-6 and yhi_ <= yC + li + 1e-6:
            typ, ys = "C", [yC]
        else:
            typ, ys = "P", pad_b
        slots.append(dict(i=i, x=[x0, x1], s=s0 + i * Pp, type=typ, y_centres=ys, lane_y=[ylo_, yhi_]))
    pad = [s_ for s_ in slots if s_["type"] == "P"]
    if pad:
        k0, k1 = pad[0]["i"], pad[-1]["i"]
        for s_ in slots[k0 : k1 + 1]:
            s_["type"], s_["y_centres"] = "P", pad_b
    boards = []  # top-layer footprints (x0, x1, y0, y1)
    for s_ in slots:
        for yc in s_["y_centres"]:
            boards.append((s_["x"][0], s_["x"][1], yc - 0.5 * BL, yc + 0.5 * BL))
    # ---- path, turns
    path = [
        dict(p=W2(A, -spec["start_back"], 0.0), z=0.0, what="start"),
        dict(p=A["P0"].copy(), z=0.0, what="ramp foot"),
        dict(p=W2(A, s0, 0.0), z=E["deck_top"], what="deck"),
        dict(p=B1.copy(), z=E["deck_top"], what="turn 1: left 45 deg (in the valley-1 trough)"),
        dict(p=B2.copy(), z=E["deck_top"], what="turn 2: right 45 deg (in the valley-1 trough)"),
        dict(p=W2(C, s_st_C, 0.0), z=E["deck_top"], what="stairs foot"),
    ]
    for j in range(2, len(T["starts"])):
        path.append(dict(p=W2(C, T["starts"][j], 0.0), z=T["tops"][j - 1], what=f"tread {j - 1}"))
    sgoal = T["s_p0"] + 0.5 * spec["platform_len"]
    path.append(dict(p=W2(C, sgoal, 0.0), z=T["tops"][-1], what="goal"))
    plen = sum(float(np.linalg.norm(b["p"] - a["p"])) for a, b in zip(path[:-1], path[1:]))
    return dict(
        E=E,
        jog=jog,
        frames=Fr,
        A=A,
        J=J,
        C=C,
        B1=B1,
        B2=B2,
        L=L,
        pull=pull,
        mitre_margin=m,
        flat=flat,
        s_h1=s_h1,
        sB1=sB1,
        halves=halves,
        s_h2=s_h2,
        s_v2=s_v2,
        s_st_C=s_st_C,
        s0=s0,
        s0_C=s0_C,
        terrace=T,
        n_slots=n_slots,
        slots=slots,
        boards=boards,
        corners=cor,
        reflex=reflex,
        runs=runs,
        path=path,
        path_len_xy=plen,
        lateral_offset=float(B2[1] - B1[1]),
        fence_y=fence_y,
    )


def on_deck(boards, poly, margin=0.003):
    """'on' if the polygon lies inside one top board (margin), 'off' if it is clear of every board (margin), else 'straddle'"""
    inside_any, touch = False, False
    xs, ys = poly[:, 0], poly[:, 1]
    for x0, x1, y0, y1 in boards:
        R = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        sep = sat_sep(R, poly)
        if sep < margin:
            touch = True
        if xs.min() >= x0 + margin and xs.max() <= x1 - margin and ys.min() >= y0 + margin and ys.max() <= y1 - margin:
            inside_any = True
    if inside_any:
        return "on"
    return "straddle" if touch else "off"


def deck_state(boards, poly, pts, margin=0.003):
    """posts / slats resting on the deck: every contact point must be inside a board (not a seam); 'off' = clear"""
    st = on_deck(boards, poly, margin)
    if st == "off":
        return "off"
    ok = all(
        any(x0 + margin <= p[0] <= x1 - margin and y0 + margin <= p[1] <= y1 - margin for (x0, x1, y0, y1) in boards)
        for p in pts
    )
    return "on" if ok else "straddle"


def plan_fence(P, spec, rail_t_of=lambda rsc: None):
    """rails / posts / fillers of every run (pure python, nominal slat dims in E) -> list of items + polygons"""
    E = P["E"]
    SL, SH, ST = E["SL"], E["SH"], E["ST"]
    li = E["l_rail_in"]
    Fr = P["frames"]
    boards = P["boards"]
    ps0 = spec["fence"]["post_scale"]
    items = []

    def post_poly(F, side, c, ps, t_rail):
        l0 = li + t_rail
        return rect(F, c - 0.5 * SH * ps, c + 0.5 * SH * ps, side * l0, side * (l0 + ST * ps))

    def post_pts(F, side, c, ps, t_rail):
        Pq = post_poly(F, side, c, ps, t_rail)
        cc = Pq.mean(0)
        return [cc + 0.7 * (q - cc) for q in Pq] + [cc]

    def support(F, side, c, t_rail):
        """post at s = c: scale + z (deck or floor) or None (straddles a deck edge / seam)"""
        for ps, z in ((ps0, 0.0), (POST_DECK_SCALE, E["deck_top"])):
            st = deck_state(boards, post_poly(F, side, c, ps, t_rail), post_pts(F, side, c, ps, t_rail))
            if z == 0.0 and st == "off":
                return ps, z
            if z > 0 and st == "on":
                return ps, z
        return None

    for R in P["runs"]:
        F = Fr[R["seg"]]
        side = R["side"]
        a, b = R["a"], R["b"]
        if R["seg"] == "J":
            continue
        # straight runs: reflex ends are where the J rail sets back (handled below); rails reach the corner point
        Lw = b - a
        best = None
        nbs = range(max(1, int(Lw / (1.15 * SL))), int(Lw / (0.85 * SL)) + 2)
        for nb in sorted(nbs, key=lambda k: abs(Lw / k - SL)):  # bays closest to the slat length first
            bay = Lw / nb
            if not (0.85 * SL - 1e-9 <= bay <= 1.15 * SL + 1e-9):
                continue
            for jit in np.r_[0.0, np.ravel([[k, -k] for k in np.arange(0.01, 0.16, 0.01)])]:
                joints = [a + i * bay for i in range(nb + 1)]
                # jitter interior joints by a common shift, bays stay within the band
                J_ = [joints[0]] + [x + jit for x in joints[1:-1]] + [joints[-1]]
                bays = np.diff(J_)
                if bays.min() < 0.85 * SL - 1e-9 or bays.max() > 1.15 * SL + 1e-9:
                    continue
                rsc = [float(bb / SL) for bb in bays]
                # post centres: start / stairs ends centred on the rail end, corner ends inside the rail span
                cs = []
                for k_, x in enumerate(J_):
                    if k_ == 0:
                        cs.append(x if R["a_kind"] == "start" else x + 0.5 * SH * POST_DECK_SCALE + CLR)
                    elif k_ == len(J_) - 1:
                        cs.append(x if R["b_kind"] == "stairs" else x - 0.5 * SH * POST_DECK_SCALE - CLR)
                    else:
                        cs.append(x)
                sup = []
                for k_, c in enumerate(cs):
                    tr = ST * max(rsc[max(0, k_ - 1)], rsc[min(k_, len(rsc) - 1)])
                    corner_end = (k_ == 0 and R["a_kind"] != "start") or (k_ == len(cs) - 1 and R["b_kind"] != "stairs")
                    if not corner_end:
                        sup.append(support(F, side, c, tr))
                        continue
                    # corner-end post: inside the rail span; slide it inwards (the rail end ray at 0.15 x half bay
                    # from the end must stay on the post)
                    sgn = 1 if k_ == 0 else -1
                    end = J_[k_]
                    ray_off = 0.15 * 0.5 * bays[0 if k_ == 0 else -1]
                    got = None
                    offs = np.linspace(
                        0.5 * SH * POST_DECK_SCALE + CLR, ray_off + 0.5 * SH * POST_DECK_SCALE - 0.012, 10
                    )
                    kind_end = R["a_kind"] if k_ == 0 else R["b_kind"]
                    if kind_end == "reflex":
                        offs = offs[::-1]  # reflex corner: as far from the corner as allowed
                    for off in offs:
                        sp_ = support(F, side, end + sgn * off, tr)
                        if sp_ is not None:
                            got = sp_
                            cs[k_] = end + sgn * off
                            break
                    sup.append(got)
                if all(s_ is not None for s_ in sup):
                    best = (nb, J_, rsc, cs, sup)
                    break
            if best:
                break
        if best is None:
            raise RuntimeError(f"no post layout for run {R['seg']}{'LR'[side < 0]}")
        nb, J_, rsc, cs, sup = best
        for k_ in range(nb):
            items.append(
                dict(kind="rail", seg=R["seg"], side=side, a=J_[k_], b=J_[k_ + 1], rsc=rsc[k_], t=ST * rsc[k_])
            )
        for k_, c in enumerate(cs):
            tr = ST * max(rsc[max(0, k_ - 1)], rsc[min(k_, len(rsc) - 1)])
            items.append(dict(kind="post", seg=R["seg"], side=side, c=c, ps=sup[k_][0], z=sup[k_][1], t_rail=tr))
    # diagonal runs: one max-scale rail + an in-line filler post at the concave end; reflex end set back from the
    # straight rail (polygon test)
    rsc_J = spec["fence"]["rail_scale_max"]
    tJ = ST * rsc_J
    LJ = SL * rsc_J
    for R in P["runs"]:
        if R["seg"] != "J":
            continue
        F = Fr["J"]
        side = R["side"]
        a, b = R["a"], R["b"]
        # straight rail at the reflex corner (A at B1 for the left side, C at B2 for the right side)
        if R["a_kind"] == "reflex":
            nb_seg, end_is_b = "A", True
        else:
            nb_seg, end_is_b = "C", False
        Fn = Fr[nb_seg]
        rn = [it for it in items if it["kind"] == "rail" and it["seg"] == nb_seg and it["side"] == side]
        rn = rn[-1] if end_is_b else rn[0]
        poly_n = rect(Fn, rn["a"], rn["b"], side * li, side * (li + rn["t"]))

        # set-back: smallest offset so the J rail (thickness tJ) is clear of the straight rail by CLR
        def jpoly(x0, x1):
            return rect(F, x0, x1, side * li, side * (li + tJ))

        if R["a_kind"] == "reflex":
            lo, hi = 0.0, 0.2
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if sat_sep(poly_n, jpoly(a + mid, a + mid + 0.3)) >= CLR:
                    hi = mid
                else:
                    lo = mid
            r0 = a + hi
            r1 = r0 + LJ
            f0 = r1 + CLR
            f1 = b - CLR  # filler at the concave end b
            fill_end = "b"
        else:
            lo, hi = 0.0, 0.2
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if sat_sep(poly_n, jpoly(b - mid - 0.3, b - mid)) >= CLR:
                    hi = mid
                else:
                    lo = mid
            r1 = b - hi
            r0 = r1 - LJ
            f1 = r0 - CLR
            f0 = a + CLR  # filler at the concave end a
            fill_end = "a"
        fl = f1 - f0
        psf = fl / SH
        if not (0.85 <= psf <= 1.15):
            raise RuntimeError(f"J filler {fl:.3f} -> scale {psf:.3f} outside the band")
        items.append(dict(kind="rail", seg="J", side=side, a=r0, b=r1, rsc=rsc_J, t=tJ, setback=round(hi, 4)))
        items.append(dict(kind="filler", seg="J", side=side, a=f0, b=f1, ps=psf, z=E["deck_top"], at=fill_end))
        # posts behind both J rail ends (inside the rail span, as far from the corners as the end rays allow)
        for end, sgn in ((r0, +1), (r1, -1)):
            hl = 0.5 * LJ
            ray_off = 0.15 * hl  # parent 'lat' rays at 0.85 of the half length
            c_lo = 0.5 * SH * POST_DECK_SCALE + CLR  # centre offset range from the rail end
            c_hi = ray_off + 0.5 * SH * POST_DECK_SCALE - 0.012
            placed = None
            offs = np.linspace(c_lo, c_hi, 12)
            if (sgn > 0 and R["a_kind"] == "reflex") or (sgn < 0 and R["b_kind"] == "reflex"):
                offs = offs[::-1]
            for off in offs:
                c = end + sgn * off
                sp_ = support(F, side, c, tJ)
                if sp_ is None or sp_[1] == 0.0:
                    continue
                pp = post_poly(F, side, c, sp_[0], tJ)
                own = lambda it: it["kind"] == "rail" and it["seg"] == "J" and it["side"] == side
                if all(
                    sat_sep(pp, ip) >= (-1e-4 if own(it) else 0.5 * CLR) for it, ip in zip(items, _polys(items, Fr, E))
                ):
                    placed = (c, sp_)
                    break
            if placed is None:
                raise RuntimeError(f"no post behind the J rail end {end:.3f} side {side}")
            items.append(dict(kind="post", seg="J", side=side, c=placed[0], ps=placed[1][0], z=placed[1][1], t_rail=tJ))
    # reflex corners: the straight runs' corner posts must clear the J posts / rails -> slide them along their rail
    polys = _polys(items, Fr, E)
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sep = sat_sep(polys[i], polys[j])
            if sep < -0.0005 and not (
                items[i]["kind"] == "rail" and items[j]["kind"] == "rail" and items[i]["seg"] == items[j]["seg"]
            ):
                bad.append((i, j, round(sep, 4)))
    return items, polys, bad


def _poly(it, Fr, E):
    F = Fr[it["seg"]]
    side = it["side"]
    li = E["l_rail_in"]
    SH, ST = E["SH"], E["ST"]
    if it["kind"] == "rail":
        return rect(F, it["a"] + 1e-4, it["b"] - 1e-4, side * li, side * (li + it["t"]))
    if it["kind"] == "filler":
        return rect(F, it["a"], it["b"], side * li, side * (li + ST * it["ps"]))
    l0 = li + it["t_rail"]
    return rect(F, it["c"] - 0.5 * SH * it["ps"], it["c"] + 0.5 * SH * it["ps"], side * l0, side * (l0 + ST * it["ps"]))


def _polys(items, Fr, E, exclude_seg_side=None):
    return [_poly(it, Fr, E) for it in items]


# ------------------------------------------------------------------ builder (needs a stage)
class SpotJogCourse(GS.SpotLevelCourse):
    def __init__(self, stage, root=GS.ROOT_PRIM):
        super().__init__(stage, root=root)

    def Sfr(self, F):
        return dict(name=F["name"], P0=F["P0"], d=F["d"], n=F["n"], yaw=F["yaw"], heading=F["heading"])

    def build(self, spec, jog=None):
        from pxr import UsdGeom

        self.spec = spec
        P = plan_jog(spec, self.dims, jog)
        self.plan = P
        E = P["E"]
        self.E = E
        UsdGeom.Xform.Define(self.stage, self.troot)
        bs = spec["board_scale"]
        t = E["t"]
        BD = E["BD"]
        A, J, C = (self.Sfr(P[k]) for k in ("A", "J", "C"))
        self.FR = dict(A=A, J=J, C=C)
        RL = G.RL
        info = dict(
            level=spec["level"], seed=spec["seed"], jog=dict(P["jog"], mitre_margin=round(P["mitre_margin"], 4))
        )
        # 1. deck: global slot grid, 0 deg boards; pad slots = two boards end to end in y (top layer seated back 8 mm)
        self.frame(A)
        byaw = self._byaw()
        for sl in P["slots"]:
            for jl in range(RL):
                for kb, yc in enumerate(sl["y_centres"]):
                    s0k = sl["s"] + (G.HEAD_SEAT if jl == RL - 1 else 0.0)
                    w = self.W(s0k + BD / 2, yc - A["P0"][1], jl * t)
                    tag = "" if len(sl["y_centres"]) == 1 else "ab"[kb]
                    self.put(
                        f"{sl['type']}_deck_L{jl}_{sl['i']:02d}{tag}",
                        "board",
                        bs,
                        byaw,
                        center=w[:2],
                        z_bottom=w[2],
                        orient="flat",
                        contacts=[("down", "flat")],
                    )
        # 2. ramp (parent recipe, frame A)
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
            target=self.W(E["s_d"] + 0.5 * G.HEAD_SEAT, 0.0, E["head"]),
            orient="flat",
            contacts=[("down", "edges")],
        )
        info["ramp"] = dict(deg=round(E["ramp_deg"], 3), head_z=round(E["head"], 4), s_d=round(E["s_d"], 4))
        # 3. crossing: hump 1 (A), valley 1 = A/J/C halves, hump 2 + valley 2 (C)
        self.frame(A)
        el = G.cross_el("hump", P["s_h1"], self.dims, spec)
        self._cross(el, "A_hump1")
        self.crossings.append(dict(el, seg="A", name="hump 1"))
        vh = []
        for h in P["halves"]:
            self.frame(self.FR[h["seg"]])
            el = G.cross_el("valley", h["s"], self.dims, spec)
            self._valley_half(el, h["side"], f"{h['seg']}_valley1{'RL'[h['side'] > 0]}")
            vh.append(dict(el, seg=h["seg"], side=h["side"], name=f"valley 1 {h['seg']}{'RL'[h['side'] > 0]}"))
        self.valley_halves = vh
        self.frame(C)
        el = G.cross_el("hump", P["s_h2"], self.dims, spec)
        self._cross(el, "C_hump2")
        self.crossings.append(dict(el, seg="C", name="hump 2"))
        el = G.cross_el("valley", P["s_v2"], self.dims, spec)
        self._cross(el, "C_valley2")
        self.crossings.append(dict(el, seg="C", name="valley 2"))
        # 4. terrace on the C lane
        Ct = dict(C, terrace=P["terrace"], s_st=P["s_st_C"])
        info["terrace"] = self._terrace(Ct)
        # 5. fence
        info["fence"] = self._fence_jog()
        info["n_parts"] = len(self.parts)
        info["crossing"] = [
            dict(
                name=c["name"], seg=c["seg"], extent_s=[round(x, 4) for x in c["extent"]], panel_deg=round(c["deg"], 3)
            )
            for c in self.crossings
        ] + [
            dict(
                name=v["name"],
                seg=v["seg"],
                side="LR"[v["side"] < 0],
                extent_s=[round(x, 4) for x in v["extent"]],
                panel_deg=round(v["deg"], 3),
            )
            for v in vh
        ]
        info["spot"] = dict(riser_layers=G.RL, riser_m=round(G.RL * E["t"], 4), ramp_deg=round(E["ramp_deg"], 3))
        path = [dict(xyz=np.round(np.r_[p["p"], GC.FLOOR_Z + p["z"]], 3).tolist(), what=p["what"]) for p in P["path"]]
        info["robot"] = dict(
            start=[*np.round(path[0]["xyz"][:2], 3).tolist(), 0.0],
            goal=path[-1]["xyz"],
            path=path,
            path_len_xy=round(P["path_len_xy"], 3),
            turns=self._turns(),
        )
        self.info = info
        return info

    def _turns(self):
        P = self.plan
        return [
            dict(
                at="A->J",
                where="valley 1 trough (cross-slope), between the A- and J-valley panels",
                xy=np.round(P["B1"], 3).tolist(),
                deg=45.0,
                dir="left",
                heading_in=0.0,
                heading_out=45.0,
            ),
            dict(
                at="J->C",
                where="valley 1 trough (cross-slope), between the J- and C-valley panels",
                xy=np.round(P["B2"], 3).tolist(),
                deg=-45.0,
                dir="right",
                heading_in=45.0,
                heading_out=0.0,
            ),
        ]

    def _valley_half(self, el, side, tag):
        """one side of a valley (support slat + panel), parent recipe (LevelCourse._cross valley branch)"""
        s_ = self.spec
        dk = self.E["deck_top"]
        yaw_al = self._yaw_al()
        PL, PD, PT = el["panel"]
        SL_, SH_, ST_ = el["support"]
        th = math.radians(el["deg"])
        ss_ = s_["support_scale"]

        def upper_anchor(P, signs, size):
            B = self.bottom_face(P, signs, size)
            return B[np.argsort(B[:, 2])[2:]].mean(0)

        vyaw = self.yaw + self.lib.flat_yaw("board")
        lc = side * (el["l_sup"] + 0.5 * ST_)
        w = self.W(el["c"], lc, dk)
        self.put(f"{tag}_slat", "slat", ss_, yaw_al, center=w[:2], z_bottom=w[2], contacts=[("down", "flat")])
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
            f"{tag}_panel",
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

    def _fence_jog(self):
        E = self.E
        fs = self.spec["fence"]
        SLs, SHs, STs = self.dims["slat"]
        items, polys, bad = plan_fence(self.plan, self.spec)
        if bad:
            raise RuntimeError(f"fence plan overlaps: {bad[:6]}")
        self.fence_items = items
        li = E["l_rail_in"]
        n_ = dict(rail=0, post=0, filler=0)
        out = []
        for it in items:
            S = self.FR[it["seg"]]
            self.frame(S)
            yaw_al = self._yaw_al()
            side = it["side"]
            tg = f"fence{it['seg']}{'LR'[side < 0]}"
            if it["kind"] == "post":
                lp = li + it["t_rail"] + 0.5 * STs * it["ps"]
                pw = self.W(it["c"], side * lp, it["z"])
                p = self.put(
                    f"{tg}_post{n_['post']}",
                    "slat",
                    it["ps"],
                    yaw_al,
                    pitch=math.pi / 2,
                    axis=tuple(self.n),
                    center=pw[:2],
                    z_bottom=pw[2],
                    contacts=[("down", "flat")],
                )
            elif it["kind"] == "filler":
                lp = li + 0.5 * STs * it["ps"]
                pw = self.W(0.5 * (it["a"] + it["b"]), side * lp, it["z"])
                p = self.put(
                    f"{tg}_filler{n_['filler']}",
                    "slat",
                    it["ps"],
                    yaw_al,
                    pitch=math.pi / 2,
                    axis=tuple(self.n),
                    center=pw[:2],
                    z_bottom=pw[2],
                    contacts=[("down", "flat")],
                )
            else:
                for k, hz in enumerate(fs["rails_above_deck"]):
                    zc = E["deck_top"] + hz
                    rw = self.W(0.5 * (it["a"] + it["b"]), side * (li + 0.5 * it["t"]), zc)
                    p = self.put(
                        f"{tg}_rail{n_['rail']}_{k}",
                        "slat",
                        it["rsc"],
                        yaw_al,
                        center=rw[:2],
                        z_bottom=zc - 0.5 * SHs * it["rsc"] + GC.FLOOR_Z,
                        contacts=[("lat", side)],
                    )
            n_[it["kind"]] += 1
            out.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in it.items()})
        return dict(
            items=out,
            rail_inner_l=round(li, 4),
            lane_width_design=round(2 * li, 4),
            rails_above_deck=fs["rails_above_deck"],
            n=n_,
        )

    # ------------------------------------------------------------ verification
    def floor_clear_jog(self, pxq, carb, step=0.1):
        """before build: down rays over the course footprint (lane frames +-(rail + post + 0.1 m), the pad) must reach the
        gym floor; hits > 2 m above the floor (basketball backboards) are recorded as overhead and passed through"""
        P = self.plan
        E = P["E"]
        bad, over, n = {}, {}, 0
        ext = E["l_rail_in"] + 0.06 + 0.1
        pts = []
        for F, a, b in (
            (P["A"], -(self.spec["start_back"] + 0.1), P["sB1"] + 0.9),
            (P["J"], -0.9, P["L"] + 0.9),
            (P["C"], -0.9, P["terrace"]["ends"][0] + 0.1),
        ):
            for s in np.arange(a, b + 1e-9, step):
                for l in np.arange(-ext, ext + 1e-9, step):
                    pts.append(W2(F, s, l))
        for x0, x1, y0, y1 in P["boards"]:
            for x in np.arange(x0 - 0.1, x1 + 0.1 + 1e-9, step):
                for y in np.arange(y0 - 0.1, y1 + 0.1 + 1e-9, step):
                    pts.append(np.array([x, y]))
        for xy in pts:
            z0 = 2.6
            n += 1
            while True:
                h = pxq.raycast_closest(carb.Float3(float(xy[0]), float(xy[1]), z0), carb.Float3(0, 0, -1), z0 + 0.4)
                if not (h and h.get("hit")):
                    bad["NO HIT"] = bad.get("NO HIT", 0) + 1
                    break
                c = str(h.get("collision", ""))
                z = float(h["position"][2])
                if z - GC.FLOOR_Z > 2.0:
                    over[c] = over.get(c, 0) + 1
                    z0 = z - 0.01
                    continue
                if "SM_floor_element" not in c or abs(z - GC.FLOOR_Z) > 0.01:
                    bad[c] = bad.get(c, 0) + 1
                break
        allx = np.array(pts)
        return dict(
            rays=n,
            ok=not bad,
            offenders=bad,
            overhead_above_2m=over,
            footprint_x=[round(float(allx[:, 0].min()), 3), round(float(allx[:, 0].max()), 3)],
            footprint_y=[round(float(allx[:, 1].min()), 3), round(float(allx[:, 1].max()), 3)],
        )

    def obb_overlaps(self, tol=0.003):
        """3D OBB separating-axis test over ALL course parts (tilted ones included): pairs that interpenetrate by more
        than tol on every separating axis"""
        boxes = []
        for p in self.parts:
            P = p["corners"]
            sg = p["signs"]
            c = P.mean(0)
            ax = []
            for i in range(3):
                v = P[sg[:, i] == 1].mean(0) - P[sg[:, i] == -1].mean(0)
                h = 0.5 * np.linalg.norm(v)
                ax.append((v / max(h * 2, 1e-12), h))
            boxes.append((p["role"], c, ax, P.min(0), P.max(0)))
        out = []
        for i in range(len(boxes)):
            ri, ci, ai, mni, mxi = boxes[i]
            for j in range(i + 1, len(boxes)):
                rj, cj, aj, mnj, mxj = boxes[j]
                if (np.minimum(mxi, mxj) - np.maximum(mni, mnj) <= tol).any():
                    continue
                axes = [a for a, _ in ai] + [a for a, _ in aj]
                for a, _ in ai:
                    for b, _ in aj:
                        cr = np.cross(a, b)
                        if np.linalg.norm(cr) > 1e-6:
                            axes.append(cr / np.linalg.norm(cr))
                depth = 1e9
                for u in axes:
                    ra = sum(h * abs(a @ u) for a, h in ai)
                    rb = sum(h * abs(b @ u) for b, h in aj)
                    pen = ra + rb - abs((cj - ci) @ u)
                    depth = min(depth, pen)
                    if depth <= tol:
                        break
                if depth > tol:
                    out.append(dict(a=ri, b=rj, depth=round(float(depth), 4)))
        return out

    def verify(self, pxq, carb):
        import gym_course_v3 as GV

        RA = self.RA
        ray = lambda o, v, Lr, skip=(): self._ray(pxq, carb, o, v, Lr, skip)
        P = self.plan
        E = self.E
        dk = E["deck_top"]
        FR = self.FR
        res = dict(parts=len(self.parts), checks=0, fail=[], max_gap=0.0)
        # (1) contacts of every structure part (parent recipe, verbatim)
        for p in self.parts:
            Pc, sg, size = p["corners"], p["signs"], p["size"]
            for kind_, mode in p["contacts"]:
                pts, dirs = [], []
                if kind_ == "down":
                    B = self.bottom_face(Pc, sg, size)
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
                        F = Pc[sg[:, i_long] == e]
                        ctr = Pc.mean(0) + 0.85 * (F.mean(0) - Pc.mean(0))
                        half_t = 0.5 * float(np.min(size)) * p["scale"]
                        pts.append(ctr + n_out * (half_t - 0.001))
                        dirs.append(n_out)
                elif kind_ == "ends":
                    i_long = int(np.argmax(size))
                    for e in (-1, 1):
                        F = Pc[sg[:, i_long] == e]
                        c0 = Pc.mean(0)
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
        # (2) overlaps: parent axis-aligned test + 3D OBB test over every part (diagonal + tilted parts included)
        res["overlaps"] = self._overlaps()
        ob = self.obb_overlaps()
        TJ = lambda o: (
            (o["a"].startswith("hand") and o["b"].startswith("plat"))
            or (o["b"].startswith("hand") and o["a"].startswith("plat"))
        )
        res["overlaps_obb"] = [o for o in ob if not TJ(o)]
        res["obb_terrace_handrail_joints"] = dict(
            pairs=[o for o in ob if TJ(o)],
            note="sloped stair handrail end entering the first platform rail (unchanged parent _terrace joinery; the same pairs are in L0 / the old L1, where the parent overlap test skips tilted parts)",
        )
        meas = {}
        # (3) profiles along the path (centre line and +-0.35 m) in every lane frame
        prof = {}
        P_A, P_J, P_C = P["A"], P["J"], P["C"]
        spans = dict(A=(-self.spec["start_back"], P["sB1"]), J=(0.0, P["L"]), C=(0.0, P["terrace"]["ends"][0] + 0.3))
        for nm, F in (("A", P_A), ("J", P_J), ("C", P_C)):
            for l in (-0.35, 0.0, 0.35):
                rows = []
                for s in np.arange(spans[nm][0], spans[nm][1], 0.02):
                    xy = W2(F, s, l)
                    h = ray(np.r_[xy, GC.FLOOR_Z + 3.0], (0, 0, -1), 4.0)
                    rows.append((round(float(s), 3), None if h is None else round(float(h["p"][2] - GC.FLOOR_Z), 4)))
                prof[f"{nm}:{l}"] = rows
        res["profile"] = prof
        c = np.array([(s, z) for s, z in prof["A:0.0"] if z is not None and z < 0.5])
        mm = (c[:, 0] > 0.06) & (c[:, 0] < E["s_d"] - 0.06)
        meas["ramp_deg"] = round(math.degrees(math.atan(np.polyfit(c[mm, 0], c[mm, 1], 1)[0])), 2)
        # deck top on the flats (A: ramp head .. hump 1; C: valley 2 .. stairs) and in the cross gaps
        el_h1 = G.cross_el("hump", P["s_h1"], self.dims, self.spec)
        el_v2 = G.cross_el("valley", P["s_v2"], self.dims, self.spec)
        flats = [("A", E["s_d"] + 0.1, P["s_h1"] - 0.1), ("C", el_v2["extent"][1] + 0.1, P["s_st_C"] - 0.1)]
        dz = []
        for nm, a, b in flats:
            cc = np.array([(s, z) for s, z in prof[f"{nm}:0.0"] if z is not None])
            m_ = (cc[:, 0] > a) & (cc[:, 0] < b)
            dz += list(cc[m_, 1])
        meas["deck_top_z"] = dict(
            median=round(float(np.median(dz)), 4),
            min=round(float(np.min(dz)), 4),
            max=round(float(np.max(dz)), 4),
            n=len(dz),
        )
        st = self.terrace["starts"]
        tops = []
        cL = np.array([(s, z) for s, z in prof["C:0.0"] if z is not None])
        for k in range(1, len(st)):
            a = st[k] + 0.05
            b = (st[k + 1] - 0.05) if k + 1 < len(st) else st[k] + 0.6
            m_ = (cL[:, 0] > a) & (cL[:, 0] < b)
            tops.append(round(float(np.median(cL[m_, 1])), 4))
        meas["tread_tops_z"] = tops
        meas["risers"] = [round(b - a, 4) for a, b in zip([meas["deck_top_z"]["median"]] + tops[:-1], tops)]
        meas["platform_z"] = tops[-1]
        # crossing panels: plane from 3 rays -> slope / pitch / roll in the part's own lane frame
        hm = []
        for p in self.parts:
            if "_panel" not in p["role"]:
                continue
            Pc, sg, size = p["corners"], p["signs"], p["size"]
            B = self.bottom_face(Pc, sg, size)
            cB = B.mean(0)
            pts = []
            for k_ in range(3):
                q = cB + 0.5 * (B[k_] - cB)
                h = ray(np.r_[q[:2], q[2] + 1.0], (0, 0, -1), 1.5)
                pts.append(h["p"])
            A_ = np.array(pts)
            nrm = np.cross(A_[1] - A_[0], A_[2] - A_[0])
            nrm = nrm / np.linalg.norm(nrm)
            if nrm[2] < 0:
                nrm = -nrm
            g = -nrm[:2] / nrm[2]
            hm.append(
                dict(
                    part=p["role"],
                    seg=p["seg"],
                    slope_deg=round(math.degrees(math.acos(nrm[2])), 2),
                    pitch_deg=round(math.degrees(math.atan(abs(g @ p["d"]))), 2),
                    roll_deg=round(math.degrees(math.atan(abs(g @ p["n"]))), 2),
                )
            )
        meas["crossing_panels"] = hm
        # surface under the bends: what the centre line / +-0.2 / +-0.45 m stations hit around B1 and B2
        bends = []
        for nm, Bp, Fin, Fout in (("B1", P["B1"], P_A, P_J), ("B2", P["B2"], P_J, P_C)):
            rows = []
            for ds in np.arange(-0.6, 0.601, 0.1):
                F = Fin if ds < 0 else Fout
                for l in (-0.45, -0.2, 0.0, 0.2, 0.45):
                    xy = Bp + F["d"] * ds + F["n"] * l
                    h = ray(np.r_[xy, GC.FLOOR_Z + 2.0], (0, 0, -1), 3.0)
                    z = None if h is None else float(h["p"][2] - GC.FLOOR_Z) - dk
                    what = (
                        None
                        if h is None
                        else (
                            "panel" if "board" in h["c"] and z > 0.004 else ("deck" if abs(z) < 0.004 else h["c"][-30:])
                        )
                    )
                    rows.append(
                        dict(ds=round(float(ds), 2), l=l, z_above_deck=None if z is None else round(z, 4), on=what)
                    )
            bends.append(
                dict(
                    bend=nm,
                    xy=np.round(Bp, 3).tolist(),
                    stations=rows,
                    frac_on_panel=round(
                        sum(1 for r in rows if r["z_above_deck"] is not None and r["z_above_deck"] > 0.004) / len(rows),
                        3,
                    ),
                )
            )
        meas["bend_surface"] = bends
        # (4) lane width: perpendicular at every rail bay centre of every run (mid-rail height) + bends
        widths = []
        for it in self.fence_items:
            if it["kind"] != "rail" or it["side"] != 1:
                continue
            F = FR[it["seg"]]
            s = 0.5 * (it["a"] + it["b"])
            o = np.r_[W2(F, s, 0.0), GC.FLOOR_Z + dk + 0.58]
            hp = ray(o, np.r_[F["n"], 0], 1.5)
            hm_ = ray(o, np.r_[-F["n"], 0], 1.5)
            if hp and hm_:
                widths.append(dict(seg=it["seg"], s=round(s, 3), w=round(hp["d"] + hm_["d"], 4)))
        ws = [w["w"] for w in widths]
        meas["lane_width"] = dict(min=min(ws), max=max(ws), n=len(ws), per_bay=widths)
        # minimum chord through stations on the centre line near the bends (72 directions, 3 rail heights)
        bw = []
        for nm, Bp, Fin, Fout in (("B1", P["B1"], P_A, P_J), ("B2", P["B2"], P_J, P_C)):
            for ds in np.arange(-0.8, 0.801, 0.1):
                F = Fin if ds < 0 else Fout
                xy = Bp + F["d"] * ds
                best = None
                for hz in (0.30, 0.58, 0.86):
                    o = np.r_[xy, GC.FLOOR_Z + dk + hz]
                    for th_ in np.radians(np.arange(0, 180, 2.5)):
                        v = np.r_[math.cos(th_), math.sin(th_), 0.0]
                        h1 = ray(o, v, 3.0)
                        h2 = ray(o, -v, 3.0)
                        if h1 and h2:
                            w_ = h1["d"] + h2["d"]
                            if best is None or w_ < best[0]:
                                best = (w_, math.degrees(th_), hz)
                bw.append(
                    dict(
                        bend=nm,
                        ds=round(float(ds), 2),
                        min_chord=round(best[0], 4) if best else None,
                        dir_deg=round(best[1], 1) if best else None,
                        at_rail_h=best[2] if best else None,
                    )
                )
            # along the mitre line through the bend
            Fm_d = Fin["d"] + Fout["d"]
            Fm_d /= np.linalg.norm(Fm_d)
            nmit = np.array([-Fm_d[1], Fm_d[0]])
            o = np.r_[Bp, GC.FLOOR_Z + dk + 0.58]
            h1 = ray(o, np.r_[nmit, 0], 2.0)
            h2 = ray(o, np.r_[-nmit, 0], 2.0)
            bw.append(dict(bend=nm, mitre_line_width=round(h1["d"] + h2["d"], 4) if h1 and h2 else None))
        meas["bend_width"] = bw
        # (5) fence continuity: horizontal rays outwards from the centre line every 2 cm at the 3 rail heights; the
        # longest run of stations whose ray escapes (> l_rail_in + 0.1) per side and height
        cont = {}
        stations = []
        for nm, F, a, b in (("A", P_A, 0.05, P["sB1"]), ("J", P_J, 0.0, P["L"]), ("C", P_C, 0.0, P["s_st_C"] - 0.05)):
            for s in np.arange(a, b, 0.02):
                stations.append((nm, F, s))
        for side in (1, -1):
            for hz in (0.30, 0.58, 0.86):
                run_, worst, esc = 0.0, 0.0, 0
                for nm, F, s in stations:
                    o = np.r_[W2(F, s, 0.0), GC.FLOOR_Z + dk + hz]
                    h = ray(o, np.r_[F["n"] * side, 0], 2.5)
                    if h is None or not h["c"].startswith(self.root):
                        run_ += 0.02
                        esc += 1
                        worst = max(worst, run_)
                    else:
                        run_ = 0.0
                cont[f"{'LR'[side < 0]}@{hz}"] = dict(
                    stations=len(stations), escaped=esc, longest_open_m=round(worst, 3)
                )
        meas["fence_continuity"] = cont
        if any(v["longest_open_m"] > 0.1 for v in cont.values()):
            res["fail"].append(dict(mode="fence gap", cont=cont))
        # (6) lane interior covered by the course (no gym floor inside the fences between the ramp head and the stairs)
        holes, n_ = [], 0
        for nm, F, a, b in (
            ("A", P_A, E["s_d"] + 0.03, P["sB1"] + 0.6),
            ("J", P_J, -0.3, P["L"] + 0.3),
            ("C", P_C, -0.3, P["s_st_C"] - 0.03),
        ):
            for s in np.arange(a, b, 0.05):
                for l in np.arange(-E["l_rail_in"] + 0.02, E["l_rail_in"] - 0.019, 0.05):
                    xy = W2(F, s, l)
                    if not (
                        abs(P["fence_y"](xy[0], 1) - xy[1]) > 0
                        and xy[1] < P["fence_y"](xy[0], 1) - 0.015
                        and xy[1] > P["fence_y"](xy[0], -1) + 0.015
                    ):
                        continue
                    n_ += 1
                    h = ray(np.r_[xy, GC.FLOOR_Z + 2.0], (0, 0, -1), 3.0)
                    if h is None or not h["c"].startswith(self.root) or float(h["p"][2] - GC.FLOOR_Z) < dk - 0.006:
                        holes.append(
                            dict(
                                xy=np.round(xy, 3).tolist(),
                                hit=None if h is None else h["c"][-50:],
                                z=None if h is None else round(float(h["p"][2] - GC.FLOOR_Z), 4),
                            )
                        )
        meas["lane_floor_covered"] = dict(rays=n_, holes=len(holes), examples=holes[:10])
        if holes:
            res["fail"].append(dict(mode="lane interior not covered by the deck", n=len(holes), ex=holes[:5]))
        # (7) fence top rail above the walking surface
        fh = []
        for p in self.parts:
            if not (p["role"].startswith("fence") and "_rail" in p["role"] and p["role"].endswith("_2")):
                continue
            cP = p["corners"].mean(0)
            h = ray(np.r_[cP[:2], cP[2] + 1.0], (0, 0, -1), 1.2)
            if h is None or not h["c"].startswith(p["path"]):
                continue
            F = FR[p["seg"]]
            s_ = float((cP[:2] - F["P0"]) @ F["d"])
            g = ray(np.r_[W2(F, s_, 0.0), GC.FLOOR_Z + 3.0], (0, 0, -1), 4.0)
            fh.append(round(float(h["p"][2] - g["p"][2]), 4))
        meas["fence_top_rail_above_walk"] = dict(min=min(fh), max=max(fh), n=len(fh)) if fh else None
        # (8) valley-half mitre clearances (2D footprints of the panels of adjacent lane frames)
        mit = []
        panels = [p for p in self.parts if "_valley1" in p["role"] and p["role"].endswith("_panel")]
        for i in range(len(panels)):
            for j in range(i + 1, len(panels)):
                if panels[i]["seg"] == panels[j]["seg"]:
                    continue
                Ai = panels[i]["corners"][:, :2]
                Aj = panels[j]["corners"][:, :2]
                hi_ = hull2d(Ai)
                hj_ = hull2d(Aj)
                mit.append(dict(a=panels[i]["role"], b=panels[j]["role"], sep_2d=round(sat_sep(hi_, hj_), 4)))
        meas["valley_mitre_sep"] = [m_ for m_ in mit if m_["sep_2d"] < 0.6]
        if any(m_["sep_2d"] < 0.0 for m_ in mit):
            res["fail"].append(
                dict(mode="valley halves overlap in plan", ex=[m_ for m_ in mit if m_["sep_2d"] < 0][:4])
            )
        res["measured"] = meas
        res["audit"] = RA.audit_no_primitives(self.stage, [self.root])
        res["ok"] = not res["fail"] and not res["overlaps"] and not res["overlaps_obb"]
        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=JOG["L"])
    a = ap.parse_args()
    sp = GS.make_spot_level_spec(1, 0)
    P = plan_jog(sp, jog=dict(L=a.L))
    E = P["E"]
    print(
        "flat",
        round(P["flat"], 3),
        "pull",
        round(P["pull"], 4),
        "mitre margin",
        round(P["mitre_margin"], 4),
        "B1",
        np.round(P["B1"], 3),
        "B2",
        np.round(P["B2"], 3),
        "offset",
        round(P["lateral_offset"], 4),
        "path",
        round(P["path_len_xy"], 3),
    )
    print(
        "h1",
        round(P["s_h1"], 3),
        "sB1",
        round(P["sB1"], 3),
        "h2(C)",
        round(P["s_h2"], 3),
        "v2(C)",
        round(P["s_v2"], 3),
        "st(C)",
        round(P["s_st_C"], 3),
    )
    for h in P["halves"]:
        F = P[h["seg"]]
        print(
            " half",
            h["seg"],
            h["side"],
            "s",
            round(h["s"], 4),
            "x",
            np.round(W2(F, h["s"], 0), 3),
            "->",
            np.round(W2(F, h["s"] + 1.2, 0), 3),
        )
    print(
        "terrace",
        {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in P["terrace"].items()},
    )
    print("slots", "".join(s_["type"] for s_ in P["slots"]), len(P["slots"]))
    for s_ in P["slots"]:
        if s_["type"] == "P":
            print("  pad slot", s_["i"], np.round(s_["x"], 3), "lane y", np.round(s_["lane_y"], 3))
    for k, v in P["corners"].items():
        print(" corner", k, np.round(v, 4), "reflex" if P["reflex"][k] else "concave")
    for R in P["runs"]:
        print(
            " run",
            R["seg"],
            R["side"],
            round(R["a"], 4),
            round(R["b"], 4),
            round(R["b"] - R["a"], 4),
            R["a_kind"],
            R["b_kind"],
        )
    items, polys, bad = plan_fence(P, sp)
    for it in items:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in it.items()})
    print("fence overlaps", bad)
    allp = np.vstack(polys)
    print(
        "fence footprint x",
        np.round([allp[:, 0].min(), allp[:, 0].max()], 3),
        "y",
        np.round([allp[:, 1].min(), allp[:, 1].max()], 3),
    )
