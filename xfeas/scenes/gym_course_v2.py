#!/usr/bin/env python
"""
GYM OBSTACLE COURSE  v2 = "v2_crouch_funnel"  (additive 2026-10-17). The v1 course (gym_course.py, make_spec(seed),
version 1) is NOT changed and stays reproducible; assets/school.usd is never written (everything is built in memory).

v2 = the seed's v1 course + two MOBILITY hurdles that need setpoint adaptation by a G1 humanoid, both on the flat deck
before the stairs. The ramp, humps, valleys, stairs and platform keep their exact v1 geometry and the lane stays at
y = 2.6. To leave a flat run before and after each hurdle, flat_a and flat_b each grow by 0.25 m (2.00 -> 2.25 m,
inside v1's own seed band 1.6-2.6). The v1 terrace code then needs one more deck board (21 instead of 20), which would
end 2.5 cm from the gym's end wall, so the whole course is shifted P0_shift = -0.20 m along x: ramp foot -22.926
(v1 -22.726), start 1.2 m before it at -23.926 - 0.2 = -24.126 (0.32 m from the wall behind), crossing ramps +0.05 m,
stairs / platform / goal +0.30 m in world x, terrace end -7.163 (0.22 m from the end wall).
The fence is rebuilt by the v1 code for the longer lane (10 bays of 1.086 m instead of 9 of 1.151 m).

  1. CROUCH  (in flat_a, between the ramp head and hump 1): one real panelled school door leaf
             (BP_school_door_13/.../InstancedSM_school_door_4_2, 2.58 x 0.73 x 0.14 m, its own material) laid flat
             ACROSS the lane above the fence, underside crouch.u (1.09 m) above the deck = 0.85 x G1's standing top
             (1.28 m). It rests on FOUR timber posts (the course's own desk apron boards
             BP_school_desk_26/SM_school_desk_2_3 standing on end on the gym floor, just outside the fence posts,
             scale = the leaf's lowest vertex over each post / 1.129 m). Normal walking collides; crouch-walk /
             crouch + lean passes (cf. D1g, 1.10 m). NOTE: same door family as the held-out D1/D1g test scenes.
  2. FUNNEL  (in flat_b, between valley 2 and the stairs): the V-shaped obstacle of Goal_scenes/obstacle_course.png.
             Two school desk tops (the course's board BP_school_desk_26/SM_school_desk_2_2, scale 1.15 -> 1.38 m tall,
             0.75 m wide, 0.03 m thick) standing upright on the deck, angled from the fence rails to the lane centre
             and converging downstream to a gap funnel.gap (0.39 m: narrower than G1's 0.47-0.55 m stance width with
             arms, wider than its 0.27-0.31 m depth) -> turn sideways and sidestep (cf. HW1). Each wing is fixed to two
             apron-board posts (scale 1.15, 1.30 m) standing on the deck in the dead zone behind it.

  spec = make_spec_v2(seed)        # spec["version"] = 2, spec["variant"] = "v2_crouch_funnel"
  B = GymCourseV2(stage); info = B.build(spec); world.reset(); ver = B.verify(pxq, carb)   (ver["v2"] = new checks)
  course_spec(version) -> (spec, builder class)  for runners; info["robot"] = start / goal / path / sections.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, copy, json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import gym_course as GC

VERSION = 2
VARIANT = "v2_crouch_funnel"
OUT_DIR = xfeas.paths.COURSES + "/gym_obstacle_course/v2_crouch_funnel"
NOMINAL_DIMS = dict(board=(1.2, 0.65, 0.0257), slat=(1.1292, 0.1874, 0.0259))
G1_TOP = 1.278  # GR00T-WBC G1 standing top (analysis/G1_CONTROLLER_BRIDGES.md table 2)


def make_spec_v2(seed=0):
    sp = GC.make_spec(seed)
    v1 = copy.deepcopy(sp)
    sp["version"] = VERSION
    sp["variant"] = VARIANT
    sp["flat_a"] = round(v1["flat_a"] + 0.25, 3)
    sp["flat_b"] = round(v1["flat_b"] + 0.25, 3)
    sp["keep_v1_P0"] = True  # P0 = v1's ramp-foot world position + P0_shift along x
    sp["P0_shift"] = -0.20
    sp["crouch"] = dict(
        kind="door_leaf", scale=1.0, u=1.09, post_scale_lim=(0.85, 1.15), post_gap=0.005, post_inset=0.03
    )
    sp["funnel"] = dict(gap=0.39, board_scale=1.15, rail_clear=0.002, post_scale=1.15, post_at=(0.35, 0.75))
    return sp


def course_spec(version=1, seed=0):
    """runner entry point: (spec, builder class) for course version 1 (v1, gym_course.py) or 2 (this file)"""
    if int(version) == 1:
        return GC.make_spec(seed), GC.GymCourse
    return make_spec_v2(seed), GymCourseV2


def _v1_P0x(spec, dims):
    v1 = dict(spec, flat_a=spec["flat_a"] - 0.25, flat_b=spec["flat_b"] - 0.25)
    tot = GC.layout(v1, dims)["s_end_req"] + 0.8
    return 0.5 * (GC.GYM["x"][0] + GC.GYM["x"][1]) - 0.5 * tot


def world_tris(prim):
    """all composed mesh triangles under prim, world frame (N, 3, 3) (fan triangulation of the faces)"""
    from pxr import Usd, UsdGeom

    T = []
    for q in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if q.IsA(UsdGeom.Mesh) and q.IsActive():
            m = UsdGeom.Mesh(q)
            pts = m.GetPointsAttr().Get()
            cnt = m.GetFaceVertexCountsAttr().Get()
            idx = m.GetFaceVertexIndicesAttr().Get()
            if not pts or not cnt or not idx:
                continue
            M = np.array(UsdGeom.Xformable(q).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            W = (np.c_[np.asarray(pts, float), np.ones(len(pts))] @ M)[:, :3]
            idx = np.asarray(idx, int)
            k = 0
            tri = []
            for c in cnt:
                for j in range(1, c - 1):
                    tri.append((idx[k], idx[k + j], idx[k + j + 1]))
                k += c
            T.append(W[np.asarray(tri)])
    return np.concatenate(T) if T else np.zeros((0, 3, 3))


def lowest_hit(T, x, y):
    """lowest z where the vertical line through (x, y) meets the triangles T (None if it misses)"""
    A, B, C = T[:, 0], T[:, 1], T[:, 2]
    d = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    ok = np.abs(d) > 1e-12
    d = np.where(ok, d, 1.0)
    w1 = ((B[:, 1] - C[:, 1]) * (x - C[:, 0]) + (C[:, 0] - B[:, 0]) * (y - C[:, 1])) / d
    w2 = ((C[:, 1] - A[:, 1]) * (x - C[:, 0]) + (A[:, 0] - C[:, 0]) * (y - C[:, 1])) / d
    w3 = 1 - w1 - w2
    m = ok & (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)
    if not m.any():
        return None
    z = w1 * A[:, 2] + w2 * B[:, 2] + w3 * C[:, 2]
    return float(z[m].min())


class GymCourseV2(GC.GymCourse):
    def build(self, spec):
        dims = self.dims
        self.spec_f = spec["funnel"]
        gym0 = GC.GYM
        if spec.get("keep_v1_P0") and abs(spec["yaw"]) < 1e-9:
            p0x = _v1_P0x(spec, dims) + spec.get("P0_shift", 0.0)
            tot = GC.layout(spec, dims)["s_end_req"] + 0.8
            xm = p0x + 0.5 * tot
            GC.GYM = dict(gym0, x=(xm - 1.0, xm + 1.0))  # build() centres the course on GYM x -> v1 P0
        try:
            info = GC.GymCourse.build(self, spec)
        finally:
            GC.GYM = gym0
        RA = self.RA
        L = self.L
        dk = L["deck_top"]
        yaw = spec["yaw"]
        SLs, SHs, STs = dims["slat"]
        sy = self.lib.size("slat")
        yaw_al = yaw + (0.0 if sy[0] >= sy[1] else math.pi / 2)
        lp = lambda P: (P[:, :2] - self.P0) @ self.n
        sp = lambda P: (P[:, :2] - self.P0) @ self.d
        fence_out = max(
            float(np.abs(lp(p["corners"])).max())
            for p in self.parts
            if p["role"].startswith("fence") and "_post" in p["role"]
        )
        rail_in = info["fence"]["rail_inner_l"]
        last_end = L["crossing"][-1]["extent"][1]

        # ---------------- 1. CROUCH: door leaf across the lane on four floor posts outside the fence
        c = spec["crouch"]
        Lw, Dw, Tw = self.lib.flat_dims("door_leaf")
        D = Dw * c["scale"]
        s_door0 = L["s_d"] + 0.5 * (spec["flat_a"] - D)
        s_dc = s_door0 + 0.5 * D
        dyaw = yaw + math.pi / 2 + self.lib.flat_yaw("door_leaf")
        w = self.W(s_dc, 0.0, dk + c["u"])
        door = self.put(
            "crouch_door", "door_leaf", c["scale"], dyaw, center=w[:2], z_bottom=w[2], orient="flat", contacts=[]
        )
        A = RA.world_points(self.stage.GetPrimAtPath(door["path"]))
        As, Al, Az = sp(A), lp(A), A[:, 2] - GC.FLOOR_Z
        TR = world_tris(self.stage.GetPrimAtPath(door["path"]))
        posts = []
        pw0 = SHs  # along-lane width at scale 1
        for side in (-1, 1):
            for e, s_side in ((-1, s_door0), (1, s_door0 + D)):
                sc = 1.10
                for _ in range(4):  # scale = lowest leaf vertex over the post's top / 1.129
                    pw, pt = pw0 * sc, STs * sc
                    s_c = s_side - e * (c["post_inset"] + 0.5 * pw)
                    l_c = side * (fence_out + c["post_gap"] + 0.5 * pt)
                    zs = []
                    for ds_ in np.linspace(-0.5 * pw, 0.5 * pw, 7):
                        for dl_ in np.linspace(-0.5 * pt, 0.5 * pt, 3):
                            xy_ = self.W(s_c + ds_, l_c + dl_)[:2]
                            zh = lowest_hit(TR, xy_[0], xy_[1])
                            assert zh is not None, ("leaf misses the post", s_c + ds_, l_c + dl_)
                            zs.append(zh - GC.FLOOR_Z)
                    T = float(min(zs))
                    Tspread = float(max(zs) - min(zs))
                    sc_new = T / SLs
                    if abs(sc_new - sc) < 1e-6:
                        break
                    sc = sc_new
                assert c["post_scale_lim"][0] <= sc <= c["post_scale_lim"][1], sc
                wp = self.W(s_c, l_c, 0.0)
                p = self.put(
                    f"crouch_post{'RL'[side > 0]}{'ab'[e > 0]}",
                    "slat",
                    sc,
                    yaw_al,
                    pitch=math.pi / 2,
                    axis=tuple(self.n),
                    center=wp[:2],
                    z_bottom=wp[2],
                    contacts=[("down", "flat")],
                )
                posts.append(
                    dict(
                        role=p["role"],
                        s=round(s_c, 4),
                        l=round(l_c, 4),
                        scale=round(sc, 4),
                        top=round(T, 4),
                        leaf_underside_spread_over_post=round(Tspread, 4),
                    )
                )
        lane = np.abs(Al) <= rail_in
        info["crouch"] = dict(
            kind="door_leaf",
            src=self.lib.catalog["door_leaf"]["src"],
            scale=c["scale"],
            s=[round(s_door0, 4), round(s_door0 + D, 4)],
            depth=round(D, 4),
            width=round(Lw * c["scale"], 4),
            thick=round(Tw * c["scale"], 4),
            u_target=c["u"],
            u_vertices_over_lane=round(float(Az[lane & (np.abs(As - s_dc) <= 0.5 * D + 0.01)].min()) - dk, 4),
            n_tris=int(len(TR)),
            run_before=round(s_door0 - L["s_d"], 4),
            run_after=round(L["crossing"][0]["extent"][0] - (s_door0 + D), 4),
            posts=posts,
            fence_post_outer_l=round(fence_out, 4),
        )

        # ---------------- 2. FUNNEL: two upright desk tops converging to a gap, each fixed to two deck posts
        f = spec["funnel"]
        bsc = f["board_scale"]
        BL1, BD1, BT1 = dims["board"]
        Hb, Wb, tb = BL1 * bsc, BD1 * bsc, BT1 * bsc  # long side vertical
        g = f["gap"]
        l_lim = rail_in - f["rail_clear"]
        th = math.asin((l_lim - 0.5 * g) / Wb)
        for _ in range(50):  # outer BACK corner 2 mm off the rail's inner face
            th = math.asin((l_lim - 0.5 * g - tb * math.cos(th)) / Wb)
        Ls = Wb * math.cos(th) + tb * math.sin(th)
        s_o = last_end + 0.5 * (spec["flat_b"] - Ls)  # outer (upstream) face corner
        s_t = s_o + Wb * math.cos(th)  # throat (downstream) face corner
        wl = lambda s, l: np.r_[self.d * s + self.n * l]  # lane vector -> world xy direction
        wings, fposts = [], []
        for side in (-1, 1):
            v = np.array([-math.cos(th), side * math.sin(th)])  # throat -> outer end, (s, l)
            nb = np.array([math.sin(th), side * math.cos(th)])  # back normal (away from the corridor)
            Ft = np.array([s_t, side * 0.5 * g])
            Cc = Ft + v * (0.5 * Wb) + nb * (0.5 * tb)
            a_w, v_w = wl(*nb), wl(*v)
            byaw = math.atan2(a_w[1], a_w[0]) + self.lib.flat_yaw("board")
            wc = self.W(Cc[0], Cc[1], dk)
            part = self.put(
                f"funnel_wing{'RL'[side > 0]}",
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
            # gap edge from the copy's own mesh: shift the wing so its innermost vertex is exactly at gap / 2
            Pw = RA.world_points(self.stage.GetPrimAtPath(part["path"]))
            delta = float((side * lp(Pw)).min()) - 0.5 * g
            rec = self.lib.placed[-1]
            dxy = -side * delta * self.n
            self.lib.shift([rec], dxy)
            part["corners"] = part["corners"] + np.r_[dxy, 0.0]
            Ft = Ft + np.array([0.0, -side * delta])
            P = part["corners"]
            ext = np.ptp(P, axis=0)
            assert abs(np.ptp(P[:, 2]) - Hb) < 0.01, (np.ptp(P[:, 2]), Hb)
            wings.append(
                dict(
                    role=part["role"],
                    mesh_edge_correction=round(-delta, 4),
                    s=[round(float(sp(P).min()), 4), round(float(sp(P).max()), 4)],
                    l=[round(float(lp(P).min()), 4), round(float(lp(P).max()), 4)],
                    height=round(float(ext[2]), 4),
                    angle_to_lane_deg=round(math.degrees(th), 3),
                )
            )
            ps = f["post_scale"]
            pw, pt = SHs * ps, STs * ps
            pyaw = math.atan2(v_w[1], v_w[0]) + (yaw_al - yaw)  # slat long axis along the wing
            for k, fr in enumerate(f["post_at"]):  # fraction from the OUTER end
                Q = Ft + v * (Wb * (1.0 - fr)) + nb * (tb + 0.5 * pt + 0.0005)
                wq = self.W(Q[0], Q[1], dk)
                pp = self.put(
                    f"funnel_post{'RL'[side > 0]}{k}",
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
                assert abs(np.ptp(PP[:, 2]) - SLs * ps) < 0.01, np.ptp(PP[:, 2])
                assert float(np.abs(lp(PP)).max()) <= 0.5 * L["BL"] - 0.003, float(np.abs(lp(PP)).max())
                fposts.append(
                    dict(
                        role=pp["role"],
                        s=round(float(Q[0]), 4),
                        l=round(float(Q[1]), 4),
                        height=round(SLs * ps, 4),
                        max_abs_l=round(float(np.abs(lp(PP)).max()), 4),
                        back_face_dir=nb.round(4).tolist(),
                        wing=part["role"],
                    )
                )
        info["funnel"] = dict(
            src=self.lib.catalog["board"]["src"],
            board_scale=bsc,
            wing_h=round(Hb, 4),
            wing_w=round(Wb, 4),
            wing_t=round(tb, 4),
            angle_to_lane_deg=round(math.degrees(th), 3),
            gap_design=g,
            s=[round(s_o, 4), round(s_o + Ls, 4)],
            s_throat=round(s_t, 4),
            mouth_width_design=round(2 * (0.5 * g + Wb * math.sin(th)), 4),
            run_before=round(s_o - last_end, 4),
            run_after=round(L["s_st"] - (s_o + Ls), 4),
            wings=wings,
            posts=fposts,
        )
        self.v2 = dict(door=door["path"], s_door=(s_door0, s_door0 + D), s_funnel=(s_o, s_o + Ls), s_throat=s_t)

        # ---------------- robot task (start / goal as v1's rule) + path + sections
        rb = info["robot"]
        path = [np.array(p_) for p_ in rb["path"]]
        ins = [
            self.W(s_door0 - 0.5, 0, dk),
            self.W(s_door0 + D + 0.4, 0, dk),
            self.W(s_o - 0.5, 0, dk),
            self.W(s_t, 0, dk),
            self.W(s_t + 0.5, 0, dk),
        ]
        pts = path[:3] + ins + path[3:]
        rb["path"] = [np.round(p_, 3).tolist() for p_ in pts]
        X = lambda s: round(float(self.W(s, 0)[0]), 3)
        secs = [
            ("lane", -1.2, 0.0),
            ("ramp", 0.0, L["s_d"]),
            ("deck", L["s_d"], s_door0),
            ("crouch", s_door0, s_door0 + D),
            ("deck", s_door0 + D, L["crossing"][0]["extent"][0]),
        ]
        for i, el in enumerate(L["crossing"]):
            secs.append((f"{el['type']} {i // 2 + 1}", el["extent"][0], el["extent"][1]))
        secs += [
            ("deck", last_end, s_o),
            ("funnel", s_o, s_o + Ls),
            ("deck", s_o + Ls, L["s_st"]),
            ("stairs", L["s_st"], L["s_p0"]),
            ("platform", L["s_p0"], self.s_end),
        ]
        rb["sections"] = [dict(name=n_, s=[round(a, 4), round(b, 4)], x=[X(a), X(b)]) for n_, a, b in secs]
        rb["course_version"] = VERSION
        rb["variant"] = VARIANT
        rb["crouch_x"] = [X(s_door0), X(s_door0 + D)]
        rb["funnel_x"] = [X(s_o), X(s_o + Ls)]
        rb["throat_xy"] = np.round(self.W(s_t, 0, dk)[:2], 3).tolist()
        info["layout"] = {k: v for k, v in L.items() if k != "crossing"}
        info["n_parts"] = len(self.parts)
        info["version"] = VERSION
        info["variant"] = VARIANT
        self.info = info
        return info

    # ------------------------------------------------------------ verification
    def verify(self, pxq, carb):
        door = self.v2["door"]

        class NoDoorDown:  # v1 profiles / measures = WALKING surface (rays down pass the leaf)
            def raycast_closest(_s, o, v, Lr):
                h = pxq.raycast_closest(o, v, Lr)
                if v.z > -0.9:
                    return h
                done = 0.0
                while h and h.get("hit") and str(h.get("collision", "")).startswith(door):
                    dd = float(h["distance"]) + 1e-4
                    done += dd
                    o = carb.Float3(o.x, o.y, o.z - dd)
                    h = pxq.raycast_closest(o, v, float(Lr - done))
                    if h and h.get("hit"):
                        h = dict(h)
                        h["distance"] = float(h["distance"]) + done
                return h

        res = GC.GymCourse.verify(self, NoDoorDown(), carb)
        L = self.L
        dk = L["deck_top"]

        def ray(o, v, Lr, skip=None):
            o = np.asarray(o, float)
            v = np.asarray(v, float)
            v = v / np.linalg.norm(v)
            done = 0.0
            for _ in range(12):
                h = pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, v)), float(Lr - done))
                if not (h and h.get("hit")):
                    return None
                c_ = str(h.get("collision", ""))
                dist = float(h["distance"])
                if skip and c_.startswith(skip):
                    o = o + v * (dist + 1e-4)
                    done += dist + 1e-4
                    continue
                return dict(d=done + dist, c=c_, p=np.asarray(h["position"], float))
            return None

        out = dict(fail=[], checks=0)
        parts = {p["role"]: p for p in self.parts}

        def chk(name, h, start_off, path_prefix):
            out["checks"] += 1
            gap = None if h is None else h["d"] - start_off
            ok = h is not None and h["c"].startswith(path_prefix) and -0.01 <= gap <= 0.005
            if not ok:
                out["fail"].append(
                    dict(
                        check=name, gap=None if gap is None else round(gap, 4), hit=None if h is None else h["c"][-60:]
                    )
                )
            return gap

        # door rests on the posts: rays UP from inside each post's top
        gaps = []
        for r_, p in parts.items():
            if r_.startswith("crouch_post"):
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
                        h = ray(np.r_[q, top - 0.02], (0, 0, 1), 0.1, skip=p["path"])
                        if h is not None and h["c"].startswith(door):
                            gp.append(h["d"] - 0.02)
                out["checks"] += 1
                ok = len(gp) == 39 and min(gp) <= 0.003 and min(gp) >= -0.01
                if not ok:
                    out["fail"].append(dict(check=f"{r_}->door", n=len(gp), min_gap=round(min(gp), 4) if gp else None))
                gaps.append(min(gp) if gp else None)
                out.setdefault("door_post_contact", {})[r_] = (
                    dict(min_gap=round(min(gp), 4), max_gap=round(max(gp), 4), n_rays=len(gp)) if gp else None
                )
        out["door_on_posts_worst_min_gap"] = (
            round(max(g_ for g_ in gaps if g_ is not None), 4) if any(g_ is not None for g_ in gaps) else None
        )
        # wing boards fixed to their posts: rays from the post's front face towards the board
        wg = []
        for fp in self.info["funnel"]["posts"]:
            p = parts[fp["role"]]
            nb = np.array(fp["back_face_dir"])
            dirw = -np.r_[self.d * nb[0] + self.n * nb[1], 0.0]
            for hz in (0.25, 0.65, 1.05):
                q = self.W(fp["s"], fp["l"], dk + hz)
                wg.append(
                    chk(
                        f"{fp['role']}->wing",
                        ray(q, dirw, 0.1, skip=p["path"]),
                        0.5 * self.dims["slat"][2] * self.spec_f["post_scale"],
                        parts[fp["wing"]]["path"],
                    )
                )
        out["wing_on_posts_max_gap"] = (
            round(max(g_ for g_ in wg if g_ is not None), 4) if any(g_ is not None for g_ in wg) else None
        )
        # CROUCH underside: rays up from 5 cm above the walking surface over the lane
        s0, s1 = self.v2["s_door"]
        U = []
        for s in np.arange(s0 - 0.05, s1 + 0.05, 0.01):
            for l in np.arange(-0.65, 0.651, 0.05):
                dn = ray(self.W(s, l, dk + 0.3), (0, 0, -1), 0.6)
                if dn is None:
                    continue
                zw = float(dn["p"][2])
                up = ray(np.r_[dn["p"][:2], zw + 0.05], (0, 0, 1), 2.0)
                if up is not None and up["c"].startswith(door):
                    U.append((s, l, float(up["p"][2]) - zw, zw - GC.FLOOR_Z))
        U = np.array(U)
        uu = U[:, 2]
        out["crouch"] = dict(
            n_rays=len(U),
            underside_above_walk_min=round(float(uu.min()), 4),
            median=round(float(np.median(uu)), 4),
            max=round(float(uu.max()), 4),
            walk_z_median=round(float(np.median(U[:, 3])), 4),
            s_covered=[round(float(U[:, 0].min()), 3), round(float(U[:, 0].max()), 3)],
            ratio_to_g1_top=round(float(uu.min()) / G1_TOP, 3),
        )
        # door top (for the image check) and covering width
        tp = ray(self.W(0.5 * (s0 + s1), 0, 3.0), (0, 0, -1), 4.0)
        out["crouch"]["leaf_top_above_walk"] = round(float(tp["p"][2] - GC.FLOOR_Z) - dk, 4) if tp else None
        # FUNNEL gap: lateral rays from the lane centre, every 5 mm along s and every 5 cm in height
        a_, b_ = self.v2["s_funnel"]
        best = None
        prof = []
        st0 = self.v2["s_throat"]
        S_ = np.unique(
            np.r_[np.arange(a_ - 0.1, b_ + 0.1, 0.005), np.arange(st0 - 0.01, st0 + 0.01, 0.0005), a_ + 0.003]
        )
        for s in S_:
            wmin = None
            for hz in np.arange(0.03, 1.40, 0.05):
                o = self.W(s, 0.0, dk + hz)
                hp = ray(o, np.r_[self.n, 0], 1.5)
                hm = ray(o, np.r_[-self.n, 0], 1.5)
                if hp and hm:
                    wdt = hp["d"] + hm["d"]
                    if wmin is None or wdt < wmin[0]:
                        wmin = (wdt, hz, hp["d"], hm["d"])
            if wmin:
                prof.append((round(float(s), 3), round(wmin[0], 4)))
                if best is None or wmin[0] < best[0]:
                    best = (wmin[0], float(s), wmin[1], wmin[2], wmin[3])
        mouth = [w_ for s_, w_ in prof if abs(s_ - (a_ + 0.003)) < 1e-3]
        hh = []
        for side in (-1, 1):
            st_ = self.v2["s_throat"] - 0.2
            l_ = side * (
                0.5 * self.info["funnel"]["gap_design"]
                + 0.2 * math.tan(math.radians(self.info["funnel"]["angle_to_lane_deg"]))
                + 0.012
            )
            ht = ray(self.W(st_, l_, 3.0), (0, 0, -1), 4.0)
            hh.append(None if ht is None else round(float(ht["p"][2] - GC.FLOOR_Z) - dk, 4))
        out["funnel"] = dict(
            gap_min=round(best[0], 4),
            gap_s=round(best[1], 4),
            gap_at_height_above_deck=round(best[2], 3),
            gap_l=[round(-best[4], 4), round(best[3], 4)],
            throat_s_design=round(self.v2["s_throat"], 4),
            mouth_width_measured=round(min(mouth), 4) if mouth else None,
            wing_top_above_deck=hh,
            width_profile=[x for x in prof if round(x[0] * 1000) % 20 == 0],
        )
        # v1 lane width check, skipping samples inside the funnel
        widths = []
        fb = self.info["fence"]
        for i in range(fb["bays"]):
            s = (i + 0.5) * fb["bay"] + 0.2
            if a_ - 0.05 <= s <= b_ + 0.05:
                continue
            o = self.W(s, 0.0, dk + 0.58)
            hp_ = ray(o, np.r_[self.n, 0], 1.5)
            hm_ = ray(o, np.r_[-self.n, 0], 1.5)
            if hp_ and hm_:
                widths.append(round(hp_["d"] + hm_["d"], 4))
        res["measured"]["lane_width"] = dict(
            min=min(widths), max=max(widths), n=len(widths), note="funnel bays skipped"
        )
        res["v2"] = out
        res["ok"] = res["ok"] and not out["fail"]
        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", type=int, default=2)
    a = ap.parse_args()
    sp, _ = course_spec(a.version, a.seed)
    print(json.dumps(sp, indent=1))
    Ly = GC.layout(sp, NOMINAL_DIMS)
    print(
        json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in Ly.items() if k != "crossing"}, default=str
        )
    )
