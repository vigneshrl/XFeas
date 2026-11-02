#!/usr/bin/env python
"""
GYM OBSTACLE COURSE L1-L5, SPOT VARIANTS (experiments/4_obstacle_course/spot/L{1..5}; baselines round 1, additive 2026-10-21).
Built on xfeas/scenes/gym_course_levels.py, which is NOT modified: SpotLevelCourse subclasses its LevelCourse and only
replaces the beam element; the module-level riser count and the beam module length are set on the imported module at run
time (no file changes). All five levels use the LevelCourse segment machinery (L1 / L2 as one straight segment with the
v1 / v3 element lists, in the v1 / v3 lane), so every level shares one builder and one verify.

Spot sizing (spotp = Spot + stowed arm, ReLIC; sources in school_course_spot.SPOT):
  * risers: deck = stairs riser = 4 desk tops (board scale 1.10, 0.0283 m each) = 0.113 m (G1: 6 tops = 0.170 m).
      0.113 m = the H2 curb class (0.11 m, Spot oracle 3/3 under fm0), inside the 0.10-0.13 m band where the Spot setpoint
      oracle climbs (gen_v1 env_climb 87/92, env_curb 22/24) and stock ReLIC mostly does not (6/92); far below the 0.17 m
      school stairs that no method climbs (0/57). 3 desk tops (0.085 m) would be the < 0.10 m band that stock often steps
      blind (16/28), i.e. no posture test. Entry ramp head (RL-1) x t = 0.085 m -> 6.8 deg.
  * beam: a crouch roof instead of the G1 plank beam on 1.24 m pillars. Two school cloakroom partitions (SM_cloakroom_2,
      scaled to 0.92 m) stand on the deck along the lane edges, inner faces 0.41 m from the centre (clear width ~0.81 m), and
      carry a roof of school desk tops (BP_school_desk_26/SM_school_desk_2_2 x1.10, the deck's own boards) seated on their
      tops: clear height 0.92 m above the deck (stock passes from 0.955 m, lowest crouch from 0.88 m; 0.92 per the coordinator, 4 cm crouch margin). The G1 pillar slat
      cannot be scaled below 0.96 m (scale limit 0.85) and a 2.55 m plank at 0.90 m would cut the fence top rail (0.98 m).
  * funnel: throat 0.68 m (Spot collision width 0.548 m + 0.13 m; N2 0.75 m passes stock 3/3).
  * everything else per level as in gym_course_levels.make_level_spec: humps / valleys, rough insets (L3+), serpentine with
      2 U-turns (L4+), oil / ice patches (L5).

  python baselines/gym_course_spot.py --level 4      # print the plan (no Isaac)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, copy, json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(THIS)
import gym_course_levels as G
import gym_course as GC

SPOT_RL = 4  # desk tops per riser
BEAM_U = float(os.environ.get("GYMSPOT_BEAM_U", 0.92))
FUNNEL_GAP = float(os.environ.get("GYMSPOT_FUNNEL_GAP", 0.68))
PARTITION_NOMINAL = (
    1.98,
    0.25,
    1.0,
)  # SM_cloakroom_2 at scale 1 (long, thick, tall); the builder uses the measured size
OUT_ROOT = xfeas.paths.COURSES + "/4_obstacle_course/spot"
ROOT_PRIM = "/World/GymCourseSpot"

G.RL = SPOT_RL
_beam_D = dict(v=PARTITION_NOMINAL[0] * BEAM_U / PARTITION_NOMINAL[2])
_element_dims0 = G.element_dims


def _element_dims_spot(dims, spec):
    E = _element_dims0(dims, spec)
    E["beam_D"] = _beam_D["v"]
    return E


G.element_dims = _element_dims_spot

# lanes: L1 = the v1 lane of G1 L1 (P0 x -22.726, terrace end x -7.687); L2 = the L3 lane (P0 x -23.60, end -7.05), 0.8 m
# longer than G1 L2's v3 lane so the 1.6 m Spot gets ~0.6 m flats between beam / humps / funnel (v3 lane: 0.47 m);
# L3 = G1's L3 lane; Spot's beam (1.78 m) and funnel (0.70 m) are longer than G1's (1.71 / 0.60 m), which would drop the
# flats below the builder's 0.15 m minimum (0.149 m; the lane cannot grow: the terrace end is quantised to whole deck slots
# and the next slot leaves the free gym floor), so L3 uses cross_gap 0.12 m between its 4 humps / valleys (G1 0.15 m).
LANES = {1: dict(P0=[-22.726, 2.60], x_end_max=-7.69), 2: dict(P0=[-23.60, 2.60], x_end_max=-7.05)}
CROSS_GAP = {3: 0.12}


def make_spot_level_spec(level=1, seed=0):
    level = int(level)
    if level <= 2:
        sp = G.make_level_spec(3, seed)  # LevelCourse spec skeleton (rough params unused without rough modules)
        sp.update(level=level)
        cross = dict(kind="crossing", types=["hump", "valley", "hump", "valley"])
        mods = [cross] if level == 1 else [dict(kind="beam"), cross, dict(kind="funnel")]
        sp["segments"] = [
            dict(
                name="A",
                heading=0.0,
                P0=LANES[level]["P0"],
                last=True,
                x_end_max=LANES[level]["x_end_max"],
                modules=mods,
            )
        ]
    else:
        sp = copy.deepcopy(G.make_level_spec(level, seed))
        if level in LANES:
            sp["segments"][0].update(LANES[level])
    sp = copy.deepcopy(sp)
    if level in CROSS_GAP:
        sp["cross_gap"] = CROSS_GAP[level]
    sp["robot"] = "spotp"
    sp["variant"] = "spot"
    sp["riser_layers"] = SPOT_RL
    sp["crouch"] = dict(
        kind="partition_roof",
        u=BEAM_U,
        roof="board",
        roof_scale=sp["board_scale"],
        partition_inset=0.004,
        n_planks=sp["crouch"]["n_planks"],
        plank_scale=sp["crouch"]["plank_scale"],
        plank_gap=sp["crouch"]["plank_gap"],
    )
    sp["funnel"] = dict(sp["funnel"], gap=FUNNEL_GAP)
    sp["spot_sizing"] = dict(
        riser_m=None,
        beam_u=BEAM_U,
        funnel_gap=FUNNEL_GAP,
        note="see module docstring; sources in school_course_spot.SPOT",
    )
    return sp


def _highest_hit(T, x, y):
    A, B, C = T[:, 0], T[:, 1], T[:, 2]
    d = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    ok = np.abs(d) > 1e-12
    d = np.where(ok, d, 1.0)
    w1 = ((B[:, 1] - C[:, 1]) * (x - C[:, 0]) + (C[:, 0] - B[:, 0]) * (y - C[:, 1])) / d
    w2 = ((C[:, 1] - A[:, 1]) * (x - C[:, 0]) + (A[:, 0] - C[:, 0]) * (y - C[:, 1])) / d
    w3 = 1 - w1 - w2
    m = ok & (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)
    return float((w1 * A[:, 2] + w2 * B[:, 2] + w3 * C[:, 2])[m].max()) if m.any() else None


class SpotLevelCourse(G.LevelCourse):
    def __init__(self, stage, root=ROOT_PRIM):
        super().__init__(stage, root=root)
        ps = self.lib.size("partition")
        self.part_dims = (float(max(ps[:2])), float(min(ps[:2])), float(ps[2]))
        _beam_D["v"] = self.part_dims[0] * BEAM_U / self.part_dims[2]
        self.troot = "/World/GymCourseSpotTerrain"

    def build(self, spec):
        info = super().build(spec)
        info["spot"] = dict(
            riser_layers=G.RL,
            riser_m=round(G.RL * self.E["t"], 4),
            ramp_deg=round(self.E["ramp_deg"], 3),
            beam_u=BEAM_U,
            funnel_gap=spec["funnel"]["gap"],
            beam_D=round(self.E["beam_D"], 4),
        )
        return info

    def _beam(self, s_door0, tag):
        """crouch roof: 2 partitions along the deck edges + desk-top roof seated on their tops"""
        import gym_course_v3 as GV

        c = self.spec["crouch"]
        E = self.E
        dk = E["deck_top"]
        Lp_, Tp_, Hp_ = self.part_dims
        psc = c["u"] / Hp_
        assert 0.85 <= psc <= 1.15, psc
        ps = self.lib.size("partition")
        yaw_p = self.yaw + (0.0 if ps[0] >= ps[1] else math.pi / 2)
        al, th = Lp_ * psc, Tp_ * psc
        l_c = 0.5 * E["BL"] - c["partition_inset"] - 0.5 * th
        walls = []
        for side in (-1, 1):
            w = self.W(s_door0 + 0.5 * al, side * l_c, dk)
            walls.append(
                self.put(
                    f"{tag}_wall{'RL'[side > 0]}", "partition", psc, yaw_p, center=w[:2], z_bottom=w[2], contacts=[]
                )
            )
        # roof boards (desk tops, long side across the lane), centred on the partitions
        bs = c["roof_scale"]
        BD = E["BD"]
        nr = max(1, int((al + 1e-6) // (BD + G.GAP)))
        Lr = nr * BD + (nr - 1) * G.GAP
        byaw = self._byaw()
        top0 = max(float(q["corners"][:, 2].max()) for q in walls)
        roof, paths = [], []
        Tw = [GV.world_tris(self.stage.GetPrimAtPath(q["path"])) for q in walls]
        for k in range(nr):
            s_k = s_door0 + 0.5 * (al - Lr) + k * (BD + G.GAP) + 0.5 * BD
            w = self.W(s_k, 0.0, 0.0)
            p = self.put(f"{tag}_roof{k}", "board", bs, byaw, center=w[:2], z_bottom=top0, orient="flat", contacts=[])
            T = GV.world_tris(self.stage.GetPrimAtPath(p["path"]))
            mn, mx = p["corners"].min(0), p["corners"].max(0)
            dz = None
            for q, Tq in zip(walls, Tw):
                for x, y in self._top_grid(q, 40, 6, 0.004):
                    if not (mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1]):
                        continue
                    zl, zt = GV.lowest_hit(T, x, y), _highest_hit(Tq, x, y)
                    if zl is None or zt is None:
                        continue
                    dz = (zl - zt) if dz is None else min(dz, zl - zt)
            if dz is None:
                raise RuntimeError(f"{tag}_roof{k}: does not overlap the partitions")
            if abs(dz) > 1e-5:
                self.parts.pop()
                self.lib.used["board"] -= 1
                self.lib.placed.pop()
                p = self.put(
                    f"{tag}_roof{k}", "board", bs, byaw, center=w[:2], z_bottom=top0 - dz, orient="flat", contacts=[]
                )
            p["rest_on"] = [q["role"] for q in walls]
            roof.append(p)
            paths.append(p["path"])
        return dict(
            seg=self.S["name"],
            kind="partition_roof",
            s=[round(s_door0, 4), round(s_door0 + al, 4)],
            depth=round(al, 4),
            roof_s=[round(s_door0 + 0.5 * (al - Lr), 4), round(s_door0 + 0.5 * (al + Lr), 4)],
            roof_len=round(Lr, 4),
            u_target=c["u"],
            partition=dict(
                src=self.lib.catalog["partition"]["src"],
                scale=round(psc, 4),
                length=round(al, 4),
                thick=round(th, 4),
                l_centre=round(l_c, 4),
                inner_face_l=round(l_c - 0.5 * th, 4),
            ),
            roof=dict(src=self.lib.catalog["board"]["src"], scale=bs, n=nr),
            walls=[q["role"] for q in walls],
            roof_roles=[q["role"] for q in roof],
            pillars=[],
            planks=[],
            paths=paths,
            d=self.d.copy(),
            n=self.n.copy(),
            P0=self.P0.copy(),
        )

    BOARD_TOP_INSET = 0.0027  # desk-top mesh top sits 2.7 mm under its bbox top (deck measured 0.1105 vs 4 x 0.0283; G1 0.1671 vs 0.1698)

    def _slick(self, sl, k, CAT, TB):
        """parent sheet, placed 1.5 mm above the REAL deck surface (the parent uses the bbox deck top, which left the
        sheets 4.1-5.3 mm above the boards -- the same failure is in the published G1 L5 verify)"""
        dk0 = self.E["deck_top"]
        self.E["deck_top"] = dk0 - self.BOARD_TOP_INSET
        try:
            return super()._slick(sl, k, CAT, TB)
        finally:
            self.E["deck_top"] = dk0

    def _tile_support_recheck(self, res, ray):
        """the parent 'tile bottom on boards' rays sit at 5 even x stations; for a 2-slot tile in a non-first segment the
        middle station falls exactly into the 2 mm seam between two deck boards and the ray reaches the gym floor
        (gap = one board, 0.0283 m; the same 12 failures are in the published G1 L4 / L5 verify). Re-check: at every
        station whose original ray fails, cast again 1.2 cm off the seam (+ then -); the offset ray must land on a DECK
        BOARD of this course (the inset's own supporting board) with a gap within [-3, +3] mm (inside the 5 mm rule).
        Every original and offset ray is recorded; parent failures of a tile are removed only if every station passes."""
        out = []
        for T in self.tiles:
            d, n, P0, s_a = T["frame"]
            Lt, Wt = T["size"]
            wl = lambda x, y, z: np.r_[P0 + d * (s_a + x) + n * y, GC.FLOOR_Z + z]
            z0 = self.E["deck_top"] - T["recess_depth"] + 0.02
            rays, bad = [], 0
            for x in np.linspace(0.05, Lt - 0.05, 5):
                for y in np.linspace(-Wt / 2 + 0.05, Wt / 2 - 0.05, 4):
                    rec = dict(x=round(float(x), 4), y=round(float(y), 4))
                    for dx in (0.0, 0.012, -0.012):
                        h = ray(wl(x + dx, y, z0), (0, 0, -1), 0.08, skip=(T["path"],))
                        g = None if h is None else round(h["d"] - 0.02, 4)
                        hit = None if h is None else h["c"]
                        role = next(
                            (
                                p_["role"]
                                for p_ in self.parts
                                if hit is not None and (hit == p_["path"] or hit.startswith(p_["path"] + "/"))
                            ),
                            None,
                        )
                        on_board = role is not None and "_deck_" in role
                        hit = role or hit
                        ok = g is not None and -0.003 <= g <= 0.003 and on_board
                        rec[{0.0: "orig", 0.012: "off_plus", -0.012: "off_minus"}[dx]] = dict(gap=g, hit=hit, ok=ok)
                        if ok:
                            rec["pass_with"] = {0.0: "orig", 0.012: "off_plus", -0.012: "off_minus"}[dx]
                            break
                    if "pass_with" not in rec:
                        bad += 1
                    rays.append(rec)
            n_off = sum(1 for r_ in rays if r_.get("pass_with", "orig") != "orig")
            out.append(dict(tile=T["path"], stations=len(rays), failed=bad, passed_only_off_seam=n_off, rays=rays))
            if bad == 0:
                res["fail"] = [
                    f
                    for f in res["fail"]
                    if not (f.get("part") == T["path"] and f.get("mode") == "tile bottom on boards")
                ]
        res["measured"]["rough_support_recheck"] = out

    def floor_clear(self, pxq, carb, step=0.1):
        """parent rule (down rays over the footprint + 0.1 m must reach the gym floor), except that hits more than
        2.0 m above the floor (the wall-mounted basketball backboard over the v1 lane) are recorded as overhead and the
        ray is continued below them"""
        P = self.plan if hasattr(self, "plan") else G.plan_layout(self.spec, self.dims)
        E = P["E"]
        bad = {}
        over = {}
        n = 0
        ext = E["l_rail_in"] + 0.06 + 0.1
        for S in P["segments"]:
            s_from = -(self.spec["start_back"] + 0.1) if S["first"] else -0.8
            for s in np.arange(s_from, S["s_end"] + 0.1, step):
                for l in np.arange(-ext, ext + 1e-9, step):
                    xy = S["P0"] + S["d"] * s + S["n"] * l
                    z0 = 2.6
                    n += 1
                    while True:
                        h = pxq.raycast_closest(
                            carb.Float3(float(xy[0]), float(xy[1]), z0), carb.Float3(0, 0, -1), z0 + 0.4
                        )
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
        return dict(rays=n, ok=not bad, offenders=bad, overhead_above_2m=over)

    @staticmethod
    def _top_grid(part, na, nb, margin):
        P = part["corners"]
        top = P[np.argsort(P[:, 2])[4:], :2]
        o = top[0]
        dd = sorted(top[1:], key=lambda q: np.linalg.norm(q - o))
        e1, e2 = dd[0] - o, dd[1] - o
        L1, L2 = np.linalg.norm(e1), np.linalg.norm(e2)
        out = []
        for i in np.linspace(margin / L1, 1 - margin / L1, nb):
            for j in np.linspace(margin / L2, 1 - margin / L2, na):
                out.append(o + e1 * i + e2 * j)
        return out

    def verify(self, pxq, carb):
        import gym_course_v3 as GV
        import school_course as SCM

        RA = self.RA
        res = super().verify(pxq, carb)
        ray = lambda o, v, Lr, skip=(): self._ray(pxq, carb, o, v, Lr, skip)
        segs = self.plan["segments"]
        bm = []
        for b in self.beams:
            S = next(s for s in segs if s["name"] == b["seg"])
            self.frame(S)
            dk = self.E["deck_top"]
            # partitions: lowest-vertex band on the deck (>= 3 supported points, COM inside their hull)
            sup_w = {}
            for role in b["walls"]:
                p = next(q for q in self.parts if q["role"] == role)
                A = RA.world_points(self.stage.GetPrimAtPath(p["path"]))
                zmin = float(A[:, 2].min())
                band = A[A[:, 2] <= zmin + 0.004]
                if len(band) > 600:
                    band = band[np.random.default_rng(0).choice(len(band), 600, replace=False)]
                sup = []
                for q in band:
                    h = ray(q + np.array([0, 0, 0.015]), (0, 0, -1), 0.06, skip=(p["path"],))
                    if h is not None and -0.01 <= h["d"] - 0.015 <= 0.005:
                        sup.append(q[:2])
                Hh = SCM._hull(np.array(sup)) if sup else np.zeros((0, 2))
                dcom = SCM._dist_to_hull(Hh, A.mean(0)[:2]) if sup else 1e9
                ok = len(sup) >= 3 and dcom <= 0.02
                sup_w[role] = dict(
                    n_band=int(len(band)),
                    n_supported=len(sup),
                    com_out=round(float(dcom), 4) if dcom < 1e8 else None,
                    z_min_above_deck=round(zmin - GC.FLOOR_Z - dk, 4),
                    ok=ok,
                )
                res["checks"] += 1
                if not ok:
                    res["fail"].append(dict(part=role, mode="partition on deck", **sup_w[role]))
            # roof boards: seated on BOTH partitions (>= 3 contact points per partition, gap -10..+5 mm)
            sup_r = {}
            for role in b["roof_roles"]:
                p = next(q for q in self.parts if q["role"] == role)
                T = GV.world_tris(self.stage.GetPrimAtPath(p["path"]))
                per = {}
                for wr in b["walls"]:
                    q = next(x for x in self.parts if x["role"] == wr)
                    g = []
                    for x, y in self._top_grid(q, 40, 6, 0.004):
                        zl = GV.lowest_hit(T, x, y)
                        if zl is None:
                            continue
                        h = ray(np.r_[x, y, zl + 0.01], (0, 0, -1), 0.06, skip=(p["path"],))
                        if h is not None and h["c"].startswith(q["path"]):
                            gp = h["d"] - 0.01
                            if -0.01 <= gp <= 0.005:
                                g.append(gp)
                    per[wr] = dict(n_contact=len(g), max_gap=round(max(g), 4) if g else None)
                ok = all(v["n_contact"] >= 3 for v in per.values())
                sup_r[role] = dict(per_partition=per, ok=ok)
                res["checks"] += 1
                if not ok:
                    res["fail"].append(dict(part=role, mode="roof on partitions", per=per))
            # clear height / width under the roof (lane interior only)
            inner = b["partition"]["inner_face_l"]
            U, Wd = [], []
            for s in np.arange(b["roof_s"][0] + 0.01, b["roof_s"][1] - 0.005, 0.02):
                for l in np.arange(-inner + 0.03, inner - 0.029, 0.05):
                    dn = ray(self.W(s, l, dk + 0.3), (0, 0, -1), 0.6)
                    if dn is None:
                        continue
                    up = ray(np.r_[dn["p"][:2], float(dn["p"][2]) + 0.02], (0, 0, 1), 2.0)
                    if up is not None and any(up["c"].startswith(x) for x in b["paths"]):
                        U.append(float(up["p"][2]) - float(dn["p"][2]))
            for s in np.linspace(b["s"][0] + 0.05, b["s"][1] - 0.05, 7):
                for hz in (0.1, 0.3, 0.5, 0.7):
                    o = self.W(s, 0.0, dk + hz)
                    hp = ray(o, np.r_[self.n, 0], 1.5)
                    hm_ = ray(o, np.r_[-self.n, 0], 1.5)
                    if hp and hm_:
                        Wd.append(hp["d"] + hm_["d"])
            U = np.array(U)
            bm.append(
                dict(
                    seg=b["seg"],
                    n_rays=int(len(U)),
                    underside_above_walk_min=round(float(U.min()), 4) if len(U) else None,
                    median=round(float(np.median(U)), 4) if len(U) else None,
                    max=round(float(U.max()), 4) if len(U) else None,
                    clear_width=dict(min=round(min(Wd), 4), max=round(max(Wd), 4)) if Wd else None,
                    roof_len=b["roof_len"],
                    partitions=sup_w,
                    roof=sup_r,
                )
            )
            if not len(U) or abs(float(U.min()) - BEAM_U) > 0.03:
                res["fail"].append(
                    dict(mode="beam underside off target", seg=b["seg"], umin=float(U.min()) if len(U) else None)
                )
        res["measured"]["beam"] = bm
        self._tile_support_recheck(res, ray)
        res["audit"] = RA.audit_no_primitives(self.stage, [self.root])
        res["ok"] = not res["fail"] and not res["overlaps"]
        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=4)
    a = ap.parse_args()
    sp = make_spot_level_spec(a.level)
    P = G.plan_layout(sp)
    print(json.dumps(G.plan_summary(P), indent=1, default=G.jsonable))
    print("E", {k: round(v, 4) for k, v in P["E"].items()})
