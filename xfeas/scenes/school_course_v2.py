#!/usr/bin/env python
"""
SCHOOL COURSE v2 (experiments/3_school_course/v2; additive 2026-10-20 -- school_course.py = v1 is NOT changed).
Same hall as v1 (ground-floor south hall, floor z 0.681, lane +x along y = -5.8), redesigned around the three skills
the released PHP parkour policy structurally lacks (crouch: no height input; strafe: no lateral input; speed change:
fixed 0.73 m/s), plus one parkour control element both can clear. Final layout, in travel order (lane +x from (-27.2, -5.8)):

  1. START STRAIGHT   3 m of clear floor.
  2. CONTROL          the v1 gym bench hurdle (SM_gym_props_5 x0.85, top 0.255 m) -- NOT closed to the walls.
  3. LOW + NARROW     a wall of low brown cabinets (BP_school_closet_3 / BP_school_bookcase_3) across the hall with ONE
                      opening 0.45 m wide, bridged on top by a real school plank (SM_literature_props_1) resting on the two
                      jamb cabinets: opening ~0.46 x 0.985 m -> crouch AND sidestep at once.
  4. OIL -> TUNNEL    a copy of the school's own oil decal (Oil_surface_02 x0.85, 2.68 m along the lane; its own material +
                      the school's PM_Oil physics material) on the approach, then a LOW TUNNEL: blue school lockers
                      (BP_school_closet_1 x1.05) LYING ON THEIR SIDE along both sides carry a continuous roof of 3 panelled
                      door leaves (InstancedSM_school_door_4_2); clear height ~1.0 m, ~2.1 m long. More fallen lockers close
                      the hall flush with the mouth (0.88-1.01 m high).
  5. SIDLE STRIP      two rows of tall wooden school wardrobes (BP_school_wardrobe_1/_2, fronts facing the strip), 0.39 m
                      apart (measured) and 2.7 m long; the entrance line is closed to both hall walls by more wardrobes.
  6. GOAL             1.4 m past the strip exit (x -9.32), outside the D1 rectangle with a 1.8 m margin.
Closures of sections 3/4 are ~1 m high (not climbable for G1 / PHP, and low enough that later sections stay visible under
the 4 m ceiling); the strip is last because its tall walls hide anything behind them. The open gym doorway on the left
wall (its leaf BP_school_door_13 swings into the lane and is turned off in memory) is blocked by 3 wardrobes; the hall's
open glazed entrance doors (BP_school_door_1, window wall, x -17.0 .. -14.2) are kept and no closure line is placed in
front of them (x -17.2 .. -14.0).

  B = SchoolCourseV2(stage, make_spec_v2()); B.prepare(); world.reset(); walls = ...; B.build(walls_at=f); ver = B.verify(pxq, carb)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import school_course as SC

ROOT_PRIM = "/World/SchoolCourseV2"
CATALOG = dict(SC.CATALOG, oil=dict(src=os.environ.get("SC2_OIL_SRC", "Oil_surface_02")))
DOOR_ZONE_X = (-17.2, -14.0)  # the hall's open glazed entrance doors (window wall): no closure line here
D1_X0 = -7.5


def make_spec_v2(**over):
    sp = dict(
        version=2,
        location="south_hall",
        lane=dict(P0=[-27.2, -5.8], yaw=0.0, floor=0.681, half=1.35),
        straight=3.0,
        clear=1.4,
        hurdle=dict(kind="bench", scale=0.85),
        oil=dict(scale=0.85, gap_to_tunnel=0.1),
        tunnel=dict(
            style="fallen_lockers",
            locker_scale=1.05,
            cab="closet3",
            cab_scale=1.14,
            n_cab=2,
            half_w=0.80,
            n_leaves=3,
            closure=["closet3", "bookcase3"],
            closure_srange=[1.08, 1.15],
        ),
        clear_after_ln=1.2,
        gym_door_block=dict(x=[-23.85, -21.2], depth=0.58, kinds=["wardrobe1", "wardrobe"]),
        low_narrow=dict(
            gap=0.45,
            cab="closet3",
            cab_scale=1.14,
            lintel="plank",
            lintel_scale=0.9,
            closure=["closet3", "bookcase3"],
            closure_srange=[1.08, 1.15],
        ),
        # gap = design distance between the wardrobe front OBB planes; 0.27 gives a MEASURED 0.39 m (door relief)
        sidle=dict(s0_min=13.75, gap=0.27, length=2.7, kinds=["wardrobe1", "wardrobe"], srange=[0.95, 1.08]),
        goal_after=1.4,
    )
    for k, v in over.items():
        if k in sp and isinstance(sp[k], dict) and isinstance(v, dict):
            sp[k].update(v)
        else:
            sp[k] = v
    return sp


class SchoolCourseV2(SC.SchoolCourse):
    def __init__(self, stage, spec, root=ROOT_PRIM):
        super().__init__(stage, spec, root)

    def init_lib(self):
        if self.lib is None:
            self.lib = self.RA.RealLib(self.stage, root=self.root_req, catalog=CATALOG)
            self.root = self.lib.root
            for k in ("bench", "door_leaf", "closet3", "bookcase3", "locker", "plank", "oil"):
                self.lib.size(k)
        return self.lib

    def length_estimate(self):
        sp = self.spec
        return sp["sidle"]["s0_min"] + sp["sidle"]["length"] + sp["goal_after"]

    # ---------------------------------------------------------------- helpers
    def front_width(self, kind):
        sz = self.lib.size(kind)
        back = self.lib.catalog[kind].get("back", "+y")
        return float(sz[1] if back in ("+x", "-x") else sz[0])

    def _rec_part(self, rc, role, section):
        R_, P_ = self.lib.orient_matrix(rc["kind"], rc.get("orient", "as_is"), rc["yaw"], rc["scale"])
        cc = (rc["mn"] + rc["mx"]) / 2
        p = dict(
            role=role,
            section=section,
            kind=rc["kind"],
            src=rc["src"],
            scale=round(rc["scale"], 4),
            orient=rc.get("orient", "as_is"),
            yaw=round(rc["yaw"], 5),
            path=str(rc["copy"]["prim"].GetPath()),
            prim=rc["copy"]["prim"],
            corners=P_ + cc,
            size=self.lib.size(rc["kind"]) * rc["scale"],
            center=cc,
        )
        self.parts.append(p)
        return p

    def line(
        self, role, section, start_sl, u, f, length, kinds, srange, depth_target=None, rotated_ok=True, max_embed=0.06
    ):
        """furniture_line in lane coordinates: start (s, l), u / f unit vectors in the lane frame (ds, dl)"""
        st = self.P0 + self.d * start_sl[0] + self.n * start_sl[1]
        U = self.d * u[0] + self.n * u[1]
        Fv = self.d * f[0] + self.n * f[1]
        pl = None
        for rot in (False, True) if rotated_ok else (False,):
            pl = self.lib.furniture_plan(length, tuple(kinds), max_embed, tuple(srange), depth_target, rot)
            if pl is not None:
                break
        if pl is None:
            raise RuntimeError(f"{role}: no furniture combination for length {length:.3f} with {kinds} {srange}")
        r = self.lib.furniture_line(st, U, Fv, length, self.F, tuple(kinds), plan=pl)
        parts = [self._rec_part(rc, f"{role}{k}", section) for k, rc in enumerate(r.pop("recs"))]
        r["parts"] = [p["role"] for p in parts]
        return r, parts

    def single(self, role, section, kind, scale, start_sl, u, f):
        """one furniture piece, front plane through start along f, its side face at start, extending along u"""
        st = self.P0 + self.d * start_sl[0] + self.n * start_sl[1]
        U = self.d * u[0] + self.n * u[1]
        Fv = self.d * f[0] + self.n * f[1]
        sz = self.lib.size(kind)
        back = self.lib.catalog[kind].get("back", "+y")
        w = sz[1] if back in ("+x", "-x") else sz[0]
        dp = sz[0] if back in ("+x", "-x") else sz[1]
        r = self.lib.furniture_line(st, U, Fv, w * scale, self.F, (kind,), plan=([(kind, False, w, dp)], scale, 0.0))
        rc = r["recs"][0]
        return self._rec_part(rc, role, section), w * scale, dp * scale

    def put_resting(self, role, section, kind, x, y, orient, yaw, scale):
        """put a piece on the floor trying both sides down (roll 0 / 180 deg about the lane axis and about the lateral
        axis); keep the pose whose lowest-vertex band (4 mm) surrounds the centroid best (a flat face down)"""
        RA = self.RA
        best = None
        for pitch, axis in (
            (0.0, (1.0, 0.0)),
            (math.pi, (float(self.d[0]), float(self.d[1]))),
            (math.pi, (float(self.n[0]), float(self.n[1]))),
        ):
            p = self.put(role, kind, x, y, self.F, orient, yaw, scale, pitch=pitch, axis=axis, section=section)
            A = RA.world_points(p["prim"])
            zmin = float(A[:, 2].min())
            band = A[A[:, 2] <= zmin + 0.004][:, :2]
            H = SC._hull(band) if len(band) >= 3 else np.zeros((0, 2))
            d = SC._dist_to_hull(H, A.mean(0)[:2]) if len(band) >= 3 else 1e9
            area = 0.0
            if len(H) >= 3:
                area = 0.5 * abs(
                    sum(H[i, 0] * H[(i + 1) % len(H), 1] - H[(i + 1) % len(H), 0] * H[i, 1] for i in range(len(H)))
                )
            key = (d > 0.005, -area)
            self.parts.pop()
            self.lib.used[kind] -= 1
            if best is None or key < best[0]:
                best = (key, pitch, axis)
        _, pitch, axis = best
        return self.put(role, kind, x, y, self.F, orient, yaw, scale, pitch=pitch, axis=axis, section=section)

    @staticmethod
    def highest_hit(T, x, y):
        A, B, C = T[:, 0], T[:, 1], T[:, 2]
        d = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
        ok = np.abs(d) > 1e-12
        d = np.where(ok, d, 1.0)
        w1 = ((B[:, 1] - C[:, 1]) * (x - C[:, 0]) + (C[:, 0] - B[:, 0]) * (y - C[:, 1])) / d
        w2 = ((C[:, 1] - A[:, 1]) * (x - C[:, 0]) + (A[:, 0] - C[:, 0]) * (y - C[:, 1])) / d
        w3 = 1 - w1 - w2
        m = ok & (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)
        return float((w1 * A[:, 2] + w2 * B[:, 2] + w3 * C[:, 2])[m].max()) if m.any() else None

    def seat(self, role, section, kind, x, y, z0, orient, yaw, scale, supports):
        """place a slab on support parts so that its underside touches the HIGHEST point of the supports' real top
        surfaces under it (triangle hits on a dense grid) -> it bears on the supports, nothing interpenetrates"""
        import gym_course_v3 as GV

        p = self.put(role, kind, x, y, z0, orient, yaw, scale, section=section)
        T = GV.world_tris(p["prim"])
        mn_p, mx_p = p["corners"].min(0), p["corners"].max(0)
        dz = None
        for q in supports:
            Tq = GV.world_tris(q["prim"])
            for a, b in self.footprint(q, 40, 10, margin=0.003):
                if not (mn_p[0] <= a <= mx_p[0] and mn_p[1] <= b <= mx_p[1]):
                    continue
                zl, zt = GV.lowest_hit(T, a, b), self.highest_hit(Tq, a, b)
                if zl is None or zt is None:
                    continue
                dd = zl - zt  # >0: slab above the support at this point
                dz = dd if dz is None else min(dz, dd)
        if dz is None:
            raise RuntimeError(f"{role}: does not overlap its supports")
        if abs(dz) > 1e-5:
            self.parts.pop()
            self.lib.used[kind] -= 1
            p = self.put(role, kind, x, y, z0 - dz, orient, yaw, scale, section=section)
        p["rest_on"] = [q["role"] for q in supports]
        return p

    # ---------------------------------------------------------------- build
    def build(self, walls=None, walls_at=None):
        RA = self.RA
        self.init_lib()
        sp = self.spec
        F = self.F
        wat = walls_at or (lambda s_, side: sp["lane"]["half"] + 1.5)
        info = dict(sections=[])
        self.closures = []
        # ---- the gym doorway on the left wall (x -23.65 .. -21.5; its open leaf BP_school_door_13 swings into the lane
        #      and is turned off in memory) is BLOCKED by a row of school wardrobes standing against the wall
        gd = sp.get("gym_door_block")
        if gd:
            sa, sb = gd["x"][0] - self.P0[0], gd["x"][1] - self.P0[0]
            wl_ = min(wat(sa - 0.4, 1), wat(sb + 0.4, 1))
            r, parts = self.line(
                "gymdoor_block_",
                "doorway_block",
                (sa, wl_ - gd["depth"] - 0.01),
                (1, 0),
                (0, -1),
                sb - sa,
                gd["kinds"],
                (0.95, 1.08),
                depth_target=gd["depth"],
                rotated_ok=False,
                max_embed=0.06,
            )
            front = wl_ - max(p_[3] for p_ in r["pieces"]) - 0.01
            info["doorway_block"] = dict(
                x=gd["x"], wall_l=round(wl_, 3), front_l=round(front, 3), pieces=r["pieces"], scale=r["scale"]
            )
            wat0 = wat
            wat = lambda s_, side, _w=wat0, _a=sa, _b=sb, _f=front: (
                _f if side > 0 and _a - 0.05 <= s_ <= _b + 0.05 else _w(s_, side)
            )
        self.wat = wat
        s = sp["straight"]
        info["sections"].append(dict(name="start_straight", s=[0.0, s], length=s))
        # ---- CONTROL: bench
        h = sp["hurdle"]
        yb = self.yaw_across(h["kind"], "as_is")
        lat, al, H = self.lat_extent(h["kind"], "as_is", yb, h["scale"])
        c = self.W(s + al / 2)
        self.put("hurdle_bench", h["kind"], c[0], c[1], F, "as_is", yb, h["scale"], section="control")
        info["sections"].append(
            dict(
                name="control_step_over",
                s=[round(s, 3), round(s + al, 3)],
                top=round(H, 4),
                width=round(lat, 3),
                what=f"gym bench {self.lib.catalog['bench']['src']} x{h['scale']}",
            )
        )
        s += al + sp["clear"]
        # ---- LOW + NARROW
        ln = sp["low_narrow"]
        s_l = s
        x_l = self.W(s_l)[0]
        if DOOR_ZONE_X[0] - 0.05 < x_l + 0.45 and x_l < DOOR_ZONE_X[1]:
            print(f"[sc2] WARNING low+narrow line x {x_l:.2f} near the door zone", flush=True)
        inner = []
        for sg in (1, -1):
            p, wdt, dpt = self.single(
                f"ln_jamb_{'L' if sg > 0 else 'R'}",
                "low_narrow",
                ln["cab"],
                ln["cab_scale"],
                (s_l, sg * 0.5 * ln["gap"]),
                (0, sg),
                (-1, 0),
            )
            inner.append(p)
            wall = wat(s_l + 0.2, sg)
            l0 = 0.5 * ln["gap"] + wdt - 0.03
            r, parts = self.line(
                f"ln_closure_{'L' if sg > 0 else 'R'}",
                "low_narrow",
                (s_l, sg * l0),
                (0, sg),
                (-1, 0),
                wall - l0,
                ln["closure"],
                ln["closure_srange"],
                depth_target=dpt,
                max_embed=0.06,
                rotated_ok=False,
            )
        jt = max(float(q["corners"][:, 2].max()) for q in inner)
        yawp = self.yaw + math.pi / 2 + self.lib.flat_yaw(ln["lintel"])
        w = self.W(s_l + 0.5 * dpt)
        lintel = self.seat(
            "ln_lintel", "low_narrow", ln["lintel"], w[0], w[1], jt, "flat", yawp, ln["lintel_scale"], inner
        )
        self.closures.append(
            dict(name="low_narrow", s=s_l, depth=dpt - 0.05, open=[(-0.5 * ln["gap"], 0.5 * ln["gap"])], z=(0.3, 0.8))
        )
        info["sections"].append(
            dict(
                name="low_narrow",
                s=[round(s_l, 3), round(s_l + dpt, 3)],
                gap_design=ln["gap"],
                height_design=round(jt - F, 4),
                jambs=dict(src=self.lib.catalog[ln["cab"]]["src"], scale=ln["cab_scale"]),
                lintel=dict(src=self.lib.catalog[ln["lintel"]]["src"], scale=ln["lintel_scale"]),
            )
        )
        s = s_l + dpt + sp["clear_after_ln"]
        # ---- TUNNEL geometry first (the oil lies on its approach)
        tn = sp["tunnel"]
        o = sp["oil"]
        # ---- OIL: copy of the school's decal, flat on the floor, centred on the approach
        R_, Po = self.lib.orient_matrix("oil", "as_is", 0.0, 1.0)
        ext = np.ptp(Po[:, :2], axis=0)
        yo = self.yaw if ext[0] >= ext[1] else self.yaw + math.pi / 2  # long side along the lane
        yo = self.yaw + (math.pi / 2 if ext[0] >= ext[1] else 0.0)  # SHORT side along the lane
        osc = o.get("scale", RA.SCALE_LIM[0])
        L0 = float(min(ext))
        o_len = L0 * osc  # along the lane
        c = self.W(s + 0.5 * o_len)
        s_t0 = s + o_len + o["gap_to_tunnel"]
        op = self.put("oil_patch", "oil", c[0], c[1], F, "as_is", yo, osc, section="oil")
        from pxr import UsdShade, UsdPhysics, Usd, UsdGeom

        pm = self.stage.GetPrimAtPath(f"{self.lib.HC.GEOM}/Looks/PM_Oil")
        for q in Usd.PrimRange(op["prim"]):
            if q.IsA(UsdGeom.Gprim) and pm.IsValid():  # the school's own oil friction (mu 0.09)
                UsdShade.MaterialBindingAPI.Apply(q).Bind(
                    UsdShade.Material(pm), UsdShade.Tokens.strongerThanDescendants, "physics"
                )
        info["sections"].append(
            dict(
                name="oil",
                s=[round(s, 3), round(s + o_len, 3)],
                what=f"school oil decal {self.lib.catalog['oil']['src']} x{osc:.3f} + PM_Oil",
                size_along_across=[round(o_len, 3), round(float(max(ext)) * osc, 3)],
            )
        )
        # ---- TUNNEL: cabinet rows (fronts inwards) + door-leaf roof + mouth closure
        rows = []
        Lrow = None
        if tn.get("style") == "fallen_lockers":
            # blue school lockers LYING ON THEIR SIDE (long axis along the lane, ~1.0 m high) carry the roof
            yk = self.yaw_along("locker", "on_side")
            lat, al, H = self.lat_extent("locker", "on_side", yk, tn["locker_scale"])
            for sg in (1, -1):
                w = self.W(s_t0 + 0.5 * al, sg * (tn["half_w"] + 0.5 * lat))
                rows.append(
                    self.put_resting(
                        f"tunnel_row_{'L' if sg > 0 else 'R'}",
                        "tunnel",
                        "locker",
                        w[0],
                        w[1],
                        "on_side",
                        yk,
                        tn["locker_scale"],
                    )
                )
            Lrow, dep = al, lat
        else:
            for sg in (1, -1):
                r, parts = self.line(
                    f"tunnel_row_{'L' if sg > 0 else 'R'}",
                    "tunnel",
                    (s_t0, sg * tn["half_w"]),
                    (1, 0),
                    (0, -sg),
                    tn["n_cab"] * self.front_width(tn["cab"]) * tn["cab_scale"] - 0.03 * (tn["n_cab"] - 1),
                    [tn["cab"]],
                    (tn["cab_scale"] - 0.002, tn["cab_scale"] + 0.002),
                    rotated_ok=False,
                )
                rows += parts
                Lrow = r["built_len"]
                dep = max(p_[3] for p_ in r["pieces"])
        s_t1 = s_t0 + Lrow
        Lw, Dw, Tw = self.lib.flat_dims("door_leaf")
        nl = tn["n_leaves"]
        ls = float(np.clip((Lrow - (nl - 1) * SC.GAP) / (nl * Dw), *RA.SCALE_LIM))
        yawl = self.yaw + math.pi / 2 + self.lib.flat_yaw("door_leaf")
        top = max(float(q["corners"][:, 2].max()) for q in rows)
        roof = []
        for il in range(nl):
            sc_ = s_t0 + 0.5 * Lrow + (il - 0.5 * (nl - 1)) * (Dw * ls + SC.GAP)
            w = self.W(sc_)
            roof.append(self.seat(f"tunnel_roof_{il}", "tunnel", "door_leaf", w[0], w[1], top, "flat", yawl, ls, rows))
        l_out = tn["half_w"] + dep
        clos = []
        for sg in (1, -1):
            wall = wat(s_t0 + 0.3, sg)
            span = wall - l_out + 0.04
            if tn.get("style") == "fallen_lockers":  # more fallen lockers side by side, flush with the mouth
                lat1, _, _ = self.lat_extent("locker", "on_side", yk, 1.0)
                best = None
                for nn in range(1, 8):
                    for tgt in (span + 0.03, span, span + 0.06, span + 0.09, span + 0.12):
                        sc_ = tgt / (nn * lat1)
                        if 0.86 <= sc_ <= 1.15 and (best is None or abs(sc_ - 1.05) < abs(best[1] - 1.05)):
                            best = (nn, sc_)
                if best is None:
                    raise RuntimeError(f"tunnel closure: no locker count for span {span:.3f}")
                nn, sc_ = best
                latc, alc, Hc = self.lat_extent("locker", "on_side", yk, sc_)
                for i in range(nn):
                    w = self.W(s_t0 + 0.5 * alc, sg * (l_out - 0.04 + (i + 0.5) * latc))
                    self.put_resting(
                        f"tunnel_closure_{'L' if sg > 0 else 'R'}{i}",
                        "tunnel",
                        "locker",
                        w[0],
                        w[1],
                        "on_side",
                        yk,
                        sc_,
                    )
                clos.append(
                    dict(
                        side="L" if sg > 0 else "R",
                        wall=round(wall, 3),
                        n=nn,
                        scale=round(sc_, 4),
                        height=round(Hc, 3),
                        built=round(nn * latc, 3),
                        what="school lockers lying on their side",
                    )
                )
            else:
                r, parts = self.line(
                    f"tunnel_closure_{'L' if sg > 0 else 'R'}",
                    "tunnel",
                    (s_t0, sg * (l_out + 0.005)),
                    (0, sg),
                    (-1, 0),
                    span,
                    tn["closure"],
                    tn["closure_srange"],
                    depth_target=0.36,
                    max_embed=0.12,
                )
                clos.append(
                    dict(
                        side="L" if sg > 0 else "R",
                        wall=round(wall, 3),
                        pieces=r["pieces"],
                        scale=r["scale"],
                        built=r["built_len"],
                    )
                )
        self.closures.append(
            dict(name="tunnel", s=s_t0, depth=0.30, open=[(-tn["half_w"], tn["half_w"])], z=(0.3, 0.8))
        )
        info["sections"].append(
            dict(
                name="tunnel",
                s=[round(s_t0, 3), round(s_t1, 3)],
                length=round(Lrow, 3),
                clear_width=round(2 * tn["half_w"], 3),
                roof=dict(src=self.lib.catalog["door_leaf"]["src"], n=nl, scale=round(ls, 4)),
                rows=dict(
                    src=self.lib.catalog["locker" if tn.get("style") == "fallen_lockers" else tn["cab"]]["src"],
                    scale=tn.get("locker_scale") if tn.get("style") == "fallen_lockers" else tn["cab_scale"],
                    depth=round(dep, 3),
                    top=round(top - F, 4),
                ),
                closure=clos,
            )
        )
        s = max(s_t1 + sp["clear"], sp["sidle"]["s0_min"])
        # ---- SIDLE STRIP: two locker rows (fronts facing the strip) + entrance closure to both walls
        sd = sp["sidle"]
        s_s = s
        rows2, dep2 = [], 0.0
        for sg in (1, -1):
            r, parts = self.line(
                f"sidle_row_{'L' if sg > 0 else 'R'}",
                "sidle",
                (s_s, sg * 0.5 * sd["gap"]),
                (1, 0),
                (0, -sg),
                sd["length"],
                sd["kinds"],
                sd["srange"],
                rotated_ok=False,
                max_embed=0.06,
            )
            rows2 += parts
            dep2 = max(dep2, max(p_[3] for p_ in r["pieces"]))
            Ls = r["built_len"]
        l_o2 = 0.5 * sd["gap"] + dep2
        for sg in (1, -1):
            wall = wat(s_s + 0.35, sg)
            r, parts = self.line(
                f"sidle_closure_{'L' if sg > 0 else 'R'}",
                "sidle",
                (s_s, sg * (l_o2 + 0.005)),
                (0, sg),
                (-1, 0),
                wall - l_o2 - 0.005,
                sd["kinds"],
                sd["srange"],
                depth_target=0.72,
                max_embed=0.10,
            )
        self.closures.append(
            dict(name="sidle", s=s_s, depth=0.6, open=[(-0.5 * sd["gap"], 0.5 * sd["gap"])], z=(0.3, 0.9, 1.6))
        )
        info["sections"].append(
            dict(
                name="sidle_strip",
                s=[round(s_s, 3), round(s_s + Ls, 3)],
                gap_design=sd["gap"],
                length=round(Ls, 3),
                what="school wardrobes " + ", ".join(self.lib.catalog[k]['src'] for k in sd['kinds']),
            )
        )
        s_goal = s_s + Ls + sp["goal_after"]
        self.s_goal = s_goal
        gx = self.W(s_goal)[0]
        if gx > D1_X0 - 1.0:
            print(f"[sc2] WARNING goal x {gx:.2f} within 1 m of the D1 rectangle", flush=True)
        path = [self.W(0.0), self.W(s_goal)]
        mids = []
        for sec in info["sections"][1:]:
            mids += [self.W(sec["s"][0] - 0.3), self.W(sec["s"][1] + 0.3)]
        path = [path[0]] + mids + [path[1]]
        info["task"] = dict(
            start=dict(xy=np.round(self.W(0)[:2], 3).tolist(), z=round(F, 3), yaw=round(self.yaw, 4)),
            goal=dict(xy=np.round(self.W(s_goal)[:2], 3).tolist(), z=round(F, 3), success_radius=0.5),
            path=[np.round(p, 3).tolist() for p in path],
            length=round(s_goal, 3),
            skills=dict(
                control_step_over="step over 0.255 m (PHP can)",
                tunnel="crouch-walk ~2.2 m under 0.99 m after a slippery oil patch (slow down)",
                low_narrow="crouch + sidestep through 0.45 x 0.99 m",
                sidle="strafe 2.9 m in a 0.39 m slot",
            ),
        )
        info["n_parts"] = len(self.parts)
        self.info = info
        return info

    # ---------------------------------------------------------------- verify
    def verify(self, pxq, carb):
        from pxr import Usd, UsdGeom, UsdShade

        RA = self.RA
        res = dict(parts=len(self.parts), support=[], fail=[], max_gap=0.0)
        # (1) support (same test as v1)
        for p in self.parts:
            if p["section"] == "oil":  # the school's oil decal is a flat textured Cube (as shipped)
                zb = float(p["corners"][:, 2].min())
                hs = [
                    self.ray(pxq, carb, np.r_[q[:2], zb + 0.05], (0, 0, -1), 0.2, skip=p["path"])
                    for q in p["corners"][:4]
                ]
                gaps = [round(h["d"] - 0.05, 4) for h in hs if h]
                ok = len(gaps) == 4 and max(gaps) <= 0.003
                res["support"].append(
                    dict(
                        part=p["role"],
                        z_min_above_floor=round(zb - self.F, 4),
                        corner_gaps=gaps,
                        ok=ok,
                        note="flat floor decal (school Oil_surface, a textured Cube as shipped)",
                    )
                )
                if not ok:
                    res["fail"].append(dict(why="oil decal not on the floor", part=p["role"], gaps=gaps))
                continue
            A = RA.world_points(p["prim"])
            zmin = float(A[:, 2].min())
            band = A[A[:, 2] <= zmin + 0.004]
            if p.get("rest_on"):
                import gym_course_v3 as GV

                T = GV.world_tris(p["prim"])
                bb = []
                for rr in p["rest_on"]:
                    q_ = next(x for x in self.parts if x["role"] == rr)
                    for x, y in self.footprint(q_, 40, 10, margin=0.003):
                        z = GV.lowest_hit(T, x, y)
                        if z is not None:
                            bb.append((x, y, z))
                band = np.array(bb) if bb else band
            if len(band) > 600:
                band = band[np.random.default_rng(0).choice(len(band), 600, replace=False)]
            sup, gaps = [], []
            for q in band:
                h = self.ray(pxq, carb, q + np.array([0, 0, 0.015]), (0, 0, -1), 0.06, skip=p["path"])
                if h is None:
                    continue
                gap = h["d"] - 0.015
                if -0.01 <= gap <= 0.005:
                    sup.append(q[:2])
                    gaps.append(gap)
            com = A.mean(0)[:2]
            Hh = SC._hull(np.array(sup)) if sup else np.zeros((0, 2))
            dcom = SC._dist_to_hull(Hh, com) if sup else 1e9
            ok = len(sup) >= 3 and dcom <= 0.02
            r = dict(
                part=p["role"],
                n_band=int(len(band)),
                n_supported=len(sup),
                max_gap=round(max(gaps), 4) if gaps else None,
                com_out_of_support_hull=round(dcom, 4) if dcom < 1e8 else None,
                z_min_above_floor=round(zmin - self.F, 4),
                ok=ok,
            )
            res["support"].append(r)
            if not ok:
                res["fail"].append(dict(why="not supported", **r))
            elif gaps:
                res["max_gap"] = max(res["max_gap"], round(max(gaps), 4))
        # (2) walking surface along the path (overhead roof / lintel skipped): visible floor mesh, oil decal or course part
        over = [p["path"] for p in self.parts if p["role"].startswith(("tunnel_roof", "ln_lintel"))]
        floor = dict(n=0, course_part=0, school_floor=0, bad=[], prims={})
        prof = []
        for l in (-0.12, 0.0, 0.12):
            for s in np.arange(-0.8, self.s_goal + 0.6, 0.05):
                o = self.W(s, l, 2.4)
                h = self.ray(pxq, carb, o, (0, 0, -1), 4.0, skip=over)
                floor["n"] += 1
                if h is None:
                    floor["bad"].append(dict(s=round(float(s), 2), l=l, why="no hit"))
                    continue
                z = float(h["p"][2] - self.F)
                if l == 0.0:
                    prof.append((round(float(s), 3), round(z, 4), h["c"][-50:]))
                if h["c"].startswith(self.root):
                    floor["course_part"] += 1
                    continue
                pr = self.stage.GetPrimAtPath(h["c"])
                if pr.IsValid() and not pr.IsA(UsdGeom.Mesh):
                    ms = [
                        q
                        for q in Usd.PrimRange(pr, Usd.TraverseInstanceProxies())
                        if q.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(q).ComputeVisibility() != UsdGeom.Tokens.invisible
                    ]
                    if ms:
                        pr = ms[0]
                vis = pr.IsValid() and UsdGeom.Imageable(pr).ComputeVisibility() != UsdGeom.Tokens.invisible
                mat = UsdShade.MaterialBindingAPI(pr).ComputeBoundMaterial()[0] if pr.IsValid() else None
                ok = (
                    pr.IsValid()
                    and pr.IsA(UsdGeom.Mesh)
                    and vis
                    and bool(mat)
                    and "GroundPlane" not in h["c"]
                    and abs(z) < 0.03
                )
                floor["prims"][h["c"]] = floor["prims"].get(h["c"], 0) + 1
                if ok:
                    floor["school_floor"] += 1
                else:
                    floor["bad"].append(dict(s=round(float(s), 2), l=l, z=round(z, 4), hit=h["c"]))
        floor["bad"] = floor["bad"][:40]
        floor["ok"] = not floor["bad"]
        res["floor"] = floor
        res["profile_centre"] = prof
        # (3) measured geometry
        secs = {x["name"]: x for x in self.info["sections"]}
        meas = {}
        t0, t1 = secs["tunnel"]["s"]
        ups, cov = [], []
        for s in np.arange(t0 - 0.3, t1 + 0.3, 0.02):
            hz = []
            for l in (-0.4, 0.0, 0.4):
                h = self.ray(pxq, carb, self.W(s, l, 0.05), (0, 0, 1), 3.0)
                hz.append(None if h is None else float(h["p"][2] - self.F))
            roofed = all(z is not None and z < 1.3 for z in hz)
            if roofed:
                cov.append(s)
                ups += hz
        meas["tunnel_clear_height"] = dict(min=round(min(ups), 4), max=round(max(ups), 4)) if ups else None
        meas["tunnel_roof_length"] = round(max(cov) - min(cov) + 0.02, 3) if cov else None
        wv = []
        for s in np.linspace(t0 + 0.1, t1 - 0.1, 5):
            for z in (0.3, 0.7):
                hp = self.ray(pxq, carb, self.W(s, 0, z), np.r_[self.n, 0], 3.0)
                hm = self.ray(pxq, carb, self.W(s, 0, z), np.r_[-self.n, 0], 3.0)
                if hp and hm:
                    wv.append(round(hp["d"] + hm["d"], 4))
        meas["tunnel_clear_width"] = dict(min=min(wv), max=max(wv)) if wv else None
        l0, l1 = secs["low_narrow"]["s"]
        hs = [self.ray(pxq, carb, self.W(sm, 0, 0.05), (0, 0, 1), 3.0) for sm in np.linspace(l0 + 0.05, l1 - 0.05, 5)]
        meas["low_narrow_height"] = dict(
            min=round(min(float(h["p"][2] - self.F) for h in hs if h), 4),
            max=round(max(float(h["p"][2] - self.F) for h in hs if h), 4),
        )
        gw = []
        for sm in np.linspace(l0 + 0.05, l1 - 0.05, 4):
            for z in (0.15, 0.4, 0.7):
                hp = self.ray(pxq, carb, self.W(sm, 0, z), np.r_[self.n, 0], 3.0)
                hm = self.ray(pxq, carb, self.W(sm, 0, z), np.r_[-self.n, 0], 3.0)
                if hp and hm:
                    gw.append(round(hp["d"] + hm["d"], 4))
        meas["low_narrow_width"] = dict(min=min(gw), max=max(gw), n=len(gw)) if gw else None
        s0, s1 = secs["sidle_strip"]["s"]
        sw, inside = [], []
        for sm in np.arange(s0 + 0.05, s1 - 0.04, 0.1):
            ws = []
            for z in (0.2, 0.6, 1.0, 1.4, 1.8):
                hp = self.ray(pxq, carb, self.W(sm, 0, z), np.r_[self.n, 0], 3.0)
                hm = self.ray(pxq, carb, self.W(sm, 0, z), np.r_[-self.n, 0], 3.0)
                if hp and hm:
                    ws.append(hp["d"] + hm["d"])
            if ws:
                sw.append(min(ws))
                if max(ws) < 0.6:
                    inside.append(sm)
        meas["sidle_width"] = dict(min=round(min(sw), 4), max=round(max(sw), 4)) if sw else None
        meas["sidle_length"] = round(max(inside) - min(inside) + 0.1, 3) if inside else None
        # (4) full-width closure: rays ALONG the lane through each closure line at lateral steps of 0.05 m, wall to wall
        clos_res = []
        for c in self.closures:
            hwl = self.wall_dist(pxq, carb, c["s"] - 0.15, 0.6, 1) or 3.5
            hwr = self.wall_dist(pxq, carb, c["s"] - 0.15, 0.6, -1) or 3.5
            ls_ = np.arange(-hwr + 0.02, hwl - 0.02, 0.05)
            free = np.ones(len(ls_), bool)
            per_z = {}
            for z in c["z"]:
                fz = np.array(
                    [
                        self.ray(pxq, carb, self.W(c["s"] - 0.15, l, z), np.r_[self.d, 0], c["depth"] + 0.25) is None
                        for l in ls_
                    ]
                )
                per_z[z] = int(fz.sum())
                free &= fz  # a PASSAGE must be free from the floor up
            openings, cur = [], None
            for l, f_ in zip(ls_, free):
                if f_ and cur is None:
                    cur = [l, l]
                elif f_:
                    cur[1] = l
                elif cur is not None:
                    openings.append((round(cur[0], 3), round(cur[1], 3)))
                    cur = None
            if cur is not None:
                openings.append((round(cur[0], 3), round(cur[1], 3)))
            bypass = []
            for a_, b_ in openings:
                designed = any(a_ >= lo - 0.2 and b_ <= hi + 0.2 for lo, hi in c["open"])
                if not designed and (b_ - a_ + 0.05) > 0.2:
                    bypass.append(dict(l=[a_, b_], width=round(b_ - a_ + 0.05, 3)))
            clos_res.append(
                dict(
                    name=c["name"],
                    s=round(c["s"], 3),
                    walls=[round(hwl, 3), round(-hwr, 3)],
                    openings=openings,
                    bypass=bypass,
                    ok=not bypass,
                )
            )
            if bypass:
                res["fail"].append(dict(why="closure bypass", name=c["name"], bypass=bypass))
        gd = self.info.get("doorway_block")
        if gd:  # the blocked gym doorway: lateral rays must hit the wardrobes
            sa, sb = gd["x"][0] - self.P0[0], gd["x"][1] - self.P0[0]
            miss = []
            for s_ in np.arange(sa + 0.03, sb - 0.02, 0.05):
                for z in (0.3, 0.9, 1.5):
                    h = self.ray(pxq, carb, self.W(s_, 1.2, z), np.r_[self.n, 0], 3.0)
                    if h is None or not h["c"].startswith(self.root):
                        miss.append((round(float(s_), 2), z, None if h is None else h["c"][-40:]))
            clos_res.append(dict(name="gym_doorway_block", ok=not miss, miss=miss[:10]))
            if miss:
                res["fail"].append(dict(why="gym doorway not blocked", miss=miss[:10]))
        meas["closure"] = clos_res
        res["measured"] = meas
        oil = [p["path"] for p in self.parts if p["section"] == "oil"]
        res["audit"] = RA.audit_no_primitives(self.stage, [self.root], skip=tuple(oil))
        res["audit_oil_decal"] = RA.audit_no_primitives(self.stage, oil)
        res["audit_note"] = (
            "the oil patch is a copy of the school's own shipped oil decal Oil_surface_02 (UsdGeom.Cube, ~1 mm, "
            "its own photographic material Oil_surface_M + PM_Oil); audited separately, everything else is Mesh"
        )
        res["ok"] = (not res["fail"]) and floor["ok"] and res["audit"]["ok"]
        return res
