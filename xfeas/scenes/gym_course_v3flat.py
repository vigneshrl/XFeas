#!/usr/bin/env python
"""
GYM OBSTACLE COURSE "v3flat" (additive 2026-10-19; v1 gym_course.py, v2 gym_course_v2.py and v3 gym_course_v3.py are NOT
changed). v3flat = v3 with a FLAT lane:
  * NO entry ramp and NO humps / valleys: those parts are simply not placed (put() skips the roles 'ramp' and 'x<i>_*').
    The deck under the former humps / valleys is v3's own flat deck (desk-top layers), unchanged.
  * NO step at the entrance: the deck is extended BACKWARDS under the start with the same 6-layer stack of school desk
    tops (3 more boards per layer, same scale / gap / material), so the robot spawns ON the deck 0.25 m from its rear
    edge (rear edge 7 cm from the gym's west wall, ray-checked) and walks a continuous flat deck to the foot of the stairs.
  * Kept, in order, at v3's exact x positions: the crouch BEAM (4 real planks SM_literature_props_1 on 8 desk-apron-board
    pillars standing on the deck), the 0.36 m FUNNEL (two upright desk tops on deck posts), the stairs + platform
    (4 risers 0.17 m, unchanged), the fences.
  * Beam variants: underside u (make_spec_v3flat(u=...)): A = 1.10 m (crouch-required hurdle for AMO, D1g height),
    B = 1.24 m (= v3).
Everything is built IN MEMORY in assets/school.usd (never written). Real school assets only (RealLib copies), every part
supported (PhysX contact rays in verify()).

  spec = make_spec_v3flat(seed=0, u=1.10, gap=0.36); B = GymCourseV3Flat(stage); info = B.build(spec); world.reset()
  ver = B.verify(pxq, carb)   # v1 + v3 checks (beam / pillars / funnel) + ver["flat"] (lane flatness, rear edge / wall, overlaps)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import copy, json, math, os, re, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import gym_course as GC
import gym_course_v3 as GV3

VERSION = "3flat"
VARIANT = "v3flat_beam_funnel_flat_lane"
OUT_DIR = xfeas.paths.COURSES + "/gym_obstacle_course/v3flat"
SKIP = re.compile(r"^(ramp$|x\d+_)")
START_S = -1.2  # robot start (lane frame), as v1/v2/v3
REAR_MARGIN = 0.25  # deck reaches >= this far behind the start


def make_spec_v3flat(seed=0, u=1.10, gap=0.36):
    sp = GV3.make_spec_v3(seed, gap=gap)
    sp["crouch"]["u"] = float(u)
    sp["version"] = VERSION
    sp["variant"] = VARIANT
    sp["flat_lane"] = dict(
        skip_roles="ramp, x<i>_* (humps / valleys)", deck_rear_extension="3 boards per layer", start_s=START_S
    )
    return sp


class GymCourseV3Flat(GV3.GymCourseV3):
    def put(self, role, *aa, **kk):
        if getattr(self, "_flat_skip", False) and SKIP.match(role):
            self.skipped.append(role)
            return dict(role=role, path="", corners=np.zeros((8, 3)))
        return GV3.GymCourseV3.put(self, role, *aa, **kk)

    def build(self, spec):
        self.skipped = []
        self._flat_skip = True
        try:
            info = GV3.GymCourseV3.build(self, spec)
        finally:
            self._flat_skip = False
        L = self.L
        dk = L["deck_top"]
        t = L["t"]
        BD = L["BD"]
        yaw = spec["yaw"]
        bs = spec["board_scale"]
        RL = spec["riser_layers"]
        byaw = yaw + math.pi / 2 + self.lib.flat_yaw("board")
        # ---- deck extension backwards (same board, scale, gap, layer stack as the v1 terrace deck)
        n_f = int(math.ceil((L["s_d"] - (START_S - REAR_MARGIN)) / (BD + GC.GAP) - 1e-6))
        front = []
        for jl in range(RL):
            s_lo = L["s_d"] + (GC.HEAD_SEAT if jl == RL - 1 else 0.0)  # front edge of the v1 deck board of this layer
            for i in range(n_f):
                sc = s_lo - GC.GAP - BD / 2 - i * (BD + GC.GAP)
                w = self.W(sc, 0.0, jl * t)
                p = GV3.GymCourseV3.put(
                    self,
                    f"deck_L{jl}_rear{i}",
                    "board",
                    bs,
                    byaw,
                    center=w[:2],
                    z_bottom=w[2],
                    orient="flat",
                    contacts=[("down", "flat")],
                )
                front.append(p)
        s_rear = L["s_d"] - n_f * (BD + GC.GAP)
        info["deck_rear"] = dict(
            n_boards_per_layer=n_f,
            layers=RL,
            s_rear_edge=round(s_rear, 4),
            x_rear_edge=round(float(self.W(s_rear, 0)[0]), 4),
            start_s=START_S,
        )
        self.front_parts = front
        info["removed_roles"] = sorted(set(self.skipped))
        info["n_removed"] = len(self.skipped)
        info.pop("ramp", None)
        info["crossing_removed"] = info.get("crossing", [])
        info["crossing"] = []
        # ---- robot task: start ON the deck, flat path to the stairs foot, then the v1 stairs
        rb = info["robot"]
        P = [np.asarray(p_, float) for p_ in rb["path"]]
        s_door0, s_door1 = self.v2["s_door"]
        s_o, s_o1 = self.v2["s_funnel"]
        st_w = self.W(START_S, 0.0, dk)
        P_flat = [st_w] + [p_ for p_ in P if p_[0] >= float(self.W(s_door0 - 0.6, 0)[0])]
        rb["path"] = [np.round(p_, 3).tolist() for p_ in P_flat]
        rb["start"] = [round(float(st_w[0]), 3), round(float(st_w[1]), 3), round(yaw, 4)]
        rb["start_z"] = round(float(st_w[2]), 4)
        X = lambda s: round(float(self.W(s, 0)[0]), 3)
        secs = [
            ("deck", START_S, s_door0),
            ("beam", s_door0, s_door1),
            ("deck", s_door1, s_o),
            ("funnel", s_o, s_o1),
            ("deck", s_o1, L["s_st"]),
            ("stairs", L["s_st"], L["s_p0"]),
            ("platform", L["s_p0"], self.s_end),
        ]
        rb["sections"] = [dict(name=n_, s=[round(a, 4), round(b, 4)], x=[X(a), X(b)]) for n_, a, b in secs]
        rb["course_version"] = VERSION
        rb["variant"] = VARIANT
        info["n_parts"] = len(self.parts)
        info["version"] = VERSION
        info["variant"] = VARIANT
        self.info = info
        return info

    def verify(self, pxq, carb):
        res = GV3.GymCourseV3.verify(self, pxq, carb)
        L = self.L
        dk = L["deck_top"]
        door = self.v2["door"]
        out = dict(fail=[], checks=0)

        def ray(o, v, Lr):
            h = pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, v)), float(Lr))
            if not (h and h.get("hit")):
                return None
            return dict(d=float(h["distance"]), c=str(h.get("collision", "")), p=np.asarray(h["position"], float))

        # 1. walking-surface profile along the whole lane (start - 0.2 m ... stairs foot), down-rays from 0.6 m above the
        #    deck (below the beam, above the wing posts' feet), lane centre and +-0.12 m (inside the 0.36 m funnel throat)
        rows = []
        s_rear = self.info["deck_rear"]["s_rear_edge"]
        for l in (-0.12, 0.0, 0.12):
            for s in np.arange(s_rear + 0.02, L["s_st"] - 0.02, 0.01):
                h = ray(self.W(s, l, dk + 0.6), (0, 0, -1), 1.2)
                z = None if h is None else float(h["p"][2] - GC.FLOOR_Z)
                rows.append(
                    (round(float(s), 3), l, None if z is None else round(z, 4), "" if h is None else h["c"][-40:])
                )
        # reference = the MEASURED walking surface (median; the nominal layer stack dk is ~3 mm above the composed meshes)
        z_ref = float(np.median([r[2] for r in rows if r[2] is not None]))
        Z = np.array([r[2] for r in rows if r[2] is not None and abs(r[2] - z_ref) <= 0.01])
        out["checks"] += 1
        flat_ok = len(Z) > 0.97 * len(rows) and float(np.abs(Z - z_ref).max()) <= 0.003
        if not flat_ok:
            out["fail"].append(
                dict(
                    check="lane_flat",
                    n=len(rows),
                    n_on_deck=len(Z),
                    max_dev=round(float(np.abs(Z - z_ref).max()), 4) if len(Z) else None,
                )
            )
        out["lane_profile"] = dict(
            n_rays=len(rows),
            n_on_deck_top=int(len(Z)),
            n_seam_or_other=int(len(rows) - len(Z)),
            deck_top_nominal=round(dk, 4),
            walk_z_median=round(z_ref, 4),
            z_min=round(float(Z.min()), 4),
            z_max=round(float(Z.max()), 4),
            max_abs_dev_from_median=round(float(np.abs(Z - z_ref).max()), 4),
            s_range=[round(s_rear + 0.02, 3), round(L["s_st"] - 0.02, 3)],
            seams_note="down-rays falling through the 2 mm gaps between desk tops (as on the v1-v3 deck)",
        )
        # 2. start pose on the deck: surface under the start footprint (+-0.25 m along, +-0.2 m across)
        zs = []
        for ds in np.linspace(-0.2, 0.2, 9):
            for dl in np.linspace(-0.2, 0.2, 5):
                h = ray(self.W(START_S + ds, dl, dk + 0.6), (0, 0, -1), 1.2)
                zs.append(None if h is None else round(float(h["p"][2] - GC.FLOOR_Z), 4))
        out["start_footprint_z"] = dict(
            min=min(z for z in zs if z is not None), max=max(z for z in zs if z is not None), n=len(zs)
        )
        # 3. rear edge vs the west wall: horizontal rays from 1 mm behind the rear edge, at every layer, 5 lateral offsets
        dw = []
        for jl in range(self.spec_rl):
            for l in np.linspace(-0.6, 0.6, 5):
                z = (jl + 0.5) * L["t"]
                h = ray(self.W(s_rear - 0.001, l, z), np.r_[-self.d, 0.0], 3.0)
                dw.append(None if h is None else round(h["d"] + 0.001, 4))
        out["checks"] += 1
        okw = all(d is not None and d > 0.01 for d in dw)
        if not okw:
            out["fail"].append(dict(check="rear_edge_wall_clearance", d=dw))
        out["rear_edge_to_wall_m"] = dict(
            min=min(d for d in dw if d is not None) if any(d is not None for d in dw) else None, n=len(dw)
        )
        # 4. overlap of every added rear board (box shrunk 3 mm) with anything that is not this board
        ov = []
        for p in self.front_parts:
            P = p["corners"]
            c = P.mean(0)
            he = 0.5 * np.ptp(P, axis=0) - 0.003
            hits = []

            def rep(hit, _h=hits, _self=p["path"]):
                cp = str(hit.collision)
                if not cp.startswith(_self):
                    _h.append(cp)
                return True

            pxq.overlap_box(
                carb.Float3(*map(float, he)), carb.Float3(*map(float, c)), carb.Float4(0, 0, 0, 1), rep, False
            )
            out["checks"] += 1
            if hits:
                ov.append(dict(part=p["role"], hits=hits[:4]))
        if ov:
            out["fail"].append(dict(check="rear_board_overlap", parts=ov))
        out["rear_board_overlaps"] = len(ov)
        out["removed_roles"] = self.info["removed_roles"]
        res["flat"] = out
        res["ok"] = res["ok"] and not out["fail"]
        return res

    @property
    def spec_rl(self):
        return int(round(self.L["deck_top"] / self.L["t"]))


def course_spec_flat(u=1.10, seed=0, gap=0.36):
    return make_spec_v3flat(seed, u=u, gap=gap), GymCourseV3Flat


if __name__ == "__main__":
    for u in (1.10, 1.24):
        sp = make_spec_v3flat(0, u=u)
        print(json.dumps(dict(u=u, crouch=sp["crouch"], funnel=sp["funnel"], flat_a=sp["flat_a"])))
